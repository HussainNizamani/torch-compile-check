"""Tests for ``action/run.sh``, the composite action's "Run torch-compile-check" step.

The step body lives in a file rather than inline in ``action/action.yml`` so it
can be executed here, which is the only way to test the thing CI actually runs:
a copy of the block pasted into a test drifts from the block, and the drift is
invisible until a workflow breaks. Every test below runs *that file* with the
same environment the YAML's ``env:`` block gives it, against the real CLI.

What they are mostly about is the M4-1 verifier's finding. Under ``set -e`` the
stage-parsing pipeline aborted the whole step on any target whose output carried
neither the "first diverges at" nor the "clean:" marker -- which is every tool
error -- so a single bad target produced no summary rows, no ``exit-code``
output, and cancelled every target after it. The three shapes named in that
report (``budget: abc``, an unknown ``--fail-on`` category, a missing target
followed by a valid one) each have a test here, and each asserts the three
things that were lost: a row per target, the later target still running, and the
outputs written.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
ACTION_DIR = REPO_ROOT / "action"
RUN_SH = ACTION_DIR / "run.sh"
ACTION_YML = ACTION_DIR / "action.yml"

# The cheapest pair of lanes that still runs the oracles end to end: aot_eager
# traces and re-runs the model without paying for inductor's codegen, and every
# assertion here is about the shell around the CLI rather than about a finding.
BACKENDS = "eager,aot_eager"

MLP = FIXTURES / "mlp.py"
GRAPH_BREAK = FIXTURES / "graph_break.py"

# action.yml's `env:` block for this step, one entry per input. The defaults
# below are that block's values when every input is left alone, so a test only
# has to name what it is changing.
DEFAULT_ENV = {
    "TARGETS": "",
    "BACKENDS": BACKENDS,
    "FAIL_ON": "numerics,alias,metadata,grad",
    "BASELINE": "",
    "WRITE_BASELINE": "",
    "MINIMIZE": "false",
    "BUDGET": "",
    "CACHE": "false",
    "JSON_OUT": "torch-compile-check-results.json",
    "EXTRA_ARGS": "",
    "ALLOW_UNIMPLEMENTED": "false",
}


class Run:
    """One execution of ``run.sh``: its exit code, job summary, and outputs."""

    def __init__(self, code: int, log: str, summary: str, outputs: dict[str, str]) -> None:
        self.code = code
        self.log = log
        self.summary = summary
        self.outputs = outputs

    @property
    def rows(self) -> list[str]:
        """The Markdown table rows, without the two header lines."""
        return [
            line
            for line in self.summary.splitlines()
            if line.startswith("|") and not line.startswith("|---")
        ][1:]

    def row(self, target: str) -> str:
        """The one row for ``target``, so a test names what it is asserting about."""
        matches = [line for line in self.rows if line.startswith(f"| `{target}` |")]
        assert len(matches) == 1, f"expected exactly one row for {target}, got {matches}"
        return matches[0]


def run_step(tmp_path: Path, **env: str) -> Run:
    """Run ``action/run.sh`` the way the composite action's step runs it."""
    summary = tmp_path / "step-summary.md"
    outputs = tmp_path / "step-output.txt"
    summary.touch()
    outputs.touch()

    environment = dict(os.environ)
    # The console script the step calls is installed beside this interpreter,
    # which is not necessarily on PATH when pytest was invoked by absolute path.
    environment["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), environment.get("PATH", "")]
    )
    environment.update(DEFAULT_ENV)
    environment.update(env)
    environment["GITHUB_STEP_SUMMARY"] = str(summary)
    environment["GITHUB_OUTPUT"] = str(outputs)
    environment["GITHUB_ACTION_PATH"] = str(ACTION_DIR)

    completed = subprocess.run(
        ["bash", str(RUN_SH)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env=environment,
    )
    parsed = dict(line.split("=", 1) for line in outputs.read_text().splitlines() if "=" in line)
    return Run(
        code=completed.returncode,
        log=completed.stdout + completed.stderr,
        summary=summary.read_text(),
        outputs=parsed,
    )


def test_a_clean_target_is_a_green_step_with_a_row_and_both_outputs(tmp_path):
    result = run_step(tmp_path, TARGETS=str(MLP), JSON_OUT="results.json")

    assert result.code == 0
    assert result.row(str(MLP)).startswith(f"| `{MLP}` | 0 | exit 0 |")
    assert "clean:" in result.row(str(MLP))
    assert result.outputs == {"exit-code": "0", "json-path": "results.json"}
    # One JSON per target, suffixed by index, as docs/action.md promises.
    assert (tmp_path / "results.1.json").is_file()


def test_a_missing_target_gets_a_row_and_the_next_target_still_runs(tmp_path):
    # The M4-1 verifier's headline case: before the fix the step aborted on the
    # first target's tool error, so the valid second target was never checked.
    result = run_step(tmp_path, TARGETS=f"no/such/file.py\n{MLP}")

    assert result.code == 2
    assert len(result.rows) == 2
    assert "| 2 | tool error: no such file: no/such/file.py |" in result.row("no/such/file.py")
    assert result.row(str(MLP)).startswith(f"| `{MLP}` | 0 | exit 0 |")
    # The worst code across the targets, not the last one's.
    assert result.outputs["exit-code"] == "2"
    assert "json-path" in result.outputs


def test_an_unparsable_budget_still_leaves_every_target_with_a_row(tmp_path):
    result = run_step(tmp_path, TARGETS=f"{MLP}\n{GRAPH_BREAK}", BUDGET="abc")

    assert result.code == 2
    assert len(result.rows) == 2
    for target in (MLP, GRAPH_BREAK):
        row = result.row(str(target))
        assert "| 2 | tool error: argument --budget: invalid float value: 'abc' |" in row
    assert result.outputs["exit-code"] == "2"


def test_an_unknown_fail_on_category_still_leaves_every_target_with_a_row(tmp_path):
    result = run_step(tmp_path, TARGETS=f"{MLP}\n{GRAPH_BREAK}", FAIL_ON="bogus")

    assert result.code == 2
    assert len(result.rows) == 2
    for target in (MLP, GRAPH_BREAK):
        assert "| 2 | tool error: unknown --fail-on category 'bogus';" in result.row(str(target))
    assert result.outputs["exit-code"] == "2"


def test_a_stale_report_is_not_read_as_a_failed_target_s_graph_health(tmp_path):
    # The graph-break cell comes out of the JSON the run writes. A file an
    # earlier invocation left at the same path would give a target that never
    # ran a break count it never measured, so the step deletes it first.
    stale = tmp_path / "results.1.json"
    stale.write_text(
        '{"schema_version": 2, "backends": [{"backend": "inductor", "reference": false, '
        '"graph": {"measured": true, "break_count": 7}}]}'
    )

    result = run_step(tmp_path, TARGETS="no/such/file.py", JSON_OUT="results.json")

    assert result.code == 2
    assert result.row("no/such/file.py").split("|")[4].strip() == "-"
    assert not stale.exists()


def test_a_boolean_input_that_is_neither_true_nor_false_is_refused(tmp_path):
    result = run_step(tmp_path, TARGETS=str(MLP), CACHE="yes")

    assert result.code == 2
    assert '::error::input cache must be "true" or "false", got "yes"' in result.log
    # Refused before any target ran, and the outputs are still written: a job
    # with continue-on-error reads exit-code to decide what to do next.
    assert result.rows == []
    assert result.outputs["exit-code"] == "2"


@pytest.mark.parametrize("name", ["MINIMIZE", "CACHE", "ALLOW_UNIMPLEMENTED"])
def test_every_boolean_input_accepts_exactly_true_and_false(tmp_path, name):
    # docs/action.md and action/README.md say "true"/"false" and nothing else;
    # this is that sentence, executed.
    refused = run_step(tmp_path, TARGETS=str(MLP), **{name: "1"})
    assert refused.code == 2
    assert 'must be "true" or "false", got "1"' in refused.log

    for value in ("true", "false"):
        # Not a real run: the point is that the value is admitted, and the
        # cheapest way to see that is the next check the step makes.
        admitted = run_step(
            tmp_path,
            TARGETS=f"{MLP}\n{GRAPH_BREAK}",
            WRITE_BASELINE="baseline.json",
            **{name: value},
        )
        assert admitted.code == 2
        assert "write-baseline takes a single target" in admitted.log


def test_write_baseline_refuses_more_than_one_target(tmp_path):
    result = run_step(
        tmp_path, TARGETS=f"{MLP}\n{GRAPH_BREAK}", WRITE_BASELINE=str(tmp_path / "b.json")
    )

    assert result.code == 2
    assert "::error::write-baseline takes a single target" in result.log
    assert result.outputs["exit-code"] == "2"


def test_an_empty_targets_input_is_a_tool_error_rather_than_an_empty_green_table(tmp_path):
    result = run_step(tmp_path, TARGETS="\n   \n")

    assert result.code == 2
    assert "::error::targets is empty" in result.log
    assert len(result.rows) == 1
    assert "no targets" in result.rows[0]
    assert result.outputs["exit-code"] == "2"


def test_an_unset_input_variable_is_named_rather_than_an_unbound_variable_crash(tmp_path):
    summary = tmp_path / "s.md"
    outputs = tmp_path / "o.txt"
    summary.touch()
    outputs.touch()
    completed = subprocess.run(
        ["bash", str(RUN_SH)],
        capture_output=True,
        text=True,
        check=False,
        cwd=tmp_path,
        env={
            "PATH": os.environ.get("PATH", ""),
            "GITHUB_STEP_SUMMARY": str(summary),
            "GITHUB_OUTPUT": str(outputs),
            "GITHUB_ACTION_PATH": str(ACTION_DIR),
            "TARGETS": str(MLP),
        },
    )

    assert completed.returncode == 2
    assert "unset input environment variable(s): BACKENDS" in completed.stderr
    assert "exit-code=2" in outputs.read_text()


def _step_env_names() -> list[str]:
    """The variable names action.yml's "Run torch-compile-check" step exports."""
    text = ACTION_YML.read_text()
    block = text[text.index("- name: Run torch-compile-check") :]
    block = block[block.index("env:") : block.index("run:")]
    return re.findall(r"^\s{8}([A-Z_]+):", block, flags=re.MULTILINE)


def test_action_yml_runs_this_script_and_exports_exactly_what_it_requires():
    # The drift guard for the split: the YAML names the inputs, the script names
    # what it needs, and neither can be changed alone without this failing.
    assert 'run: bash "$GITHUB_ACTION_PATH/run.sh"' in ACTION_YML.read_text()

    required = re.search(
        r"^REQUIRED_ENV=\(\n(.*?)^\)$", RUN_SH.read_text(), flags=re.MULTILINE | re.DOTALL
    )
    assert required is not None
    assert sorted(required.group(1).split()) == sorted(_step_env_names())
    assert sorted(_step_env_names()) == sorted(DEFAULT_ENV)
