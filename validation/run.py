"""The real-world validation runner.

PLAN.md "Real-world validation set": a reproducible runner that exercises
torch-compile-check on public, CPU-runnable models -- rather than the fixture-
sized targets ``tests/fixtures/`` and the bug-shaped ``cases/`` corpus use
-- so the tool's false-positive rate and default tolerances can be judged
against real architectures. Every target lives in ``validation/targets/``
as a plain discovery-convention file (PLAN.md "Discovery convention"); this
script's job is only to run ``torch-compile-check`` on each one, record what came
back, and regenerate the table in ``docs/validation.md``.

``torchvision`` and ``transformers`` are validation-only extras: neither is
a ``pyproject.toml`` dependency (``docs/validation.md`` "Extras" says why),
so a target that needs a missing package is skipped, not treated as a tool
error -- ``hf_tiny_bert.py`` is the only target that currently needs one.

"Missing" covers two shapes, and the second one cost a run. A package that
is not installed at all is caught up front by :func:`importlib.util.find_spec`.
A package that *is* installed but cannot be imported the way the target
imports it is caught after the fact: ``transformers`` 5 moved
``from transformers import BertModel`` behind a lazy import that raises
``ModuleNotFoundError``, ``find_spec`` said the package was there, and the
target came back as a tool error rather than a skip (M4-2 estate run). Both
are the same fact about the environment and both are reported as skipped
with the reason, which is why the ``validation`` extra pins
``transformers<5``: a skip says the row was not measured, and a tool error
says the tool broke.

Every real run is reported verbatim. A finding on a real model is not
tuned away by adjusting a target or a tolerance to make it disappear; it is
either a real torch-compile-check finding (kept, and worth a look) or grounds to
open an issue against this file if it turns out to be this file's bug
rather than the compiler's.

Usage::

    python validation/run.py                    # every target
    python validation/run.py --targets tv_resnet18,train_step_mlp
"""

from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TARGETS_DIR = Path(__file__).resolve().parent / "targets"
RESULTS_DIR = Path(__file__).resolve().parent / "results"
VALIDATION_DOC = REPO_ROOT / "docs" / "validation.md"
README = REPO_ROOT / "README.md"

BACKENDS = "eager,aot_eager,inductor"

STAGE_RE = re.compile(r"^stage\n\s*(.+)$", re.MULTILINE)
FINDING_RE = re.compile(r"^  (\w+)  \((\d+) fail\)", re.MULTILINE)
EXIT_ERROR = 2

# How an import failure of a validation extra shows up in the CLI's one-line
# tool error: discovery re-raises the target's own exception with its class
# name in the sentence (``torch_compile_check.discover``). Matching the class name
# *and* the package the target declared is what keeps this from swallowing a
# real tool error that merely happens to mention importing.
IMPORT_ERROR_RE = re.compile(r"\b(ModuleNotFoundError|ImportError)\b")


@dataclass(frozen=True)
class TargetSpec:
    """One entry in ``validation/targets/``, and how to run it."""

    filename: str
    numerics_sensitive: bool
    """Whether to add ``--fp64-oracle``: PLAN.md "The oracle blind spot"'s
    fp64 eager reference is most useful on the floating-point-heavy
    convolution and attention stacks, so it is added for those and skipped
    for the training-step target, where the point is the grad path rather
    than numerics."""

    requires_package: str | None = None
    """A package this target imports beyond ``torch`` itself, checked with
    :func:`importlib.util.find_spec` before invoking ``torch-compile-check`` at
    all, so a missing extra is a clean skip rather than a subprocess crash
    the report would otherwise have to parse an explanation out of."""


TARGET_SPECS: tuple[TargetSpec, ...] = (
    TargetSpec("tv_resnet18.py", numerics_sensitive=True, requires_package="torchvision"),
    TargetSpec("tv_mobilenet_v3_small.py", numerics_sensitive=True, requires_package="torchvision"),
    TargetSpec("tv_efficientnet_b0.py", numerics_sensitive=True, requires_package="torchvision"),
    TargetSpec("tv_vit_b_16_tiny.py", numerics_sensitive=True, requires_package="torchvision"),
    TargetSpec("hf_tiny_bert.py", numerics_sensitive=False, requires_package="transformers"),
    TargetSpec("train_step_mlp.py", numerics_sensitive=False),
)


@dataclass
class TargetResult:
    """What one target's run produced, in the shape the table and the JSON
    file both read from."""

    name: str
    status: str
    """One of ``"clean"``, ``"finding"``, ``"tool_error"``, or ``"skipped"``."""

    exit_code: int | None
    findings_by_oracle: dict[str, int] = field(default_factory=dict)
    stage: str | None = None
    seconds: float | None = None
    reason: str | None = None
    """Why a target was skipped, or a one-line summary of a tool error."""

    def to_json(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "exit_code": self.exit_code,
            "findings_by_oracle": self.findings_by_oracle,
            "stage": self.stage,
            "seconds": self.seconds,
            "reason": self.reason,
        }


def run_target(spec: TargetSpec) -> TargetResult:
    """Run one target through ``torch-compile-check``, or skip it cleanly."""
    name = spec.filename.removesuffix(".py")
    if (
        spec.requires_package is not None
        and importlib.util.find_spec(spec.requires_package) is None
    ):
        return TargetResult(
            name=name,
            status="skipped",
            exit_code=None,
            reason=f"{spec.requires_package} not installed (validation-only extra)",
        )

    path = TARGETS_DIR / spec.filename
    args = [
        sys.executable,
        "-m",
        "torch_compile_check.cli",
        str(path),
        "--backends",
        BACKENDS,
        "--color",
        "never",
    ]
    if spec.numerics_sensitive:
        args.append("--fp64-oracle")

    started = time.perf_counter()
    completed = subprocess.run(args, capture_output=True, text=True, cwd=REPO_ROOT)
    elapsed = time.perf_counter() - started
    out = completed.stdout

    if completed.returncode == EXIT_ERROR:
        first_line = next((line for line in completed.stderr.splitlines() if line.strip()), "")
        if spec.requires_package is not None and _extra_failed_to_import(
            completed.stderr + completed.stdout, spec.requires_package
        ):
            # Installed, and still not importable the way this target imports
            # it. That is a fact about the environment, exactly like the
            # package being absent, so it reads as a skip with the reason
            # rather than as the tool falling over.
            return TargetResult(
                name=name,
                status="skipped",
                exit_code=None,
                seconds=elapsed,
                reason=(
                    f"{spec.requires_package} is installed but this target could not "
                    f"import it: {first_line or 'exit 2 with no stderr line to report'}"
                ),
            )
        return TargetResult(
            name=name,
            status="tool_error",
            exit_code=completed.returncode,
            seconds=elapsed,
            reason=first_line or "exit 2 with no stderr line to report",
        )

    stage_match = STAGE_RE.search(out)
    stage = stage_match.group(1) if stage_match else None
    findings = {oracle: int(count) for oracle, count in FINDING_RE.findall(out)}
    status = "finding" if completed.returncode == 1 else "clean"

    return TargetResult(
        name=name,
        status=status,
        exit_code=completed.returncode,
        findings_by_oracle=findings,
        stage=stage,
        seconds=elapsed,
    )


def _extra_failed_to_import(output: str, package: str) -> bool:
    """Did this run fail because the declared extra would not import?

    Both halves are required. An ``ImportError`` on its own could be the tool
    failing to import something of its own, and the package name on its own
    appears in every message that quotes the target's path.
    """
    return bool(IMPORT_ERROR_RE.search(output)) and package in output


def environment() -> dict[str, str]:
    """Provenance for the report: torch version, git hash, and architecture.

    Imported directly rather than parsed out of a target's stdout, so
    provenance is available even when every target is skipped.
    """
    import torch

    git_version = getattr(torch.version, "git_version", None) or "unknown"
    return {
        "torch_version": torch.__version__,
        "torch_git": git_version[:12],
        "python_version": platform.python_version(),
        "machine": platform.machine(),
        "platform": platform.platform(),
    }


def render_table(results: list[TargetResult]) -> str:
    """The target | exit | findings by oracle | stage | seconds table."""
    lines = [
        "| Target | Status | Exit | Findings by oracle | Stage | Seconds |",
        "|---|---|---|---|---|---|",
    ]
    for result in results:
        if result.status == "skipped":
            # A target skipped up front never ran and has no time; one skipped
            # because its extra would not import did, and hiding that would
            # make the two look like the same measurement.
            seconds = "--" if result.seconds is None else f"{result.seconds:.1f}"
            lines.append(f"| `{result.name}` | skipped | -- | -- | {result.reason} | {seconds} |")
            continue
        if result.status == "tool_error":
            lines.append(
                f"| `{result.name}` | tool error | {result.exit_code} | -- | {result.reason} | "
                f"{result.seconds:.1f} |"
            )
            continue
        findings = (
            ", ".join(
                f"{oracle} ({count})" for oracle, count in sorted(result.findings_by_oracle.items())
            )
            or "none"
        )
        stage = result.stage or "(no stage line)"
        lines.append(
            f"| `{result.name}` | {result.status} | {result.exit_code} | {findings} | {stage} | "
            f"{result.seconds:.1f} |"
        )
    return "\n".join(lines)


def render_doc(results: list[TargetResult], env: dict[str, str], run_date: str) -> str:
    """The whole ``docs/validation.md`` body."""
    return f"""# Real-world validation

PLAN.md "Real-world validation set": torch-compile-check run against public,
CPU-runnable models -- not the bug-shaped fixtures in `cases/` or the tiny
fixtures in `tests/fixtures/` -- so its false-positive rate and default
tolerances can be judged against real architectures. Generated by
`validation/run.py`; do not hand-edit this file, edit the runner or a
target and regenerate instead.

## How to run

```console
python -m pip install -e ".[dev]"
python -m pip install -e ".[validation]"  # torchvision + transformers<5
python validation/run.py
```

Install `torchvision` from the same index as torch (see
`docs/cross-arch.md`); a PyPI `torchvision` beside a `torch+cpu` wheel
fails to load its compiled ops, and every `tv_*` target then reports a
tool error rather than a result.

`--targets name1,name2` runs a subset; the default is every target under
`validation/targets/`. A result JSON lands in `validation/results/`, named
`<date>-<arch>-<torch version>.json`, and this file is regenerated from the
latest run.

## What "clean" means

A target is "clean" when every backend's checks against eager pass with
no `fail`-severity finding (PLAN.md "Reports"), which is
`torch-compile-check`'s own exit 0. All five oracles run as of this table --
numerics, alias, metadata, grad, and graph -- so every one of them is
part of its "clean" reading. Graph health is the one that is
informational by default: without `--baseline` or `--fullgraph`, neither
of which this suite passes, a graph break is reported as context and
never as a verdict.

## Tolerance policy in force

Default tolerances (PLAN.md "numerics"), no target-specific overrides.
Numerics-sensitive targets (the four vision/ViT targets) add
`--fp64-oracle`, which widens a deep copy of the model and its floating
inputs to float64 and runs a third eager reference at that width, so a
divergence can be told apart from "both eager and compiled are imprecise
at fp32" rather than reported as a compiler bug. `train_step_mlp` and
`hf_tiny_bert` do not: the training-step target is here for the grad path,
not numerics, and the fp64 pass has nothing extra to add to a well-
conditioned two-layer MLP; the BERT target's embedding lookup is integer-
indexed and unaffected by the pass either way, so it is left off for speed
rather than for a numerics reason.

## Extras

`torchvision` and `transformers` are **not** `pyproject.toml` dependencies
-- they exist only to build validation targets, and `torch-compile-check` itself
has exactly one runtime dependency (`torch`), per PLAN.md "Engineering
decisions". They are the `validation` extra instead.

A target whose extra is missing is reported below as "skipped", not as a
tool error, and that covers both ways an extra can be missing: not
installed at all, and installed but not importable the way the target
imports it. The second one is why the extra pins `transformers<5` --
5.x moved `from transformers import BertModel` behind a lazy import that
raises `ModuleNotFoundError`, `importlib.util.find_spec` reported the
package as present, and `hf_tiny_bert` came back as a tool error, which
reads as "torch-compile-check is broken" rather than "this environment cannot
build the target". `hf_tiny_bert.py`'s own docstring has the detail on
what importing it directly looks like without `transformers` installed.

Point `torch-compile-check` at `validation/targets/hf_tiny_bert.py` directly,
outside of `validation/run.py`, in an environment without `transformers`,
and the CLI itself has no suite to fall back on: it exits `2` with
`torch-compile-check: importing hf_tiny_bert.py raised
ModuleNotFoundError: No module named 'transformers'`, a tool error rather
than a skip, because there is nothing to compare without a successful
import (`docs/usage.md`). `pip install "torch-compile-check[validation]"`
installs what every target in this table needs.

## Provenance

Run on {run_date}. torch `{env["torch_version"]}` (git `{env["torch_git"]}`),
Python {env["python_version"]}, `{env["machine"]}` (`{env["platform"]}`).
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` for every invocation (matches the
CLI's own default), backends `{BACKENDS}`.

## Results

{render_table(results)}

The table above is regenerated by `python validation/run.py`.
Cross-architecture results and the parity method live in
`docs/cross-arch.md`.
"""


def update_readme(link_line: str) -> None:
    """Add the validation doc's one link line to the README, once."""
    text = README.read_text()
    if "docs/validation.md" in text:
        return
    marker = "## Blind spot, stated up front"
    if marker not in text:
        raise SystemExit(
            f"README.md: expected section {marker!r} to anchor the new link, not found"
        )
    text = text.replace(marker, f"{link_line}\n\n{marker}", 1)
    README.write_text(text)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--targets",
        default="all",
        help="comma-separated target names (without .py), or 'all' (default)",
    )
    args = parser.parse_args(argv)

    if args.targets == "all":
        specs = TARGET_SPECS
    else:
        wanted = {name.strip() for name in args.targets.split(",") if name.strip()}
        specs = tuple(spec for spec in TARGET_SPECS if spec.filename.removesuffix(".py") in wanted)
        missing = wanted - {spec.filename.removesuffix(".py") for spec in specs}
        if missing:
            parser.error(f"unknown target(s): {', '.join(sorted(missing))}")

    env = environment()
    results = [run_target(spec) for spec in specs]

    run_date = datetime.date.today().isoformat()
    RESULTS_DIR.mkdir(exist_ok=True)
    result_path = RESULTS_DIR / f"{run_date}-{env['machine']}-{env['torch_version']}.json"
    result_path.write_text(
        json.dumps(
            {"date": run_date, "environment": env, "results": [r.to_json() for r in results]},
            indent=2,
        )
        + "\n"
    )

    VALIDATION_DOC.write_text(render_doc(results, env, run_date))
    update_readme(
        "## Validation against real models\n\n"
        "torch-compile-check is run against public, CPU-runnable models "
        "(torchvision, a tiny HF BERT, a training step) in "
        "[`docs/validation.md`](docs/validation.md), regenerated by "
        "[`validation/run.py`](validation/run.py)."
    )

    print(f"wrote {result_path.relative_to(REPO_ROOT)}")
    print(f"wrote {VALIDATION_DOC.relative_to(REPO_ROOT)}")
    for result in results:
        print(f"{result.name}: {result.status}" + (f" ({result.reason})" if result.reason else ""))

    return 1 if any(r.status == "tool_error" for r in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
