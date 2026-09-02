"""Tests for ``validation/run.py``, the real-world validation runner.

PLAN.md "Real-world validation set" makes this script the thing that decides
whether a public model is reported as clean, as a finding, or as not measured at
all -- and ``docs/validation.md`` is generated from what it decides. The one
mistake it must not make is calling a row something it did not measure, so these
tests are about the boundary between the three: a skip, a finding, and a tool
error.

The targets here are written into a temporary directory rather than taken from
``validation/targets/``, because the real ones pull torchvision or transformers
and take a minute each. ``BACKENDS`` is narrowed for the one test that runs a
model, for the same reason: what is under test is the runner's reading of an
exit code, not inductor.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import validation.run as validation
from validation.run import TargetSpec, render_table, run_target

REPO_ROOT = Path(__file__).resolve().parents[1]

# An installed, importable, torch-free stand-in for `transformers`: what
# matters is that find_spec finds it, so the target gets as far as running and
# the import failure has to be recognised after the fact.
INSTALLED = "json"


@pytest.fixture
def targets(tmp_path, monkeypatch):
    """A temporary ``validation/targets/`` directory."""
    monkeypatch.setattr(validation, "TARGETS_DIR", tmp_path)
    return tmp_path


def test_a_target_whose_extra_is_not_installed_is_skipped_before_it_runs(targets):
    (targets / "absent_extra.py").write_text("model = None\ninputs = ()\n")

    result = run_target(
        TargetSpec(
            "absent_extra.py",
            numerics_sensitive=False,
            requires_package="no_such_package_9c1f",
        )
    )

    assert result.status == "skipped"
    assert result.exit_code is None
    assert result.seconds is None, "nothing ran, so there is no time to report"
    assert result.reason == "no_such_package_9c1f not installed (validation-only extra)"


def test_an_extra_that_is_installed_but_will_not_import_is_a_skip_and_not_a_tool_error(targets):
    # The M4-2 estate run: transformers 5 moved `from transformers import
    # BertModel` behind a lazy import that raises, find_spec said the package
    # was present, and hf_tiny_bert came back as a tool error -- which reads as
    # "compile-check is broken" rather than "this environment cannot build the
    # target". Reproduced here with a package that is always installed and an
    # import from it that never works.
    (targets / "broken_extra.py").write_text(
        f"from {INSTALLED} import BertModel\n\nmodel = BertModel\ninputs = ()\n"
    )

    result = run_target(
        TargetSpec("broken_extra.py", numerics_sensitive=False, requires_package=INSTALLED)
    )

    assert result.status == "skipped"
    assert result.exit_code is None
    assert result.reason is not None
    assert result.reason.startswith(f"{INSTALLED} is installed but this target could not import it")
    assert "ImportError" in result.reason
    # It did run, so unlike the skip above it has a measured time.
    assert result.seconds is not None


def test_a_tool_error_that_is_not_an_import_failure_is_still_a_tool_error(targets):
    # The other side of the same boundary: a target that declares an extra and
    # fails for its own reasons must not be laundered into a skip.
    (targets / "no_entry.py").write_text("value = 3\n")

    result = run_target(
        TargetSpec("no_entry.py", numerics_sensitive=False, requires_package=INSTALLED)
    )

    assert result.status == "tool_error"
    assert result.exit_code == 2
    assert result.reason


def test_a_target_that_runs_clean_is_reported_clean(targets, monkeypatch):
    monkeypatch.setattr(validation, "BACKENDS", "eager,aot_eager")
    (targets / "tiny.py").write_text(
        "import torch\n"
        "from torch import nn\n"
        "\n"
        "torch.manual_seed(0)\n"
        "model = nn.Linear(4, 4)\n"
        "model.eval()\n"
        "\n"
        "\n"
        "def get_inputs():\n"
        "    return (torch.randn(2, 4),)\n"
    )

    result = run_target(TargetSpec("tiny.py", numerics_sensitive=False))

    assert result.status == "clean"
    assert result.exit_code == 0
    assert result.findings_by_oracle == {}
    assert result.stage is not None
    assert result.stage.startswith("clean:")
    assert result.seconds is not None


def test_the_table_puts_a_skipped_target_s_reason_where_a_reader_will_see_it(targets):
    result = run_target(
        TargetSpec("gone.py", numerics_sensitive=False, requires_package="no_such_package_9c1f")
    )

    table = render_table([result])

    assert "| `gone.py`" not in table, "the row is keyed by the target name, not the filename"
    assert "| `gone` | skipped | -- | -- | no_such_package_9c1f not installed" in table


def test_the_validation_extra_pins_transformers_below_five():
    # The pin is the other half of the fix above: the runner degrades to a skip
    # on transformers 5, and the extra installs a version the target can
    # actually use, so `pip install -e ".[validation]"` produces a real row.
    extras = re.search(
        r"^validation = \[(.*?)^\]", (REPO_ROOT / "pyproject.toml").read_text(), re.M | re.S
    )
    assert extras is not None, "pyproject.toml has no [project.optional-dependencies] validation"
    assert '"transformers<5"' in extras.group(1)
    assert '"torchvision"' in extras.group(1)
