"""Every corpus case exists in two shapes (`cases/README.md`): a standalone
RED/GREEN script and a discovery-convention twin the tool itself can run.
This module is the twin's contract: for each pair, the standalone script's
own RED/GREEN verdict on the installed torch is the ground truth, and
`torch-compile-check <twin>` must agree with it -- RED means exit 1 with the
finding named in the twin's docstring, GREEN means exit 0 clean.

The standalone scripts are run in a subprocess (they are not import-safe as
a module: each defines its own `main()` and calls `sys.exit`), the twins
through `torch-compile-check`'s own `main()` in-process, following the pattern
`test_cli.py::test_the_tool_reports_the_copyback_alias_case_end_to_end`
already uses for `alias_copyback.py`. A standalone script that exits 2
(crashed outright) skips the pair rather than asserting anything -- a crash
means the case cannot even establish RED or GREEN on this torch, which is
not the tool's failure to catch.

Two entries below name the same file in both columns:
`alias_view_slice_scatter_copyback` and
`alias_diagonal_scatter_index_put_chain`, the reviewer-reported siblings of
`alias_slice_scatter_copyback` (`cases/markers.py`), have no separate
discovery-convention twin -- each file exposes the module-level `fn` and
`inputs` PLAN.md's discovery convention looks for, in addition to the
standalone script shape, so it is its own twin.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from torch_compile_check.cli import EXIT_FINDING, EXIT_OK, main

REPO_ROOT = Path(__file__).resolve().parents[1]
CASES = REPO_ROOT / "cases"

# standalone: the RED/GREEN script FINDINGS.md keys on.
# twin: the discovery-convention file torch-compile-check runs directly.
# extra_args: flags torch-compile-check needs to exercise the same shape the
#   standalone script checks (PLAN.md "CLI surface for v1"); empty when the
#   default invocation already matches.
# red_stage_backend: the backend named in the stage verdict's
#   "first diverges at <backend>" line when the standalone script is RED.
#   Deterministic across torch builds because it follows the tool's fixed
#   lane order (eager, aot_eager, inductor), not anything environment
#   -specific -- see each twin's docstring for why that backend and not
#   another.
TWINS = (
    pytest.param(
        "dtype_int8_matmul_promotion.py",
        "dtype_promotion.py",
        (),
        "inductor",
        id="dtype_int8_matmul_promotion",
    ),
    pytest.param(
        "alias_slice_scatter_copyback.py",
        "alias_copyback.py",
        (),
        "inductor",
        id="alias_slice_scatter_copyback",
    ),
    pytest.param(
        "alias_noop_view_identity.py",
        "alias_noop_view.py",
        (),
        "inductor",
        id="alias_noop_view_identity",
    ),
    pytest.param(
        "distributions_validation_branch.py",
        "distributions_binomial_kl.py",
        ("--fullgraph",),
        "aot_eager",
        id="distributions_validation_branch",
    ),
    pytest.param(
        "numerics_cpu_inductor_miscompile.py",
        "numerics_polyjuice_minmax.py",
        ("--dynamic",),
        "inductor",
        id="numerics_cpu_inductor_miscompile",
    ),
    # Reviewer-reported siblings of alias_slice_scatter_copyback (2026-09-03),
    # not part of the original C-1 slice and with no separate twin file: each
    # exposes the module-level `fn` and `inputs` PLAN.md's discovery
    # convention looks for in the same file as its own standalone script, so
    # it names itself in both columns here.
    pytest.param(
        "alias_view_slice_scatter_copyback.py",
        "alias_view_slice_scatter_copyback.py",
        (),
        "inductor",
        id="alias_view_slice_scatter_copyback",
    ),
    pytest.param(
        "alias_diagonal_scatter_index_put_chain.py",
        "alias_diagonal_scatter_index_put_chain.py",
        (),
        "inductor",
        id="alias_diagonal_scatter_index_put_chain",
    ),
)


@pytest.mark.parametrize(("standalone", "twin", "extra_args", "red_stage_backend"), TWINS)
def test_twin_agrees_with_the_standalone_scripts_verdict(
    capsys, standalone, twin, extra_args, red_stage_backend
):
    completed = subprocess.run(
        [sys.executable, str(CASES / standalone)],
        capture_output=True,
        text=True,
    )
    if completed.returncode == 2:
        pytest.skip(
            f"{standalone} crashed on this torch (exit 2), so it establishes "
            "neither RED nor GREEN here"
        )
    assert completed.returncode in (0, 1), (
        f"{standalone} exited {completed.returncode}, expected 0 (GREEN), 1 "
        f"(RED), or 2 (crash); stdout={completed.stdout!r} stderr={completed.stderr!r}"
    )
    is_red = completed.returncode == 1

    code = main([str(CASES / twin), *extra_args, "--color", "never"])
    out = capsys.readouterr().out

    if is_red:
        assert code == EXIT_FINDING, (
            f"{standalone} was RED ({completed.stdout.strip()!r}) but "
            f"torch-compile-check {twin} exited {code}, not {EXIT_FINDING}"
        )
        assert f"first diverges at {red_stage_backend}" in out, out
    else:
        assert code == EXIT_OK, (
            f"{standalone} was GREEN ({completed.stdout.strip()!r}) but "
            f"torch-compile-check {twin} exited {code}, not {EXIT_OK}"
        )
        assert "clean:" in out, out
