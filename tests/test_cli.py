"""Tests for the CLI surface: the two working paths, and the parsed-only rest."""

from __future__ import annotations

import importlib
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from compile_check import __version__
from compile_check.cli import (
    EXIT_ERROR,
    EXIT_FINDING,
    EXIT_OK,
    build_parser,
    format_probe_table,
    format_run_only,
    main,
    parse_fail_on,
)
from compile_check.env import PROBED_APIS
from compile_check.oracles import Finding
from compile_check.report.terminal import DEFAULT_MAX_FINDINGS
from compile_check.results import BackendResult, RunSet

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
CASES = REPO_ROOT / "cases"

PACKAGE_MODULES = [
    "compile_check.cli",
    "compile_check.discover",
    "compile_check.env",
    "compile_check.localize",
    "compile_check.minimize",
    "compile_check.oracles",
    "compile_check.oracles.alias",
    "compile_check.oracles.grad",
    "compile_check.oracles.graph",
    "compile_check.oracles.metadata",
    "compile_check.oracles.numerics",
    "compile_check.report",
    "compile_check.report.json",
    "compile_check.report.markdown",
    "compile_check.report.pytest_case",
    "compile_check.report.terminal",
    "compile_check.results",
    "compile_check.runner",
]


def test_version_prints_version(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"compile-check {__version__}"


def test_version_matches_pyproject():
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "(?P<version>[^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert match.group("version") == __version__


def test_probe_exits_zero_and_prints_one_row_per_api(capsys):
    assert main(["--probe"]) == EXIT_OK
    lines = capsys.readouterr().out.splitlines()
    # header, rule, then one row per probed API
    assert len(lines) == len(PROBED_APIS) + 2
    assert lines[0].split() == ["api", "status"]
    for name, line in zip(PROBED_APIS, lines[2:], strict=True):
        assert line.startswith(name)
        assert line.split()[-1] in {"present", "absent"}


@pytest.mark.parametrize("flag", ["--json", "--md"])
def test_probe_warns_that_report_flags_are_ignored(capsys, flag):
    assert main(["--probe", flag, "out"]) == EXIT_OK
    captured = capsys.readouterr()
    assert captured.err.strip() == f"compile-check: {flag} ignored with --probe"
    assert captured.out.startswith("api")


def test_probe_alone_warns_about_nothing(capsys):
    assert main(["--probe"]) == EXIT_OK
    assert capsys.readouterr().err == ""


def test_format_probe_table_two_columns():
    table = format_probe_table({"torch.compile": True, "torch.nope": False})
    assert table.splitlines()[2:] == [
        "torch.compile  present",
        "torch.nope     absent",
    ]


def test_a_path_that_does_not_exist_is_a_tool_error(capsys):
    assert main(["model.py"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "no such file: model.py" in err


def test_no_arguments_exits_two_with_the_usage(capsys):
    assert main([]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert "needs a target" in err
    assert "usage: compile-check" in err


def test_full_v1_flag_surface_parses():
    args = build_parser().parse_args(
        [
            "model.py",
            "--entry",
            "mymod:model",
            "--inputs",
            "mymod:get_inputs",
            "--backends",
            "eager,aot_eager,aot_eager_decomp_partition,inductor",
            "--device",
            "cuda",
            "--json",
            "out.json",
            "--md",
            "report.md",
            "--fail-on",
            "numerics,alias,metadata,grad,graph",
            "--fullgraph",
            "--dynamic",
            "--rtol",
            "1e-5",
            "--atol",
            "1e-8",
            "--seed",
            "1234",
            "--allow-caches",
            "--fp64-oracle",
            "--budget",
            "600",
            "--baseline",
            "graph-baseline.json",
            "--no-grad",
            "--share-module",
            "--max-findings",
            "3",
            "--color",
            "never",
        ]
    )
    assert args.path == "model.py"
    assert args.entry == "mymod:model"
    assert args.inputs == "mymod:get_inputs"
    assert args.backends == "eager,aot_eager,aot_eager_decomp_partition,inductor"
    assert args.device == "cuda"
    assert args.json == "out.json"
    assert args.md == "report.md"
    assert args.fail_on == "numerics,alias,metadata,grad,graph"
    assert args.fullgraph is True
    assert args.dynamic is True
    assert args.rtol == pytest.approx(1e-5)
    assert args.atol == pytest.approx(1e-8)
    assert args.seed == 1234
    assert args.allow_caches is True
    assert args.fp64_oracle is True
    assert args.budget == pytest.approx(600.0)
    assert args.baseline == "graph-baseline.json"
    assert args.no_grad is True
    assert args.share_module is True
    assert args.max_findings == 3
    assert args.color == "never"


def test_defaults_match_the_plan():
    args = build_parser().parse_args([])
    assert args.backends == "eager,aot_eager,inductor"
    assert args.device == "cpu"
    assert args.fail_on == "numerics,alias,metadata,grad"
    assert args.seed == 0
    assert args.fullgraph is False
    assert args.dynamic is False
    assert args.no_grad is False
    assert args.share_module is False
    assert args.max_findings == DEFAULT_MAX_FINDINGS
    assert args.color == "auto"


def test_bad_device_is_a_tool_error():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["model.py", "--device", "tpu"])
    assert excinfo.value.code == EXIT_ERROR


def test_every_module_imports():
    for name in PACKAGE_MODULES:
        assert importlib.import_module(name) is not None


def test_stubs_raise_not_implemented():
    # discover.py and runner.py landed in M1-1, the numerics and metadata
    # oracles in M1-2, localize.py plus report/terminal.py in M1-3, the alias
    # oracle in M2-1 and the grad oracle in M2-2; each is covered by its own
    # test module. What is left below is what M3 still owes.
    from compile_check import minimize
    from compile_check.oracles import graph
    from compile_check.report import json as json_report
    from compile_check.report import markdown, pytest_case

    with pytest.raises(NotImplementedError):
        minimize.minimize(None, None, lambda _fn, _inputs: True)
    with pytest.raises(NotImplementedError):
        graph.check({}, {})
    with pytest.raises(NotImplementedError):
        json_report.dump({}, Path("out.json"))
    with pytest.raises(NotImplementedError):
        markdown.render({})
    with pytest.raises(NotImplementedError):
        pytest_case.emit({})


def test_importing_the_package_does_not_import_torch():
    # PLAN.md engineering decision: --version must not pay for the torch import.
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, compile_check.cli; print('torch' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert completed.stdout.strip() == "False"


def test_console_script_is_installed():
    script = Path(sys.executable).parent / "compile-check"
    if not script.exists():  # pragma: no cover - non-editable or non-venv install
        pytest.skip(f"console script not found at {script}")
    completed = subprocess.run([str(script), "--version"], capture_output=True, text=True)
    assert completed.returncode == 0
    assert completed.stdout.strip() == f"compile-check {__version__}"


def test_run_only_is_hidden_from_help():
    # A developer path, not part of the v1 surface in PLAN.md.
    assert "--run-only" not in build_parser().format_help()
    assert build_parser().parse_args(["m.py", "--run-only"]).run_only is True


def test_run_only_prints_a_row_and_the_outputs_per_backend(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager,aot_eager"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "target     mlp:model" in out
    assert "eager[0] torch.float32 (4, 4)" in out
    assert "aot_eager[0] torch.float32 (4, 4)" in out
    # Both lanes recorded a wall time and the parameter grads.
    assert out.count("ok") >= 2
    assert "4 parameters" in out


def test_run_only_reports_the_oracles_and_finds_nothing_on_a_clean_model(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager,aot_eager"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    # The four that run, and the one that does not exist yet: "not checked"
    # must not read as "checked and clean".
    assert "oracles    numerics, alias, metadata, grad" in out
    assert "not implemented yet, nothing checked: graph" in out
    assert "findings\n  none" in out


def test_fail_on_narrows_the_verdict_not_the_checks(capsys):
    # The CEO decision behind M1-3: --fail-on selects which categories turn a
    # finding into exit 1. Every implemented oracle runs whatever it says, so a
    # narrowed exit rule never narrows what the report looked at.
    code = main(
        [str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager", "--fail-on", "metadata"]
    )
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "oracles    numerics, alias, metadata, grad" in out
    assert "fail-on    metadata\n" in out


def test_run_only_rejects_an_unknown_fail_on_category(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--fail-on", "numerics,typo"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Traceback" not in err
    assert "unknown --fail-on category 'typo'" in err
    assert "numerics, alias, metadata, grad, graph" in err


def test_parse_fail_on_keeps_the_order_and_drops_duplicates():
    assert parse_fail_on("metadata, numerics ,metadata") == ["metadata", "numerics"]
    assert parse_fail_on(",") == []
    with pytest.raises(ValueError, match="unknown --fail-on categories 'a', 'b'"):
        parse_fail_on("a,b")


def test_format_run_only_lists_the_findings_it_is_given():
    runset = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        results={"eager": BackendResult(backend="eager", outputs=[1])},
    )
    findings = [
        Finding(
            oracle="metadata",
            backend="inductor",
            output_index=0,
            severity="fail",
            message="dtype differs: eager torch.int8, inductor torch.int64",
            details={"field": "dtype"},
        ),
        Finding(
            oracle="numerics",
            backend="inductor",
            output_index=None,
            severity="warn",
            message="something structural",
            details={},
        ),
    ]
    out = format_run_only(runset, findings, fail_on=["numerics", "metadata"])

    assert "[fail] metadata inductor[0] dtype differs" in out
    assert "[warn] numerics inductor[-] something structural" in out
    assert "  none" not in out


def test_run_only_reports_a_raising_model_as_a_tool_error(capsys):
    code = main([str(FIXTURES / "raises.py"), "--run-only", "--backends", "eager"])
    out = capsys.readouterr().out
    assert code == EXIT_ERROR
    assert "raised RuntimeError" in out
    assert "this target is broken on purpose" in out


def test_run_only_reports_a_discovery_failure_on_stderr(capsys):
    code = main([str(FIXTURES / "empty_target.py"), "--run-only"])
    assert code == EXIT_ERROR
    assert "no entry point found" in capsys.readouterr().err


def test_run_only_without_a_target_is_a_tool_error(capsys):
    assert main(["--run-only"]) == EXIT_ERROR
    assert "needs a target" in capsys.readouterr().err


def test_run_only_rejects_an_unknown_backend(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager,bogus"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert "'bogus'" in err
    assert "unknown backend" in err


def test_run_only_rejects_an_empty_backend_list(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", ","])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Traceback" not in err
    assert "no backends requested" in err


def test_run_only_rejects_cuda_when_torch_reports_none(capsys):
    import torch

    if torch.cuda.is_available():  # pragma: no cover - depends on the machine
        pytest.skip("this machine has CUDA, so the unavailable path cannot run")
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--device", "cuda"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Traceback" not in err
    assert len(err.strip().splitlines()) == 1
    assert "cuda" in err
    assert "no CUDA device" in err


def test_run_only_turns_an_unexpected_error_into_a_tool_error(capsys, monkeypatch):
    from compile_check import runner as runner_module

    def boom(*args, **kwargs):
        raise ValueError("something the tool did not expect")

    monkeypatch.setattr(runner_module, "run_all", boom)
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager"])
    err = capsys.readouterr().err
    assert code == EXIT_ERROR
    assert "Traceback" not in err
    assert "ValueError: something the tool did not expect" in err


def test_a_multi_line_error_is_reported_on_one_line(capsys, monkeypatch):
    from compile_check import runner as runner_module

    def boom(*args, **kwargs):
        raise RuntimeError("first line\nsecond line\nthird line")

    monkeypatch.setattr(runner_module, "run_all", boom)
    assert main([str(FIXTURES / "mlp.py"), "--run-only"]) == EXIT_ERROR
    err = capsys.readouterr().err
    assert len(err.strip().splitlines()) == 1
    assert "first line" in err
    assert "second line" not in err
    assert "+2 more lines" in err


def test_a_model_that_raises_still_reports_per_backend(capsys):
    # The boundary must not swallow this one: eager failing is a result the
    # report shows, and only then exit 2.
    code = main([str(FIXTURES / "raises.py"), "--run-only", "--backends", "eager"])
    captured = capsys.readouterr()
    assert code == EXIT_ERROR
    assert "raised RuntimeError" in captured.out
    assert captured.err == ""


def test_run_only_without_an_eager_lane_says_nothing_was_compared(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "aot_eager"])
    out = capsys.readouterr().out
    assert code == EXIT_OK
    assert "findings\n  not checked: this run has no eager lane" in out


# --- the main path ----------------------------------------------------------


def test_the_main_path_reports_a_clean_model_and_exits_zero(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--backends", "eager,aot_eager", "--color", "never"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    assert out.startswith(f"compile-check {__version__}   target mlp:model")
    assert "clean: no backend diverged from eager across 1 lane" in out
    assert "findings\n  none" in out
    # The environment block a parity comparison needs (PLAN.md
    # "Cross-architecture parity is a feature").
    assert "machine   " in out
    assert "caches    disabled (force_disable_caches=True)" in out
    assert "\033[" not in out


def test_the_tool_reports_the_copyback_alias_case_end_to_end(capsys):
    # The whole tool on a corpus case: discovery, three lanes, the oracles, the
    # stage verdict, the exit code. PLAN.md "alias" names 195451 as the bug this
    # oracle exists for, and the case returns identical values in both worlds,
    # so nothing but the alias oracle can catch it.
    code = main([str(CASES / "alias_copyback.py"), "--color", "never"])
    out = capsys.readouterr().out

    if code == EXIT_OK:  # pragma: no cover - depends on the torch build
        assert "clean:" in out
        pytest.skip("this torch does not reproduce 195451, so the case is green here")

    assert code == EXIT_FINDING
    assert "alias  (1 fail)" in out
    assert "inductor returned input[0] itself as output[0]" in out
    # Both relations travel with the finding, so the claim can be checked.
    assert "eager_relation" in out
    assert "compiled_relation" in out
    # aot_eager agrees with eager, which is what makes the verdict inductor's.
    assert "first diverges at inductor, which implicates inductor lowering/codegen" in out
    # Nothing else fired, so the alias category is what drove the exit code, and
    # naming only that category still exits 1.
    assert "numerics  yes      pass              pass" in out
    assert "metadata  yes      pass              pass" in out
    assert main([str(CASES / "alias_copyback.py"), "--fail-on", "alias"]) == EXIT_FINDING
    capsys.readouterr()


def test_a_backend_that_really_raised_exits_one_whatever_fail_on_says(capsys):
    # The live half of the rule that
    # test_a_compiled_lane_that_raised_exits_one_whatever_fail_on_says reaches by
    # monkeypatching run_backend: a compiled lane that raises while eager does
    # not is exit 1 regardless of --fail-on. The fixture registers a backend that
    # raises when torch.compile asks it to compile, so the failure is real and
    # does not depend on an op that happens to be broken on some torch.
    # Importing it here is what puts the name in the registry before the CLI
    # validates --backends.
    from compile_check.discover import import_target_module

    fixture = FIXTURES / "compile_only_raises.py"
    backend = import_target_module(str(fixture)).BACKEND
    code = main(
        [
            str(fixture),
            "--backends",
            f"eager,{backend}",
            # Not one of the categories a finding could fall into: the exit code
            # below is the exception rule, not an oracle's.
            "--fail-on",
            "graph",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_FINDING
    assert "raised BackendCompilerFailed" in out
    assert "this backend raises on purpose" not in out  # the report names the type, not the stack
    # The eager lane is healthy, so this is a compile-only failure and not a
    # broken model, which would have been exit 2.
    assert "eager " in out
    assert "the model raised" not in out
    assert f"first diverges at {backend}" in out
    # No oracle ran against a lane that produced nothing.
    assert "findings\n  none" in out


def test_a_backend_the_target_registers_survives_a_cold_run(tmp_path):
    # M2-2 housekeeping. The test above can only pass because importing the
    # fixture in this process put the backend in the registry before the CLI
    # looked; a user gets no such favour. In a fresh interpreter the name does
    # not exist until discovery imports the target, and validating --backends
    # before that rejected a backend that was about to exist -- as exit 2, a
    # tool error, where the truth is a compiled lane that raised, exit 1.
    from compile_check.discover import import_target_module

    fixture = FIXTURES / "compile_only_raises.py"
    backend = import_target_module(str(fixture)).BACKEND
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "codegen")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compile_check.cli",
            str(fixture),
            "--backends",
            f"eager,{backend}",
            "--color",
            "never",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == EXIT_FINDING, completed.stderr
    assert "unknown backend" not in completed.stderr
    assert "raised BackendCompilerFailed" in completed.stdout
    assert f"first diverges at {backend}" in completed.stdout


def test_fail_on_grad_drives_the_exit_code_on_a_backward_only_divergence(capsys):
    # The grad oracle's half of the --fail-on rule, on a real run: this backend
    # runs the traced graph unchanged and raises in the backward, so the only
    # divergence in the report is a grad one. It fails the run when grad is a
    # --fail-on category and not otherwise, and it is a finding either way.
    from compile_check.discover import import_target_module

    fixture = FIXTURES / "backward_raises.py"
    backend = import_target_module(str(fixture)).BACKEND
    argv = [str(fixture), "--backends", f"eager,{backend}", "--color", "never"]

    assert main([*argv, "--fail-on", "grad"]) == EXIT_FINDING
    out = capsys.readouterr().out
    assert "grad      yes      1 fail" in out
    assert "the backward pass raised RuntimeError" in out
    # The forward pass is bit-identical, so nothing else fired.
    assert "numerics  no       pass" in out

    assert main([*argv, "--fail-on", "numerics"]) == EXIT_OK
    narrowed = capsys.readouterr().out
    # Narrowing the exit rule never narrows what was checked: the finding is
    # still in the report, it just does not decide the exit code.
    assert "grad      no       1 fail" in narrowed


def test_no_grad_says_the_check_was_switched_off(capsys):
    # An oracle that was turned off must not read as an oracle that found
    # nothing, which is the same rule the report applies to the graph oracle.
    clean = main([str(FIXTURES / "mlp.py"), "--backends", "eager,aot_eager", "--color", "never"])
    with_grad = capsys.readouterr().out
    assert clean == EXIT_OK
    assert "grad      yes      pass" in with_grad

    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,aot_eager",
            "--no-grad",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    # info never fails a run, whatever --fail-on says.
    assert code == EXIT_OK
    assert "grad      yes      1 info" in out
    assert "--no-grad switched the backward pass off" in out
    assert "  run       backends eager, aot_eager   seed 0" in out
    assert "grad off" in out


def test_a_backend_nothing_registers_is_still_a_tool_error(tmp_path):
    # The other half: deferring the decision is not the same as dropping it.
    env = dict(os.environ)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "codegen")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compile_check.cli",
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,inducter",
            "--color",
            "never",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == EXIT_ERROR
    assert "unknown backend 'inducter'" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_the_main_path_reports_a_broken_model_as_the_model_and_exits_two(capsys):
    code = main([str(FIXTURES / "raises.py"), "--backends", "eager,aot_eager", "--color", "never"])
    captured = capsys.readouterr()

    assert code == EXIT_ERROR
    assert "the model raised RuntimeError under eager" in captured.out
    assert "this target is broken on purpose" in captured.out
    # Not "checked and clean": nothing was compared at all.
    assert "not checked: nothing was compared" in captured.out
    assert "findings\n  none" not in captured.out
    # A recorded result, not a crash, so nothing goes to stderr.
    assert captured.err == ""


def test_a_synthetic_divergence_exits_one_and_names_the_stage(capsys, monkeypatch):
    # The oracles cannot be made to fail on a correct model on demand, so the
    # divergence is injected at the one seam that matters here: an oracle that
    # reports. Everything downstream -- localization, the report, the exit code
    # -- is the real thing.
    from compile_check import oracles as oracles_module

    class AlwaysFails:
        name = "metadata"

        def compare(self, eager, other, cfg):
            del eager, cfg
            return [
                Finding(
                    oracle="metadata",
                    backend=other.backend,
                    output_index=0,
                    severity="fail",
                    message="dtype differs: eager torch.int8, aot_eager torch.int64",
                    details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
                )
            ]

    monkeypatch.setitem(oracles_module.ORACLES, "metadata", AlwaysFails())
    code = main([str(FIXTURES / "mlp.py"), "--backends", "eager,aot_eager", "--color", "never"])
    out = capsys.readouterr().out

    assert code == EXIT_FINDING
    assert "first diverges at aot_eager, which implicates capture/AOTAutograd/decomposition" in out
    # PLAN.md "Where divergence appears is not always where the fix belongs".
    assert "not necessarily where the fix belongs" in out
    assert "the bug is in" not in out
    assert "[fail] aot_eager output[0]" in out
    assert "dtype differs" in out


def test_fail_on_decides_the_exit_code_without_narrowing_the_report(capsys, monkeypatch):
    from compile_check import oracles as oracles_module

    class AlwaysFails:
        name = "metadata"

        def compare(self, eager, other, cfg):
            del eager, cfg
            return [
                Finding(
                    oracle="metadata",
                    backend=other.backend,
                    output_index=0,
                    severity="fail",
                    message="dtype differs",
                    details={},
                )
            ]

    monkeypatch.setitem(oracles_module.ORACLES, "metadata", AlwaysFails())
    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,aot_eager",
            "--fail-on",
            "numerics",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    # The metadata oracle still ran and its finding is still reported...
    assert "[fail] aot_eager output[0]" in out
    assert "first diverges at aot_eager" in out
    assert "metadata  no " in out
    # ...but it is not a --fail-on category, so it does not fail the run.
    assert code == EXIT_OK


def test_a_compiled_lane_that_raised_exits_one_whatever_fail_on_says(capsys, monkeypatch):
    # A lane that could not run is not a lane that passed, and an exception
    # belongs to no oracle category, so --fail-on cannot switch this off.
    from compile_check import runner as runner_module
    from compile_check.results import CapturedException

    real_run_backend = runner_module.run_backend

    def sabotage(fn, example_inputs, backend, **kwargs):
        result = real_run_backend(fn, example_inputs, backend, **kwargs)
        if backend != "eager":
            result.exception = CapturedException(
                type="BackendCompilerFailed", message="synthetic", traceback=("...",)
            )
            result.outputs = []
        return result

    monkeypatch.setattr(runner_module, "run_backend", sabotage)
    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,aot_eager",
            "--fail-on",
            "graph",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_FINDING
    assert "raised BackendCompilerFailed" in out
    assert "first diverges at aot_eager" in out


def test_a_run_without_an_eager_lane_is_a_tool_error(capsys):
    # PLAN.md "Runner semantics" makes eager the reference world; a run with no
    # reference must not report clean.
    code = main([str(FIXTURES / "mlp.py"), "--backends", "aot_eager", "--color", "never"])
    out = capsys.readouterr().out

    assert code == EXIT_ERROR
    assert "no eager lane" in out
    assert "not checked: nothing was compared" in out


def test_the_main_path_shares_the_boundary_with_run_only(capsys):
    # Carry-over (0): one _guarded_run(), so a bad backend name reads the same
    # on both paths and neither shows a traceback.
    plain = main([str(FIXTURES / "mlp.py"), "--backends", "eager,bogus"])
    plain_err = capsys.readouterr().err
    hidden = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager,bogus"])
    hidden_err = capsys.readouterr().err

    assert plain == hidden == EXIT_ERROR
    assert plain_err == hidden_err
    assert "Traceback" not in plain_err
    assert "unknown backend 'bogus'" in plain_err


def test_the_main_path_says_which_flags_it_ignored(capsys, tmp_path):
    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager",
            "--json",
            str(tmp_path / "out.json"),
            "--md",
            str(tmp_path / "report.md"),
            "--color",
            "never",
        ]
    )
    err = capsys.readouterr().err

    assert code == EXIT_OK
    assert "--json is not implemented yet (it lands in M3), ignored" in err
    assert "--md is not implemented yet (it lands in M3), ignored" in err
    assert not (tmp_path / "out.json").exists()


def test_no_grad_switches_the_backward_pass_off(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--run-only", "--backends", "eager", "--no-grad"])
    out = capsys.readouterr().out

    assert code == EXIT_OK
    # The grads block only appears for a lane that ran a backward.
    assert "grads" not in out


def test_color_always_paints_and_never_does_not(capsys):
    main([str(FIXTURES / "mlp.py"), "--backends", "eager", "--color", "always"])
    painted = capsys.readouterr().out
    main([str(FIXTURES / "mlp.py"), "--backends", "eager", "--color", "never"])
    plain = capsys.readouterr().out

    assert "\033[" in painted
    assert "\033[" not in plain

    # Wall times are the one thing that legitimately differs between two runs
    # of the same target, so they are normalised out; everything else, colour
    # included, must be identical once the escapes are stripped.
    def normalise(report):
        return re.sub(r"\d+\.\d{4}s", "<time>", re.sub(r"\033\[[0-9;]*m", "", report))

    assert normalise(painted) == normalise(plain)


def test_color_auto_follows_the_terminal_and_no_color(monkeypatch):
    from compile_check.cli import _use_color

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    assert _use_color("auto") is True
    assert _use_color("never") is False

    monkeypatch.setenv("NO_COLOR", "1")
    assert _use_color("auto") is False
    assert _use_color("always") is True

    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr(sys.stdout, "isatty", lambda: False)
    assert _use_color("auto") is False


def test_max_findings_caps_each_oracle_group(capsys, monkeypatch):
    from compile_check import oracles as oracles_module

    class ThreeFailures:
        name = "metadata"

        def compare(self, eager, other, cfg):
            del eager, cfg
            return [
                Finding(
                    oracle="metadata",
                    backend=other.backend,
                    output_index=index,
                    severity="fail",
                    message=f"synthetic divergence {index}",
                    details={},
                )
                for index in range(3)
            ]

    monkeypatch.setitem(oracles_module.ORACLES, "metadata", ThreeFailures())
    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,aot_eager",
            "--max-findings",
            "1",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_FINDING
    assert "synthetic divergence 0" in out
    assert "synthetic divergence 1" not in out
    assert "2 more metadata findings not shown (--max-findings 1)" in out


def test_the_cache_variable_is_set_before_the_run_and_recorded_in_the_report(capsys, monkeypatch):
    # PLAN.md "Runner semantics": set before torch does any compiling. In this
    # process torch is already imported by the time a test runs, so what is
    # asserted here is that main() sets the variable and that the report says
    # which mode was in force.
    from compile_check.runner import CACHE_ENV_VAR

    monkeypatch.delenv(CACHE_ENV_VAR, raising=False)
    assert main([str(FIXTURES / "mlp.py"), "--backends", "eager", "--color", "never"]) == EXIT_OK
    assert os.environ[CACHE_ENV_VAR] == "1"
    assert "caches    disabled" in capsys.readouterr().out


def test_a_negative_max_findings_is_a_tool_error(capsys):
    code = main([str(FIXTURES / "mlp.py"), "--max-findings", "-1"])
    err = capsys.readouterr().err

    assert code == EXIT_ERROR
    assert "--max-findings must not be negative, got -1" in err


def test_max_findings_zero_counts_without_printing(capsys, monkeypatch):
    from compile_check import oracles as oracles_module

    class OneFailure:
        name = "metadata"

        def compare(self, eager, other, cfg):
            del eager, cfg
            return [
                Finding(
                    oracle="metadata",
                    backend=other.backend,
                    output_index=0,
                    severity="fail",
                    message="synthetic divergence",
                    details={},
                )
            ]

    monkeypatch.setitem(oracles_module.ORACLES, "metadata", OneFailure())
    code = main(
        [
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager,aot_eager",
            "--max-findings",
            "0",
            "--color",
            "never",
        ]
    )
    out = capsys.readouterr().out

    assert code == EXIT_FINDING
    assert "synthetic divergence" not in out
    assert "metadata  (1 fail)" in out
    assert "1 more metadata finding not shown (--max-findings 0)" in out


def test_a_fresh_process_disables_the_caches_before_torch_is_imported(tmp_path):
    # The in-process test above can only show that main() sets the variable;
    # this one shows it is set early enough to matter. The child starts with
    # TORCHINDUCTOR_FORCE_DISABLE_CACHES unset, so if main() set it after the
    # first import torch the config would read False and the report would say
    # so. PLAN.md "Runner semantics": verified that setting the variable makes
    # torch._inductor.config.force_disable_caches read True.
    from compile_check.runner import CACHE_ENV_VAR

    env = dict(os.environ)
    env.pop(CACHE_ENV_VAR, None)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "codegen")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compile_check.cli",
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager",
            "--color",
            "never",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert "caches    disabled (force_disable_caches=True)" in completed.stdout


def test_allow_caches_leaves_the_variable_alone_and_the_report_says_so(tmp_path):
    from compile_check.runner import CACHE_ENV_VAR

    env = dict(os.environ)
    env.pop(CACHE_ENV_VAR, None)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(tmp_path / "codegen")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "compile_check.cli",
            str(FIXTURES / "mlp.py"),
            "--backends",
            "eager",
            "--allow-caches",
            "--color",
            "never",
        ],
        capture_output=True,
        text=True,
        env=env,
    )

    assert completed.returncode == EXIT_OK, completed.stderr
    assert "ENABLED (force_disable_caches=False, --allow-caches)" in completed.stdout
