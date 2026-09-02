"""Tests for the CLI surface: the two working paths, and the parsed-only rest."""

from __future__ import annotations

import importlib
import re
import subprocess
import sys
from pathlib import Path

import pytest

from compile_check import __version__
from compile_check.cli import (
    EXIT_ERROR,
    EXIT_OK,
    build_parser,
    format_probe_table,
    main,
)
from compile_check.env import PROBED_APIS

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

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


def test_unknown_path_exits_two(capsys):
    assert main(["model.py"]) == EXIT_ERROR
    assert "not implemented yet" in capsys.readouterr().err


def test_no_arguments_exits_two(capsys):
    assert main([]) == EXIT_ERROR
    assert "not implemented yet" in capsys.readouterr().err


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


def test_defaults_match_the_plan():
    args = build_parser().parse_args([])
    assert args.backends == "eager,aot_eager,inductor"
    assert args.device == "cpu"
    assert args.fail_on == "numerics,alias,metadata,grad"
    assert args.seed == 0
    assert args.fullgraph is False
    assert args.dynamic is False


def test_bad_device_is_a_tool_error():
    with pytest.raises(SystemExit) as excinfo:
        build_parser().parse_args(["model.py", "--device", "tpu"])
    assert excinfo.value.code == EXIT_ERROR


def test_every_module_imports():
    for name in PACKAGE_MODULES:
        assert importlib.import_module(name) is not None


def test_stubs_raise_not_implemented():
    # discover.py and runner.py landed in M1-1, the numerics and metadata
    # oracles in M1-2, and each is covered by its own test module; what is left
    # below is what M2 and M3 still owe.
    from compile_check import localize, minimize
    from compile_check.oracles import alias, grad, graph
    from compile_check.report import json as json_report
    from compile_check.report import markdown, pytest_case, terminal

    with pytest.raises(NotImplementedError):
        localize.implicated_stage("inductor")
    with pytest.raises(NotImplementedError):
        minimize.minimize(None, None, lambda _fn, _inputs: True)
    for oracle in (alias, grad, graph):
        with pytest.raises(NotImplementedError):
            oracle.check({}, {})
    with pytest.raises(NotImplementedError):
        terminal.render({})
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
