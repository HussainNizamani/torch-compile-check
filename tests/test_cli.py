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
    assert "not implemented in M0" in capsys.readouterr().err


def test_no_arguments_exits_two(capsys):
    assert main([]) == EXIT_ERROR
    assert "not implemented in M0" in capsys.readouterr().err


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
    # discover.py and runner.py landed in M1-1 and are covered by their own
    # test modules; what is left below is what M1-2 and M1-3 still owe.
    from compile_check import localize, minimize
    from compile_check.oracles import alias, grad, graph, metadata, numerics
    from compile_check.report import json as json_report
    from compile_check.report import markdown, pytest_case, terminal

    with pytest.raises(NotImplementedError):
        localize.implicated_stage("inductor")
    with pytest.raises(NotImplementedError):
        minimize.minimize(None, None, lambda _fn, _inputs: True)
    for oracle in (numerics, alias, metadata, grad, graph):
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
