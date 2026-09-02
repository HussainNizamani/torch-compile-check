"""argparse surface, exit codes.

PLAN.md "Package layout": ``cli.py`` -- argparse surface, exit codes.

The whole v1 flag surface from PLAN.md "CLI surface for v1" is parsed here, so
that the surface is fixed and testable from M0 onwards. Two paths do real work
in M0, ``--version`` and ``--probe``; every other invocation exits 2, which is
the plan's exit code for a tool error.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Mapping, Sequence

from compile_check import __version__
from compile_check.env import probe_apis
from compile_check.oracles import ORACLES

PROG = "compile-check"

# PLAN.md "CLI surface for v1", exit codes: 0 clean, 1 at least one finding in a
# --fail-on category, 2 tool error.
EXIT_OK = 0
EXIT_FINDING = 1
EXIT_ERROR = 2

DEFAULT_BACKENDS = "eager,aot_eager,inductor"
# PLAN.md "GitHub Action": the correctness categories always fail; the graph
# oracle is informational unless it is asked for explicitly.
DEFAULT_FAIL_ON = "numerics,alias,metadata,grad"
DEFAULT_SEED = 0

NOT_IMPLEMENTED_MESSAGE = (
    f"{PROG}: not implemented in M0 (scaffold only); the paths that work are --version and --probe"
)

_EPILOG = """\
exit codes:
  0  clean
  1  at least one finding in a --fail-on category
  2  tool error (import failure, discovery failure, backend unavailable,
     model raised in eager)
"""


def build_parser() -> argparse.ArgumentParser:
    """Build the v1 argument parser."""
    parser = argparse.ArgumentParser(
        prog=PROG,
        description=(
            "Differential testing for torch.compile: run a model under eager and "
            "under compiled backends and report whether compilation changed the answers."
        ),
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        nargs="?",
        help="python file holding the model or function under test",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"{PROG} {__version__}",
        help="print the version and exit",
    )
    parser.add_argument(
        "--probe",
        action="store_true",
        help="print the torch API probe as a table and exit",
    )
    parser.add_argument(
        "--entry",
        metavar="module:callable",
        help="override discovery, name the model or function directly",
    )
    parser.add_argument(
        "--inputs",
        metavar="module:callable",
        help="override discovery, name the input factory",
    )
    parser.add_argument(
        "--backends",
        metavar="LIST",
        default=DEFAULT_BACKENDS,
        help=(
            "which backends to run, comma separated "
            f"(default: {DEFAULT_BACKENDS}; aot_eager_decomp_partition is an "
            "optional fourth lane for finer stage localization)"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="device to place the model and inputs on (default: cpu)",
    )
    parser.add_argument(
        "--json",
        metavar="OUT.JSON",
        help="write the versioned JSON result",
    )
    parser.add_argument(
        "--md",
        metavar="REPORT.MD",
        help="write the Markdown issue draft",
    )
    parser.add_argument(
        "--fail-on",
        metavar="LIST",
        default=DEFAULT_FAIL_ON,
        help=(
            "which oracle categories turn a finding into exit code 1, comma "
            f"separated from {','.join(ORACLES)} (default: {DEFAULT_FAIL_ON})"
        ),
    )
    parser.add_argument(
        "--fullgraph",
        action="store_true",
        help="pass fullgraph=True to torch.compile (default: False)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        help="add a second pass with dynamic=True",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        help="override the numerics relative tolerance for every dtype",
    )
    parser.add_argument(
        "--atol",
        type=float,
        help="override the numerics absolute tolerance for every dtype",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed (default: {DEFAULT_SEED})",
    )
    parser.add_argument(
        "--allow-caches",
        action="store_true",
        help="do not set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1",
    )
    parser.add_argument(
        "--fp64-oracle",
        action="store_true",
        help="add an fp64 eager reference run to the numerics oracle",
    )
    parser.add_argument(
        "--budget",
        metavar="SECONDS",
        type=float,
        help="wall-clock ceiling for the whole run, for CI use",
    )
    parser.add_argument(
        "--baseline",
        metavar="FILE",
        help="stored graph-health baseline, so the graph oracle fails on new breaks only",
    )
    return parser


def format_probe_table(probe: Mapping[str, bool]) -> str:
    """Render ``probe_apis()`` output as a two-column table."""
    width = max((len(name) for name in probe), default=len("api"))
    lines = [f"{'api':<{width}}  status", f"{'-' * width}  -------"]
    lines += [
        f"{name:<{width}}  {'present' if present else 'absent'}" for name, present in probe.items()
    ]
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the ``compile-check`` console script."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.probe:
        # --probe is a diagnostic that prints a table and exits; it produces no
        # run, so it has nothing to write to the report files. Silently dropping
        # them looked like a failed write, so say so.
        for flag, value in (("--json", args.json), ("--md", args.md)):
            if value is not None:
                print(f"{PROG}: {flag} ignored with --probe", file=sys.stderr)
        print(format_probe_table(probe_apis()))
        return EXIT_OK

    print(NOT_IMPLEMENTED_MESSAGE, file=sys.stderr)
    return EXIT_ERROR


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
