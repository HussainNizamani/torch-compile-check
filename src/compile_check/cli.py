"""argparse surface, exit codes.

PLAN.md "Package layout": ``cli.py`` -- argparse surface, exit codes.

The whole v1 flag surface from PLAN.md "CLI surface for v1" is parsed here. The
main path runs the target under every requested backend, compares the lanes with
the oracles, localizes the divergence to a compilation stage, prints the
terminal report, and picks an exit code. ``--version`` and ``--probe`` are
diagnostics that print and stop; ``--run-only`` is a hidden developer path that
dumps the raw records without a verdict.

Two decisions about ``--fail-on``, because the flag is easy to misread. It
selects which oracle *categories* turn a finding into exit code 1; it does not
select which oracles run. Every implemented oracle runs on every run, so the
report always shows the whole picture and only the verdict is configurable. And
a compiled backend that raised while eager did not is exit code 1 whatever
``--fail-on`` says: an exception belongs to no oracle category, and a lane that
could not run is not a lane that passed.

Torch is imported only inside functions, after the cache environment variable
has been set in ``main()``, and only on a path that actually runs a model. A
test asserts that importing this module does not import torch.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from compile_check import __version__
from compile_check.env import probe_apis
from compile_check.localize import StageVerdict, localize
from compile_check.oracles import ORACLE_NAMES, ORACLES, Finding, OracleConfig, run_oracles
from compile_check.report.terminal import DEFAULT_MAX_FINDINGS, render
from compile_check.results import RunSet

PROG = "compile-check"

log = logging.getLogger("compile_check")

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
COLOR_CHOICES = ("auto", "always", "never")

# Flags PLAN.md fixes in the v1 surface whose implementation lands later. Parsed
# from M0 so the surface does not move, and reported as ignored rather than
# silently accepted: a user who passed --json and got no file must be told why.
_PENDING_FLAGS: tuple[tuple[str, str], ...] = (
    ("json", "--json"),
    ("md", "--md"),
    ("budget", "--budget"),
    ("baseline", "--baseline"),
)

_EPILOG = """\
exit codes:
  0  clean
  1  at least one fail-severity finding in a --fail-on category, or a compiled
     backend that raised while eager did not
  2  tool error (import failure, discovery failure, backend unavailable,
     model raised in eager)

environment:
  TORCHINDUCTOR_FORCE_DISABLE_CACHES
     set to 1 by this tool before torch is imported, unless --allow-caches was
     passed; a run must measure the current compiler, not an artifact an earlier
     run cached
  TORCHINDUCTOR_CACHE_DIR
     where inductor writes the code it generates. Disabling the caches stops
     torch reading that directory, not writing to it, and nothing prunes it;
     point this at a scratch directory if disk is tight
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
    # Hidden: a developer path that runs discovery and the runner and prints
    # what came back, so the runner can be exercised end to end before M1-3
    # wires the oracles and the real report. Not part of the v1 surface.
    parser.add_argument(
        "--run-only",
        action="store_true",
        help=argparse.SUPPRESS,
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
            f"separated from {','.join(ORACLE_NAMES)} (default: {DEFAULT_FAIL_ON})"
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
        "--no-grad",
        action="store_true",
        help=(
            "skip the backward pass; without it one backward runs on a "
            "deterministic scalar reduction whenever anything requires grad"
        ),
    )
    parser.add_argument(
        "--share-module",
        action="store_true",
        help=(
            "run every backend against one module object instead of giving each "
            "lane its own deep copy; saves the memory of one copy of the weights "
            "and lets a buffer written by the forward pass leak into the next lane"
        ),
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
        help=(
            "do not set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1; the run gets faster "
            "and starts measuring whatever an earlier run cached, so the report "
            "records which mode was in force"
        ),
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
    # Presentation, not semantics: neither of the two below changes what is
    # checked or what the exit code is, only how much of it reaches the
    # terminal. That is why they are not in PLAN.md's flag table, which fixes
    # the semantic surface.
    parser.add_argument(
        "--max-findings",
        metavar="N",
        type=int,
        default=DEFAULT_MAX_FINDINGS,
        help=(
            "how many findings to print per oracle (default: "
            f"{DEFAULT_MAX_FINDINGS}, 0 for counts only); the rest are counted, "
            "never dropped"
        ),
    )
    parser.add_argument(
        "--color",
        choices=COLOR_CHOICES,
        default="auto",
        help=(
            "colourise the report (default: auto, which means colour when stdout "
            "is a terminal and NO_COLOR is unset)"
        ),
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

    if not args.allow_caches:
        # PLAN.md "Runner semantics": the caches are disabled by setting this
        # before torch does any compiling. Parts of the cache machinery read the
        # variable when torch is imported, and every torch import in this
        # package is lazy and happens below this line, which makes main() the
        # one place early enough. The import is local for the same reason: it
        # must not be able to drag torch in at module scope.
        from compile_check.runner import CACHE_ENV_VAR

        os.environ[CACHE_ENV_VAR] = "1"

    if args.probe:
        # --probe is a diagnostic that prints a table and exits; it produces no
        # run, so it has nothing to write to the report files. Silently dropping
        # them looked like a failed write, so say so.
        for flag, value in (("--json", args.json), ("--md", args.md)):
            if value is not None:
                print(f"{PROG}: {flag} ignored with --probe", file=sys.stderr)
        print(format_probe_table(probe_apis()))
        return EXIT_OK

    if args.max_findings < 0:
        # Not clamped silently: a negative cap is a typo, and a report that
        # quietly showed everything would hide it.
        return _tool_error(
            f"--max-findings must not be negative, got {args.max_findings} "
            "(0 prints the counts and none of the findings)"
        )

    if args.path is None:
        print(
            f"{PROG}: needs a target, a python file or a dotted module holding "
            "the model or function under test",
            file=sys.stderr,
        )
        parser.print_usage(sys.stderr)
        return EXIT_ERROR

    if args.run_only:
        return run_only(args)

    return run(args)


def run(args: argparse.Namespace) -> int:
    """The main path: run, compare, localize, report, exit code.

    Everything before the report goes through :func:`_guarded_run`, so a typo in
    ``--backends`` and a torch internal blowing up are reported the same way
    here as on the developer path, in one sentence and with no traceback.
    """
    for attribute, flag in _PENDING_FLAGS:
        if getattr(args, attribute) is not None:
            print(
                f"{PROG}: {flag} is not implemented yet (it lands in M3), ignored", file=sys.stderr
            )

    runset, fail_on = _guarded_run(args)
    if runset is None:
        return EXIT_ERROR

    cfg = OracleConfig(
        rtol=args.rtol,
        atol=args.atol,
        fp64=args.fp64_oracle,
        fp64_reference=runset.fp64,
    )
    findings = _compare_backends(runset, cfg)
    verdict = localize(runset, findings)
    print(
        render(
            runset,
            findings,
            verdict,
            fail_on=fail_on,
            max_findings=args.max_findings,
            color=_use_color(args.color),
        )
    )
    return _exit_code(findings, verdict, fail_on)


def _exit_code(
    findings: Sequence[Finding],
    verdict: StageVerdict,
    fail_on: Sequence[str],
) -> int:
    """PLAN.md "CLI surface for v1" exit codes, decided in one place.

    Three rules turn a verdict into one of the plan's three codes.

    Nothing was compared -- eager raised, or no eager lane ran -- is exit 2.
    PLAN.md "Runner semantics" makes eager the reference world and lists "model
    raised in eager" among the tool errors; a run with no reference is the same
    situation reached a different way, and reporting either as clean would be a
    lie of the worst kind for a testing tool.

    A compiled lane that raised while eager did not is exit 1 regardless of
    ``--fail-on``. The flag names oracle categories, an exception belongs to
    none of them, and a lane that could not run is not a lane that passed.

    Otherwise exit 1 when some fail-severity finding belongs to a ``--fail-on``
    category. ``warn`` and ``info`` never fail a run: PLAN.md "metadata" makes a
    contiguous-to-contiguous stride change a legitimate layout choice, and
    PLAN.md "The oracle blind spot" makes the fp64 distance context rather than
    a verdict.
    """
    if not verdict.compared:
        return EXIT_ERROR
    if any(entry.raised is not None for entry in verdict.backends):
        return EXIT_FINDING
    if any(f.severity == "fail" and f.oracle in fail_on for f in findings):
        return EXIT_FINDING
    return EXIT_OK


def _use_color(choice: str) -> bool:
    """Whether the report is painted, from ``--color`` and the terminal.

    ``auto`` follows the two conventions a user expects: colour when stdout is a
    terminal, and never when ``NO_COLOR`` is set in the environment, so a report
    piped into a file or a CI log stays plain text.
    """
    if choice == "always":
        return True
    if choice == "never":
        return False
    return sys.stdout.isatty() and not os.environ.get("NO_COLOR")


def _guarded_run(args: argparse.Namespace) -> tuple[RunSet | None, list[str]]:
    """Validate the flags, load the target, and run every backend.

    M1-1 review carry-over (0): the main path and ``--run-only`` share one
    exception boundary and one pair of validators. A user who typed a backend
    name wrong gets the same sentence on both paths, and the stack that produced
    it goes to the debug log rather than to the terminal.

    The order is deliberate. The flags are checked first, because a typo should
    be reported before a possibly slow import of the user's module; then the
    module, because a broken target should be reported before anything compiles.
    ``validate_device`` grows a branch here when ``--device`` gains ``mps`` or
    ``xpu``; the choices in :func:`build_parser` and that function are the two
    places a device is admitted.

    Returns:
        The runset and the parsed ``--fail-on`` categories, or ``(None, [])``
        once a one-line tool error has been printed to stderr.
    """
    from compile_check.discover import DiscoveryError, load_target
    from compile_check.runner import RunnerError, run_all, validate_backends, validate_device

    backends = [name.strip() for name in args.backends.split(",") if name.strip()]
    try:
        # Its own boundary, so that a ValueError out of the run below still
        # reports as the unexpected error it is rather than as a bad flag.
        fail_on = parse_fail_on(args.fail_on)
    except ValueError as exc:
        _tool_error(str(exc))
        return None, []

    try:
        validate_backends(backends)
        validate_device(args.device)
        target = load_target(args.path, entry=args.entry, inputs=args.inputs)
        runset = run_all(
            target,
            backends,
            device=args.device,
            seed=args.seed,
            fullgraph=args.fullgraph,
            dynamic=args.dynamic,
            grad=not args.no_grad,
            disable_caches=not args.allow_caches,
            fp64=args.fp64_oracle,
            share_module=args.share_module,
        )
    except (DiscoveryError, RunnerError) as exc:
        # Ours, with a message written for a user: print it as it is.
        _tool_error(str(exc))
        return None, []
    except Exception as exc:
        # Not ours: a torch internal, a bad --entry object, an OSError. The
        # class name carries information a bare message would not, and the
        # traceback is still available with logging turned up.
        log.debug("the run failed", exc_info=True)
        _tool_error(f"{type(exc).__name__}: {exc}")
        return None, []
    return runset, fail_on


def parse_fail_on(spec: str) -> list[str]:
    """Split and check ``--fail-on``, against the real oracle names.

    Args:
        spec: the comma-separated flag value.

    Returns:
        The requested categories, in the order given, duplicates dropped.

    Raises:
        ValueError: a category is not one of
            :data:`compile_check.oracles.ORACLE_NAMES`.
    """
    names = list(dict.fromkeys(name.strip() for name in spec.split(",") if name.strip()))
    unknown = [name for name in names if name not in ORACLE_NAMES]
    if unknown:
        raise ValueError(
            f"unknown --fail-on categor{'ies' if len(unknown) > 1 else 'y'} "
            f"{', '.join(repr(name) for name in unknown)}; the oracles are "
            f"{', '.join(ORACLE_NAMES)}"
        )
    return names


def run_only(args: argparse.Namespace) -> int:
    """Run discovery, the runner, and the oracles, then dump the raw records.

    The developer path behind ``--run-only``: no localization, no report, and no
    verdict, so nothing here decides whether a run is clean. Findings are
    printed and deliberately do not change the exit code; that wiring is
    :func:`run`. Exit 2 means the tool could not get as far as a reference
    result, which is PLAN.md's rule that a model raising in eager is a tool
    error.
    """
    runset, fail_on = _guarded_run(args)
    if runset is None:
        return EXIT_ERROR

    cfg = OracleConfig(
        rtol=args.rtol,
        atol=args.atol,
        fp64=args.fp64_oracle,
        fp64_reference=runset.fp64,
    )
    print(format_run_only(runset, _compare_backends(runset, cfg), fail_on=fail_on))

    eager = runset.eager
    if eager is not None and not eager.ok:
        return EXIT_ERROR
    return EXIT_OK


def _compare_backends(runset: RunSet, cfg: OracleConfig) -> list[Finding]:
    """Run every implemented oracle over every non-eager lane.

    Every oracle runs on every run, whatever ``--fail-on`` says. The flag
    decides which categories turn a finding into exit code 1, not which checks
    happen: a report that quietly stopped looking at aliasing because the user
    narrowed the exit-code rule would be a report that says less than it seems
    to.

    Without an eager lane there is nothing to compare against: PLAN.md "Runner
    semantics" makes eager the reference world, and a run of ``--backends
    inductor`` alone is a run with no reference, not a clean run.
    """
    eager = runset.eager
    if eager is None:
        log.warning("no eager lane in this run, so the oracles have no reference to compare with")
        return []
    findings: list[Finding] = []
    for result in runset.others:
        findings.extend(run_oracles(eager, result, cfg))
    return findings


def _tool_error(message: str) -> int:
    """Print one line on stderr and return the tool-error exit code.

    One line: a multi-line torch message is truncated to its first line with a
    marker, so the terminal never shows a wall of stack-shaped text on a path
    whose whole point is that it is not a crash.
    """
    lines = [line for line in message.splitlines() if line.strip()]
    first = lines[0] if lines else message
    if len(lines) > 1:
        first = f"{first} [+{len(lines) - 1} more lines, run with debug logging for the traceback]"
    print(f"{PROG}: {first}", file=sys.stderr)
    return EXIT_ERROR


def format_run_only(
    runset: RunSet,
    findings: Sequence[Finding] = (),
    *,
    fail_on: Sequence[str] = (),
) -> str:
    """Render a :class:`~compile_check.results.RunSet` for the developer path."""
    env = runset.env
    lines = [
        f"target     {runset.target_name}",
        f"device     {runset.device}   seed {runset.seed}   "
        f"fullgraph {runset.fullgraph}   dynamic {runset.dynamic}",
        f"torch      {env.get('torch_version')} on {env.get('machine')}, "
        f"python {env.get('python_version')}",
        f"caches     force_disable_caches={env.get('inductor_force_disable_caches')}",
    ]
    lines.append(f"oracles    {_format_oracles()}")
    if fail_on:
        # Named separately from the oracles that ran, because the two are
        # different questions: what was checked, and what would fail the run.
        lines.append(f"fail-on    {', '.join(fail_on)}")
    lines += [
        "",
        f"{'backend':<28}{'outputs':>8}{'first call':>13}{'second call':>13}  status",
    ]
    rows = list(runset.results.items())
    if runset.fp64 is not None:
        # Labelled as what it is: a reference, not a lane under test.
        rows.append((f"{runset.fp64.backend} (reference)", runset.fp64))
    for name, result in rows:
        # `result.exception is None` rather than `result.ok`: the property says
        # the same thing, but a reader that narrows types cannot see through it.
        status = "ok" if result.exception is None else f"raised {result.exception.type}"
        lines.append(
            f"{name:<28}{len(result.outputs):>8}"
            f"{_seconds(result.first_call_s):>13}{_seconds(result.second_call_s):>13}  {status}"
        )

    lines.append("")
    lines.append("findings")
    if runset.eager is None:
        # Not the same statement as "none": without the reference world nothing
        # was compared, and a report must not let those two read alike.
        lines.append("  not checked: this run has no eager lane to compare against")
    elif not findings:
        lines.append("  none")
    else:
        lines.extend(f"  {_format_finding(finding)}" for finding in findings)

    lines.append("")
    lines.append("outputs")
    for name, result in runset.results.items():
        if not result.outputs:
            lines.append(f"  {name}: none")
            continue
        for index, leaf in enumerate(result.outputs):
            lines.append(f"  {name}[{index}] {_describe(leaf)}")

    grads = [(name, r) for name, r in runset.results.items() if r.grad_ran]
    if grads:
        lines.append("")
        lines.append("grads")
        for name, result in grads:
            recorded = sum(1 for g in result.input_grads if g is not None)
            lines.append(
                f"  {name}: {recorded} of {len(result.input_grads)} inputs, "
                f"{len(result.param_grads)} parameters"
            )

    failures = [(name, r) for name, r in runset.results.items() if r.exception is not None]
    if failures:
        lines.append("")
        lines.append("exceptions")
        for name, result in failures:
            assert result.exception is not None
            lines.append(f"  {name}: {result.exception.type}: {result.exception.message}")
            lines.extend(f"    {line}" for line in result.exception.traceback)
    return "\n".join(lines)


def _seconds(value: float | None) -> str:
    """Format a wall time, or a dash when the call did not happen."""
    return "-" if value is None else f"{value:.4f}s"


def _format_oracles() -> str:
    """Name the oracles that ran, and the ones that do not exist yet.

    An oracle that has not been written must not read as an oracle that found
    nothing, which is the difference between "checked and clean" and "not
    checked".
    """
    live = [name for name in ORACLE_NAMES if name in ORACLES]
    pending = [name for name in ORACLE_NAMES if name not in ORACLES]
    text = ", ".join(live) if live else "none implemented yet"
    if pending:
        text += f"   (not implemented yet, nothing checked: {', '.join(pending)})"
    return text


def _format_finding(finding: Finding) -> str:
    """One finding on one line: severity, oracle, which output, message."""
    index = "-" if finding.output_index is None else str(finding.output_index)
    return f"[{finding.severity}] {finding.oracle} {finding.backend}[{index}] {finding.message}"


def _describe(leaf: Any) -> str:
    """One-line description of an output leaf, tensor or not."""
    dtype = getattr(leaf, "dtype", None)
    shape = getattr(leaf, "shape", None)
    if dtype is None or shape is None:
        return f"{type(leaf).__name__} {leaf!r}"
    return f"{dtype} {tuple(shape)}"


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
