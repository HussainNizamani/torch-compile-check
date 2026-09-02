"""JSON report.

PLAN.md "Reports": JSON is versioned with a top-level ``schema_version``
integer, bumped on any incompatible field change. It carries the environment
block (architecture always included, see cross-architecture parity), the run
configuration, and one record per backend per oracle with a machine-readable
finding list. This is the CI-consumable artifact, and it is the unit of
comparison for cross-architecture parity.

The module is named after the artifact it writes, following PLAN.md "Package
layout"; the stdlib ``json`` module stays reachable from it because imports are
absolute.

The schema, version 1, which this docstring commits to and :func:`validate`
enforces::

    {
      "schema_version": 1,
      "tool":        {"name": str, "version": str},
      "target":      {"name": str, "file": str|null,
                      "entry": str|null, "inputs": str|null},
      "environment": {"torch_version": str|null, "torch_git_version": str|null,
                      "python_version": str|null, "platform": str|null,
                      "machine": str|null, "cpu_flags": str|null,
                      "cuda_available": bool|null,
                      "inductor_force_disable_caches": bool|null},
      "run":         {"device": str, "seed": int, "backends": [str],
                      "fullgraph": bool, "dynamic": bool, "grad": bool,
                      "fp64": bool, "module": str, "share_module": bool,
                      "target_is_module": bool, "module_copy_error": str|null,
                      "fail_on": [str], "grad_tol_factor": float,
                      "rtol": float|null, "atol": float|null,
                      "baseline": str|null},
      "backends":    [{"backend": str, "reference": bool, "ok": bool,
                       "outputs": int, "first_call_s": float|null,
                       "second_call_s": float|null, "exception": E|null,
                       "second_call_exception": E|null, "grad_ran": bool,
                       "grad_error": E|null, "graph": G|null}],
      "findings":    [{"oracle": str, "backend": str, "output_index": int|null,
                       "severity": "fail"|"warn"|"info", "message": str,
                       "details": object}],
      "verdict":     {"stage": str, "first_divergent_backend": str|null,
                      "summary": str, "note": str|null, "clean": bool,
                      "compared": bool, "backends": [B]},
      "counts":      {"fail": int, "warn": int, "info": int},
      "exit_code":   int|null
    }

    E = {"type": str, "message": str, "traceback": [str]}
    G = {"measured": bool, "graph_count": int, "break_count": int,
         "breaks": [{"reason": str, "summary": str, "user_frame": str|null}],
         "op_count": int, "compile_times": str|null, "recompiles": int,
         "unique_graphs_before": int|null, "unique_graphs_after": int|null,
         "explain_error": E|null}
    B = {"backend": str, "fail": int, "warn": int, "info": int,
         "graph_fail": int, "raised": str|null, "raised_on_repeat": str|null}

Two things are deliberately not in it. There is no timestamp: PLAN.md
"Cross-architecture parity is a feature" makes parity in v1 "running the tool on
two machines and diffing the two JSON files", and a field that differs on every
run for no reason makes that diff worse. And there is no ``compare`` support
here: a first-class ``compile-check compare a.json b.json`` subcommand is v0.2,
so v1 emits the artifact and leaves the diffing to ``diff``.

Validation is hand-rolled, in :func:`validate`, because PLAN.md "Engineering
decisions" keeps torch the only runtime dependency and a schema this size does
not earn ``jsonschema``.

Nothing here imports torch: the artifact is built from the records the runner
and the oracles already produced.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from compile_check import __version__
from compile_check.localize import StageVerdict
from compile_check.oracles import DEFAULT_GRAD_TOL_FACTOR, ORACLE_NAMES, SEVERITIES, Finding
from compile_check.oracles.graph import summarise_reason
from compile_check.results import BackendResult, CapturedException, GraphHealth, RunSet
from compile_check.runner import FP64_BACKEND

__all__ = ["SCHEMA_VERSION", "build", "dump", "validate"]

log = logging.getLogger("compile_check")

SCHEMA_VERSION = 1
"""Bumped on any incompatible field change, per PLAN.md "Reports"."""

# Wall times are rounded to the same four decimals the terminal report prints.
# A CI artifact that carried sixteen digits of a number nobody compares would
# only add noise to the parity diff the schema exists for.
_TIME_DIGITS = 4

_TYPE_NAMES = {
    type(None): "null",
    bool: "boolean",
    int: "integer",
    float: "number",
    str: "string",
    list: "array",
    dict: "object",
}


def build(
    runset: RunSet,
    findings: Sequence[Finding],
    verdict: StageVerdict,
    *,
    fail_on: Sequence[str] = (),
    grad_tol_factor: float = DEFAULT_GRAD_TOL_FACTOR,
    rtol: float | None = None,
    atol: float | None = None,
    baseline: str | None = None,
    fp64: bool = False,
    exit_code: int | None = None,
) -> dict[str, Any]:
    """Build the JSON document for one run.

    Args:
        runset: the run, from :func:`compile_check.runner.run_all`.
        findings: every finding the oracles produced.
        verdict: the stage verdict, from :func:`compile_check.localize.localize`.
        fail_on: the ``--fail-on`` categories that decided the exit code.
        grad_tol_factor: ``--grad-tol-factor``, recorded for the same reason the
            terminal report prints it: a clean grad row means a different thing
            at 10x than at 1x.
        rtol: ``--rtol``, or ``None`` for the per-dtype defaults.
        atol: ``--atol``, likewise.
        baseline: the ``--baseline`` path, when one was in force.
        fp64: whether ``--fp64-oracle`` added the reference lane.
        exit_code: the code the CLI is about to return, so a CI job reading the
            artifact does not have to re-derive it.

    Returns:
        The document, ready for :func:`dump`. Every value is JSON-native; a
        finding detail that is not is coerced to its string form rather than
        breaking the write.
    """
    lanes = [(name, result, False) for name, result in runset.results.items()]
    if runset.fp64 is not None:
        # Labelled as what it is. PLAN.md "The oracle blind spot" makes the fp64
        # pass a reference the numerics oracle reads, never a lane under test,
        # so a consumer counting failures has to be able to skip it.
        lanes.append((FP64_BACKEND, runset.fp64, True))

    source = runset.target_source
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": {"name": "compile-check", "version": __version__},
        "target": {
            "name": runset.target_name,
            "file": source.file if source is not None else None,
            "entry": source.entry if source is not None else None,
            "inputs": source.inputs if source is not None else None,
        },
        # Passed through as collected, so that the JSON block and the terminal
        # block cannot come to disagree about the machine a run happened on.
        "environment": {key: _jsonable(value) for key, value in runset.env.items()},
        "run": {
            "device": runset.device,
            "seed": runset.seed,
            "backends": runset.backends,
            "fullgraph": runset.fullgraph,
            "dynamic": runset.dynamic,
            "grad": runset.grad,
            "fp64": fp64,
            # What happened to the module, not what was asked for: a run whose
            # lanes shared one object because the deep copy failed has to say so
            # in the artifact as well as on the terminal.
            "module": runset.module_handling,
            "share_module": runset.share_module,
            "target_is_module": runset.target_is_module,
            "module_copy_error": runset.module_copy_error,
            "fail_on": list(fail_on),
            "grad_tol_factor": grad_tol_factor,
            "rtol": rtol,
            "atol": atol,
            "baseline": baseline,
        },
        "backends": [_backend(name, result, reference) for name, result, reference in lanes],
        "findings": [_finding(finding) for finding in findings],
        "verdict": _verdict(verdict),
        "counts": {
            severity: sum(1 for finding in findings if finding.severity == severity)
            for severity in SEVERITIES
        },
        "exit_code": exit_code,
    }


def dump(result: Mapping[str, Any], path: Path) -> None:
    """Write a run result to *path* as versioned JSON.

    The document is validated before it is written. A report that does not match
    its own schema is a bug in this tool, and writing it anyway would hand a CI
    job an artifact its consumer cannot read.

    Args:
        result: the document, from :func:`build`.
        path: the file to write.

    Raises:
        ValueError: the document does not match the schema.
        OSError: the file could not be written.
    """
    problems = validate(result)
    if problems:
        raise ValueError(
            f"the JSON report does not match schema version {SCHEMA_VERSION}: "
            + "; ".join(problems)
        )
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False: json.dump would otherwise write NaN and Infinity, which
    # are not JSON and which a strict parser on the other side of a CI artifact
    # upload refuses. _jsonable has already turned any non-finite number into
    # its string form, so reaching that error means a value got past it.
    path.write_text(
        json.dumps(result, indent=2, sort_keys=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def validate(document: Any) -> list[str]:
    """Check a document against the schema in this module's docstring.

    Hand-rolled, and shaped as a list of problems rather than an exception, so
    that a test or a consumer sees every mismatch at once instead of the first.

    Args:
        document: the parsed JSON, or the mapping :func:`build` returned.

    Returns:
        One sentence per problem, empty when the document is valid.
    """
    problems: list[str] = []
    if not isinstance(document, dict):
        return [f"the document is a {_name(document)}, expected an object"]

    _fields(
        problems,
        "",
        document,
        {
            "schema_version": (int,),
            "tool": (dict,),
            "target": (dict,),
            "environment": (dict,),
            "run": (dict,),
            "backends": (list,),
            "findings": (list,),
            "verdict": (dict,),
            "counts": (dict,),
            "exit_code": (int, type(None)),
        },
    )
    if problems:
        # Everything below indexes into those keys; reporting "run.device is
        # missing" when `run` itself is a list would be noise on top of noise.
        return problems

    if document["schema_version"] != SCHEMA_VERSION:
        problems.append(
            f"schema_version is {document['schema_version']}, this build writes {SCHEMA_VERSION}"
        )

    _fields(problems, "tool.", document["tool"], {"name": (str,), "version": (str,)})
    _fields(
        problems,
        "target.",
        document["target"],
        {
            "name": (str,),
            "file": (str, type(None)),
            "entry": (str, type(None)),
            "inputs": (str, type(None)),
        },
    )
    _fields(
        problems,
        "environment.",
        document["environment"],
        {
            "torch_version": (str, type(None)),
            "torch_git_version": (str, type(None)),
            "python_version": (str, type(None)),
            "platform": (str, type(None)),
            "machine": (str, type(None)),
            "cpu_flags": (str, type(None)),
            "cuda_available": (bool, type(None)),
            "inductor_force_disable_caches": (bool, type(None)),
        },
    )
    _fields(
        problems,
        "run.",
        document["run"],
        {
            "device": (str,),
            "seed": (int,),
            "backends": (list,),
            "fullgraph": (bool,),
            "dynamic": (bool,),
            "grad": (bool,),
            "fp64": (bool,),
            "module": (str,),
            "share_module": (bool,),
            "target_is_module": (bool,),
            "module_copy_error": (str, type(None)),
            "fail_on": (list,),
            "grad_tol_factor": (int, float),
            "rtol": (int, float, type(None)),
            "atol": (int, float, type(None)),
            "baseline": (str, type(None)),
        },
    )
    _fields(problems, "counts.", document["counts"], dict.fromkeys(SEVERITIES, (int,)))

    for index, entry in enumerate(document["backends"]):
        _backend_problems(problems, f"backends[{index}]", entry)
    for index, entry in enumerate(document["findings"]):
        _finding_problems(problems, f"findings[{index}]", entry)
    _verdict_problems(problems, document["verdict"])
    problems.extend(_unserialisable("", document))
    return problems


def _backend(name: str, result: BackendResult, reference: bool) -> dict[str, Any]:
    """One lane's record: what it produced, how long it took, how it failed."""
    return {
        "backend": name,
        "reference": reference,
        "ok": result.ok,
        "outputs": len(result.outputs),
        "first_call_s": _seconds(result.first_call_s),
        "second_call_s": _seconds(result.second_call_s),
        "exception": _exception(result.exception),
        # Its own field, as on the record: a lane that raises on the first call
        # produced nothing, and a lane that answers once and then throws
        # produced an answer that does not reproduce.
        "second_call_exception": _exception(result.second_call_exception),
        "grad_ran": result.grad_ran,
        "grad_error": _exception(result.grad_error),
        "graph": _graph(result.graph_health),
    }


def _graph(health: GraphHealth | None) -> dict[str, Any] | None:
    """One lane's graph health, or ``None`` for a lane that was never compiled.

    ``None`` and ``measured: false`` are two different statements -- no graphs
    to measure, and graphs that could not be measured -- and a consumer that
    read either as "no graph breaks" would draw the wrong conclusion.
    """
    if health is None:
        return None
    return {
        "measured": health.measured,
        "graph_count": health.graph_count,
        "break_count": health.break_count,
        "breaks": [
            {
                "reason": item.reason,
                # The same one-line identity a baseline stores and compares on,
                # from the one function that writes it, so an artifact and a
                # baseline file cannot disagree about what a break is called.
                "summary": summarise_reason(item.reason),
                "user_frame": item.user_frame,
            }
            for item in health.breaks
        ],
        "op_count": health.op_count,
        "compile_times": health.compile_times,
        "recompiles": health.recompiles,
        "unique_graphs_before": health.unique_graphs_before,
        "unique_graphs_after": health.unique_graphs_after,
        "explain_error": _exception(health.explain_error),
    }


def _finding(finding: Finding) -> dict[str, Any]:
    """One finding, with its details coerced to JSON-native values."""
    return {
        "oracle": finding.oracle,
        "backend": finding.backend,
        "output_index": finding.output_index,
        "severity": finding.severity,
        "message": finding.message,
        "details": {str(key): _jsonable(value) for key, value in finding.details.items()},
    }


def _verdict(verdict: StageVerdict) -> dict[str, Any]:
    """The stage verdict and the per-lane counts behind it."""
    return {
        "stage": verdict.stage,
        "first_divergent_backend": verdict.first_divergent_backend,
        "summary": verdict.summary,
        "note": verdict.note,
        "clean": verdict.clean,
        # "checked and clean" against "not checked", which is the distinction
        # the whole report is built to keep visible.
        "compared": verdict.compared,
        "backends": [
            {
                "backend": entry.backend,
                "fail": entry.fail,
                "warn": entry.warn,
                "info": entry.info,
                "graph_fail": entry.graph_fail,
                "raised": entry.raised.type if entry.raised is not None else None,
                "raised_on_repeat": (
                    entry.raised_on_repeat.type if entry.raised_on_repeat is not None else None
                ),
            }
            for entry in verdict.backends
        ],
    }


def _exception(exception: CapturedException | None) -> dict[str, Any] | None:
    """One captured exception, or ``None`` when nothing raised."""
    if exception is None:
        return None
    return {
        "type": exception.type,
        "message": exception.message,
        "traceback": list(exception.traceback),
    }


def _seconds(value: float | None) -> float | None:
    """A wall time rounded for an artifact, or ``None`` if the call never ran."""
    return None if value is None else round(value, _TIME_DIGITS)


def _jsonable(value: Any) -> Any:
    """Coerce one value to something ``json.dumps`` writes and a parser reads.

    A finding's ``details`` is documented as strings, numbers, and lists, and
    every oracle in this build keeps to that. This is the guard for the one that
    does not: an artifact is worth more with one field stringified than not
    written at all, so an unknown object becomes its ``str()`` rather than a
    ``TypeError`` out of the report. Non-finite floats go the same way, because
    ``NaN`` and ``Infinity`` are Python's JSON extension and not JSON.
    """
    if value is None or isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple | set | frozenset):
        return [_jsonable(item) for item in value]
    return str(value)


def _fields(
    problems: list[str],
    prefix: str,
    value: Any,
    expected: Mapping[str, tuple[type, ...]],
) -> None:
    """Check one object's fields, reporting a missing or wrongly typed one."""
    if not isinstance(value, dict):
        problems.append(
            f"{prefix.rstrip('.') or 'the document'} is a {_name(value)}, expected an object"
        )
        return
    for name, types in expected.items():
        if name not in value:
            problems.append(f"{prefix}{name} is missing")
            continue
        found = value[name]
        # bool is an int in Python, so an integer field would accept `true`
        # silently; a count of `true` findings is a typo, not a count of one.
        if isinstance(found, bool) and bool not in types:
            problems.append(f"{prefix}{name} is a boolean, expected {_expected(types)}")
        elif not isinstance(found, types):
            problems.append(f"{prefix}{name} is a {_name(found)}, expected {_expected(types)}")


def _backend_problems(problems: list[str], prefix: str, entry: Any) -> None:
    """One entry of the ``backends`` array."""
    _fields(
        problems,
        f"{prefix}.",
        entry,
        {
            "backend": (str,),
            "reference": (bool,),
            "ok": (bool,),
            "outputs": (int,),
            "first_call_s": (int, float, type(None)),
            "second_call_s": (int, float, type(None)),
            "exception": (dict, type(None)),
            "second_call_exception": (dict, type(None)),
            "grad_ran": (bool,),
            "grad_error": (dict, type(None)),
            "graph": (dict, type(None)),
        },
    )


def _finding_problems(problems: list[str], prefix: str, entry: Any) -> None:
    """One entry of the ``findings`` array."""
    _fields(
        problems,
        f"{prefix}.",
        entry,
        {
            "oracle": (str,),
            "backend": (str,),
            "output_index": (int, type(None)),
            "severity": (str,),
            "message": (str,),
            "details": (dict,),
        },
    )
    if not isinstance(entry, dict):
        return
    # The two closed vocabularies: an unknown oracle name or severity means the
    # artifact came from a build whose findings this one cannot count.
    if isinstance(entry.get("oracle"), str) and entry["oracle"] not in ORACLE_NAMES:
        problems.append(
            f"{prefix}.oracle is {entry['oracle']!r}, not one of {', '.join(ORACLE_NAMES)}"
        )
    if isinstance(entry.get("severity"), str) and entry["severity"] not in SEVERITIES:
        problems.append(
            f"{prefix}.severity is {entry['severity']!r}, not one of {', '.join(SEVERITIES)}"
        )


def _verdict_problems(problems: list[str], verdict: Any) -> None:
    """The ``verdict`` object and its per-lane summaries."""
    _fields(
        problems,
        "verdict.",
        verdict,
        {
            "stage": (str,),
            "first_divergent_backend": (str, type(None)),
            "summary": (str,),
            "note": (str, type(None)),
            "clean": (bool,),
            "compared": (bool,),
            "backends": (list,),
        },
    )
    if not isinstance(verdict, dict) or not isinstance(verdict.get("backends"), list):
        return
    for index, entry in enumerate(verdict["backends"]):
        _fields(
            problems,
            f"verdict.backends[{index}].",
            entry,
            {
                "backend": (str,),
                "fail": (int,),
                "warn": (int,),
                "info": (int,),
                "graph_fail": (int,),
                "raised": (str, type(None)),
                "raised_on_repeat": (str, type(None)),
            },
        )


def _unserialisable(prefix: str, value: Any) -> list[str]:
    """Every value in the document that ``json.dumps`` could not write.

    Walked rather than left to ``json.dumps`` to raise, because the point of
    :func:`validate` is to say what is wrong and where, and a ``TypeError``
    naming a class says neither.
    """
    if value is None or isinstance(value, bool | int | str):
        return []
    if isinstance(value, float):
        return (
            []
            if math.isfinite(value)
            else [f"{prefix or 'the document'} is {value}, which is not valid JSON"]
        )
    if isinstance(value, dict):
        problems: list[str] = []
        for key, item in value.items():
            if not isinstance(key, str):
                problems.append(f"{prefix or 'the document'} has a non-string key {key!r}")
            problems.extend(_unserialisable(f"{prefix}.{key}" if prefix else str(key), item))
        return problems
    if isinstance(value, list):
        problems = []
        for index, item in enumerate(value):
            problems.extend(_unserialisable(f"{prefix}[{index}]", item))
        return problems
    return [f"{prefix or 'the document'} is a {_name(value)}, which is not valid JSON"]


def _name(value: Any) -> str:
    """A value's type, in the JSON vocabulary where there is one."""
    return _TYPE_NAMES.get(type(value), type(value).__name__)


def _expected(types: tuple[type, ...]) -> str:
    """The accepted types of one field, as one phrase."""
    return " or ".join(_TYPE_NAMES.get(item, item.__name__) for item in types)
