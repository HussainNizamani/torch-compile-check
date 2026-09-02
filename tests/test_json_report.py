"""Tests for the JSON report and its hand-rolled validator.

Built from synthetic records for the same reason the terminal report's tests
are: the artifact is a function of the records, so it can be pinned exactly
without paying for a compile. The one thing torch is needed for -- that a real
run's document validates -- is in tests/test_cli.py, where the run already
happens.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from compile_check import __version__
from compile_check.localize import localize
from compile_check.minimize import Kept, Minimization, Shrink, Stub
from compile_check.oracles import Finding
from compile_check.report.json import SCHEMA_VERSION, build, dump, validate
from compile_check.results import (
    BackendResult,
    CapturedException,
    GraphBreak,
    GraphHealth,
    RunSet,
    TargetSource,
)

ENV = {
    "torch_version": "2.14.0+cpu",
    "torch_git_version": "08187d9e0fba1234567890",
    "python_version": "3.10.12",
    "platform": "Linux-6.8.0-1057-oracle-aarch64-with-glibc2.35",
    "machine": "aarch64",
    "cpu_flags": "asimd asimdhp asimddp",
    "cuda_available": False,
    "inductor_force_disable_caches": True,
}

FINDING = Finding(
    oracle="metadata",
    backend="inductor",
    output_index=0,
    severity="fail",
    message="dtype differs: eager torch.int8, inductor torch.int64",
    details={"field": "dtype", "expected": "torch.int8", "got": "torch.int64"},
)


@pytest.fixture
def runset() -> RunSet:
    """Two lanes, one of them compiled, with graph health on the compiled one."""
    built = RunSet(
        target_name="dtype_promotion:fn",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        target_is_module=False,
        target_source=TargetSource(
            file="cases/dtype_promotion.py",
            text="import torch\n\n\ndef fn(a):\n    return a\n\n\ninputs = (torch.ones(2),)\n",
            entry="fn",
            inputs="inputs",
        ),
        env=dict(ENV),
    )
    built.results["eager"] = BackendResult(
        backend="eager", outputs=[1], first_call_s=0.0005, second_call_s=0.0001
    )
    built.results["inductor"] = BackendResult(
        backend="inductor",
        outputs=[1],
        first_call_s=3.5,
        second_call_s=0.0004,
        graph_health=GraphHealth(
            graph_count=1,
            break_count=1,
            breaks=(GraphBreak(reason="Failed to trace builtin operator", user_frame="m.py:3"),),
            op_count=3,
            unique_graphs_before=1,
            unique_graphs_after=1,
        ),
    )
    return built


def document(runset: RunSet, findings=(FINDING,), **kwargs) -> dict:
    """The artifact for one run, with the verdict derived as the CLI does."""
    return build(runset, list(findings), localize(runset, list(findings)), **kwargs)


def test_a_document_validates_against_its_own_schema(runset):
    assert validate(document(runset)) == []


def test_the_document_survives_a_json_round_trip(runset, tmp_path):
    path = tmp_path / "out.json"
    dump(document(runset), path)

    # Read back the way a CI consumer reads it, with the strict parser: the
    # artifact has to be JSON, not Python's superset of it.
    reloaded = json.loads(path.read_text(encoding="utf-8"), parse_constant=_no_constants)
    assert validate(reloaded) == []
    assert reloaded == document(runset)


def _no_constants(name: str) -> object:  # pragma: no cover - only runs on a bad document
    raise AssertionError(f"the artifact carries {name}, which is not valid JSON")


def test_the_top_level_shape_is_the_documented_one(runset):
    built = document(runset, exit_code=1)

    assert built["schema_version"] == SCHEMA_VERSION
    assert built["tool"] == {"name": "compile-check", "version": __version__}
    assert built["target"] == {
        "name": "dtype_promotion:fn",
        "file": "cases/dtype_promotion.py",
        "entry": "fn",
        "inputs": "inputs",
    }
    assert built["environment"] == ENV
    assert built["counts"] == {"fail": 1, "warn": 0, "info": 0}
    assert built["exit_code"] == 1


def test_every_lane_carries_its_timings_and_its_exceptions(runset):
    runset.results["inductor"].exception = CapturedException(
        type="RuntimeError", message="boom", traceback=("line one",)
    )
    runset.results["inductor"].second_call_exception = CapturedException(
        type="AssertionError", message="second", traceback=()
    )
    built = document(runset, findings=())

    lanes = {entry["backend"]: entry for entry in built["backends"]}
    assert lanes["eager"]["first_call_s"] == pytest.approx(0.0005)
    assert lanes["eager"]["exception"] is None
    assert lanes["inductor"]["exception"] == {
        "type": "RuntimeError",
        "message": "boom",
        "traceback": ["line one"],
    }
    assert lanes["inductor"]["second_call_exception"]["type"] == "AssertionError"
    assert validate(built) == []


def test_graph_health_carries_the_baseline_summary_of_every_break(runset):
    graph = {entry["backend"]: entry["graph"] for entry in document(runset)["backends"]}

    # None for a lane that was never compiled, which is not the same statement
    # as a lane with no graph breaks.
    assert graph["eager"] is None
    assert graph["inductor"]["break_count"] == 1
    assert graph["inductor"]["breaks"] == [
        {
            "reason": "Failed to trace builtin operator",
            "summary": "Failed to trace builtin operator",
            "user_frame": "m.py:3",
        }
    ]


def test_the_fp64_reference_is_carried_and_labelled(runset):
    runset.fp64 = BackendResult(backend="eager_fp64", outputs=[1], first_call_s=0.02)
    built = document(runset)

    reference = [entry for entry in built["backends"] if entry["reference"]]
    assert [entry["backend"] for entry in reference] == ["eager_fp64"]
    # It is not a lane under test, so it is not in the run's backend list either.
    assert built["run"]["backends"] == ["eager", "inductor"]


def test_the_run_block_records_the_module_handling_that_happened(runset):
    runset.target_is_module = True
    runset.module_copy_error = "TypeError: cannot pickle 'module' object"
    built = document(runset)

    assert built["run"]["module"] == (
        "shared across every lane (deep copy failed: TypeError: cannot pickle 'module' object)"
    )
    assert built["run"]["module_copy_error"] == "TypeError: cannot pickle 'module' object"
    assert built["run"]["share_module"] is False


def test_the_run_block_records_the_comparison_knobs(runset):
    built = document(
        runset,
        fail_on=["numerics", "graph"],
        grad_tol_factor=1.0,
        rtol=1e-5,
        atol=1e-8,
        baseline="baseline.json",
        fp64=True,
    )

    assert built["run"]["fail_on"] == ["numerics", "graph"]
    assert built["run"]["grad_tol_factor"] == pytest.approx(1.0)
    assert built["run"]["rtol"] == pytest.approx(1e-5)
    assert built["run"]["atol"] == pytest.approx(1e-8)
    assert built["run"]["baseline"] == "baseline.json"
    assert built["run"]["fp64"] is True


def test_the_verdict_and_its_per_lane_counts_are_carried(runset):
    verdict = document(runset)["verdict"]

    assert verdict["stage"] == "inductor lowering/codegen"
    assert verdict["first_divergent_backend"] == "inductor"
    assert verdict["clean"] is False
    assert verdict["compared"] is True
    counts = {entry["backend"]: entry for entry in verdict["backends"]}
    assert counts["inductor"]["fail"] == 1
    assert counts["inductor"]["graph_fail"] == 0
    assert counts["eager"]["raised"] is None


def test_a_detail_value_that_is_not_json_is_coerced_rather_than_dropped(runset):
    finding = Finding(
        oracle="numerics",
        backend="inductor",
        output_index=0,
        severity="fail",
        message="values differ",
        details={
            "field": "values",
            "rtol": float("nan"),
            "shape": (2, 3),
            "leaf": object(),
        },
    )
    built = document(runset, findings=[finding])
    details = built["findings"][0]["details"]

    # NaN and Infinity are Python's JSON extension, not JSON; a tuple is a list;
    # anything else becomes its str(), because an artifact with one field
    # stringified is worth more than no artifact.
    assert details["rtol"] == "nan"
    assert details["shape"] == [2, 3]
    assert details["leaf"].startswith("<object object")
    assert validate(built) == []


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda d: d.pop("counts"), "counts is missing"),
        (lambda d: d.update(schema_version="1"), "schema_version is a string"),
        (lambda d: d.update(exit_code=True), "exit_code is a boolean"),
        (lambda d: d["run"].update(seed=None), "run.seed is a null"),
        (lambda d: d["environment"].pop("machine"), "environment.machine is missing"),
        (lambda d: d["findings"][0].update(oracle="vibes"), "findings[0].oracle is 'vibes'"),
        (lambda d: d["findings"][0].update(severity="bad"), "findings[0].severity is 'bad'"),
        (lambda d: d["backends"][0].update(outputs="two"), "backends[0].outputs is a string"),
        (lambda d: d["verdict"].update(clean="yes"), "verdict.clean is a string"),
        (
            lambda d: d["verdict"]["backends"][0].update(fail=None),
            "verdict.backends[0].fail is a null",
        ),
    ],
)
def test_the_validator_names_what_is_wrong_and_where(runset, mutate, expected):
    built = document(runset)
    mutate(built)
    problems = validate(built)

    assert any(problem.startswith(expected) for problem in problems), problems


def test_a_document_of_the_wrong_version_is_rejected(runset):
    built = document(runset)
    built["schema_version"] = SCHEMA_VERSION + 1

    problems = validate(built)

    assert len(problems) == 1
    assert problems[0].startswith(
        f"schema_version is {SCHEMA_VERSION + 1}, this build writes {SCHEMA_VERSION}"
    )


def test_a_real_version_one_document_is_rejected_by_its_version_and_not_by_a_missing_key(runset):
    # M3-3 verifier: a v1 artifact used to be rejected for "minimized is
    # missing", which is true and useless -- it describes a v1 document as a
    # damaged v2 one. The version is now checked first and reported alone.
    built = document(runset)
    del built["minimized"]
    built["schema_version"] = 1

    problems = validate(built)

    assert len(problems) == 1
    assert problems[0].startswith(f"schema_version is 1, this build writes {SCHEMA_VERSION}")
    assert "minimized" not in problems[0]


def test_something_that_is_not_an_object_at_all_is_rejected():
    assert validate([1, 2, 3]) == ["the document is a array, expected an object"]


def test_a_value_json_cannot_write_is_named_with_its_path(runset):
    built = document(runset)
    built["run"]["device"] = object()

    assert any("run.device is a object" in problem for problem in validate(built))


def test_dump_refuses_to_write_a_document_that_does_not_match_the_schema(tmp_path):
    path = tmp_path / "out.json"

    with pytest.raises(ValueError, match="does not match schema version"):
        dump({"schema_version": SCHEMA_VERSION}, path)
    assert not path.exists()


def test_dump_creates_the_directory_and_ends_the_file_with_a_newline(runset, tmp_path):
    path = tmp_path / "nested" / "dir" / "out.json"
    dump(document(runset), path)

    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_a_run_with_no_source_still_has_a_target_block():
    bare = RunSet(
        target_name="m:model",
        device="cpu",
        seed=0,
        fullgraph=False,
        dynamic=False,
        grad=True,
        env=dict(ENV),
    )
    bare.results["eager"] = BackendResult(backend="eager", outputs=[1])
    built = document(bare, findings=())

    assert built["target"] == {"name": "m:model", "file": None, "entry": None, "inputs": None}
    assert validate(built) == []


def test_the_artifact_carries_no_timestamp(runset, tmp_path):
    # PLAN.md "Cross-architecture parity is a feature": parity in v1 is two
    # machines and a diff, so a field that differs on every run for no reason
    # would make the comparison the schema exists for worse.
    path = tmp_path / "out.json"
    dump(document(runset), path)
    text = path.read_text(encoding="utf-8")

    assert "timestamp" not in text
    assert "generated_at" not in text
    assert Path(path).exists()


# --- the minimized section (schema 2) ---------------------------------------


MINIMIZED = Minimization(
    finding=FINDING,
    reproduced=True,
    shrinks=(Shrink(index=0, before=(8, 4), after=(1, 4)),),
    stubs=(Stub(path="net.0", module="Linear"),),
    kept=(Kept(path="net.1", module="ReLU", reason="the finding did not survive"),),
    notes=("a note",),
    steps=4,
    seconds=1.25,
    partial=True,
    partial_reason="the --budget of 5s ran out after 4 candidate re-runs",
    handoff="TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4",
)


def test_the_schema_version_is_two_because_minimized_was_added():
    # PLAN.md "Reports": the integer is bumped on any incompatible field
    # change, and a v1 document has no `minimized` key at all.
    assert SCHEMA_VERSION == 2


def test_a_run_without_the_minimizer_writes_null_rather_than_an_empty_record(runset):
    built = document(runset)
    assert built["minimized"] is None
    assert validate(built) == []


def test_a_minimized_run_carries_every_documented_field(runset):
    built = document(runset, minimized=MINIMIZED)
    assert validate(built) == []

    minimized = built["minimized"]
    assert minimized["attempted"] is True
    assert minimized["reason"] is None
    assert minimized["reproduced"] is True
    assert minimized["changed"] is True
    assert minimized["finding"] == {
        "oracle": "metadata",
        "backend": "inductor",
        "output_index": 0,
        "severity": "fail",
        "field": "dtype",
    }
    assert minimized["shrinks"] == [{"index": 0, "before": [8, 4], "after": [1, 4]}]
    assert minimized["stubs"] == [{"path": "net.0", "module": "Linear"}]
    assert minimized["kept"] == [
        {"path": "net.1", "module": "ReLU", "reason": "the finding did not survive"}
    ]
    assert minimized["notes"] == ["a note"]
    assert minimized["steps"] == 4
    assert minimized["seconds"] == 1.25
    assert minimized["partial"] is True
    assert minimized["partial_reason"].startswith("the --budget of 5s ran out")
    assert "TORCHDYNAMO_REPRO_AFTER=aot" in minimized["handoff"]


def test_a_run_the_minimizer_had_nothing_to_do_on_is_not_null(runset):
    # The distinction the field exists to keep: null is "not run", and this is
    # "run, and there was nothing to work from".
    built = document(runset, minimized=Minimization.not_attempted("no fail-severity finding"))
    assert validate(built) == []
    assert built["minimized"]["attempted"] is False
    assert built["minimized"]["finding"] is None
    assert built["minimized"]["reason"] == "no fail-severity finding"


def test_the_minimized_section_round_trips_through_json(runset, tmp_path):
    path = tmp_path / "out.json"
    dump(document(runset, minimized=MINIMIZED), path)
    reloaded = json.loads(path.read_text(encoding="utf-8"))
    assert reloaded["minimized"] == document(runset, minimized=MINIMIZED)["minimized"]


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (lambda m: m.pop("stubs"), "minimized.stubs is missing"),
        (lambda m: m.update(steps="4"), "minimized.steps is a string, expected integer"),
        (lambda m: m.update(partial=None), "minimized.partial is a null, expected boolean"),
        (
            lambda m: m["stubs"].append({"path": "x"}),
            "minimized.stubs[1].module is missing",
        ),
        (
            lambda m: m["finding"].update(oracle=2),
            "minimized.finding.oracle is a integer, expected string",
        ),
    ],
)
def test_a_broken_minimized_section_is_reported_field_by_field(runset, mutate, expected):
    built = document(runset, minimized=MINIMIZED)
    mutate(built["minimized"])
    assert expected in validate(built)


def test_a_document_that_claims_version_two_without_the_minimized_key_is_rejected(runset):
    # What the bump is for: `minimized` is not optional in a v2 document, so a
    # v1 artifact relabelled as v2 -- or a writer that forgot the key -- is
    # named rather than silently read as a run with no minimizer. A document
    # that says version 1 honestly is rejected by its version instead, one test
    # above.
    built = document(runset)
    del built["minimized"]
    assert "minimized is missing" in validate(built)
