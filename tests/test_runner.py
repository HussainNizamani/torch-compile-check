"""Tests for the multi-backend runner.

conftest.py has already set TORCHINDUCTOR_FORCE_DISABLE_CACHES=1, before torch
was imported, which is the only moment at which it can be set.
"""

from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import pytest
import torch

from compile_check.discover import Target, load_target
from compile_check.results import TRACEBACK_LINES, BackendResult, RunSet
from compile_check.runner import (
    ABLATION_LADDER,
    CACHE_ENV_VAR,
    FP64_BACKEND,
    RunnerError,
    available_backends,
    run_all,
    run_backend,
    run_fp64_reference,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


@pytest.fixture(scope="module")
def mlp_runset():
    """One eager-plus-inductor run of the MLP, shared: inductor costs seconds."""
    target = load_target(str(FIXTURES / "mlp.py"))
    return run_all(target, ["eager", "inductor"], seed=0)


def test_the_cache_variable_was_set_before_torch_was_imported():
    assert os.environ[CACHE_ENV_VAR] == "1"
    assert torch._inductor.config.force_disable_caches is True


def test_the_session_writes_its_generated_code_somewhere_of_its_own():
    # M1-1 review carry-over (b): disabling the caches stops inductor reading an
    # artifact, not writing one, and the default directory is never cleaned up.
    cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    assert cache_dir, "conftest.py must point the codegen output at its own directory"
    assert Path(cache_dir).is_dir()
    assert Path(cache_dir) != Path(tempfile.gettempdir())


def test_runset_shape(mlp_runset):
    assert isinstance(mlp_runset, RunSet)
    assert mlp_runset.backends == ["eager", "inductor"]
    assert mlp_runset.target_name == "mlp:model"
    assert isinstance(mlp_runset.eager, BackendResult)
    assert [result.backend for result in mlp_runset.others] == ["inductor"]
    assert mlp_runset.env["torch_version"] == torch.__version__
    assert mlp_runset.env["inductor_force_disable_caches"] is True


def test_eager_and_inductor_agree_on_count_dtypes_and_shapes(mlp_runset):
    eager = mlp_runset.results["eager"]
    inductor = mlp_runset.results["inductor"]
    assert eager.ok, eager.exception
    assert inductor.ok, inductor.exception
    assert len(eager.outputs) == len(inductor.outputs) == 1
    assert [t.dtype for t in eager.outputs] == [t.dtype for t in inductor.outputs]
    assert [t.shape for t in eager.outputs] == [t.shape for t in inductor.outputs]
    assert eager.output_spec == inductor.output_spec
    # Not the numerics oracle (that is M1-2), but a compiled MLP that disagreed
    # with eager here would mean the harness itself is wrong.
    torch.testing.assert_close(inductor.outputs[0], eager.outputs[0])


def test_both_calls_are_timed(mlp_runset):
    for result in mlp_runset.results.values():
        assert result.first_call_s is not None
        assert result.first_call_s > 0
        assert result.second_call_s is not None
        assert result.second_call_s > 0
    # Compiling costs seconds; the second call is the same graph again.
    assert (
        mlp_runset.results["inductor"].first_call_s > mlp_runset.results["inductor"].second_call_s
    )


def test_outputs_are_kept_as_clones_and_as_references(mlp_runset):
    for result in mlp_runset.results.values():
        assert len(result.output_refs) == len(result.outputs)
        for clone, reference in zip(result.outputs, result.output_refs, strict=True):
            assert clone is not reference
            torch.testing.assert_close(clone, reference.detach())


def test_parameter_grads_are_recorded_per_name(mlp_runset):
    for result in mlp_runset.results.values():
        assert result.grad_ran is True
        assert result.grad_error is None
        assert set(result.param_grads) == {
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
        }
        assert result.param_grads["net.0.weight"].shape == (16, 8)
        # The MLP's input does not require grad, so nothing is recorded for it.
        assert result.input_grads == [None]


def test_input_grads_are_recorded_when_the_input_requires_grad():
    x = torch.randn(3, 4, requires_grad=True)
    target = Target(fn=torch.sin, example_inputs=(x,), name="inline:sin")
    result = run_all(target, ["eager"]).results["eager"]

    assert result.grad_ran is True
    assert result.input_grads[0] is not None
    torch.testing.assert_close(result.input_grads[0], torch.cos(x.detach()))
    # The clone is a leaf that requires grad, and it is not the caller's tensor.
    assert result.input_refs[0] is not x
    assert result.input_refs[0].requires_grad is True
    assert result.input_refs[0].is_leaf is True
    assert x.grad is None


def test_grad_can_be_switched_off():
    x = torch.randn(3, 4, requires_grad=True)
    target = Target(fn=torch.sin, example_inputs=(x,), name="inline:sin")
    result = run_all(target, ["eager"], grad=False).results["eager"]

    assert result.grad_ran is False
    assert result.input_grads == []
    assert result.param_grads == {}


def test_a_raising_target_is_captured_not_propagated():
    target = load_target(str(FIXTURES / "raises.py"))
    runset = run_all(target, ["eager", "aot_eager"])

    for result in runset.results.values():
        assert result.ok is False
        assert result.outputs == []
        assert result.exception is not None
        assert result.exception.type == "RuntimeError"
        assert "broken on purpose" in result.exception.message
        assert 0 < len(result.exception.traceback) <= TRACEBACK_LINES
        assert result.exception.traceback[0].startswith("Traceback")
        # The lane is still timed, so a report can say how long it took to fail.
        assert result.first_call_s is not None
        assert result.second_call_s is None
        # The repeat call never happened, so there is nothing to record for it.
        assert result.second_call_exception is None


def test_a_failing_repeat_call_is_recorded_separately(caplog):
    # PLAN.md "Runner semantics": each backend is called twice, and the second
    # call is what shows a recompile. A lane that answers once and then raises
    # produced a result that does not reproduce, which is not the same thing as
    # a lane that never produced one.
    class SecondCallRaises(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = 0

        def forward(self, x):
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("this target answers exactly once")
            return x * 2

    with caplog.at_level(logging.WARNING, logger="compile_check"):
        result = run_backend(SecondCallRaises(), (torch.ones(3),), "eager", grad=False)

    assert result.ok
    assert result.exception is None
    torch.testing.assert_close(result.outputs[0], torch.full((3,), 2.0))
    assert result.second_call_s is None
    assert result.second_call_exception is not None
    assert result.second_call_exception.type == "RuntimeError"
    assert "answers exactly once" in result.second_call_exception.message
    assert 0 < len(result.second_call_exception.traceback) <= TRACEBACK_LINES
    assert "raised RuntimeError on the second" in caplog.text


def test_inputs_are_isolated_between_backends():
    target = load_target(str(FIXTURES / "mutating.py"))
    original = target.example_inputs[0].clone()
    runset = run_all(target, ["eager", "aot_eager"])

    eager = runset.results["eager"]
    other = runset.results["aot_eager"]
    assert eager.ok, eager.exception
    assert other.ok, other.exception

    # Each backend saw the same clean input...
    for result in (eager, other):
        torch.testing.assert_close(result.inputs_before[0], original)
    # ...and each mutated only its own clone.
    for result in (eager, other):
        torch.testing.assert_close(result.inputs_after[0], torch.zeros_like(original))
        assert result.input_refs[0] is not target.example_inputs[0]
    # The target's own tensor is untouched, so a second run_all would be identical.
    torch.testing.assert_close(target.example_inputs[0], original)
    # Both backends computed the sum of the clean input, not of a zeroed one.
    torch.testing.assert_close(eager.outputs[0], original.sum())
    torch.testing.assert_close(other.outputs[0], original.sum())


def test_the_layout_of_every_input_is_recorded_around_the_call():
    target = load_target(str(FIXTURES / "mutating.py"))
    result = run_all(target, ["eager"]).results["eager"]

    assert result.ok, result.exception
    before, after = result.input_meta_before[0], result.input_meta_after[0]
    assert before is not None
    assert after is not None
    assert before.shape == (3, 4)
    assert before.stride == (4, 1)
    assert before.dtype == "torch.float32"
    assert before.storage_offset == 0
    assert before.storage_ptr != 0
    # zero_() writes in place: the values moved and the layout did not.
    assert (after.shape, after.stride, after.dtype) == (before.shape, before.stride, before.dtype)
    assert after.data_ptr == before.data_ptr
    assert not torch.equal(result.inputs_before[0], result.inputs_after[0])


def test_an_in_place_resize_shows_up_in_the_layout_records():
    # The record exists because a clone cannot stand in for it: Tensor.clone()
    # keeps the values and is free to pick its own stride, so a resize_ or an
    # as_strided_ is only visible in what the runner read off the input itself.
    def resize_the_input(x: torch.Tensor) -> torch.Tensor:
        total = x.sum()
        x.resize_(2, 6)
        return total

    target = Target(fn=resize_the_input, example_inputs=(torch.ones(3, 4),), name="inline:resize")
    result = run_all(target, ["eager"], grad=False).results["eager"]

    assert result.ok, result.exception
    before, after = result.input_meta_before[0], result.input_meta_after[0]
    assert before is not None
    assert after is not None
    assert (before.shape, before.stride) == ((3, 4), (4, 1))
    assert (after.shape, after.stride) == ((2, 6), (6, 1))


def test_a_non_tensor_input_leaf_has_no_layout_record():
    target = Target(
        fn=lambda x, scale: x * scale,
        example_inputs=(torch.ones(2), 3),
        name="inline:scaled",
    )
    result = run_all(target, ["eager"]).results["eager"]

    assert result.ok, result.exception
    assert [meta is None for meta in result.input_meta_before] == [False, True]
    assert len(result.input_meta_after) == len(result.inputs_after)


def test_the_layout_records_survive_a_backend_that_raised():
    target = load_target(str(FIXTURES / "raises.py"))
    result = run_all(target, ["eager"]).results["eager"]

    assert not result.ok
    # The mutation oracle must still be able to say what the inputs looked like
    # when a lane blew up halfway through writing into them.
    assert len(result.input_meta_before) == 1
    assert len(result.input_meta_after) == 1


def test_keyword_inputs_are_passed_as_keywords():
    target = load_target(str(FIXTURES / "kwargs_target.py"))
    result = run_all(target, ["eager"]).results["eager"]

    assert result.ok
    torch.testing.assert_close(result.outputs[0], torch.full((2, 3), 3.0))


def test_non_tensor_output_leaves_survive_the_flatten():
    target = Target(
        fn=lambda x: (x * 2, 3, "annotation"),
        example_inputs=(torch.ones(2),),
        name="inline:mixed",
    )
    result = run_all(target, ["eager"]).results["eager"]

    assert result.ok
    assert len(result.outputs) == 3
    torch.testing.assert_close(result.outputs[0], torch.full((2,), 2.0))
    assert result.outputs[1] == 3
    assert result.outputs[2] == "annotation"


def test_dict_outputs_keep_their_structure():
    target = load_target(str(FIXTURES / "fn_target.py"))
    runset = run_all(target, ["eager", "aot_eager"])

    eager = runset.results["eager"]
    other = runset.results["aot_eager"]
    assert eager.ok, eager.exception
    assert other.ok, other.exception
    assert eager.output_spec == other.output_spec
    assert [t.dtype for t in eager.outputs] == [torch.float32, torch.int64]
    assert [t.dtype for t in other.outputs] == [torch.float32, torch.int64]


def test_seeding_is_reapplied_per_backend():
    # A target that reads the global RNG returns the same thing under every
    # backend only because the runner reseeds before each one.
    target = Target(fn=lambda: torch.rand(4), example_inputs=(), name="inline:rand")
    runset = run_all(target, ["eager", "aot_eager"], seed=7)

    torch.testing.assert_close(
        runset.results["eager"].outputs[0],
        runset.results["aot_eager"].outputs[0],
    )


def test_the_compiler_is_reset_once_per_backend(monkeypatch):
    # PLAN.md "Runner semantics": each backend runs after a reset, so no
    # compiled artifact or guard from a previous backend is reused.
    calls = []
    real_reset = torch.compiler.reset

    def counting_reset():
        calls.append(len(calls))
        real_reset()

    monkeypatch.setattr(torch.compiler, "reset", counting_reset)
    target = Target(fn=torch.sin, example_inputs=(torch.ones(3),), name="inline:sin")
    run_all(target, ["eager", "aot_eager"])

    assert len(calls) == 2


def test_unknown_backends_are_rejected_before_anything_runs():
    target = Target(fn=torch.sin, example_inputs=(torch.ones(3),), name="inline:sin")
    with pytest.raises(RunnerError) as excinfo:
        run_all(target, ["eager", "bogus"])
    message = str(excinfo.value)
    assert "'bogus'" in message
    assert "eager, aot_eager, aot_eager_decomp_partition, inductor" in message


def test_available_backends_covers_the_ablation_ladder():
    available = available_backends()
    assert set(ABLATION_LADDER) <= set(available)
    # PLAN.md: eager and aot_eager carry the debug tag, so a validator that
    # forgot exclude_tags=() would miss them.
    assert "eager" in available
    assert "aot_eager" in available


def test_an_unavailable_device_is_rejected_before_anything_runs():
    if torch.cuda.is_available():  # pragma: no cover - depends on the machine
        pytest.skip("this machine has CUDA, so the unavailable path cannot run")
    target = Target(fn=torch.sin, example_inputs=(torch.ones(3),), name="inline:sin")
    with pytest.raises(RunnerError) as excinfo:
        run_all(target, ["eager"], device="cuda")
    assert "no CUDA device" in str(excinfo.value)


def test_the_fp64_reference_runs_beside_the_backends_and_widens_nothing_else():
    # PLAN.md "The oracle blind spot": an extra eager run at float64 width.
    # The model the backends ran is the caller's own object, so widening it in
    # place would corrupt every later run of the same target.
    target = load_target(str(FIXTURES / "mlp.py"))
    runset = run_all(target, ["eager"], fp64=True)

    assert runset.fp64 is not None
    assert runset.fp64.backend == FP64_BACKEND
    assert runset.fp64.ok, runset.fp64.exception
    assert runset.fp64.outputs[0].dtype == torch.float64
    assert runset.fp64.grad_ran is False
    # Not a lane: it is absent from the backend list and from `others`.
    assert FP64_BACKEND not in runset.results
    assert runset.backends == ["eager"]
    assert next(target.fn.parameters()).dtype == torch.float32
    assert runset.results["eager"].outputs[0].dtype == torch.float32


def test_no_fp64_reference_is_asked_for_unless_the_flag_is_set():
    target = load_target(str(FIXTURES / "mlp.py"))
    assert run_all(target, ["eager"]).fp64 is None


def test_a_target_that_cannot_be_copied_leaves_the_fp64_reference_unset(caplog):
    class Uncopyable(torch.nn.Module):
        def __deepcopy__(self, memo):
            raise RuntimeError("this module refuses to be copied")

        def forward(self, x):
            return x * 2

    with caplog.at_level(logging.WARNING, logger="compile_check"):
        assert run_fp64_reference(Uncopyable(), (torch.ones(3),)) is None
    assert "no fp64 reference" in caplog.text


def test_a_plain_callable_needs_no_copy_to_run_at_float64():
    result = run_fp64_reference(torch.sin, (torch.ones(3),))

    assert result is not None
    assert result.ok, result.exception
    assert result.outputs[0].dtype == torch.float64
    torch.testing.assert_close(result.outputs[0], torch.sin(torch.ones(3, dtype=torch.float64)))
