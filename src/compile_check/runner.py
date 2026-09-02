"""Seeding, cloning, reset, per-backend execution.

PLAN.md "Package layout": ``runner.py`` -- seeding, cloning, reset, per-backend
execution.

PLAN.md "Runner semantics": the runner establishes that any difference it
reports comes from the backend and not from the harness. The RNG is seeded
identically before every run, inputs are deep cloned per backend, each backend
runs after ``torch.compiler.reset()``, caches are disabled by default, the eager
run is the reference world, and each backend is called twice so the graph oracle
can see whether a recompile happened.

Torch is imported inside the functions, never at module scope, so that importing
this module (and therefore ``compile_check.cli``) does not pay for the torch
import; a test in tests/test_cli.py enforces it.
"""

from __future__ import annotations

import copy
import functools
import importlib
import logging
import operator
import os
import platform
import random
import time
import traceback
from collections.abc import Sequence
from typing import Any

from compile_check.discover import Target
from compile_check.env import collect_environment
from compile_check.results import (
    TRACEBACK_LINES,
    BackendResult,
    CapturedException,
    RunSet,
    TensorMeta,
)

__all__ = [
    "ABLATION_LADDER",
    "CACHE_ENV_VAR",
    "FP64_BACKEND",
    "RunnerError",
    "available_backends",
    "run_all",
    "run_backend",
    "run_fp64_reference",
    "validate_backends",
    "validate_device",
]

log = logging.getLogger("compile_check")

# PLAN.md "Verified API surface": torch._inductor.config.force_disable_caches
# reads this at import time, so the CLI sets it before importing torch.
CACHE_ENV_VAR = "TORCHINDUCTOR_FORCE_DISABLE_CACHES"

# PLAN.md "Stage localization": the ablation ladder, in order. Every backend the
# installed torch registers is accepted; these four are the ones the tool is
# about, so they are what an error message names.
ABLATION_LADDER: tuple[str, ...] = (
    "eager",
    "aot_eager",
    "aot_eager_decomp_partition",
    "inductor",
)

# PLAN.md "The oracle blind spot": the optional fp64 eager reference. It is not
# a torch.compile backend and never goes through the registry; the name exists
# so a report can label the row, and so run_backend knows to call the target
# directly the way it does for eager.
FP64_BACKEND = "eager_fp64"

# The lanes the runner calls directly instead of compiling.
EAGER_BACKENDS: tuple[str, ...] = ("eager", FP64_BACKEND)


class RunnerError(Exception):
    """A run cannot be set up at all: unknown backend, unavailable device.

    PLAN.md "CLI surface for v1" lists "backend unavailable" among the exit
    code 2 conditions, alongside import and discovery failure. This is the
    exception the CLI turns into that exit code, with the message printed as a
    single line and no traceback: an unknown backend name is a typo by the user,
    and a torch stack tells them nothing a sentence cannot.

    It is deliberately not what a *model* failing raises. A target that blows up
    inside eager is recorded on its BackendResult and reported per backend; only
    a run that could not be started reaches this.
    """


def available_backends() -> list[str]:
    """Every backend name ``torch.compile`` accepts on this install, sorted.

    PLAN.md "Verified API surface": ``eager`` and ``aot_eager`` carry the
    ``debug`` tag, so a plain ``list_backends()`` omits them and a validator
    must pass ``exclude_tags=()``. ``eager`` is added anyway because the runner
    treats it as the reference world and calls the target directly for it,
    without going through the registry at all.
    """
    dynamo = importlib.import_module("torch._dynamo")
    return sorted({"eager", *dynamo.list_backends(exclude_tags=())})


def validate_backends(backends: Sequence[str], *, defer_unknown: bool = False) -> list[str]:
    """Check every requested backend name before anything is compiled.

    Args:
        backends: the requested names, in run order.
        defer_unknown: report an empty request, but treat a name the registry
            does not hold as a question for later rather than as an error. The
            registry is not fixed at the moment the flags are parsed: a target
            module is free to call ``torch._dynamo.register_backend`` when it is
            imported, and a cold run that rejected the name before discovery
            rejected a backend that was about to exist. The CLI passes this on
            its up-front pass and lets :func:`run_all`, which runs after the
            target has been imported and before anything is compiled, be the
            check that refuses.

    Returns:
        The requested names, unchanged.

    Raises:
        RunnerError: naming the unknown backends and listing the known ones.
    """
    if not backends:
        raise RunnerError("no backends requested")
    known = available_backends()
    unknown = [name for name in backends if name not in known]
    if unknown and defer_unknown:
        log.debug(
            "backend%s %s %s not registered yet; deferring the decision until the "
            "target has been imported, which may register %s",
            "s" if len(unknown) > 1 else "",
            ", ".join(repr(name) for name in unknown),
            "are" if len(unknown) > 1 else "is",
            "them" if len(unknown) > 1 else "it",
        )
        return list(backends)
    if unknown:
        raise RunnerError(
            f"unknown backend{'s' if len(unknown) > 1 else ''} "
            f"{', '.join(repr(name) for name in unknown)}; the ablation ladder is "
            f"{', '.join(ABLATION_LADDER)} "
            f"({len(known)} backends are registered on this torch, see "
            "torch._dynamo.list_backends(exclude_tags=()))"
        )
    return list(backends)


def validate_device(device: str) -> str:
    """Check the requested device is usable before anything is placed on it.

    Raises:
        RunnerError: the device was requested and torch cannot provide it.
    """
    torch = importlib.import_module("torch")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RunnerError(
            f"device {device!r} was requested but torch {torch.__version__} on "
            f"{platform.machine()} reports no CUDA device is available"
        )
    return device


def run_all(
    target: Target,
    backends: Sequence[str],
    *,
    device: str = "cpu",
    seed: int = 0,
    fullgraph: bool = False,
    dynamic: bool = False,
    grad: bool = True,
    disable_caches: bool = True,
    fp64: bool = False,
    share_module: bool = False,
) -> RunSet:
    """Run ``target`` under every backend in ``backends`` and record the results.

    Args:
        target: what to run, from :func:`compile_check.discover.load_target`.
        backends: backend names in ablation-ladder order, conventionally
            ``eager`` first because it is the reference world.
        device: where the model and the inputs are placed.
        seed: reapplied before every backend, not once per run.
        fullgraph: passed to ``torch.compile``.
        dynamic: when true, passed to ``torch.compile`` as ``dynamic=True``.
        grad: run one backward pass when anything in the run requires grad.
        disable_caches: force the inductor caches off, which is the default and
            what the CLI does unless ``--allow-caches`` was passed.
        fp64: add the ``eager_fp64`` reference run of ``--fp64-oracle``. It is
            recorded on :attr:`~compile_check.results.RunSet.fp64`, not among
            the backends, and a target that cannot be run at float64 leaves it
            ``None`` rather than failing the run.
        share_module: hand every lane the same ``nn.Module`` object instead of
            its own deep copy (``--share-module``). See :func:`_lane_module` for
            what the copy buys and what it costs.

    Returns:
        One :class:`~compile_check.results.BackendResult` per backend, in the
        order given, plus the environment block. A backend that raised is a
        recorded result, not an exception out of this function.

    Raises:
        RunnerError: the run could not be set up, because a backend name is not
            registered on this torch or the device is unavailable.
    """
    torch = importlib.import_module("torch")
    # Before a single compile, and after the target has been imported: a typo in
    # --backends or a CUDA request on a CPU-only box is a setup error, and
    # finding it after the eager lane has already run wastes the run and buries
    # the message. This is the pass that refuses -- the CLI's earlier one defers
    # unknown names, because importing the target is what registers them.
    validate_backends(backends)
    validate_device(device)
    _configure_caches(disable_caches)

    fn = target.fn
    if isinstance(fn, torch.nn.Module):
        # nn.Module.to is in place, so this moves the user's own module once
        # rather than per backend; every backend must see the same weights.
        fn = fn.to(device)

    runset = RunSet(
        target_name=target.name,
        device=device,
        seed=seed,
        fullgraph=fullgraph,
        dynamic=dynamic,
        grad=grad,
        share_module=share_module,
        env=collect_environment(),
    )
    if fp64:
        runset.fp64 = run_fp64_reference(
            fn,
            target.example_inputs,
            kwargs=target.kwargs,
            device=device,
            seed=seed,
        )
    for backend in backends:
        log.debug("running backend %s", backend)
        runset.results[backend] = run_backend(
            _lane_module(torch, fn, share_module),
            target.example_inputs,
            backend,
            kwargs=target.kwargs,
            device=device,
            seed=seed,
            fullgraph=fullgraph,
            dynamic=dynamic,
            grad=grad,
        )
    return runset


def _lane_module(torch: Any, fn: Any, share: bool) -> Any:
    """The module object one lane runs against: its own copy, or the shared one.

    PLAN.md "Runner semantics" isolates the *inputs* per backend so that a
    mutation by one lane is invisible to the next. A module's own state needs
    the same treatment and did not have it until M2-2: a buffer that the forward
    pass writes to -- a step counter, BatchNorm running statistics in train mode
    -- is carried into the next lane, so the second lane computes with a state
    the first one left behind and the numerics oracle reports a divergence that
    the backend did not cause. Each lane therefore gets its own deep copy.

    The cost is memory: one extra copy of the parameters and buffers is alive
    while a lane runs, so a run peaks at the user's module plus one copy rather
    than at the module alone. It is bounded by the model, not by the number of
    lanes -- the previous lane's copy is dropped when the next one is made --
    and ``--share-module`` turns it off for a model too large to copy, at the
    price of the false divergence above.

    A plain callable is handed back as it is. A function has no state to isolate,
    and PLAN.md's discovery convention makes ``nn.Module`` the stateful case.
    """
    if share or not isinstance(fn, torch.nn.Module):
        return fn
    try:
        return copy.deepcopy(fn)
    except Exception as exc:
        # Not fatal: a module that refuses to be copied still runs, and a lane
        # that shares state is better than a run that does not happen. Said out
        # loud, because it is the condition under which a numerics finding may
        # be the harness's doing rather than the backend's.
        log.warning(
            "this module could not be deep copied (%s: %s), so every lane shares one "
            "object; a stateful forward pass can make the lanes disagree by itself",
            type(exc).__name__,
            exc,
        )
        return fn


def run_fp64_reference(
    fn: Any,
    example_inputs: Sequence[Any],
    *,
    kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    seed: int = 0,
) -> BackendResult | None:
    """Run the target once in float64 eager, as a reference for the numerics oracle.

    PLAN.md "The oracle blind spot": eager is the reference world, so a bug that
    lives in eager is invisible. The partial mitigation, borrowed from
    ``benchmarks/dynamo/common.py``, is a third computation at float64 width;
    comparing both the fp32 eager result and the compiled result against it
    separates "compiled is wrong" from "both are imprecise".

    The module is deep copied before it is widened, so the run under test keeps
    the float32 weights every backend saw. A target that cannot be copied or
    cannot run at float64 (a kernel with no double implementation is the usual
    one) is not an error: the reference is simply unavailable, and the oracle
    says nothing rather than something wrong.

    Args:
        fn: the target, already placed on ``device`` by :func:`run_all`.
        example_inputs: the same inputs the backends were given.
        kwargs: the target's keyword inputs.
        device: where the reference run is placed.
        seed: reapplied before the reference run, as for every other lane.

    Returns:
        A :class:`~compile_check.results.BackendResult` named
        :data:`FP64_BACKEND`, or ``None`` when the target could not be widened.
    """
    torch = importlib.import_module("torch")
    try:
        widened = _to_float64(torch, fn)
    except Exception as exc:
        log.warning("no fp64 reference: the target could not be copied (%s)", exc)
        return None

    def widen(value: Any) -> Any:
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            return value.detach().to(torch.float64)
        return value

    args, call_kwargs = torch.utils._pytree.tree_map(
        widen, (tuple(example_inputs), dict(kwargs or {}))
    )
    # grad is off: the fp64 pass exists to give the numerics oracle a reference
    # for the forward values, and PLAN.md's grad oracle compares eager against
    # the compiled lanes, not against fp64.
    result = run_backend(
        widened,
        args,
        FP64_BACKEND,
        kwargs=call_kwargs,
        device=device,
        seed=seed,
        grad=False,
    )
    if not result.ok:
        assert result.exception is not None
        log.warning(
            "the fp64 reference run raised %s: %s",
            result.exception.type,
            result.exception.message.splitlines()[0] if result.exception.message else "",
        )
    return result


def _to_float64(torch: Any, fn: Any) -> Any:
    """A float64 copy of the target, leaving the caller's own module alone.

    A plain callable is returned as it is: it has no weights to widen, and the
    inputs it is given are already float64.
    """
    if not isinstance(fn, torch.nn.Module):
        return fn
    return copy.deepcopy(fn).double()


def run_backend(
    fn: Any,
    example_inputs: Sequence[Any],
    backend: str,
    *,
    kwargs: dict[str, Any] | None = None,
    device: str = "cpu",
    seed: int = 0,
    fullgraph: bool = False,
    dynamic: bool = False,
    grad: bool = True,
) -> BackendResult:
    """Run ``fn`` under one backend and return everything the oracles compare.

    Every input this backend sees is cloned from ``example_inputs`` here, not
    inherited from the previous backend, which is what makes a mutation by one
    backend invisible to the next.

    ``backend`` is a torch.compile backend name, or one of
    :data:`EAGER_BACKENDS`, which are called directly instead.
    """
    torch = importlib.import_module("torch")
    result = BackendResult(backend=backend)

    _seed_everything(torch, seed)
    args, call_kwargs = _clone_inputs(torch, tuple(example_inputs), dict(kwargs or {}), device)
    leaves, spec = torch.utils._pytree.tree_flatten((args, call_kwargs))
    result.input_spec = spec
    result.input_refs = list(leaves)
    result.inputs_before = [_snapshot(torch, leaf) for leaf in leaves]
    result.input_meta_before = [_describe(torch, leaf) for leaf in leaves]

    # PLAN.md "Runner semantics": no compiled artifact or guard from a previous
    # backend is reused. torch.compiler.reset() is the public spelling of
    # torch._dynamo.reset(); the fallback is there for a torch old enough to
    # have only the private one.
    _reset_compiler(torch)
    _zero_grads(torch, fn, leaves)

    # Neither of these two goes through torch.compile: eager is the reference
    # world and eager_fp64 is the same call at another width.
    if backend in EAGER_BACKENDS:
        call = fn
    else:
        call = torch.compile(
            fn,
            backend=backend,
            fullgraph=fullgraph,
            # torch.compile's own default is dynamic=None, "decide automatically";
            # dynamic=False would force static shapes, which is a third mode and
            # not what "the user did not pass --dynamic" means.
            dynamic=True if dynamic else None,
        )

    started = time.perf_counter()
    try:
        outputs = call(*args, **call_kwargs)
    except Exception as exc:
        result.first_call_s = time.perf_counter() - started
        result.exception = _capture(exc)
        _record_inputs_after(torch, result, leaves)
        log.debug("backend %s raised %s", backend, type(exc).__name__)
        return result
    result.first_call_s = time.perf_counter() - started

    out_leaves, out_spec = torch.utils._pytree.tree_flatten(outputs)
    result.output_spec = out_spec
    result.output_refs = list(out_leaves)
    result.outputs = [_snapshot(torch, leaf) for leaf in out_leaves]
    # Taken after the measured call and before the repeat call, so an input
    # mutation is recorded once rather than twice (M2's mutation oracle).
    _record_inputs_after(torch, result, leaves)

    # PLAN.md "Runner semantics": each backend is called twice with the same
    # inputs; the second call exists so the graph oracle can see a recompile.
    # Its output is deliberately discarded, but a failure is not: a lane that
    # answers once and then raises is recorded on the result so the graph oracle
    # and the report can say so.
    started = time.perf_counter()
    try:
        call(*args, **call_kwargs)
        result.second_call_s = time.perf_counter() - started
    except Exception as exc:
        result.second_call_exception = _capture(exc)
        log.warning(
            "backend %s succeeded on the first call and raised %s on the second: %s",
            backend,
            type(exc).__name__,
            exc,
        )

    if grad:
        _run_backward(torch, fn, result)
    return result


def _run_backward(torch: Any, fn: Any, result: BackendResult) -> None:
    """One backward on a deterministic scalar reduction, grads recorded.

    PLAN.md "grad": reduce the outputs to a scalar with a fixed rule (the sum of
    every floating point output element, in traversal order, integer and bool
    outputs skipped), call backward, then compare both the values and the set of
    tensors that received a grad at all.
    """
    if not _anything_requires_grad(torch, fn, result.input_refs):
        return
    reduced = [
        leaf.float().sum()
        for leaf in result.output_refs
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point() and leaf.requires_grad
    ]
    if not reduced:
        log.debug("backend %s: no differentiable output, backward skipped", result.backend)
        return

    try:
        functools.reduce(operator.add, reduced).backward()
    except Exception as exc:
        result.grad_error = _capture(exc)
        log.debug("backend %s: backward raised %s", result.backend, type(exc).__name__)
        return

    result.grad_ran = True
    result.input_grads = [_grad_of(torch, leaf) for leaf in result.input_refs]
    if isinstance(fn, torch.nn.Module):
        result.param_grads = {
            name: grad
            for name, parameter in fn.named_parameters()
            if (grad := _grad_of(torch, parameter)) is not None
        }


def _configure_caches(disable: bool) -> None:
    """Force the inductor caches off, and say so if it was asked for too late.

    Setting the config attribute covers the compiles this process is about to
    do. The environment variable matters as well because parts of the cache
    machinery read it when torch is imported, which is why the CLI sets it in
    ``main()`` before its own torch import and the test suite sets it in
    conftest.py.
    """
    # Typed as Any because torch's config modules build their attributes at
    # import time, so a static reader does not see this one.
    config: Any = importlib.import_module("torch._inductor.config")
    if not disable:
        log.debug(
            "caches left as the environment set them, force_disable_caches=%s",
            config.force_disable_caches,
        )
        return
    if os.environ.get(CACHE_ENV_VAR) != "1":
        log.warning(
            "%s was not set before torch was imported; setting "
            "torch._inductor.config.force_disable_caches now, but a cache that read "
            "the variable at import time may already be enabled",
            CACHE_ENV_VAR,
        )
    config.force_disable_caches = True


def _reset_compiler(torch: Any) -> None:
    """Drop every compiled artifact and guard from an earlier backend."""
    dynamo = importlib.import_module("torch._dynamo")
    reset = getattr(torch.compiler, "reset", None) or dynamo.reset
    reset()


def _seed_everything(torch: Any, seed: int) -> None:
    """Seed torch, Python, and numpy when it is present.

    PLAN.md "Runner semantics": the RNG is seeded before every run with the same
    seed, covering ``torch.manual_seed`` and the Python and numpy generators
    when numpy is present. numpy is an optional dependency, so its absence is
    not an error.
    """
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        numpy = importlib.import_module("numpy")
    except ImportError:
        return
    numpy.random.seed(seed)


def _clone_inputs(
    torch: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device: str,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Deep clone the example inputs for one backend.

    The clone preserves dtype, stride, and ``requires_grad``, and lands on
    ``device``. It is built through ``detach`` so that the result is a leaf: a
    plain ``clone()`` of a tensor that requires grad is not a leaf, and a
    non-leaf never gets a ``.grad`` for the grad oracle to read.
    """
    target_device = torch.device(device)

    def clone(value: Any) -> Any:
        if not isinstance(value, torch.Tensor):
            return _deepcopy(value)
        cloned = value.detach().clone()
        if cloned.device != target_device:
            cloned = cloned.to(target_device)
        if value.requires_grad:
            cloned.requires_grad_(True)
        return cloned

    cloned_args, cloned_kwargs = torch.utils._pytree.tree_map(clone, (args, kwargs))
    return tuple(cloned_args), dict(cloned_kwargs)


def _record_inputs_after(torch: Any, result: BackendResult, leaves: Sequence[Any]) -> None:
    """Snapshot the inputs as the call left them, values and layout together.

    Both halves are taken at the same moment on purpose: the alias oracle reads
    them as one pair, and a value snapshot taken before the repeat call beside a
    layout read after it would describe a state that never existed.
    """
    result.inputs_after = [_snapshot(torch, leaf) for leaf in leaves]
    result.input_meta_after = [_describe(torch, leaf) for leaf in leaves]


def _snapshot(torch: Any, value: Any) -> Any:
    """A detached clone of a tensor, or a copy of a non-tensor leaf."""
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    return _deepcopy(value)


def _describe(torch: Any, value: Any) -> TensorMeta | None:
    """Where one leaf's bytes are, as plain data, or ``None``.

    ``None`` covers both a leaf that is not a tensor and a tensor whose layout
    refuses one of these questions -- a sparse tensor has no ``stride()``, a
    meta tensor no storage. PLAN.md "alias" needs all of them together to decide
    an overlap, so a partial record would be worse than none: it would let a
    fact the runner could not read look like a fact that changed.
    """
    if not isinstance(value, torch.Tensor):
        return None
    try:
        return TensorMeta(
            shape=tuple(value.shape),
            stride=tuple(value.stride()),
            dtype=str(value.dtype),
            storage_offset=value.storage_offset(),
            data_ptr=value.data_ptr(),
            storage_ptr=value.untyped_storage().data_ptr(),
        )
    except Exception as exc:
        log.debug("no layout record for a %s leaf: %s", type(value).__name__, exc)
        return None


def _deepcopy(value: Any) -> Any:
    """``copy.deepcopy``, falling back to the object for anything uncopyable."""
    try:
        return copy.deepcopy(value)
    except Exception as exc:
        log.debug("keeping a reference to an uncopyable leaf (%s): %s", type(value).__name__, exc)
        return value


def _grad_of(torch: Any, tensor: Any) -> Any:
    """The ``.grad`` of a tensor as a detached clone, or ``None``."""
    if not isinstance(tensor, torch.Tensor) or tensor.grad is None:
        return None
    return tensor.grad.detach().clone()


def _zero_grads(torch: Any, fn: Any, leaves: Sequence[Any]) -> None:
    """Clear every ``.grad`` this backend could write to.

    PLAN.md "grad": grads are zeroed before each backward so a leaked
    accumulation cannot be mistaken for a divergence. The input clones are fresh
    and so already have none; the parameters are shared across backends and do
    not.
    """
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            leaf.grad = None
    if isinstance(fn, torch.nn.Module):
        for parameter in fn.parameters():
            parameter.grad = None


def _anything_requires_grad(torch: Any, fn: Any, leaves: Sequence[Any]) -> bool:
    """Whether the grad oracle activates for this run."""
    if any(isinstance(leaf, torch.Tensor) and leaf.requires_grad for leaf in leaves):
        return True
    if isinstance(fn, torch.nn.Module):
        return any(parameter.requires_grad for parameter in fn.parameters())
    return False


def _capture(exc: Exception) -> CapturedException:
    """Record an exception with the head of its traceback."""
    formatted = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return CapturedException(
        type=type(exc).__name__,
        message=str(exc),
        traceback=tuple(formatted.splitlines()[:TRACEBACK_LINES]),
    )
