"""The records a run produces, and nothing else.

PLAN.md "Package layout" does not name this module; it is the shared vocabulary
that :mod:`compile_check.runner`, the five oracles, and the three reports all
speak, and putting it in ``runner.py`` would have made every oracle import the
runner.

Two rules hold for everything here. Nothing imports torch, at module scope or at
all, so an oracle or a report can be imported without paying for the torch
import (PLAN.md "Engineering decisions", enforced for ``cli.py`` by a test);
tensors are therefore typed ``Any``. And every field is data the oracles read,
not a view onto live state: what is stored is either a detached clone taken at a
known moment, or a reference that is documented as a reference because an oracle
needs object identity (PLAN.md "alias": "Python object identity is recorded
separately").
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["BackendResult", "CapturedException", "RunSet", "TensorMeta"]

# PLAN.md M1-1 brief: an exception is recorded with its first 20 traceback lines,
# enough to name the failing frame without pasting a whole inductor stack into a
# terminal report.
TRACEBACK_LINES = 20


@dataclass(frozen=True)
class CapturedException:
    """A backend that raised, recorded rather than propagated.

    A compiled backend raising is a result, not a crash: it is exactly what the
    stage localization in PLAN.md "Stage localization" reads. Only the eager
    lane raising is fatal, and that decision belongs to the CLI, not here.
    """

    type: str
    """Exception class name, e.g. ``"RuntimeError"``."""

    message: str
    """``str(exc)``, which for a torch backend error is usually multi-line."""

    traceback: tuple[str, ...]
    """The first :data:`TRACEBACK_LINES` lines of the formatted traceback."""


@dataclass(frozen=True)
class TensorMeta:
    """Where one tensor's bytes were, at one moment, as plain data.

    PLAN.md "alias" needs `storage_offset`, `stride`, `shape`, and the element
    size to decide whether two tensors overlap, and PLAN.md "Verified API
    surface" confirms every field below against the installed wheel. A clone
    cannot stand in for this: `Tensor.clone()` keeps the values but is free to
    pick its own layout, so the stride of a snapshot is the snapshot's stride
    and not the input's.

    Recorded before and after each call so the alias oracle of M2 can tell a
    value mutation (`copy_`) from a metadata mutation (`resize_`,
    `as_strided_`), which no comparison of the two clones can.
    """

    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    """``str(tensor.dtype)``, so this record never holds a torch object."""

    storage_offset: int
    data_ptr: int
    """The tensor's own first byte. An address, so it is comparable within one
    run (did this call reallocate?) and meaningless between two."""

    storage_ptr: int
    """``untyped_storage().data_ptr()``: which buffer the tensor lives in."""


@dataclass
class BackendResult:
    """Everything one backend produced, in the form the oracles compare."""

    backend: str
    """``eager``, ``aot_eager``, ``aot_eager_decomp_partition``, ``inductor``."""

    outputs: list[Any] = field(default_factory=list)
    """Flattened output leaves as detached clones.

    The numerics, metadata, and grad oracles read these. They are clones so that
    a later backend, or a backward pass, cannot change what an earlier backend
    is recorded as having returned.
    """

    output_refs: list[Any] = field(default_factory=list)
    """The same leaves as live references, in the same order as :attr:`outputs`.

    The alias oracle (M2) needs storage identity and Python object identity, and
    a clone destroys both, so the originals are kept alongside the clones.
    """

    output_spec: Any = None
    """The ``torch.utils._pytree`` ``TreeSpec`` the outputs flattened with.

    Compared before the leaves are: two runs that return different structures
    have diverged whatever the leaf values say.
    """

    output_requires_grad: list[bool] = field(default_factory=list)
    """``requires_grad`` of every output leaf, index-aligned with :attr:`outputs`.

    Recorded rather than read back off :attr:`outputs`, because those are
    ``detach()``ed clones and a detached tensor answers ``False`` whatever the
    tensor it was cloned from said. PLAN.md "metadata" compares this field per
    output; the metadata oracle reads it from here for exactly that reason, and
    the runner's backward pass reads it to decide which outputs to reduce.
    """

    inputs_before: list[Any] = field(default_factory=list)
    """Flattened input leaves as detached clones, taken before the first call."""

    inputs_after: list[Any] = field(default_factory=list)
    """The same leaves as detached clones, taken after the measured call.

    ``inputs_before`` versus ``inputs_after`` is the mutation oracle of M2
    (PLAN.md "alias": the set of inputs whose bytes changed across the call).
    The snapshot is taken after the first call and before the repeat call, so a
    mutation is recorded once rather than applied twice.
    """

    input_meta_before: list[TensorMeta | None] = field(default_factory=list)
    """Layout of every input leaf before the first call, index-aligned with
    :attr:`inputs_before`. ``None`` for a leaf that is not a tensor, or whose
    layout does not answer (a sparse or nested tensor): a fact the runner could
    not read must not read as a fact that changed."""

    input_meta_after: list[TensorMeta | None] = field(default_factory=list)
    """The same, taken with :attr:`inputs_after`. The pair is what tells a
    ``resize_`` or an ``as_strided_`` from a plain in-place write."""

    input_refs: list[Any] = field(default_factory=list)
    """The input leaves as live references, in the same order.

    The alias oracle (M2) relates outputs to inputs by storage and by object
    identity, and neither survives a clone.
    """

    input_spec: Any = None
    """The ``TreeSpec`` the inputs flattened with."""

    first_call_s: float | None = None
    """Wall time of compile plus the first call, the measured one."""

    second_call_s: float | None = None
    """Wall time of the second call, so M3 can see a recompile."""

    exception: CapturedException | None = None
    """Set when the backend raised; :attr:`outputs` is then empty."""

    second_call_exception: CapturedException | None = None
    """Set when the first call succeeded and the repeat call raised.

    Its own field rather than a second value on :attr:`exception`, because the
    two say different things. A backend that raises on the first call produced
    no result at all; a backend that answers once and then throws produced a
    result that is not reproducible, which is a graph-health fact the graph
    oracle (M3) reads off a recompile that went wrong. Stage localization treats
    only :attr:`exception` as "this lane did not run".
    """

    input_grads: list[Any | None] = field(default_factory=list)
    """``.grad`` clone per input leaf, index-aligned with :attr:`inputs_before`.

    ``None`` where that leaf received no gradient, which is itself compared:
    PLAN.md "grad" makes the set of tensors that got a grad at all part of the
    contract.
    """

    param_grads: dict[str, Any] = field(default_factory=dict)
    """``.grad`` clone per parameter, keyed by ``named_parameters()`` name."""

    grad_ran: bool = False
    """Whether a backward pass actually ran, so a missing grad is not read as a
    divergence when the oracle simply did not activate."""

    grad_error: CapturedException | None = None
    """Set when the forward succeeded and the backward raised."""

    @property
    def ok(self) -> bool:
        """Whether the forward pass completed."""
        return self.exception is None

    @property
    def grads(self) -> dict[str, Any]:
        """Every gradient this run produced, keyed by a label naming its tensor.

        The two records above are shaped for the runner that writes them: one
        index-aligned list with a hole per input that got nothing, and one dict
        per parameter name. The grad oracle asks a different question -- which
        tensors ended up with a gradient at all, and are the two lanes' answers
        the same set -- and it must be able to put the answer in a sentence.
        This is that view: sparse, labelled, and the one place the labels are
        written, so a message and a set comparison cannot drift apart.

        The values are the same clone objects the two records hold, so reading
        this costs a dict and no tensor memory.
        """
        labelled = {
            f"input[{index}]": grad
            for index, grad in enumerate(self.input_grads)
            if grad is not None
        }
        labelled.update({f"parameter {name}": grad for name, grad in self.param_grads.items()})
        return labelled

    @property
    def grad_present(self) -> tuple[str, ...]:
        """The label of every tensor that received a gradient, in record order.

        PLAN.md "grad" makes this set part of the contract: it must be identical
        in both lanes, and a tensor that got a gradient in one world and not in
        the other is a divergence whatever the gradients that did arrive say.
        """
        return tuple(self.grads)


@dataclass
class RunSet:
    """One run of one target across every requested backend.

    The order of :attr:`results` is the order the backends ran, which is the
    ablation ladder order of PLAN.md "Stage localization"; localization in M1-3
    walks it and reports the first lane to diverge.
    """

    target_name: str
    device: str
    seed: int
    fullgraph: bool
    dynamic: bool
    grad: bool
    share_module: bool = False
    """``--share-module``: every lane ran against one ``nn.Module`` object.

    Recorded because it changes what a divergence means. With the default, each
    lane deep copies the module and a buffer the forward pass writes to cannot
    leak from one lane into the next; with the flag on, it can, and a numerics
    finding may be the harness's doing. A report that did not say which mode
    produced it would leave that ambiguous.
    """

    results: dict[str, BackendResult] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)
    """``env.collect_environment()``, taken after torch was imported and after
    the cache setting was applied, so it records the process that actually ran."""

    fp64: BackendResult | None = None
    """The ``eager_fp64`` reference run, when ``--fp64-oracle`` asked for one.

    Kept off :attr:`results` on purpose. PLAN.md "The oracle blind spot" makes
    this a reference the numerics oracle reads, not a lane under test: it must
    never appear in :attr:`others`, in the backend table, or as the backend a
    stage verdict names. ``None`` when the flag was off or the target could not
    be run at float64.
    """

    @property
    def backends(self) -> list[str]:
        """The backends that ran, in order."""
        return list(self.results)

    @property
    def eager(self) -> BackendResult | None:
        """The reference world, or ``None`` if eager was not among the backends.

        PLAN.md "Runner semantics": the eager run is the reference world. Every
        oracle compares against this one.
        """
        return self.results.get("eager")

    @property
    def others(self) -> list[BackendResult]:
        """Every non-eager result, in run order."""
        return [result for name, result in self.results.items() if name != "eager"]
