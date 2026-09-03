"""The records a run produces, and nothing else.

PLAN.md "Package layout" does not name this module; it is the shared vocabulary
that :mod:`torch_compile_check.runner`, the five oracles, and the three reports all
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

__all__ = [
    "BackendResult",
    "CapturedException",
    "GraphBreak",
    "GraphHealth",
    "RunSet",
    "TargetSource",
    "TensorMeta",
]

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


@dataclass(frozen=True)
class TargetSource:
    """Where the target came from, and the text it was written as.

    Recorded for the two M3-2 reports that quote the user's own code rather than
    describing it: PLAN.md "Reports" wants the minimal repro inline in the
    Markdown draft, and PLAN.md "Regression test emission" wants the same case
    as a test method. Both need the source, and neither can get it from the
    records the oracles read, which are numbers and labels by design.

    One record rather than five fields on :class:`RunSet`, because the five are
    one fact -- what discovery resolved -- and a report that has any of them
    wants the rest. ``None`` on a :class:`RunSet` means the run was not built
    through :func:`torch_compile_check.discover.load_target` (a hand-built record, or
    a target with no file), which is not the same as a target whose file could
    not be read: that one arrives with :attr:`text` at ``None``.
    """

    file: str | None
    """The target's ``__file__``, for a report to name. ``None`` for a module
    with no file, e.g. one built in memory."""

    text: str | None
    """The file's source, verbatim. ``None`` when it could not be read."""

    entry: str | None
    """The attribute discovery resolved as the entry point, as an expression
    against the target module: ``model``, ``fn``, or whatever ``--entry`` named.

    ``None`` when the entry point lives in another module, which ``--entry
    other:thing`` allows: the name would not resolve inside a repro built from
    this file, and a report that wrote it anyway would hand out code that does
    not run.
    """

    inputs: str | None
    """The attribute the inputs came from: ``inputs``, ``get_inputs``, or what
    ``--inputs`` named. ``None`` under the same rule as :attr:`entry`."""

    keyword_inputs: tuple[str, ...] = ()
    """The keyword names the target is called with, when the inputs were a
    mapping. Empty for the ordinary positional case. A report that emits a call
    has to know which of the two it is writing."""


@dataclass(frozen=True)
class GraphBreak:
    """One place Dynamo gave up and fell back to the interpreter.

    PLAN.md "graph": graph breaks are not bugs, they explain why a user is not
    getting the speedup they expect. What makes one actionable is the pair below
    -- why Dynamo stopped, and which line of the user's own code it stopped at
    -- so both are recorded and neither is derived from the other.
    """

    reason: str
    """Dynamo's own explanation, verbatim and usually a dozen lines.

    Kept whole rather than trimmed here: the hints under the first line are what
    a user acts on, the Markdown report of M3-2 wants them, and the graph oracle
    is where the one-line identity a baseline compares on is derived from.
    """

    user_frame: str | None
    """``file:line in function`` of the last user frame under the break.

    The *last* frame, which is the one closest to the break: the head of the
    stack is whatever entry point the model was called through and is the same
    for every break in a run. ``None`` when Dynamo recorded no user stack.
    """


@dataclass(frozen=True)
class GraphHealth:
    """What one compiled lane's graphs looked like (PLAN.md "graph").

    Filled by :func:`torch_compile_check.runner.run_backend` from
    ``torch._dynamo.explain`` and from ``counters['stats']['unique_graphs']``
    sampled around the repeat call, and read by the graph oracle. Not recorded
    for ``eager`` or ``eager_fp64``: neither is compiled, so neither has graphs.
    """

    graph_count: int = 0
    """How many graphs Dynamo produced. ``0`` means it captured nothing at all."""

    break_count: int = 0
    """``ExplainOutput.graph_break_count``, floored at zero and at :attr:`breaks`.

    torch computes it as ``graph_count - 1``, which is right in the middle and
    wrong at both ends: a callable Dynamo captured no graph for comes back as
    ``-1``, and a break whose resumption produced no second graph comes back one
    lower than the number of reasons recorded. The floors are applied once, in
    :func:`torch_compile_check.runner._graph_health`, so that a negative count cannot
    read as an improvement against a baseline and a break with a reason is never
    counted as no break at all.
    """

    breaks: tuple[GraphBreak, ...] = ()
    """One entry per reason ``explain`` reported, in the order it reported them.

    May be shorter than :attr:`break_count`: the count is torch's arithmetic
    over the graphs and the reasons are what the tracer managed to record. Never
    longer, because of the floor above.
    """

    op_count: int = 0
    """How many operators ended up in the captured graphs."""

    compile_times: str | None = None
    """``torch._dynamo.utils.compile_times(repr="str")`` as ``explain`` returned it.

    Process-cumulative rather than per lane -- torch accumulates it and
    ``reset()`` does not clear it -- so it is context for a report to print and
    never a number this tool subtracts. The per-lane wall time is
    :attr:`BackendResult.first_call_s`, measured around the first compiled call.
    """

    unique_graphs_before: int | None = None
    """``counters['stats']['unique_graphs']`` read just before the repeat call."""

    unique_graphs_after: int | None = None
    """The same counter just after it. ``None`` on either side means the counter
    could not be read, which is not the same as a counter that did not move."""

    explain_error: CapturedException | None = None
    """Set when the ``explain`` pass itself raised.

    A target that raises raises under ``explain`` too, and a lane whose graph
    health could not be measured must not read as a lane with no graph breaks.
    """

    @property
    def measured(self) -> bool:
        """Whether the ``explain`` pass produced counts at all."""
        return self.explain_error is None

    @property
    def recompiled(self) -> bool:
        """Whether the repeat call with identical inputs compiled a new graph.

        PLAN.md "graph": the counter incrementing on a second call with
        identical inputs means the model recompiled when it should not have. An
        unreadable counter answers ``False``, because "not known" must not be
        reported as "it happened".
        """
        before, after = self.unique_graphs_before, self.unique_graphs_after
        if before is None or after is None:
            return False
        return after > before

    @property
    def recompiles(self) -> int:
        """How many graphs the repeat call added, or ``0`` when it added none."""
        before, after = self.unique_graphs_before, self.unique_graphs_after
        if before is None or after is None:
            return 0
        return max(0, after - before)


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

    graph_health: GraphHealth | None = None
    """What ``torch._dynamo.explain`` and the recompile counter said, or ``None``.

    ``None`` for a lane that was never compiled -- ``eager``, ``eager_fp64`` --
    and for a hand-built record. The graph oracle reads it, and treats ``None``
    as "not measured" rather than as "no graph breaks", which are the two
    answers a report must never let read alike.
    """

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

    target_is_module: bool = True
    """Whether the target was an ``nn.Module`` at all.

    A plain callable has no parameters or buffers to isolate, so the deep copy
    :attr:`share_module` switches off never applied to it in the first place.
    Recorded so the report can say that instead of claiming a copy that never
    happened.
    """

    module_copy_error: str | None = None
    """Why the per-lane deep copy did not happen, when it was meant to.

    ``<ExceptionType>: <message>`` for a module that refused ``copy.deepcopy``,
    which leaves every lane sharing one object exactly as ``--share-module``
    does. Until M3-1 that fallback was a warning in the log while the report's
    environment block still said "deep copied per lane"; a run whose lanes may
    have leaked state into each other has to say so where the evidence is read.
    """

    target_source: TargetSource | None = None
    """Where the target came from and what it was written as, or ``None``.

    Filled by :func:`torch_compile_check.runner.run_all` from the discovered target,
    and read only by the Markdown draft and the test emitter of M3-2, which are
    the two reports that quote the user's code. Nothing here is compared.
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
    def module_handling(self) -> str:
        """What actually happened to the module, not what the flags asked for.

        Three states, and the M3 brief's point is that the third one used to be
        invisible. ``--share-module`` is a choice; a module that refused
        ``copy.deepcopy`` gets the same sharing without having chosen it, and
        until M3-1 that showed in the report as "deep copied per lane" with the
        reason only in a warning log. A run whose lanes may have leaked state
        into each other has to say so where the evidence is read.

        A target that is not an ``nn.Module`` gets neither sentence: a function
        has no parameters or buffers, so there was never a copy to make or skip.

        Written here rather than in one of the reports because both the terminal
        block and the JSON environment block have to say it, and two sentences
        that could drift apart would let a report contradict its own artifact.
        """
        if not self.target_is_module:
            return "not copied: the target is a plain callable, with no state to isolate"
        if self.share_module:
            return "shared across every lane (--share-module)"
        if self.module_copy_error is not None:
            return f"shared across every lane (deep copy failed: {self.module_copy_error})"
        return "deep copied per lane"

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
