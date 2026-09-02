"""Submodule delta debug, input shrink, minifier handoff.

PLAN.md "Package layout": ``minimize.py`` -- submodule delta debug, input
shrink, minifier handoff.

PLAN.md "Minimizer, v1": v1 works at the module and input level -- delta
debugging over ``nn.Module`` children, plus input shrinking, with the FX graph
level handed off to torch's built-in accuracy minifier. It runs only after a
finding, and it is allowed to give up.

Four passes, and the plan's own numbering for them.

1. Submodule delta debugging. Every child is replaced, one at a time, with
   ``torch.nn.Identity``; the replacement is kept when the same finding still
   reproduces and reverted when it does not. What survives is the part of the
   model the finding needs, which is what turns "my 200-layer model is wrong"
   into "this one block is wrong". Not every module is stubbable, so the pass
   records the subtrees it could not replace rather than failing.
2. Input shrinking. The leading dimension of the input tensors is halved while
   the finding survives, down to one. Other dimensions are untouched in v1,
   because shrinking a feature dimension usually changes which kernel is
   selected.
3. Backend bisection. Already done: it is the ablation ladder that
   :mod:`compile_check.localize` walked to produce the stage verdict, and this
   module neither repeats nor second-guesses it.
4. Built-in minifier handoff, for the FX level. :func:`handoff_note` writes the
   two environment variables and says what they do. It is deliberately *not*
   executed: the accuracy minifier compares numbers only, it can end with
   "Input graph did not fail the tester", and a tool that silently spent
   minutes on a minifier that then declined would be worse than one that hands
   over the command.

Two decisions about the order and the cost, because both are visible in a
report.

Input shrinking runs before the delta debugging even though the plan numbers it
second. Every candidate the delta pass evaluates is a full run of two lanes, and
running them on the smallest inputs that still reproduce is the cheapest way to
pay for the pass that has one candidate per child.

The minimizer re-runs *two* lanes per candidate -- the eager reference and the
one lane the finding was reported against -- and no others. PLAN.md "Runner
semantics" makes eager the reference world, so a candidate cannot be judged
without it, and the other lanes on the ladder have already said what they had to
say. Each lane gets its own module copy, exactly as
:func:`compile_check.runner.run_all` gives one to each backend.

Torch is imported inside the functions, never at module scope, for the reason
:mod:`compile_check.runner` gives.
"""

from __future__ import annotations

import copy
import importlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from compile_check.discover import Target
from compile_check.oracles import Finding, OracleConfig, run_oracles
from compile_check.results import RunSet
from compile_check.runner import lane_module, run_backend

__all__ = [
    "MAX_STEPS",
    "Budget",
    "Kept",
    "Minimization",
    "Outcome",
    "Reproduces",
    "Shrink",
    "Stub",
    "finding_key",
    "handoff_note",
    "minimize",
    "reproducer",
    "shrink_inputs",
    "stub_children",
]

log = logging.getLogger("compile_check")

# How many candidates the minimizer may evaluate when ``--budget`` gives it no
# wall clock to work to. A candidate is two lane runs, one of which compiles, so
# this is the ceiling that keeps a minimizer on a deep model from running longer
# than the run it is minimizing. It is deliberately generous: PLAN.md's pass is
# one candidate per child plus a handful per input, and a model with more than a
# hundred children is one where --budget is the right knob anyway.
MAX_STEPS = 100

# PLAN.md "Relationship to PyTorch's built-in tools" and "Minimizer, v1": the
# accuracy minifier's two knobs, read from the environment by
# torch/_dynamo/config.py. Written here as the exact strings a user pastes.
REPRO_AFTER = "TORCHDYNAMO_REPRO_AFTER=aot"
REPRO_LEVEL = "TORCHDYNAMO_REPRO_LEVEL=4"


@dataclass(frozen=True)
class Outcome:
    """What one candidate re-run did, as the two facts a caller acts on.

    A bool would collapse the two ways a candidate can fail to reproduce, and
    the difference is the whole content of the delta-debugging report: a child
    whose replacement made the model raise is a child a passthrough does not fit
    (PLAN.md "Minimizer, v1": "an identity or a shape-preserving stub"), and a
    child whose replacement ran fine and lost the finding is the child the
    finding lives in.
    """

    reproduced: bool
    """Whether the same finding -- see :func:`finding_key` -- came back."""

    error: str | None = None
    """The exception type a lane raised, when one did. ``None`` when both ran."""


Reproduces = Callable[[Any, "tuple[Any, ...]", "dict[str, Any]"], Outcome]
"""``reproduces(fn, args, kwargs) -> Outcome``: does this candidate still show it?

A parameter rather than something this module derives, so the two passes below
are testable without compiling anything, and so a caller can supply a cheaper
predicate than :func:`reproducer`'s two lane runs.
"""


@dataclass(frozen=True)
class Shrink:
    """One input leaf whose leading dimension the minimizer halved down."""

    index: int
    """Position in the flattened ``(args, kwargs)``, the same index the runner's
    input records and the alias oracle's ``input[N]`` labels use."""

    before: tuple[int, ...]
    after: tuple[int, ...]


@dataclass(frozen=True)
class Stub:
    """One child module replaced by a passthrough, with the finding intact."""

    path: str
    """Its dotted name under the target, as ``named_modules()`` gives it."""

    module: str
    """The class name that was there, so a report can say what was dropped."""


@dataclass(frozen=True)
class Kept:
    """One child the minimizer could not replace, and why it could not."""

    path: str
    module: str
    reason: str
    """Written for a reader: either a passthrough did not fit, or the finding
    lives in this child."""


class Budget:
    """How much work the minimizer may do, and the sentence saying it stopped.

    Two ceilings, because they answer different questions. ``--budget`` is the
    wall clock a CI job is willing to spend and is the one a user sets;
    :data:`MAX_STEPS` is the ceiling that applies when nobody set one, so a
    minimizer cannot run away on a model with a thousand children.

    Exhausting either is not a failure. It leaves the result *partial*, which
    every report says out loud: a partially minimized case is still smaller than
    the original, and a report that presented it as the smallest reproducer
    would be claiming something it did not check.
    """

    def __init__(self, seconds: float | None = None, steps: int = MAX_STEPS) -> None:
        self.seconds = seconds
        self.steps = steps
        self.started = time.perf_counter()
        self.spent = 0
        self.stopped: str | None = None

    @property
    def elapsed(self) -> float:
        """Wall time since this budget was opened."""
        return time.perf_counter() - self.started

    def spend(self) -> bool:
        """Take one candidate's worth of budget, or refuse and record why.

        Checked *before* the candidate runs, so a ``--budget`` is a ceiling on
        what the minimizer starts rather than on what it has already finished.
        """
        if self.stopped is not None:
            return False
        if self.spent >= self.steps:
            self.stopped = (
                f"the ceiling of {self.steps} candidate re-runs was reached "
                "(--budget SECONDS is the other way to bound this)"
            )
            return False
        if self.seconds is not None and self.elapsed >= self.seconds:
            self.stopped = (
                f"the --budget of {self.seconds:g}s ran out after "
                f"{self.spent} candidate re-run{'' if self.spent == 1 else 's'}"
            )
            return False
        self.spent += 1
        return True


@dataclass(frozen=True)
class Minimization:
    """What the minimizer did, in the form every report renders.

    Data only, like :mod:`compile_check.results`: no live module and no tensor,
    so the record survives into the JSON artifact without either having to be
    re-derived. The minimized *target* itself is not carried, because nothing
    downstream runs it -- the test emitter writes the two changes below into the
    file it generates, and a reader runs that.
    """

    finding: Finding | None = None
    """The finding this pass was aimed at, or ``None`` when none was attempted."""

    reason: str | None = None
    """Why nothing was attempted. Set exactly when :attr:`finding` is ``None``."""

    reproduced: bool = False
    """Whether the finding came back on a re-run of the *unchanged* target.

    Checked first and outside the budget: without it a report could show an
    empty minimization and leave a reader thinking the case resisted shrinking,
    when in fact the finding did not reproduce at all the second time. A run
    that answers ``False`` here stops before either pass.
    """

    shrinks: tuple[Shrink, ...] = ()
    stubs: tuple[Stub, ...] = ()
    kept: tuple[Kept, ...] = ()

    notes: tuple[str, ...] = ()
    """One sentence per thing that could not be done, in the order it was tried."""

    steps: int = 0
    """Candidates evaluated, not counting the unbudgeted control re-run."""

    seconds: float = 0.0
    partial: bool = False
    """Whether a ceiling stopped the pass before it ran out of candidates."""

    partial_reason: str | None = None
    handoff: str = ""
    """The accuracy-minifier note of PLAN.md's step 4, never executed."""

    @property
    def attempted(self) -> bool:
        """Whether there was a finding to minimize at all."""
        return self.finding is not None

    @property
    def changed(self) -> bool:
        """Whether anything actually got smaller."""
        return bool(self.shrinks or self.stubs)

    @property
    def summary(self) -> str:
        """One sentence: what came out, for a report heading or a draft."""
        if self.finding is None:
            return f"nothing to minimize: {self.reason}"
        if not self.reproduced:
            return "nothing was minimized: the finding did not reproduce on a re-run"
        if not self.changed:
            return "nothing could be reduced: every input and every child is load-bearing"
        parts = []
        if self.stubs:
            parts.append(
                f"{len(self.stubs)} child module{'' if len(self.stubs) == 1 else 's'} "
                "replaced with torch.nn.Identity()"
            )
        if self.shrinks:
            parts.append(f"{len(self.shrinks)} input{'' if len(self.shrinks) == 1 else 's'} shrunk")
        return f"{' and '.join(parts)}{', partial' if self.partial else ''}"

    @classmethod
    def not_attempted(cls, reason: str) -> Minimization:
        """The record for a run there was nothing to minimize in.

        A record rather than ``None``, because "the minimizer was asked and had
        nothing to work from" and "the minimizer was not asked" are two
        different statements and a report has to be able to make the first one.
        """
        return cls(finding=None, reason=reason)


def finding_key(finding: Finding) -> tuple[str, str, int | None, str | None]:
    """The identity a candidate has to keep for the finding to count as the same.

    The M3-3 brief's rule: the same oracle, the same output index, and the same
    field. The lane is in there too, because a divergence that moved from
    ``inductor`` to ``aot_eager`` is a different diagnosis and not a smaller
    version of the same one.

    Deliberately *not* the message or the values. A shrunk input changes the
    first differing element and the numbers on both sides of it, and a key that
    included them would call every successful shrink a different finding.
    """
    field_name = finding.details.get("field")
    return (
        finding.oracle,
        finding.backend,
        finding.output_index,
        field_name if isinstance(field_name, str) else None,
    )


def handoff_note(finding: Finding) -> str:
    """PLAN.md step 4: the built-in accuracy minifier, handed over rather than run.

    Two wordings, because the tool has to be honest about which one applies.
    The minifier shrinks an FX graph that already fails an *accuracy* test, so
    for a numerics divergence it is the next step; for an aliasing, metadata,
    gradient, or graph finding it compares the wrong thing entirely, and telling
    a user to run it would send them after an "Input graph did not fail the
    tester" that means nothing about their bug.
    """
    if finding.oracle == "numerics":
        return (
            "torch's own accuracy minifier can take this down to the FX graph level. Run the "
            f"target again with {REPRO_AFTER} {REPRO_LEVEL} set in the environment and it will "
            "write a minified repro directory. compile-check does not run it: the minifier can "
            'end with "Input graph did not fail the tester", and a pass that declines is worth '
            "less than the module and the inputs above."
        )
    return (
        f"torch's accuracy minifier ({REPRO_AFTER} {REPRO_LEVEL}) compares numbers only, so it "
        f"would not isolate this {finding.oracle} finding; the two variables are here because "
        "they are the next step for a numerics divergence, not for this one. compile-check does "
        "not run it either way."
    )


def reproducer(
    runset: RunSet,
    finding: Finding,
    cfg: OracleConfig,
) -> Reproduces:
    """Build the predicate that answers whether one candidate still shows ``finding``.

    Two lanes per call, run exactly the way :func:`compile_check.runner.run_all`
    ran them -- same device, same seed, same ``fullgraph`` and ``dynamic``, own
    module copy each -- and then the one oracle that produced the finding, so
    that a candidate is judged by the same rule the report was written from.

    Two things are switched off, both because nothing the comparison reads
    depends on them. Graph health costs a third trace through the target and
    only the graph oracle reads it, so it is measured only when the finding is a
    graph finding. The backward pass is run only for a gradient finding, for the
    same reason: PLAN.md's other four oracles compare the forward call.

    A candidate that will not run at all is not an error. The eager lane
    refusing a stubbed child is exactly how "a passthrough does not fit here"
    reaches the report, so it comes back as an :class:`Outcome` with the
    exception type on it.
    """
    key = finding_key(finding)
    lane = finding.backend
    measure_graphs = finding.oracle == "graph"
    grad = runset.grad and finding.oracle == "grad"

    def check(fn: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Outcome:
        reference = run_backend(
            lane_module(fn, share=runset.share_module)[0],
            args,
            "eager",
            kwargs=kwargs,
            device=runset.device,
            seed=runset.seed,
            grad=grad,
            measure_graphs=False,
        )
        if not reference.ok:
            assert reference.exception is not None
            log.debug("candidate rejected: eager raised %s", reference.exception.type)
            return Outcome(reproduced=False, error=reference.exception.type)

        other = run_backend(
            lane_module(fn, share=runset.share_module)[0],
            args,
            lane,
            kwargs=kwargs,
            device=runset.device,
            seed=runset.seed,
            fullgraph=runset.fullgraph,
            dynamic=runset.dynamic,
            grad=grad,
            measure_graphs=measure_graphs,
        )
        found = any(
            finding_key(candidate) == key
            for candidate in run_oracles(reference, other, cfg, names=[finding.oracle])
        )
        error = other.exception.type if other.exception is not None else None
        return Outcome(reproduced=found, error=error)

    return check


def shrink_inputs(
    fn: Any,
    args: Sequence[Any],
    kwargs: dict[str, Any],
    reproduces: Reproduces,
    *,
    budget: Budget,
) -> tuple[tuple[Any, ...], dict[str, Any], list[Shrink], list[str]]:
    """PLAN.md step 2: halve the leading dimension while the finding survives.

    Leaves that share a leading dimension are halved *together* first. A batched
    model is given a batch of activations, a batch of masks, and a batch of
    labels, and halving one of them alone makes the model raise rather than
    reproduce; halving the group is the step that actually shrinks such a
    target. Each leaf is then offered a halving of its own, so a leaf whose
    leading dimension is not the batch is not held back by one that is.

    Only the leading dimension moves, per PLAN.md: shrinking a feature dimension
    usually changes which kernel is selected, which would make the minimized
    case a different case.

    Args:
        fn: the target, passed through to the predicate unchanged.
        args: the positional inputs.
        kwargs: the keyword inputs.
        reproduces: the predicate, see :data:`Reproduces`.
        budget: the ceiling; every candidate spends one unit of it.

    Returns:
        The smallest inputs that still reproduced, one :class:`Shrink` per leaf
        that moved, and a note for each reason nothing did.
    """
    torch = importlib.import_module("torch")
    pytree = torch.utils._pytree
    leaves, spec = pytree.tree_flatten((tuple(args), dict(kwargs)))
    before = [_shape(torch, leaf) for leaf in leaves]

    def halve(group: list[int]) -> None:
        """Halve every leaf in ``group`` together, while the finding survives."""
        nonlocal leaves
        while True:
            size = _leading(torch, leaves[group[0]])
            if size <= 1:
                return
            # Floor division, and never below one: PLAN.md's loop runs "until
            # halving stops reproducing or the dimension reaches one", and a
            # zero-length input is not a smaller case, it is a different one.
            smaller = max(1, size // 2)
            if not budget.spend():
                return
            candidate = list(leaves)
            for index in group:
                candidate[index] = _head(torch, candidate[index], smaller)
            new_args, new_kwargs = pytree.tree_unflatten(candidate, spec)
            if not reproduces(fn, tuple(new_args), dict(new_kwargs)).reproduced:
                return
            leaves = candidate

    # Largest first, so the dimension with the most to give is tried while there
    # is the most budget left.
    sizes = sorted({size for size in (_leading(torch, leaf) for leaf in leaves) if size > 1})
    for size in reversed(sizes):
        group = [index for index, leaf in enumerate(leaves) if _leading(torch, leaf) == size]
        if not group:
            # Every leaf that had this leading dimension has already been
            # halved past it by an earlier, larger group.
            continue
        halve(group)
        if len(group) > 1:
            for index in group:
                if _leading(torch, leaves[index]) > 1:
                    halve([index])

    after = [_shape(torch, leaf) for leaf in leaves]
    shrinks = [
        Shrink(index=index, before=was, after=now)
        for index, (was, now) in enumerate(zip(before, after, strict=True))
        if was != now and was is not None and now is not None
    ]
    new_args, new_kwargs = pytree.tree_unflatten(leaves, spec)
    return tuple(new_args), dict(new_kwargs), shrinks, _shrink_notes(before, shrinks)


def _shrink_notes(before: Sequence[tuple[int, ...] | None], shrinks: Sequence[Shrink]) -> list[str]:
    """Why nothing shrank, in the words the M3-3 brief asks the report to carry."""
    if shrinks:
        return []
    if not any(shape is not None for shape in before):
        return ["no input is a tensor, so there was no dimension to shrink"]
    # `if shape` filters out both a non-tensor leaf (None) and a scalar one
    # (an empty shape), neither of which has a leading dimension to index.
    if not any(shape[0] > 1 for shape in before if shape):
        return [
            "no input has a leading dimension above 1, so there was nothing to halve "
            '(v1 shrinks the leading dimension only, see PLAN.md "Minimizer, v1")'
        ]
    return [
        "every input's leading dimension is load-bearing: halving it stopped reproducing "
        "the finding, so the inputs are unchanged"
    ]


def stub_children(
    fn: Any,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    reproduces: Reproduces,
    *,
    budget: Budget,
) -> tuple[Any, list[Stub], list[Kept], list[str]]:
    """PLAN.md step 1: delta debugging over the module tree.

    ``named_modules()`` walks parents before their children, and a child under a
    subtree that was successfully replaced is skipped, so the pass is greedy
    from the top: replacing a whole block costs one candidate and saves one per
    module inside it.

    A replacement is kept when the same finding still reproduces and reverted
    when it does not, which is the two-line definition of delta debugging. The
    work happens on a deep copy: the target the caller handed in is the object
    the run under report was made with, and a minimizer that edited it would
    change what the rest of the process is describing.

    Returns:
        The stubbed module, the replacements that stuck, the children that could
        not be replaced with the reason for each, and a note per reason the pass
        did not run at all.
    """
    torch = importlib.import_module("torch")
    if not isinstance(fn, torch.nn.Module):
        return fn, [], [], ["the target is a plain callable, so it has no children to replace"]
    try:
        work = copy.deepcopy(fn)
    except Exception as exc:
        # The same fallback the runner takes for a module it cannot copy: say so
        # and do less, rather than edit the caller's own object.
        log.debug("no delta debugging: the module could not be deep copied (%s)", exc)
        return (
            fn,
            [],
            [],
            [
                f"the module could not be deep copied ({type(exc).__name__}: {exc}), so no "
                "child was replaced -- editing the target the report describes is not an option"
            ],
        )

    stubs: list[Stub] = []
    kept: list[Kept] = []
    children = [(path, child) for path, child in work.named_modules() if path]
    for path, child in children:
        if any(path == stub.path or path.startswith(f"{stub.path}.") for stub in stubs):
            continue
        if isinstance(child, torch.nn.Identity):
            # Already a passthrough. Replacing it would be a candidate spent to
            # learn nothing, and reporting it as stubbed would claim a reduction
            # that was in the model to begin with.
            continue
        if not budget.spend():
            break
        original = _replace(work, path, torch.nn.Identity())
        outcome = reproduces(work, args, kwargs)
        if outcome.reproduced:
            stubs.append(Stub(path=path, module=type(original).__name__))
            continue
        _replace(work, path, original)
        kept.append(
            Kept(
                path=path,
                module=type(original).__name__,
                reason=(
                    f"replacing it raised {outcome.error}, so a passthrough does not fit there"
                    if outcome.error is not None
                    else "the finding did not survive the replacement, so it lives in here"
                ),
            )
        )

    notes = [] if children else ["the target module has no children to replace"]
    return work, stubs, kept, notes


def _replace(module: Any, path: str, replacement: Any) -> Any:
    """Swap one child in place and hand back what was there.

    ``getattr``/``setattr`` rather than ``set_submodule``: the name of a
    ``Sequential`` child is ``"0"``, which is not an attribute a reader can
    write, and ``nn.Module.__setattr__`` registers a module under whatever
    string it is given.
    """
    parent_path, _, name = path.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    original = getattr(parent, name)
    setattr(parent, name, replacement)
    return original


def minimize(
    target: Target,
    runset: RunSet,
    finding: Finding,
    cfg: OracleConfig,
    *,
    budget: float | None = None,
    steps: int = MAX_STEPS,
) -> Minimization:
    """Shrink a reproducing case as far as it still reproduces.

    Args:
        target: what the run was made from, from
            :func:`compile_check.discover.load_target`. Its module has already
            been placed on the device by the runner, so the candidates run where
            the finding was found.
        runset: the run the finding came from, for the lane settings the
            candidates have to keep.
        finding: the divergence to keep alive, conventionally the same one the
            regression-test emitter writes about.
        cfg: the oracle configuration the finding was produced under.
        budget: ``--budget``, a wall-clock ceiling in seconds, or ``None``.
        steps: the candidate ceiling that applies when there is no ``--budget``.

    Returns:
        The record: the stubbed children, the shrunk inputs, the subtrees that
        could not be replaced, and the minifier handoff note. It never carries a
        verdict: minimizing changes what a report shows, never what it decided.
    """
    started = time.perf_counter()
    allowance = Budget(seconds=budget, steps=steps)
    reproduces = reproducer(runset, finding, cfg)
    handoff = handoff_note(finding)

    # The control, and outside the budget on purpose: it is the one re-run
    # without which the record would be a claim about nothing. A finding that
    # does not come back here is a finding the minimizer must not pretend to
    # have shrunk.
    control = reproduces(target.fn, tuple(target.example_inputs), dict(target.kwargs))
    if not control.reproduced:
        return Minimization(
            finding=finding,
            reproduced=False,
            notes=(
                "the finding did not reproduce on a re-run of the unchanged target"
                + (f" ({control.error} was raised this time)" if control.error else "")
                + ", so nothing was minimized; a finding that does not reproduce twice is "
                "not one to file",
            ),
            seconds=time.perf_counter() - started,
            handoff=handoff,
        )

    args, kwargs, shrinks, notes = shrink_inputs(
        target.fn,
        tuple(target.example_inputs),
        dict(target.kwargs),
        reproduces,
        budget=allowance,
    )
    _, stubs, kept, stub_notes = stub_children(
        target.fn, args, kwargs, reproduces, budget=allowance
    )
    return Minimization(
        finding=finding,
        reproduced=True,
        shrinks=tuple(shrinks),
        stubs=tuple(stubs),
        kept=tuple(kept),
        notes=(*notes, *stub_notes),
        steps=allowance.spent,
        seconds=time.perf_counter() - started,
        partial=allowance.stopped is not None,
        partial_reason=allowance.stopped,
        handoff=handoff,
    )


def _leading(torch: Any, leaf: Any) -> int:
    """The leading dimension of one leaf, or ``0`` for anything without one.

    Zero rather than ``None`` so the callers can compare sizes without a guard;
    nothing is ever halved below one, so a zero can never be mistaken for a
    dimension worth shrinking.
    """
    if not isinstance(leaf, torch.Tensor) or leaf.dim() == 0:
        return 0
    return int(leaf.shape[0])


def _shape(torch: Any, leaf: Any) -> tuple[int, ...] | None:
    """One leaf's shape, or ``None`` when it is not a tensor."""
    if not isinstance(leaf, torch.Tensor):
        return None
    return tuple(int(size) for size in leaf.shape)


def _head(torch: Any, leaf: Any, size: int) -> Any:
    """The first ``size`` slices of one leaf, as a fresh leaf tensor.

    Detached and cloned rather than sliced: a slice of a tensor that requires
    grad is not a leaf, and a non-leaf never gets the ``.grad`` PLAN.md "grad"
    compares. ``requires_grad`` is restored so the candidate is the same kind of
    input the original was.
    """
    del torch  # the leaf answers all of this itself
    shrunk = leaf[:size].detach().clone()
    if leaf.requires_grad:
        shrunk.requires_grad_(True)
    return shrunk
