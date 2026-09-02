"""Alias and mutation oracle.

PLAN.md "Oracles": compares storage identity and overlap among all outputs and
inputs, the input mutation set, and Python object identity; passes when the
compiled relation equals the eager relation exactly. It is the oracle for the
195451 inductor reinplacing bug and for 191449 / PR 191844, AOTAutograd
aliased-output identity.

PLAN.md "alias": two tensors are related when they share an untyped storage and
their byte ranges overlap -- storage identity alone is not sufficient, since two
disjoint views of one buffer share a data pointer. Object identity is recorded
separately, because "output 0 is the same object as input 1" is a stronger
contract than "output 0 aliases input 1", and 191449 lives in that gap.

The three rules this oracle fails on, stated once so they are not re-derived
from the code:

* an alias that one lane has and the other does not. Added under compile is the
  195451 shape -- a functional result that comes back as the mutated input --
  and dropped under compile breaks the contract just as squarely, since a caller
  that writes through the returned view expects the write to land.
* an object identity that one lane has and the other does not. Both directions
  are 191449: two outputs collapsed into one object, and one object split into
  two.
* an input mutated by one lane and not by the other, in its values or in its
  layout.

Storage sharing without overlap is recorded and compared, but at ``info``: two
disjoint views of one buffer are not aliases in the sense that matters, and a
compiled backend packing two unrelated outputs into one allocation is a
legitimate choice a user cannot observe. The same goes for
``torch._debug_has_internal_overlap``, which PLAN.md "alias" asks for as context
per tensor: it is reported when the two lanes disagree, and it is never a
verdict.

What is compared is the *relation*, never an address. Two runs allocate at
different addresses by definition, so every fact here is a comparison between
tensors of one run -- is this output the same object as that input, do these two
byte ranges overlap -- and only the answers cross between the lanes.

Torch is imported inside the functions, never at module scope.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass
from typing import Any

from compile_check.oracles.base import Finding, OracleConfig, Severity, align_outputs
from compile_check.results import BackendResult, TensorMeta

__all__ = [
    "META_FIELDS",
    "AliasOracle",
    "AliasRelation",
    "Link",
    "Mutation",
    "relation",
]

log = logging.getLogger("compile_check")

# The layout fields an in-place metadata mutation moves: resize_ changes the
# shape and the stride, as_strided_ and set_ can change all four. Addresses are
# deliberately not here; they are compared within a run and never between two.
META_FIELDS: tuple[str, ...] = ("shape", "stride", "dtype", "storage_offset")


@dataclass(frozen=True)
class Link:
    """How two tensors of one run are related.

    The three facts are ordered by strength: being the same object implies
    sharing a storage, which is what makes an overlap possible at all. They are
    kept apart rather than collapsed into one "aliases" boolean because the
    report has to be able to say which of them changed.
    """

    left: str
    """Entity label, ``output[i]`` or ``input[k]``."""

    right: str
    same_object: bool = False
    same_storage: bool = False
    overlaps: bool = False

    @property
    def aliases(self) -> bool:
        """PLAN.md "alias": a shared storage *and* overlapping byte ranges."""
        return self.same_storage and self.overlaps

    @property
    def related(self) -> bool:
        """Whether this pair is worth recording at all."""
        return self.same_object or self.same_storage or self.overlaps

    def describe(self) -> str:
        """One entry of the relation, as the report prints it."""
        facts = []
        if self.same_object:
            facts.append("same object")
        if self.same_storage:
            facts.append("overlapping" if self.overlaps else "same storage, disjoint")
        elif self.overlaps:  # pragma: no cover - overlap is only computed within a storage
            facts.append("overlapping")
        return f"{self.left}~{self.right} {'/'.join(facts)}" if facts else ""


@dataclass(frozen=True)
class Mutation:
    """What one call did to one input leaf, in place."""

    label: str
    values_changed: bool | None
    """``None`` when the two snapshots could not be compared at all, which is
    not the same answer as "unchanged" and is never reported as one.

    A layout move counts as a value change as well: after a ``resize_`` the
    bytes the tensor covers are not the bytes it covered, and the two snapshots
    are not equal. The layout finding beside it is what says which happened."""

    metadata_changed: tuple[str, ...] = ()
    """The :data:`META_FIELDS` an in-place ``resize_`` or ``as_strided_`` moved."""

    reallocated: bool = False
    """The leaf's first byte moved, so the call reallocated behind it. Context
    for a layout mutation, not a fact compared between lanes: where an allocator
    puts a buffer is not part of the contract."""

    before: str = ""
    after: str = ""
    """The layout on each side, rendered, for the report's detail line."""

    @property
    def happened(self) -> bool:
        """Whether this input was mutated in any way the oracle compares."""
        return bool(self.values_changed) or bool(self.metadata_changed)

    def describe(self) -> str:
        """One entry of the mutation set, as the report prints it."""
        if self.values_changed is None:
            return f"{self.label} mutation unknown"
        what = []
        if self.values_changed:
            what.append("values")
        what.extend(self.metadata_changed)
        if not what:
            return ""
        if self.reallocated:
            what.append("reallocated")
        return f"{self.label} mutated ({', '.join(what)})"


@dataclass(frozen=True)
class AliasRelation:
    """One run's alias relation and mutation set, as the oracle compares them.

    :attr:`links` is sparse: only related pairs are kept, since a relation *is*
    its related pairs and a full n-squared grid of "unrelated" would be the same
    statement written out longhand. :meth:`link` answers for any pair.
    """

    backend: str
    outputs: int
    """How many output leaves the relation covers."""

    inputs: int
    """How many input leaves it covers."""

    links: tuple[Link, ...] = ()
    mutations: tuple[Mutation, ...] = ()
    """One per input leaf, in order, mutated or not."""

    self_overlap: tuple[int | None, ...] = ()
    """``torch._debug_has_internal_overlap`` per output leaf: 0 no, 1 yes, 2
    undecidable, ``None`` when the tensor could not answer."""

    def link(self, left: str, right: str) -> Link:
        """The link for one pair, or an unrelated one when it was not recorded."""
        for entry in self.links:
            if entry.left == left and entry.right == right:
                return entry
        return Link(left=left, right=right)

    def mutation(self, label: str) -> Mutation | None:
        """The mutation record for one input label."""
        for entry in self.mutations:
            if entry.label == label:
                return entry
        return None

    def describe(self) -> list[str]:
        """The whole relation as report-ready strings, aliases then mutations."""
        entries = [text for link in self.links if (text := link.describe())]
        entries += [text for mutation in self.mutations if (text := mutation.describe())]
        entries += [
            f"output[{index}] {'self-overlapping' if overlap == 1 else 'self-overlap undecidable'}"
            for index, overlap in enumerate(self.self_overlap)
            if overlap
        ]
        return entries or ["no aliases, no mutations"]


class AliasOracle:
    """Do the two worlds share, return, and mutate the same tensors?"""

    name: str = "alias"

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Compare the two runs' alias relations and mutation sets.

        ``cfg`` is unused: an alias either holds or it does not, so there is no
        tolerance to apply. The parameter stays because it is the
        :class:`~compile_check.oracles.base.Oracle` protocol.
        """
        del cfg
        torch = importlib.import_module("torch")
        _leaves, findings = align_outputs(eager, other, self.name)
        if not eager.ok or not other.ok:
            # align_outputs has already decided there is nothing to compare; the
            # exception itself is stage localization's to report, not an oracle's.
            return findings

        expected = relation(torch, eager)
        got = relation(torch, other)
        # A lane that returned more leaves than the other has already been
        # reported by align_outputs; comparing the leaves both lanes have is
        # still worth doing, so the relation is compared over the common prefix.
        outputs = min(expected.outputs, got.outputs)
        inputs = min(expected.inputs, got.inputs)
        # Both relations, side by side, on every finding this oracle produces:
        # an alias difference is a statement about two whole relations, and a
        # reader who cannot see the other one cannot check the claim.
        context = {
            "eager_relation": expected.describe(),
            "compiled_relation": got.describe(),
        }

        findings.extend(self._compare_links(expected, got, outputs, inputs, context))
        findings.extend(self._compare_mutations(expected, got, inputs, context))
        findings.extend(self._compare_self_overlap(expected, got, outputs, context))
        return findings

    def _compare_links(
        self,
        expected: AliasRelation,
        got: AliasRelation,
        outputs: int,
        inputs: int,
        context: dict[str, Any],
    ) -> list[Finding]:
        """Every pair, in canonical order: outputs against outputs, then inputs."""
        findings = []
        for left, right in _pair_labels(outputs, inputs):
            finding = self._compare_link(
                expected.link(left, right),
                got.link(left, right),
                got.backend,
                context,
            )
            if finding is not None:
                findings.append(finding)
        return findings

    def _compare_link(
        self,
        expected: Link,
        got: Link,
        backend: str,
        context: dict[str, Any],
    ) -> Finding | None:
        """One pair, or ``None`` when the two lanes agree about it."""
        if expected == got:
            return None
        left, right = expected.left, expected.right
        index = _output_index(left)
        severity: Severity = "fail"

        if expected.same_object != got.same_object:
            field = "identity_added" if got.same_object else "identity_dropped"
            mutator, other_lane = (backend, "eager") if got.same_object else ("eager", backend)
            if right.startswith("input["):
                # The 195451 shape: the lane handed back the input object itself
                # rather than the independent result the model computed.
                message = (
                    f"{mutator} returned {right} itself as {left} and {other_lane} "
                    "returned a distinct object"
                )
            else:
                # The 191449 shape, in both directions: two outputs collapsed
                # into one object, or one object split into two.
                shared = " that share a storage" if expected.aliases or got.aliases else ""
                message = (
                    f"{mutator} returned one object for {left} and {right} and "
                    f"{other_lane} returned distinct objects{shared}"
                )
        elif expected.aliases != got.aliases:
            field = "alias_added" if got.aliases else "alias_dropped"
            message = (
                f"{backend} {left} aliases {right} (same storage, overlapping bytes); "
                f"eager's do not"
                if got.aliases
                else f"eager {left} aliases {right} (same storage, overlapping bytes); "
                f"{backend}'s do not"
            )
        else:
            # Same storage, disjoint bytes: recorded because a reader chasing an
            # aliasing bug wants to see it, never a verdict, because two
            # non-overlapping views of one buffer cannot be told apart from two
            # separate buffers by anything a user can write.
            field = "buffer_sharing"
            severity = "info"
            sharer, other_lane = (backend, "eager") if got.same_storage else ("eager", backend)
            message = (
                f"{sharer} put {left} and {right} in one buffer with disjoint byte "
                f"ranges and {other_lane} did not, which is an allocation choice "
                "rather than an alias"
            )

        return Finding(
            oracle="alias",
            backend=backend,
            output_index=index,
            severity=severity,
            message=message,
            details={
                "field": field,
                "expected": expected.describe() or f"{left}~{right} unrelated",
                "got": got.describe() or f"{left}~{right} unrelated",
                **context,
            },
        )

    def _compare_mutations(
        self,
        expected: AliasRelation,
        got: AliasRelation,
        inputs: int,
        context: dict[str, Any],
    ) -> list[Finding]:
        """The input mutation set, per input leaf."""
        findings = []
        for index in range(inputs):
            label = f"input[{index}]"
            before = expected.mutation(label)
            after = got.mutation(label)
            if before is None or after is None:  # pragma: no cover - inputs is their min
                continue
            findings.extend(self._compare_mutation(before, after, got.backend, context))
        return findings

    def _compare_mutation(
        self,
        expected: Mutation,
        got: Mutation,
        backend: str,
        context: dict[str, Any],
    ) -> list[Finding]:
        """One input leaf: its values, then its layout."""
        label = expected.label
        findings = []
        if expected.values_changed is None or got.values_changed is None:
            if expected.values_changed != got.values_changed:
                findings.append(
                    self._finding(
                        backend,
                        "info",
                        f"whether {label} was mutated could not be compared: the two "
                        "snapshots of it are not comparable in one of the lanes",
                        "mutation_unknown",
                        expected,
                        got,
                        context,
                    )
                )
        elif expected.values_changed != got.values_changed:
            mutator, other_lane = (backend, "eager") if got.values_changed else ("eager", backend)
            findings.append(
                self._finding(
                    backend,
                    "fail",
                    f"{mutator} mutated {label} in place and {other_lane} did not",
                    "mutation_added" if got.values_changed else "mutation_dropped",
                    expected,
                    got,
                    context,
                )
            )

        if expected.metadata_changed != got.metadata_changed:
            added = tuple(f for f in got.metadata_changed if f not in expected.metadata_changed)
            dropped = tuple(f for f in expected.metadata_changed if f not in got.metadata_changed)
            if added and not dropped:
                field = "metadata_mutation_added"
                message = (
                    f"{backend} changed the {', '.join(added)} of {label} in place "
                    f"({got.before} -> {got.after}) and eager did not"
                )
            elif dropped and not added:
                field = "metadata_mutation_dropped"
                message = (
                    f"eager changed the {', '.join(dropped)} of {label} in place "
                    f"({expected.before} -> {expected.after}) and {backend} did not"
                )
            else:
                field = "metadata_mutation_differs"
                message = (
                    f"the two lanes changed {label} in place in different ways: eager "
                    f"moved its {', '.join(expected.metadata_changed)} and {backend} "
                    f"moved its {', '.join(got.metadata_changed)}"
                )
            findings.append(self._finding(backend, "fail", message, field, expected, got, context))
        return findings

    def _compare_self_overlap(
        self,
        expected: AliasRelation,
        got: AliasRelation,
        outputs: int,
        context: dict[str, Any],
    ) -> list[Finding]:
        """PLAN.md "alias": internal overlap per tensor, as context only."""
        findings = []
        for index in range(outputs):
            mine, theirs = expected.self_overlap[index], got.self_overlap[index]
            if mine == theirs or mine is None or theirs is None:
                continue
            findings.append(
                Finding(
                    oracle=self.name,
                    backend=got.backend,
                    output_index=index,
                    severity="info",
                    message=(
                        f"output[{index}] overlaps itself in eager and not in {got.backend}"
                        if mine
                        else f"output[{index}] overlaps itself in {got.backend} and not in eager, "
                        "which changes what a write through it means"
                    ),
                    details={
                        "field": "internal_overlap",
                        "expected": mine,
                        "got": theirs,
                        **context,
                    },
                )
            )
        return findings

    def _finding(
        self,
        backend: str,
        severity: Severity,
        message: str,
        field: str,
        expected: Mutation,
        got: Mutation,
        context: dict[str, Any],
    ) -> Finding:
        """A mutation finding, which belongs to an input rather than an output."""
        return Finding(
            oracle=self.name,
            backend=backend,
            # PLAN.md's Finding indexes outputs; a mutated input belongs to the
            # run, and the label in the message says which input it was.
            output_index=None,
            severity=severity,
            message=message,
            details={
                "field": field,
                "expected": expected.describe() or f"{expected.label} not mutated",
                "got": got.describe() or f"{got.label} not mutated",
                **context,
            },
        )


def relation(torch: Any, result: BackendResult) -> AliasRelation:
    """Build the alias relation and mutation set of one run.

    Reads the live references the runner kept (PLAN.md "Runner semantics" keeps
    the original output objects precisely for this), never the clones: a clone
    has neither the storage nor the object identity of what it copied.

    Args:
        torch: the imported torch module.
        result: the run to describe.

    Returns:
        The relation, sparse in its links: unrelated pairs are not stored.
    """
    outputs = [_facts(torch, leaf) for leaf in result.output_refs]
    inputs = [_facts(torch, leaf) for leaf in result.input_refs]
    entities = {
        **{f"output[{index}]": facts for index, facts in enumerate(outputs)},
        **{f"input[{index}]": facts for index, facts in enumerate(inputs)},
    }
    links = []
    for left, right in _pair_labels(len(outputs), len(inputs)):
        link = _link(left, right, entities[left], entities[right])
        if link.related:
            links.append(link)
    return AliasRelation(
        backend=result.backend,
        outputs=len(outputs),
        inputs=len(inputs),
        links=tuple(links),
        mutations=tuple(_mutations(torch, result)),
        self_overlap=tuple(facts.internal_overlap for facts in outputs),
    )


@dataclass(frozen=True)
class _Facts:
    """What one tensor answers about where its bytes are.

    Every field is optional, and a missing one is a fact not read rather than a
    fact that is false: a meta tensor has no storage to point at, and a sparse
    one has no stride to compute a byte range from.
    """

    tensor: Any = None
    storage_ptr: int | None = None
    byte_range: tuple[int, int] | None = None
    internal_overlap: int | None = None


def _facts(torch: Any, leaf: Any) -> _Facts:
    """Read one leaf, tensor or not, without ever letting torch raise out."""
    if not isinstance(leaf, torch.Tensor):
        return _Facts()
    return _Facts(
        tensor=leaf,
        storage_ptr=_storage_ptr(leaf),
        byte_range=_byte_range(leaf),
        internal_overlap=_internal_overlap(torch, leaf),
    )


def _storage_ptr(tensor: Any) -> int | None:
    """Which buffer a tensor lives in, or ``None`` when that is not an answer.

    A storage holding no bytes is deliberately not an identity: an empty tensor
    can report a null data pointer, and two unrelated empty tensors sharing that
    null would read as one buffer in both lanes and as an alias in neither.
    """
    try:
        storage = tensor.untyped_storage()
        if storage.nbytes() == 0:
            return None
        return int(storage.data_ptr())
    except Exception as exc:
        log.debug("no storage pointer for this tensor: %s", exc)
        return None


def _byte_range(tensor: Any) -> tuple[int, int] | None:
    """The half-open byte range a tensor reaches into its storage.

    PLAN.md "alias": computed from ``storage_offset``, ``stride``, ``shape``,
    and the element size. ``None`` when the tensor reaches nothing (any size-0
    dimension) or cannot say, and a ``None`` range never overlaps anything.
    """
    try:
        element_size = tensor.element_size()
        low = high = tensor.storage_offset()
        for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
            if size == 0:
                return None
            span = (size - 1) * stride
            # Both directions, because a negative stride reaches backwards from
            # the offset; torch does not hand those out today, and a range that
            # silently assumed otherwise would be wrong rather than unsupported.
            low += min(span, 0)
            high += max(span, 0)
        return int(low * element_size), int((high + 1) * element_size)
    except Exception as exc:
        log.debug("no byte range for this tensor: %s", exc)
        return None


def _internal_overlap(torch: Any, tensor: Any) -> int | None:
    """``torch._debug_has_internal_overlap``: 0 no, 1 yes, 2 undecidable."""
    try:
        return int(torch._debug_has_internal_overlap(tensor))
    except Exception as exc:  # pragma: no cover - a layout the helper refuses
        log.debug("no internal-overlap answer for this tensor: %s", exc)
        return None


def _link(left: str, right: str, first: _Facts, second: _Facts) -> Link:
    """Relate two tensors of one run."""
    if first.tensor is None or second.tensor is None:
        # At least one leaf is not a tensor: it has no storage to share, and
        # non-tensor leaves are the numerics oracle's business.
        return Link(left=left, right=right)
    same_storage = first.storage_ptr is not None and first.storage_ptr == second.storage_ptr
    return Link(
        left=left,
        right=right,
        same_object=first.tensor is second.tensor,
        same_storage=same_storage,
        overlaps=same_storage and _overlap(first.byte_range, second.byte_range),
    )


def _overlap(first: tuple[int, int] | None, second: tuple[int, int] | None) -> bool:
    """Whether two half-open byte ranges intersect."""
    if first is None or second is None:
        return False
    return first[0] < second[1] and second[0] < first[1]


def _mutations(torch: Any, result: BackendResult) -> list[Mutation]:
    """The mutation record of every input leaf, in order.

    PLAN.md "alias": the values are compared over the bytes, so a call that
    writes back what was already there is deliberately not counted as a
    mutation. The layout is compared over the records the runner took beside
    those snapshots, since a clone keeps the values of a tensor but not
    necessarily its stride.
    """
    mutations = []
    for index, before in enumerate(result.inputs_before):
        after = result.inputs_after[index] if index < len(result.inputs_after) else None
        meta_before = _meta(result.input_meta_before, index)
        meta_after = _meta(result.input_meta_after, index)
        mutations.append(
            Mutation(
                label=f"input[{index}]",
                values_changed=_values_changed(torch, before, after),
                metadata_changed=_metadata_changed(meta_before, meta_after),
                reallocated=(
                    meta_before is not None
                    and meta_after is not None
                    and meta_before.data_ptr != meta_after.data_ptr
                ),
                before=_render(meta_before),
                after=_render(meta_after),
            )
        )
    return mutations


def _meta(records: list[TensorMeta | None], index: int) -> TensorMeta | None:
    """One layout record, or ``None`` when the runner did not take it."""
    return records[index] if index < len(records) else None


def _values_changed(torch: Any, before: Any, after: Any) -> bool | None:
    """Whether a leaf's value changed across the call, or ``None`` if unknown."""
    if isinstance(before, torch.Tensor) and isinstance(after, torch.Tensor):
        try:
            # equal() is False rather than an error on a shape or dtype change,
            # which is the answer this wants: the tensor is not what it was.
            return not bool(torch.equal(before, after))
        except Exception as exc:
            log.debug("two snapshots of an input could not be compared: %s", exc)
            return None
    if before is None and after is None:
        return False
    try:
        return not bool(before == after)
    except Exception as exc:
        log.debug("two snapshots of a non-tensor input could not be compared: %s", exc)
        return None


def _metadata_changed(before: TensorMeta | None, after: TensorMeta | None) -> tuple[str, ...]:
    """Which of :data:`META_FIELDS` an in-place call moved."""
    if before is None or after is None:
        return ()
    return tuple(field for field in META_FIELDS if getattr(before, field) != getattr(after, field))


def _render(meta: TensorMeta | None) -> str:
    """A layout record as the report prints it."""
    if meta is None:
        return "unknown"
    return (
        f"{meta.dtype.removeprefix('torch.')}{tuple(meta.shape)} "
        f"stride {tuple(meta.stride)} offset {meta.storage_offset}"
    )


def _pair_labels(outputs: int, inputs: int) -> list[tuple[str, str]]:
    """Every compared pair, in one canonical order.

    The relation and the comparison walk the same list, so a link is built and
    read under the same key. Output against output first, since that is where an
    identity collapse shows, then every output against every input, which is
    where a reinplacing bug shows. Input against input is not compared: the
    runner clones the inputs per backend from one source, so that half of the
    relation is the harness's own doing rather than the backend's.
    """
    labels = [f"output[{index}]" for index in range(outputs)]
    pairs = [
        (labels[first], labels[second])
        for first in range(outputs)
        for second in range(first + 1, outputs)
    ]
    pairs += [(label, f"input[{index}]") for label in labels for index in range(inputs)]
    return pairs


def _output_index(label: str) -> int | None:
    """The output index a label names, for the finding's ``output_index``."""
    if not label.startswith("output["):
        return None  # pragma: no cover - the left half of a pair is always an output
    return int(label[len("output[") : -1])
