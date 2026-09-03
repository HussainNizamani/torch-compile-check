"""Metadata oracle.

PLAN.md "Oracles": compares dtype, shape, stride, ``requires_grad``, device, and
contiguity, per output; passes on exact equality on every field. It is the
oracle that catches 191308, int8 matmul silently promoted to int64.

PLAN.md "metadata": stride is compared but reported at a lower severity than
dtype and shape, because a layout change alone is usually a performance decision
rather than a correctness defect. It still appears in the report, since a stride
change combined with an alias change is how a reinplacing bug presents.

The stride rule, stated once so it is not re-derived from the code: a stride
difference where **both** tensors report ``is_contiguous()`` is a ``warn``, and
any other stride difference is a ``fail``. Two contiguous tensors of the same
shape are indistinguishable through indexing, and their strides can still differ
legitimately -- a size-1 or size-0 dimension has no meaningful stride, and a
compiled kernel is free to pick another one. Once either side is not contiguous
the stride is observable: it decides what a view sees and what a mutation
through that view writes.

``requires_grad`` is the one field not read off the tensor in front of it. What
this oracle compares are the runner's output *clones*, and a clone is taken with
``detach()``, so it answers ``False`` however the tensor it copied was built;
before M2-2 that made this field vacuous on every real run. The runner now
records the flag off the live output leaf, and that record wins wherever it
exists. A hand-built pair carrying no record still falls back to the tensor,
which is what comparing two tensors made on the spot is meant to mean.

Torch is imported inside the functions, never at module scope.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from typing import Any

from torch_compile_check.oracles.base import Finding, OracleConfig, Severity, align_outputs
from torch_compile_check.results import BackendResult

__all__ = ["FIELDS", "MetadataOracle"]

log = logging.getLogger("torch_compile_check")

_UNAVAILABLE = object()

# Compared in this order, which is the order the report lists them in: the two
# fields that are almost always the defect first, then layout facts.
FIELDS: tuple[str, ...] = (
    "dtype",
    "shape",
    "stride",
    "requires_grad",
    "device",
    "is_contiguous",
    "layout",
)


class MetadataOracle:
    """Do the two worlds return tensors of the same shape, dtype, and layout?"""

    name: str = "metadata"

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Compare every output leaf field by field.

        ``cfg`` is unused: metadata equality is exact, so there is no tolerance
        to apply. The parameter stays because it is the
        :class:`~torch_compile_check.oracles.base.Oracle` protocol.
        """
        del cfg
        torch = importlib.import_module("torch")
        pairs, findings = align_outputs(eager, other, self.name)
        for index, expected, got in pairs:
            findings.extend(
                self._compare_leaf(
                    torch,
                    index,
                    expected,
                    got,
                    other.backend,
                    _recorded_requires_grad(eager, index),
                    _recorded_requires_grad(other, index),
                )
            )
        return findings

    def _compare_leaf(
        self,
        torch: Any,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
        expected_requires_grad: bool | None = None,
        got_requires_grad: bool | None = None,
    ) -> list[Finding]:
        """One output leaf."""
        expected_is_tensor = isinstance(expected, torch.Tensor)
        got_is_tensor = isinstance(got, torch.Tensor)
        if not expected_is_tensor or not got_is_tensor:
            return self._compare_types(index, expected, got, backend)

        expected_fields = _describe(torch, expected, expected_requires_grad)
        got_fields = _describe(torch, got, got_requires_grad)
        findings = []
        for name in FIELDS:
            expected_value = expected_fields[name]
            got_value = got_fields[name]
            if expected_value is _UNAVAILABLE or got_value is _UNAVAILABLE:
                # A layout that does not answer for this field on one side or
                # the other; the layout difference itself is reported below.
                log.debug("output %d: %s is unavailable on one side", index, name)
                continue
            if expected_value == got_value:
                continue
            severity: Severity = "fail"
            note = ""
            if (
                name == "stride"
                and expected_fields["is_contiguous"] is True
                and got_fields["is_contiguous"] is True
            ):
                severity = "warn"
                note = (
                    ", but both tensors are contiguous, so this is a layout choice "
                    "rather than an observable difference"
                )
            findings.append(
                Finding(
                    oracle=self.name,
                    backend=backend,
                    output_index=index,
                    severity=severity,
                    message=(
                        f"{name} differs: eager {_show(expected_value)}, "
                        f"{backend} {_show(got_value)}{note}"
                    ),
                    details={"field": name, "expected": expected_value, "got": got_value},
                )
            )
        return findings

    def _compare_types(
        self,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
    ) -> list[Finding]:
        """At least one side is not a tensor: compare what the leaf even is.

        A model that returns a tensor under eager and an int under inductor has
        broken the contract before any field can be compared. Two non-tensor
        leaves of the same type are the numerics oracle's business, since ``==``
        is the whole comparison there.
        """
        if type(expected) is type(got):
            return []
        return [
            Finding(
                oracle=self.name,
                backend=backend,
                output_index=index,
                severity="fail",
                message=(
                    f"output type differs: eager returned a {type(expected).__name__}, "
                    f"{backend} returned a {type(got).__name__}"
                ),
                details={
                    "field": "type",
                    "expected": type(expected).__name__,
                    "got": type(got).__name__,
                },
            )
        ]


def _recorded_requires_grad(result: BackendResult, index: int) -> bool | None:
    """What the runner saw on the live output leaf, or ``None`` if it took no record.

    ``None`` is what a hand-built :class:`~torch_compile_check.results.BackendResult`
    gives, and it means "ask the tensor", not "the tensor did not require grad".
    """
    if index < len(result.output_requires_grad):
        return result.output_requires_grad[index]
    return None


def _describe(torch: Any, tensor: Any, requires_grad: bool | None = None) -> dict[str, Any]:
    """Every compared field of one tensor, as report-ready values.

    Values are strings, ints, bools, and lists rather than torch objects, so a
    finding's ``details`` survives the JSON report of M3 unchanged. A field a
    layout refuses to answer (a sparse or nested tensor and ``stride()``) is
    recorded as unavailable rather than as an error: the layout difference is
    the finding, and a traceback out of an oracle is not.

    Args:
        torch: the imported torch module.
        tensor: the leaf to describe.
        requires_grad: the runner's record for this leaf, which replaces what
            the tensor says. ``None`` leaves the tensor to answer. See this
            module's docstring for why the tensor cannot be trusted for it.
    """
    fields: dict[str, Any] = {}
    readers: dict[str, Callable[[], Any]] = {
        "dtype": lambda: str(tensor.dtype),
        "shape": lambda: list(tensor.shape),
        "stride": lambda: list(tensor.stride()),
        "requires_grad": lambda: bool(tensor.requires_grad),
        # PLAN.md "metadata" compares the device; the type is what is compared,
        # not the index, so cuda:0 and cuda:1 are not a divergence of this run.
        "device": lambda: tensor.device.type,
        "is_contiguous": lambda: bool(tensor.is_contiguous()),
        "layout": lambda: str(tensor.layout),
    }
    for name, read in readers.items():
        try:
            fields[name] = read()
        except Exception as exc:
            log.debug("%s is unavailable on this tensor: %s", name, exc)
            fields[name] = _UNAVAILABLE
    if requires_grad is not None:
        fields["requires_grad"] = requires_grad
    return fields


def _show(value: Any) -> str:
    """Render a field value the way a terminal report wants to read it."""
    if isinstance(value, list):
        return f"({', '.join(str(item) for item in value)})"
    return str(value)
