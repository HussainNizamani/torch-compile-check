"""Numerics oracle.

PLAN.md "Oracles": compares output tensor values, per output; passes when
``assert_close`` holds within the per-dtype tolerance, plus NaN and inf position
parity. It is the oracle for the 190765 CPU inductor miscompile and for the
194593 and 194596 divergent validation branches.

PLAN.md "numerics": the comparison runs with ``check_dtype=False`` and
``check_stride=False``, because dtype and stride are the metadata oracle's job
and reporting one divergence twice hides which one is the real defect. A
tolerance-level difference is never called a bug; a NaN or inf that appears or
disappears is, because that is a category difference rather than a rounding one.

PLAN.md "The oracle blind spot": with ``--fp64-oracle`` the runner adds an
``eager_fp64`` reference and this oracle reports, at ``info`` severity, when
eager itself has already drifted from it. That is the difference between "the
compiled lane is wrong" and "both lanes are imprecise", and it is the first
question a reviewer asks about a tool that treats eager as ground truth.

Torch is imported inside the functions, never at module scope, so that importing
``compile_check.oracles`` (which ``cli.py`` does) does not pay for it.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from compile_check.oracles.base import Finding, OracleConfig, Severity, align_outputs
from compile_check.results import BackendResult

__all__ = [
    "FALLBACK_TOLERANCES",
    "Mismatch",
    "NumericsOracle",
    "compare_tensors",
    "resolve_tolerances",
]

log = logging.getLogger("compile_check")

# PLAN.md "Verified API surface", measured against the installed wheel: the
# per-dtype defaults behind assert_close. Used only when
# torch.testing._comparison.default_tolerances is missing, which is a private
# API the plan flags as such; the live function is preferred so that a torch
# that retunes its own defaults retunes ours. Keyed by str(dtype) because this
# module must not import torch to hold a table. Anything absent is exact, which
# is what integer and bool dtypes want.
FALLBACK_TOLERANCES: dict[str, tuple[float, float]] = {
    "torch.float16": (1e-3, 1e-5),
    "torch.bfloat16": (1.6e-2, 1e-5),
    "torch.float32": (1.3e-6, 1e-5),
    "torch.float64": (1e-7, 1e-7),
    "torch.complex32": (1e-3, 1e-5),
    "torch.complex64": (1.3e-6, 1e-5),
    "torch.complex128": (1e-7, 1e-7),
}


def resolve_tolerances(
    torch: Any,
    dtypes: Sequence[Any],
    cfg: OracleConfig,
    *,
    factor: float = 1.0,
) -> tuple[float, float]:
    """The ``(rtol, atol)`` this comparison runs with.

    ``--rtol`` and ``--atol`` override per dtype and independently: passing only
    ``--atol`` keeps the default relative tolerance rather than dropping it to
    zero, because a user tightening one knob did not ask to tighten the other.

    Args:
        torch: the imported torch module.
        dtypes: every dtype taking part in the comparison. Both are passed even
            though ``check_dtype=False``, because ``assert_close`` promotes and
            then uses the loosest tolerance of the dtypes involved.
        cfg: the run configuration holding the overrides.
        factor: scales both tolerances, last, after the overrides. The grad
            oracle passes ``cfg.grad_tol_factor`` here
            (:data:`~compile_check.oracles.base.DEFAULT_GRAD_TOL_FACTOR`); every
            other caller leaves it at 1. Applied after the overrides on purpose:
            it is a property of what is being compared, not of what the user
            asked for, so ``--atol 1e-6 --grad-tol-factor 10`` compares
            gradients at 1e-5 and outputs at 1e-6, which is the policy stated in
            one place rather than two rules that interact.

    Returns:
        The relative and absolute tolerance.
    """
    rtol, atol = _default_tolerances(torch, dtypes)
    if cfg.rtol is not None:
        rtol = cfg.rtol
    if cfg.atol is not None:
        atol = cfg.atol
    return rtol * factor, atol * factor


@dataclass(frozen=True)
class Mismatch:
    """Why two tensors did not compare equal, in the words a report prints.

    Not a :class:`~compile_check.oracles.base.Finding`: it carries no oracle
    name, no backend, and no output index, because the caller is what knows
    those. It is the value comparison's answer, and whichever oracle asked turns
    it into a finding of its own.
    """

    message: str
    """One line, ``assert_close``'s own words wherever it had any."""

    details: dict[str, Any]
    """The tolerances the decision was made with, and the two dtypes."""


def compare_tensors(
    torch: Any,
    expected: Any,
    got: Any,
    cfg: OracleConfig,
    *,
    tol_factor: float = 1.0,
) -> Mismatch | None:
    """Compare two tensors under PLAN.md's numerics rule, or say why not.

    The one place the rule lives, so that the grad oracle of M2-2 compares a
    gradient exactly the way the numerics oracle compares an output: the same
    per-dtype tolerances, the same ``--rtol`` and ``--atol`` overrides, dtype
    and stride left to the metadata oracle, and a NaN in the same place on both
    sides counted as agreement rather than as a mismatch.

    ``assert_close`` writes a better mismatch message than anything built here
    would: element counts, the greatest absolute and relative difference, and
    where each occurs. It is used for its message, which is why the comparison
    is a try/except rather than a boolean.

    Args:
        torch: the imported torch module.
        expected: the reference tensor, from the eager lane.
        got: the tensor under test.
        cfg: the run's tolerances.
        tol_factor: scales both tolerances, for a caller whose comparison is
            looser by policy. The grad oracle is the only one today; see
            :func:`resolve_tolerances`.

    Returns:
        ``None`` when the two agree within tolerance, or the :class:`Mismatch`
        that says how they did not. A pair the comparison could not walk at all
        is a mismatch too, with the error in its message: a value the tool could
        not check must not read as a value that passed.
    """
    rtol, atol = resolve_tolerances(torch, (expected.dtype, got.dtype), cfg, factor=tol_factor)
    details: dict[str, Any] = {
        "rtol": rtol,
        "atol": atol,
        "expected_dtype": str(expected.dtype),
        "got_dtype": str(got.dtype),
    }
    if tol_factor != 1.0:
        # Said out loud in the finding, because the same two tensors compared
        # under the output rule would have been a fail and a reader has to be
        # able to see which rule produced the number above.
        details["tol_factor"] = tol_factor
    try:
        torch.testing.assert_close(
            got,
            expected,
            rtol=rtol,
            atol=atol,
            # PLAN.md "numerics": NaN parity is its own finding in the numerics
            # oracle, so here a NaN in the same place on both sides is agreement
            # rather than a value mismatch.
            equal_nan=True,
            check_device=False,
            # dtype and stride belong to the metadata oracle.
            check_dtype=False,
            check_stride=False,
        )
    except AssertionError as exc:
        details["assert_close"] = str(exc)
        return Mismatch(message=_one_line(str(exc)), details=details)
    except Exception as exc:
        # Not a mismatch: a meta tensor, a layout assert_close cannot walk.
        details["error"] = f"{type(exc).__name__}: {exc}"
        return Mismatch(
            message=f"values could not be compared: {type(exc).__name__}: {_one_line(str(exc))}",
            details=details,
        )
    return None


def _default_tolerances(torch: Any, dtypes: Sequence[Any]) -> tuple[float, float]:
    """The per-dtype defaults, from torch when it exposes them."""
    if not dtypes:
        return 0.0, 0.0
    try:
        comparison = importlib.import_module("torch.testing._comparison")
    except ImportError:  # pragma: no cover - torch without the private module
        comparison = None
    live = getattr(comparison, "default_tolerances", None)
    if live is not None:
        try:
            rtol, atol = live(*dtypes)
        except Exception as exc:  # pragma: no cover - a torch that changed the signature
            log.debug("default_tolerances(%s) failed, using the table: %s", dtypes, exc)
        else:
            return float(rtol), float(atol)
    pairs = [FALLBACK_TOLERANCES.get(str(dtype), (0.0, 0.0)) for dtype in dtypes]
    return max(rtol for rtol, _ in pairs), max(atol for _, atol in pairs)


class NumericsOracle:
    """Do the two worlds compute the same numbers?"""

    name: str = "numerics"

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Compare every output leaf, then the fp64 reference if there is one."""
        torch = importlib.import_module("torch")
        pairs, findings = align_outputs(eager, other, self.name)
        for index, expected, got in pairs:
            findings.extend(self._compare_leaf(torch, index, expected, got, other.backend, cfg))
        findings.extend(self._compare_fp64(torch, eager, other, cfg))
        return findings

    def _compare_leaf(
        self,
        torch: Any,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """One output leaf, tensor or not."""
        expected_is_tensor = isinstance(expected, torch.Tensor)
        got_is_tensor = isinstance(got, torch.Tensor)
        if expected_is_tensor != got_is_tensor:
            # The metadata oracle names the type difference; this one says why
            # no value comparison happened, so the report is not silently short
            # one output.
            return [
                self._finding(
                    index,
                    backend,
                    "fail",
                    f"values cannot be compared: eager returned a "
                    f"{type(expected).__name__} and {backend} returned a {type(got).__name__}",
                    {"expected_type": type(expected).__name__, "got_type": type(got).__name__},
                )
            ]
        if not expected_is_tensor:
            return self._compare_non_tensor(index, expected, got, backend)

        findings = self._compare_masks(torch, index, expected, got, backend)
        findings.extend(self._compare_values(torch, index, expected, got, backend, cfg))
        return findings

    def _compare_non_tensor(
        self,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
    ) -> list[Finding]:
        """A non-tensor leaf: an int, a string, a None a model returned alongside.

        Compared with ``==``, because there is no tolerance to apply and no
        dtype to promote. Anything whose ``==`` does not answer with a bool (a
        numpy array is the usual one) is reported rather than guessed at.
        """
        try:
            same = bool(expected == got)
        except Exception as exc:
            return [
                self._finding(
                    index,
                    backend,
                    "fail",
                    f"non-tensor output could not be compared with ==: {type(exc).__name__}: {exc}",
                    {"expected": repr(expected), "got": repr(got)},
                )
            ]
        if same:
            return []
        return [
            self._finding(
                index,
                backend,
                "fail",
                f"non-tensor output differs: eager returned {expected!r}, "
                f"{backend} returned {got!r}",
                {"expected": repr(expected), "got": repr(got)},
            )
        ]

    def _compare_masks(
        self,
        torch: Any,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
    ) -> list[Finding]:
        """NaN and inf position parity, as findings of their own.

        PLAN.md "numerics": a NaN appearing or disappearing is a category
        difference rather than a rounding difference, and ``assert_close`` with
        ``equal_nan=False`` would report it as a plain value mismatch. So the
        boolean masks are compared for exact positional equality, separately
        from the values, and the value comparison keeps ``equal_nan=True``.
        """
        if not any(t.is_floating_point() or t.is_complex() for t in (expected, got)):
            # isnan on an integer tensor is all-False in both worlds; comparing
            # those masks would only add noise.
            return []
        findings = []
        for label, predicate in (("NaN", torch.isnan), ("inf", torch.isinf)):
            finding = self._mask_finding(torch, index, expected, got, backend, label, predicate)
            if finding is not None:
                findings.append(finding)
        return findings

    def _mask_finding(
        self,
        torch: Any,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
        label: str,
        predicate: Any,
    ) -> Finding | None:
        """One mask comparison, or ``None`` when the masks agree."""
        try:
            # .cpu() so that a device difference (which check_device=False lets
            # the value comparison ignore) does not turn into a comparison error
            # here; on a CPU run it is a no-op.
            expected_mask = predicate(expected).cpu()
            got_mask = predicate(got).cpu()
            if expected_mask.shape != got_mask.shape:
                # A shape divergence; assert_close and the metadata oracle both
                # report it, and an elementwise mask comparison cannot.
                return None
            differing = expected_mask != got_mask
            count = int(differing.sum())
        except Exception as exc:
            log.debug("output %d: %s mask comparison failed: %s", index, label, exc)
            return None
        if count == 0:
            return None
        first = differing.nonzero()
        return self._finding(
            index,
            backend,
            "fail",
            f"{label} positions differ in {count} of {expected_mask.numel()} elements: "
            f"eager has {int(expected_mask.sum())} {label} values, "
            f"{backend} has {int(got_mask.sum())}",
            {
                "field": f"{label.lower()}_mask",
                "differing_elements": count,
                "expected_count": int(expected_mask.sum()),
                "got_count": int(got_mask.sum()),
                "first_differing_index": first[0].tolist() if first.numel() else None,
            },
        )

    def _compare_values(
        self,
        torch: Any,
        index: int,
        expected: Any,
        got: Any,
        backend: str,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """One output leaf's values, as a finding or nothing.

        The rule itself is :func:`compare_tensors`, so that a gradient and an
        output are compared by the same code; this turns its answer into a
        finding that knows which output and which backend it belongs to.
        """
        mismatch = compare_tensors(torch, expected, got, cfg)
        if mismatch is None:
            return []
        return [self._finding(index, backend, "fail", mismatch.message, mismatch.details)]

    def _compare_fp64(
        self,
        torch: Any,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Is eager itself already off the fp64 reference?

        PLAN.md "The oracle blind spot": the fp64 pass separates two cases the
        two-way comparison conflates. Compiled being further from fp64 than
        eager is a compiled defect; both drifting similarly means the
        eager-versus-compiled gap is accumulated rounding. Neither is a verdict,
        so the finding is ``info`` and never fails a run: it is context for the
        numerics findings above it.
        """
        reference = cfg.fp64_reference
        if not cfg.fp64 or reference is None:
            return []
        if not reference.ok:
            log.debug("fp64 reference raised %s, skipping", reference.exception)
            return []

        findings = []
        for index, expected in enumerate(eager.outputs):
            if index >= len(reference.outputs):
                break
            exact = reference.outputs[index]
            got = other.outputs[index] if index < len(other.outputs) else None
            eager_error = _distance(torch, expected, exact)
            if eager_error is None:
                continue
            rtol, atol = resolve_tolerances(torch, (expected.dtype,), cfg)
            if torch.allclose(
                expected.to(torch.float64),
                exact.to(torch.float64),
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ):
                continue
            got_error = _distance(torch, got, exact)
            verdict = (
                "both imprecise"
                if got_error is None or got_error <= eager_error
                else f"{other.backend} is further from fp64 than eager is"
            )
            findings.append(
                self._finding(
                    index,
                    other.backend,
                    "info",
                    f"eager itself deviates from the fp64 reference by "
                    f"{eager_error:.3g} (rtol={rtol:g}, atol={atol:g}); "
                    f"{other.backend} deviates by "
                    f"{'n/a' if got_error is None else format(got_error, '.3g')}: {verdict}",
                    {
                        "field": "fp64_reference",
                        "eager_max_abs_diff": eager_error,
                        "got_max_abs_diff": got_error,
                        "rtol": rtol,
                        "atol": atol,
                        "verdict": verdict,
                    },
                )
            )
        return findings

    def _finding(
        self,
        index: int | None,
        backend: str,
        severity: Severity,
        message: str,
        details: dict[str, Any],
    ) -> Finding:
        """A finding stamped with this oracle's name."""
        return Finding(
            oracle=self.name,
            backend=backend,
            output_index=index,
            severity=severity,
            message=message,
            details=details,
        )


def _distance(torch: Any, value: Any, exact: Any) -> float | None:
    """Greatest absolute difference from the fp64 reference, in float64.

    ``None`` when the pair is not comparable that way: a non-tensor leaf, an
    integer output (which fp64 does not change), or a shape divergence, all of
    which the ordinary comparisons above already report.
    """
    if not isinstance(value, torch.Tensor) or not isinstance(exact, torch.Tensor):
        return None
    if not (value.is_floating_point() and exact.is_floating_point()):
        return None
    if value.shape != exact.shape:
        return None
    try:
        difference = (value.to(torch.float64) - exact.to(torch.float64)).abs()
        return float(difference.max()) if difference.numel() else 0.0
    except Exception as exc:  # pragma: no cover - a layout float64 cannot subtract
        log.debug("fp64 distance failed: %s", exc)
        return None


def _one_line(message: str) -> str:
    """Collapse a multi-line torch message into one line for the report."""
    return "; ".join(line.strip() for line in message.splitlines() if line.strip())
