"""Gradients oracle.

PLAN.md "Oracles": compares the ``.grad`` of every input and parameter after one
backward on a deterministic scalar reduction, and the set of tensors that
received a grad at all; grad values pass the numerics rule and the grad presence
set must be identical. Its bug class is backward-only divergence and partitioner
bugs.

PLAN.md "grad": the oracle activates when any input or parameter has
``requires_grad``. Grads are zeroed before each backward so a leaked
accumulation cannot be mistaken for a divergence. Only one backward step is run;
multi-step training loop correctness is out of scope for v1.

Three rules, stated once so they are not re-derived from the code:

* a backward that raised in one lane and not in the other. Reported on its own
  and nothing else is: a lane whose backward did not finish has no gradients for
  a reason that is already known, and listing every missing one underneath would
  bury the sentence that says why.
* the presence set. A tensor that ended up with a gradient in one world and not
  in the other is a divergence whatever the gradients that did arrive say, and
  the finding names the parameter.
* the values, through :func:`torch_compile_check.oracles.numerics.compare_tensors`, so
  a gradient is compared by the same rule an output is: the same per-dtype
  tolerances, the same ``--rtol`` and ``--atol`` overrides, and then one
  deliberate loosening. Both tolerances are multiplied by
  ``cfg.grad_tol_factor``
  (:data:`~torch_compile_check.oracles.base.DEFAULT_GRAD_TOL_FACTOR`, ``10``, settable
  with ``--grad-tol-factor``), because a gradient is a sum over every path that
  reaches a tensor and compilation is free to fuse and reassociate that sum. The
  M2-2 verification measured the boundary of it: a compiled resnet18 backward
  sat about 1.24e-5 from eager's against a float32 atol of 1e-5, close enough
  that the same run came back clean or failing depending on the last bit. Ten
  clears that and does not clear every model -- see
  :data:`~torch_compile_check.oracles.base.DEFAULT_GRAD_TOL_FACTOR` for what a whole
  resnet18 backward actually needs, which is why the number is a flag. The
  output tolerances are untouched -- the reason for the looser rule is the
  backward's accumulation and nothing else, and lending it to the forward
  comparison would blunt the oracle that catches 190765.

What this oracle deliberately does not check is the ``requires_grad`` flag on
the outputs. That is a field of PLAN.md "metadata", the metadata oracle compares
it per output, and reporting one divergence from two oracles hides which one is
the real defect.

Torch is imported inside the functions, never at module scope.
"""

from __future__ import annotations

import importlib
import logging
from typing import Any

from torch_compile_check.oracles.base import Finding, OracleConfig, Severity
from torch_compile_check.oracles.numerics import compare_tensors
from torch_compile_check.results import BackendResult, CapturedException

__all__ = ["GradOracle"]

log = logging.getLogger("torch_compile_check")


class GradOracle:
    """Do the two worlds differentiate to the same thing?"""

    name: str = "grad"

    def compare(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Compare the two runs' gradients, and the set of tensors that got one."""
        if not cfg.grad:
            # An oracle that was switched off must not read as an oracle that
            # found nothing, which is the same rule the report applies to an
            # oracle that has not been written yet.
            return [
                self._finding(
                    other.backend,
                    "info",
                    "gradients were not compared: --no-grad switched the backward pass off",
                    {"field": "grad_disabled"},
                )
            ]
        if not eager.ok or not other.ok:
            # A lane that raised in the forward pass never reached a backward.
            # The exception is stage localization's to report, not an oracle's.
            log.debug(
                "grad: nothing to compare, %s raised",
                "eager" if not eager.ok else other.backend,
            )
            return []

        errors = self._compare_errors(eager, other)
        if errors:
            return errors

        findings = self._compare_presence(eager, other)
        findings.extend(self._compare_values(eager, other, cfg))
        return findings

    def _compare_errors(self, eager: BackendResult, other: BackendResult) -> list[Finding]:
        """One lane's backward raised and the other's did not.

        Both raising is agreement, not a divergence: the model is what cannot be
        differentiated, and PLAN.md's contract is about compilation changing an
        answer rather than about the answer being available.
        """
        if (eager.grad_error is None) == (other.grad_error is None):
            if eager.grad_error is not None:
                log.debug(
                    "grad: both lanes' backward raised (%s, %s), which is not a divergence",
                    eager.grad_error.type,
                    other.grad_error.type if other.grad_error else None,
                )
            return []

        raised_in_compiled = other.grad_error is not None
        error = other.grad_error if raised_in_compiled else eager.grad_error
        assert error is not None
        lane, quiet = (other.backend, "eager") if raised_in_compiled else ("eager", other.backend)
        return [
            self._finding(
                other.backend,
                "fail",
                f"the backward pass raised {error.type} under {lane} and completed "
                f"under {quiet}: {_one_line(error.message)}",
                {
                    "field": "grad_error_added" if raised_in_compiled else "grad_error_dropped",
                    "expected": _error_label(eager.grad_error),
                    "got": _error_label(other.grad_error),
                    "traceback": list(error.traceback[:3]),
                },
            )
        ]

    def _compare_presence(self, eager: BackendResult, other: BackendResult) -> list[Finding]:
        """PLAN.md "grad": the set of tensors that received a grad, made equal."""
        expected, got = eager.grad_present, other.grad_present
        mine, theirs = set(expected), set(got)
        findings = [
            self._presence_finding(other.backend, label, expected, got, dropped=True)
            for label in expected
            if label not in theirs
        ]
        findings += [
            self._presence_finding(other.backend, label, expected, got, dropped=False)
            for label in got
            if label not in mine
        ]
        return findings

    def _presence_finding(
        self,
        backend: str,
        label: str,
        expected: tuple[str, ...],
        got: tuple[str, ...],
        *,
        dropped: bool,
    ) -> Finding:
        """One tensor that got a gradient in one lane only."""
        producer, other_lane = ("eager", backend) if dropped else (backend, "eager")
        return self._finding(
            backend,
            "fail",
            f"{producer} produced a gradient for {label} and {other_lane} did not",
            {
                "field": "grad_missing" if dropped else "grad_extra",
                "tensor": label,
                "expected": "a gradient" if dropped else "none",
                "got": "none" if dropped else "a gradient",
                # Counts rather than the two whole sets: a model with sixty
                # parameters would otherwise put sixty labels under every one of
                # sixty findings, and the label above is what a reader acts on.
                "eager_grads": len(expected),
                "compiled_grads": len(got),
            },
        )

    def _compare_values(
        self,
        eager: BackendResult,
        other: BackendResult,
        cfg: OracleConfig,
    ) -> list[Finding]:
        """Every gradient both lanes produced, through the numerics rule."""
        torch = importlib.import_module("torch")
        mine, theirs = eager.grads, other.grads
        findings = []
        for label, expected in mine.items():
            got = theirs.get(label)
            if got is None:
                # Already reported by the presence comparison, in a sentence
                # that says more than a failed value comparison would.
                continue
            mismatch = compare_tensors(torch, expected, got, cfg, tol_factor=cfg.grad_tol_factor)
            if mismatch is None:
                continue
            findings.append(
                self._finding(
                    other.backend,
                    "fail",
                    f"the gradient of {label} differs: {mismatch.message}",
                    {"field": "grad_values", "tensor": label, **mismatch.details},
                )
            )
        return findings

    def _finding(
        self,
        backend: str,
        severity: Severity,
        message: str,
        details: dict[str, Any],
    ) -> Finding:
        """A finding stamped with this oracle's name.

        ``output_index`` is always ``None``. A gradient belongs to an input or a
        parameter rather than to an output, the label in the message is what
        says which, and pointing at an output index would point at the wrong
        tensor.
        """
        return Finding(
            oracle=self.name,
            backend=backend,
            output_index=None,
            severity=severity,
            message=message,
            details=details,
        )


def _error_label(error: CapturedException | None) -> str:
    """How a lane's backward ended, for the detail line."""
    return "completed" if error is None else error.type


def _one_line(message: str) -> str:
    """Collapse a multi-line torch message into one line for the report."""
    return "; ".join(line.strip() for line in message.splitlines() if line.strip())
