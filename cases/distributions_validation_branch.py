"""`torch.distributions.kl_divergence` between two `Binomial` distributions
cannot be captured under `torch.compile(fullgraph=True)` when `total_count`
is a tensor, because `_kl_binomial_binomial` branches on a tensor condition
(`if (p.total_count < q.total_count).any():`).

Issue: https://github.com/pytorch/pytorch/issues/194593 (sibling:
https://github.com/pytorch/pytorch/issues/194596, same class, Wishart.mean)

This is a fullgraph compile-support gap, not a numerical mismatch: the
default `fullgraph=False` mode graph-breaks at the branch and falls back to
eager for that piece, so it returns the correct answer. Confirmed against
the issue text and reproduced directly (not the speculative "validation
branch" framing from the case's own planning notes -- the concrete fault is
Dynamo's "Data-dependent branching" (gb0170) graph break on
`(p.total_count < q.total_count).any()` inside `_kl_binomial_binomial`, not
a `_validate_args` divergence).

Oracle: graph (fullgraph capturability) -- and, incidentally, numerics if a
caller compares the fullgraph=True path's crash to eager's success.

First diverging backend: both `aot_eager` and `inductor` fail identically
under fullgraph=True per the issue report, since the failure is in Dynamo's
tracing (symbolic control flow), upstream of backend codegen; this file
only exercises the default `inductor` backend.

Known-bad torch versions: RED (fullgraph=True raises where eager succeeds)
reproduced here on torch 2.15.0.dev20260831+cpu (git
cbf102a9aec0f6f83466e0584e66d9a96ab613f6), aarch64, CPU. The issue also
reports the same failure on nightly 2.15.0.dev20260821+cu130 (x86_64,
CUDA-capable box, per the reporter); not independently re-run on that build
here. As of 2026-09-02 no fix has merged; the issue's only comment records a
maintainer picking it up with a "keep the eager check, guard it for
compile" direction, unmerged.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import sys

import torch
from torch.distributions import Binomial, kl_divergence


def build():
    def fn(p_count, p_prob, q_count, q_prob):
        return kl_divergence(
            Binomial(total_count=p_count, probs=p_prob),
            Binomial(total_count=q_count, probs=q_prob),
        )

    example_inputs = (
        torch.tensor([5.0]),
        torch.tensor([0.2]),
        torch.tensor([3.0]),
        torch.tensor([0.4]),
    )
    return fn, example_inputs


def _git_hash():
    # The provenance the RED/GREEN line needs is torch's own build commit,
    # not this repo's -- a case can be RED/GREEN purely because of which
    # torch checkout it ran against, and that is what the parity table in
    # FINDINGS.md keys on.
    git_version = getattr(torch.version, "git_version", None)
    if not git_version:
        return "unknown"
    return git_version[:7]


def check(eager_out, compiled_out, inputs):
    """`compiled_out` here is a (raised: bool, value_or_exception) pair from
    the fullgraph=True attempt, since the RED condition is "raises where
    eager succeeds", not a value mismatch.
    """
    raised, payload = compiled_out
    if raised:
        return True, (
            f"fullgraph=True raised where eager succeeded (eager={eager_out.tolist()}): "
            f"{type(payload).__name__}: {str(payload)[:160]}"
        )
    if not torch.equal(payload, eager_out):
        return True, f"fullgraph=True returned a value but it diverges from eager: eager={eager_out.tolist()} compiled={payload.tolist()}"
    return False, f"fullgraph=True captured the graph and matched eager: {payload.tolist()}"


def _try_fullgraph(fn, example_inputs, backend_name):
    try:
        value = torch.compile(fn, backend=backend_name, fullgraph=True)(*example_inputs)
        return (False, value)
    except Exception as exc:  # noqa: BLE001 - intentionally broad, this is the thing under test
        return (True, exc)


def _report(case, backend_name, eager_out, compiled_result, extra=""):
    is_red, message = check(eager_out, compiled_result, None)
    status = "RED" if is_red else "GREEN"
    print(
        f"{status} {case} torch={torch.__version__} git={_git_hash()} "
        f"arch={platform.machine()} backend={backend_name} :: {message}{extra}"
    )
    return is_red


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="extra backend to also check, e.g. aot_eager")
    args = parser.parse_args()

    case = "distributions_validation_branch"
    fn, example_inputs = build()

    eager_out = fn(*example_inputs)
    compiled_result = _try_fullgraph(fn, example_inputs, "inductor")

    # Context probe: default (fullgraph=False) mode is expected to graph-break
    # and still match eager, showing the bug is specific to fullgraph=True.
    try:
        no_fullgraph_out = torch.compile(fn, backend="inductor")(*example_inputs)
        no_fullgraph_matches = torch.equal(no_fullgraph_out, eager_out)
    except Exception:  # noqa: BLE001
        no_fullgraph_matches = False

    is_red = _report(
        case, "inductor", eager_out, compiled_result,
        extra=f" (fullgraph=False matches eager: {no_fullgraph_matches})",
    )
    exit_code = 1 if is_red else 0

    if args.backend:
        extra_result = _try_fullgraph(fn, example_inputs, args.backend)
        _report(case, args.backend, eager_out, extra_result)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
