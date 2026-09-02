"""CPU Inductor returned wrong, sometimes nondeterministic values for one
member of a family of mathematically equivalent float32 graphs, while the
simplest and the most complex members of the same equivalence class both
compiled correctly.

Issue: https://github.com/pytorch/pytorch/issues/190765 (closed as
completed 2026-07-27; no linked fix PR was found via `gh issue view`, so the
closing commit is not identified here)

Contract violated: numerics. `((X @ A)^T + (X @ B)^T)^T == X @ A + X @ B ==
X @ (A + B)` by transpose-distributes-over-addition, double-transpose
elimination, and matmul linearity -- eager agrees these are equal. Under
`torch.compile(backend="inductor", dynamic=True)`, the "strict intermediate"
graph (the transpose/add/transpose form) returned incorrect and, across
repeated calls with identical inputs, non-reproducible values, while the
"minimum endpoint" (`X @ (A + B)`) and a much more complex 125-cost
equivalent graph both compiled correctly. This is exactly the shape of bug
that endpoint-only differential testing (only testing the simplest and most
complex equivalent forms) would miss.

Oracle: numerics (equality against eager, plus a same-input determinism
check across repeated compiled calls).

First diverging backend: not established by the issue for `aot_eager`; the
report only exercises `backend="inductor", dynamic=True`. Not independently
tested against `aot_eager` here.

Known-bad torch versions: the issue reports RED on torch 2.13.0, macOS
Apple Silicon (arm64). Re-run here on torch 2.15.0.dev20260831+cpu (git
cbf102a9aec0f6f83466e0584e66d9a96ab613f6), aarch64 Linux, CPU: GREEN --
compiled output matched eager and was deterministic across 4 repeated
calls, consistent with the issue being closed as "completed" on
2026-07-27 (before this torch build's date), i.e. this looks fixed upstream
by the time of this nightly. No specific fixing commit/PR was identified
via `gh issue view 190765 --repo pytorch/pytorch`; the issue body alone was
used to transcribe this repro, verbatim in structure, exactly as the issue's
"Direct reproducer for the failing intermediate" section.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import sys

import torch

_VALUES = torch.tensor([-3.0, -1.0, 1.0, 2.0, 4.0], dtype=torch.float32)


def _cyclic_tensor(shape):
    n = 1
    for dim in shape:
        n *= dim
    return _VALUES[torch.arange(n) % len(_VALUES)].reshape(shape).clone()


def _postprocess(z):
    z = torch.flatten(z)
    z = torch.repeat_interleave(z, repeats=2)
    return torch.flip(z, dims=[0])


def _intermediate_graph(x, a, b):
    left = (x @ a).transpose(0, 1)
    right = (x @ b).transpose(0, 1)
    return _postprocess((left + right).transpose(0, 1))


def build():
    x = _cyclic_tensor((8, 6))
    a = _cyclic_tensor((6, 8))
    b = _cyclic_tensor((6, 8))
    return _intermediate_graph, (x, a, b)


def _git_hash():
    # The provenance the RED/GREEN line needs is torch's own build commit,
    # not this repo's -- a case can be RED/GREEN purely because of which
    # torch checkout it ran against, and that is what the parity table in
    # FINDINGS.md keys on.
    git_version = getattr(torch.version, "git_version", None)
    if not git_version:
        return "unknown"
    return git_version[:7]


def check(eager_out, compiled_outs, inputs):
    """`compiled_outs` is a list of outputs from repeated compiled calls
    with identical inputs (the issue's determinism check needs more than
    one run).
    """
    mismatches = [i for i, out in enumerate(compiled_outs) if not torch.equal(out, eager_out)]
    nondeterministic = any(
        not torch.equal(compiled_outs[0], out) for out in compiled_outs[1:]
    )

    if mismatches:
        return True, (
            f"compiled output diverged from eager on run(s) {mismatches} of "
            f"{len(compiled_outs)}; eager[:8]={eager_out[:8].tolist()} "
            f"compiled[:8] on first mismatch={compiled_outs[mismatches[0]][:8].tolist()}"
            + (" (also nondeterministic across repeated runs)" if nondeterministic else "")
        )
    if nondeterministic:
        return True, (
            f"compiled output matched eager on run 0 but was nondeterministic "
            f"across {len(compiled_outs)} repeated calls with identical inputs"
        )
    return False, f"compiled output matched eager and was deterministic across {len(compiled_outs)} runs"


def _report(case, backend_name, eager_out, compiled_outs):
    is_red, message = check(eager_out, compiled_outs, None)
    status = "RED" if is_red else "GREEN"
    print(
        f"{status} {case} torch={torch.__version__} git={_git_hash()} "
        f"arch={platform.machine()} backend={backend_name} :: {message}"
    )
    return is_red


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="extra backend to also check, e.g. aot_eager")
    args = parser.parse_args()

    case = "numerics_cpu_inductor_miscompile"
    fn, example_inputs = build()

    try:
        eager_out = fn(*example_inputs)
        compiled_fn = torch.compile(fn, backend="inductor", dynamic=True)
        compiled_outs = [compiled_fn(*example_inputs) for _ in range(4)]
    except Exception as exc:  # noqa: BLE001 - a crash in the case itself, not a RED finding
        print(f"CRASH {case} :: {type(exc).__name__}: {exc}")
        sys.exit(2)

    is_red = _report(case, "inductor", eager_out, compiled_outs)
    exit_code = 1 if is_red else 0

    if args.backend:
        extra_fn = torch.compile(fn, backend=args.backend, dynamic=True)
        extra_outs = [extra_fn(*example_inputs) for _ in range(4)]
        _report(case, args.backend, eager_out, extra_outs)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
