"""CPU Inductor silently promotes the output dtype of some int8 batched
matmuls to int64; eager, the `eager` backend, and `aot_eager` all keep
int8.

Issue: https://github.com/pytorch/pytorch/issues/191308

Contract violated: dtype. `compile(f)(x)` must produce the same dtype as
`f(x)`, not just the same numeric values. The reporter's ablation
(native eager / backend="eager" / backend="aot_eager" all return int8;
backend="inductor" returns int64, with both dynamic=True and dynamic=False)
pins the divergence to Inductor's codegen for this shape family, not to
functionalization or Dynamo's dispatch.

Idiom used here is the exact reproducer from the issue (int8 3-D batched
matmul, shape (1,1,2) @ (1,2,2)); confirmed against the issue text via
`gh issue view 191308`, not independently re-derived, so it is not marked
"to be confirmed."

Oracle: metadata (dtype).

First diverging backend: `inductor`. `aot_eager` matches eager.

Known-bad torch versions: RED reproduced here on torch
2.15.0.dev20260831+cpu (git cbf102a9aec0f6f83466e0584e66d9a96ab613f6),
aarch64, CPU. The issue also reports RED on 2.13.0 and on a 2.14.0
nightly (x86_64, per the reporter); not independently re-run on those
versions here. No fix is known to have merged as of 2026-09-02.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import subprocess
import sys

import torch


def build():
    def model(a, b):
        return torch.matmul(a, b)

    a = torch.ones((1, 1, 2), dtype=torch.int8)
    b = torch.ones((1, 2, 2), dtype=torch.int8)
    return model, (a, b)


def _git_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def check(eager_out, compiled_out, inputs):
    if compiled_out.dtype != eager_out.dtype:
        return True, (
            f"compiled dtype {compiled_out.dtype} != eager dtype "
            f"{eager_out.dtype} (values: eager={eager_out.tolist()} "
            f"compiled={compiled_out.tolist()})"
        )
    return False, f"compiled dtype matches eager: {compiled_out.dtype}"


def _report(case, backend_name, eager_out, compiled_out):
    is_red, message = check(eager_out, compiled_out, None)
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

    case = "dtype_int8_matmul_promotion"
    fn, example_inputs = build()

    try:
        eager_out = fn(*example_inputs)
        compiled_out = torch.compile(fn, backend="inductor", fullgraph=True, dynamic=False)(*example_inputs)
    except Exception as exc:  # noqa: BLE001 - a crash in the case itself, not a RED finding
        print(f"CRASH {case} :: {type(exc).__name__}: {exc}")
        sys.exit(2)

    is_red = _report(case, "inductor", eager_out, compiled_out)
    exit_code = 1 if is_red else 0

    if args.backend:
        extra_out = torch.compile(fn, backend=args.backend, fullgraph=True, dynamic=False)(*example_inputs)
        _report(case, args.backend, eager_out, extra_out)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
