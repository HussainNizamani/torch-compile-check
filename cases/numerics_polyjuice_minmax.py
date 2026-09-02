"""190765 as a compile-check target: a fixed miscompile, expected GREEN.

Issue: https://github.com/pytorch/pytorch/issues/190765 (closed as
completed 2026-07-27). Fix PR:
https://github.com/pytorch/pytorch/pull/190966 ("Fixes #190765") -- a
negativity guard on `ModularIndexing`'s term-stripping in Inductor's sympy
simplification layer. PLAN.md "numerics" names the contract this violated
before the fix: `((X @ A)^T + (X @ B)^T)^T == X @ A + X @ B` by
transpose-distributes-over-addition, double-transpose elimination, and
matmul linearity -- eager agrees these are equal, and a fixed torch's
compiled lanes must agree too.

Twin file, deliberately, on the pattern of the other twins in this
directory. `cases/numerics_cpu_inductor_miscompile.py` is the corpus entry
for the same bug: a standalone RED/GREEN script that FINDINGS.md keys on,
transcribed verbatim in structure from the issue's own "Direct reproducer
for the failing intermediate" section, including its own determinism check
across four repeated compiled calls (`build()`/`check()`, driven by its own
`main()`). This file is the same "strict intermediate" reproducer -- the
member of the equivalence class that miscompiled, not the simpler
`X @ (A + B)` or the more complex 125-cost form that both compiled correctly
even pre-fix -- written to the discovery convention of PLAN.md, a
module-level `fn` and `inputs`, so that
`compile-check cases/numerics_polyjuice_minmax.py` runs it through the tool
itself.

Version marker. `numerics_cpu_inductor_miscompile.py` names the file
"polyjuice" in this twin's own filename for the same reason the corpus entry
does not: torch is polymorphic in the equivalence class it picks, and this
twin exercises the one member of that class the issue found wrong. Measured
on torch `2.14.0+cpu` (git `08187d9`, aarch64, CPU-only, caches disabled),
which postdates #190966: `compile-check cases/numerics_polyjuice_minmax.py`
is exit 0, clean, no findings, matching the standalone script's GREEN on
this build. The issue reports RED on torch 2.13.0 (pre-fix); a torch that
predates #190966 is expected to turn this case exit 1 (numerics finding on
`inductor`) instead. `--dynamic` reproduces the original report's exact
invocation (`backend="inductor", dynamic=True`) and is GREEN here as well;
the plain invocation without `--dynamic` is enough to demonstrate the fix
holds, since the equality this case checks does not depend on it.
"""

from __future__ import annotations

import torch

_VALUES = torch.tensor([-3.0, -1.0, 1.0, 2.0, 4.0], dtype=torch.float32)


def _cyclic_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    n = 1
    for dim in shape:
        n *= dim
    return _VALUES[torch.arange(n) % len(_VALUES)].reshape(shape).clone()


def _postprocess(z: torch.Tensor) -> torch.Tensor:
    z = torch.flatten(z)
    z = torch.repeat_interleave(z, repeats=2)
    return torch.flip(z, dims=[0])


def fn(x: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """The "strict intermediate" reproducer from the issue, unchanged."""
    left = (x @ a).transpose(0, 1)
    right = (x @ b).transpose(0, 1)
    return _postprocess((left + right).transpose(0, 1))


inputs = (
    _cyclic_tensor((8, 6)),
    _cyclic_tensor((6, 8)),
    _cyclic_tensor((6, 8)),
)
