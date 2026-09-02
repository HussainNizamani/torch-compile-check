# FINDINGS

Ground truth for the five regression-corpus cases in `cases/`. Filled only
from runs actually executed for slice C-1 (see the PR's Evidence Report for
full command/output transcripts). All runs below used
`/tmp/pruefer_venv` (Python 3.10.12, torch `2.15.0.dev20260831+cpu`,
git `cbf102a9aec0f6f83466e0584e66d9a96ab613f6`, aarch64 Linux, CPU-only,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`), the only torch build available in
this environment at build time. No other torch version, architecture, or
accelerator was run.

| Case | Issue | PR | Status upstream | Oracle | First diverging backend | RED on (torch versions run) | GREEN on (torch versions run) |
|---|---|---|---|---|---|---|---|
| `alias_slice_scatter_copyback.py` | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [#195484](https://github.com/pytorch/pytorch/pull/195484) (open) | open, unmerged | alias | not run (no backend split observed on this box; not established) | not run | 2.15.0.dev20260831+cpu, aarch64, CPU (default `inductor`, and `--backend aot_eager`) |
| `alias_noop_view_identity.py` | [#191449](https://github.com/pytorch/pytorch/issues/191449) | [#191844](https://github.com/pytorch/pytorch/pull/191844) (merged 2026-09-02T03:45:57Z, commit `a3586f0018`) | merged 2026-09-02 | alias | `inductor` (RED); `aot_eager` did not reproduce on this box | 2.15.0.dev20260831+cpu, aarch64, CPU (default `inductor`) | 2.15.0.dev20260831+cpu, aarch64, CPU (`--backend aot_eager` only) |
| `dtype_int8_matmul_promotion.py` | [#191308](https://github.com/pytorch/pytorch/issues/191308) | none found | open, unfixed | metadata (dtype) | `inductor` | 2.15.0.dev20260831+cpu, aarch64, CPU (default `inductor`) | 2.15.0.dev20260831+cpu, aarch64, CPU (`--backend aot_eager`) |
| `distributions_validation_branch.py` | [#194593](https://github.com/pytorch/pytorch/issues/194593) (sibling [#194596](https://github.com/pytorch/pytorch/issues/194596)) | none found | open, unfixed | graph (fullgraph capturability) | not established (both `inductor` and `aot_eager` raised under `fullgraph=True`) | 2.15.0.dev20260831+cpu, aarch64, CPU (`fullgraph=True`, `inductor` and `--backend aot_eager`) | 2.15.0.dev20260831+cpu, aarch64, CPU (default `fullgraph=False` mode, context probe only, not a separate case file) |
| `numerics_cpu_inductor_miscompile.py` | [#190765](https://github.com/pytorch/pytorch/issues/190765) | none found (issue closed as completed 2026-07-27, no linked fix commit) | closed (completed) | numerics | not run (matched eager on this box; no divergence observed) | not run | 2.15.0.dev20260831+cpu, aarch64, CPU (default `inductor`, and `--backend aot_eager`) |

## Notes on surprises (not predicted by the slice brief)

- **195451 did not reproduce here.** The brief expected RED (fix PR 195484 is
  still open). The exact reproducer from the issue text, run standalone
  outside this repo first, also came back GREEN on this torch build/arch.
  Working theory: `should_reinplace_scatter()`'s reinplacing heuristic does
  not fire for this shape/dtype on aarch64 CPU inductor the way it evidently
  does on the reporter's setup. Not investigated further; flagged as a risk.
- **191449's `aot_eager` backend did not reproduce**, even though the issue's
  own root-cause diagnosis places the bug in AOTAutograd metadata
  classification (backend-independent). Only `inductor` showed the shape
  corruption on this box.
- **190765 is GREEN**, consistent with the issue being closed "completed" on
  2026-07-27, before this torch build's Aug 31 date. No specific fixing
  commit was identified (`gh issue view` returned no linked PR); the closure
  is circumstantial evidence of a fix, not confirmation of one.
- **194593's contract violation is a `fullgraph=True` graph break
  ("Data-dependent branching", gb0170)**, not literally a `_validate_args`
  divergence as the slice brief speculated. Root cause and the branch line
  (`_kl_binomial_binomial`'s `if (p.total_count < q.total_count).any():`)
  were confirmed directly against the issue text.
