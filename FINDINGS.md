# FINDINGS

Ground truth for the five regression-corpus cases in `cases/`. Filled only
from runs actually executed for slice C-1 (see the PR's Evidence Report for
full command/output transcripts).

**Canonical verification run:** `/tmp/baumeister_clean_venv`, a venv created
fresh for this purpose (`pip install --index-url
https://download.pytorch.org/whl/nightly/cpu torch`, nothing else on top),
Python 3.10.12, torch `2.15.0.dev20260901+cpu`, git
`279f79e09c3f3ef458061013bda2d2f483c02cae`, aarch64 Linux, CPU-only,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. This matches the CEO's independent
ashburn run bit for bit. No other architecture or accelerator was run.

**Environment-integrity note, earlier draft used a contaminated venv.** An
earlier pass verified against a pre-existing `/tmp/pruefer_venv` (torch
`2.15.0.dev20260831+cpu`) and got a false GREEN on
`alias_slice_scatter_copyback.py` (issue 195451). Root cause: that venv's
`torch/_inductor/fx_passes/reinplace.py` had a hand-patched
`should_reinplace_scatter()` with an escape-check guard (function name
`_scatter_result_escapes_graph`) that does not exist in the official
pytorch/pytorch source at the exact commit that venv's own
`torch.version.git_version` reported (confirmed by diffing the installed
file against `gh api repos/pytorch/pytorch/contents/...?ref=<commit>`), and
does not exist anywhere in current pytorch/pytorch main either (`gh api
search/code` found zero matches for the guard's function name). Every other
file spot-checked in that venv -- the files touched by the fixes behind the
other four cases -- matched official source exactly; only this one file was
altered. All five cases below were re-run in the clean venv; only case 1's
RED/GREEN result changed (GREEN to RED). Do not reuse `/tmp/pruefer_venv`
for verification without first diffing its inductor/functorch sources
against the official repository at its reported commit.

| Case | Issue | PR | Status upstream | Oracle | First diverging backend | RED on (torch versions run) | GREEN on (torch versions run) |
|---|---|---|---|---|---|---|---|
| `alias_slice_scatter_copyback.py` | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [#195484](https://github.com/pytorch/pytorch/pull/195484) (open) | open, unmerged | alias | not established (only `inductor` run; `--backend aot_eager` available but not separately confirmed against this build) | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`) | not run |
| `alias_noop_view_identity.py` | [#191449](https://github.com/pytorch/pytorch/issues/191449) | [#191844](https://github.com/pytorch/pytorch/pull/191844) (merged 2026-09-02T03:45:57Z, commit `a3586f0018`) | merged 2026-09-02 | alias | `inductor` (fix location: AOTAutograd -- see note below on why those differ) | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`) -- this build predates the 03:45 UTC merge | 2.15.0.dev20260901+cpu, aarch64, CPU (`--backend aot_eager` only) |
| `dtype_int8_matmul_promotion.py` | [#191308](https://github.com/pytorch/pytorch/issues/191308) | none found | open, unfixed | metadata (dtype) | `inductor` | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`) | 2.15.0.dev20260901+cpu, aarch64, CPU (`--backend aot_eager`) |
| `distributions_validation_branch.py` | [#194593](https://github.com/pytorch/pytorch/issues/194593) (sibling [#194596](https://github.com/pytorch/pytorch/issues/194596)) | none found | open, unfixed | graph (fullgraph capturability) | not established (both `inductor` and `aot_eager` raised under `fullgraph=True`) | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, `fullgraph=True`, `inductor` and `--backend aot_eager`) | 2.15.0.dev20260901+cpu, aarch64, CPU (default `fullgraph=False` mode, context probe only, not a separate case file) |
| `numerics_cpu_inductor_miscompile.py` | [#190765](https://github.com/pytorch/pytorch/issues/190765) | [#190966](https://github.com/pytorch/pytorch/pull/190966) ("Fixes #190765"; `ModularIndexing` negativity guard) | fixed by #190966 | numerics | not run (matched eager; no divergence observed) | expected on torch <= 2.13.x (pre-fix; not independently re-run here) | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`, and `--backend aot_eager`) -- confirmed the guard is present in this build's `torch/utils/_sympy/functions.py`, diffed against official source at the same commit: match |

## Notes on surprises (not predicted by the slice brief)

- **195451 IS RED, as the brief expected** -- once run against a clean
  install. See the environment-integrity note above for why an earlier pass
  said GREEN. PR 195484 (the fix) is still open, unmerged.
- **191449: first diverging backend is `inductor`, fix location is
  AOTAutograd -- confirmed, and these are two different things, not a
  contradiction.** Re-run on the 0901 nightly: both MWEs (resize-on-view and
  the two-object identity check) are RED under `inductor`, GREEN under
  `aot_eager`. The fix (PR 191844) lives entirely in AOTAutograd's
  `run_functionalized_fw_and_collect_metadata` -- backend-independent code
  -- yet the symptom is only observable when the backend actually hands
  back one buffer object for two logical outputs; `aot_eager`'s eager
  kernels return distinct view objects regardless of AOTAutograd's
  (mis)classification, so the bug never surfaces there even pre-fix. A
  stage verdict (which backend diverges) says where the divergence becomes
  *visible*, never where the bug *lives* -- the general form of this rule is
  now in PLAN.md's stage-localization section.
- **190765 GREEN is expected, not a surprise.** Fixed upstream by
  [PR #190966](https://github.com/pytorch/pytorch/pull/190966) ("Fixes
  #190765"): a negativity guard on `ModularIndexing`'s term-stripping in
  Inductor's sympy simplification, so the strip is only applied when every
  surviving term is provably nonnegative. Confirmed present in the clean
  venv's `torch/utils/_sympy/functions.py`, and that file was diffed byte-
  for-byte against the official pytorch/pytorch source at the same commit
  (match) -- unlike case 1, this fix is genuinely upstream, not a local
  patch. RED expected on any torch build that predates it (e.g. 2.13.x, per
  the issue's own environment); GREEN expected on any build that contains
  it.
- **194593's contract violation is a `fullgraph=True` graph break
  ("Data-dependent branching", gb0170)**, not literally a `_validate_args`
  divergence as the slice brief speculated. Root cause and the branch line
  (`_kl_binomial_binomial`'s `if (p.total_count < q.total_count).any():`)
  were confirmed directly against the issue text.
