# FINDINGS

Ground truth for the seven regression-corpus cases in `cases/`: the five
filled from runs actually executed for slice C-1 (see that PR's Evidence
Report for full command/output transcripts), plus two reviewer-reported
siblings of `alias_slice_scatter_copyback.py` added 2026-09-03 (their own
runs are below, under "Reviewer-reported siblings of 195451").

**Canonical verification run:** `/tmp/baumeister_clean_venv`, a venv created
fresh for this purpose (`pip install --index-url
https://download.pytorch.org/whl/nightly/cpu torch`, nothing else on top),
Python 3.10.12, torch `2.15.0.dev20260901+cpu`, git
`279f79e09c3f3ef458061013bda2d2f483c02cae`, aarch64 Linux, CPU-only,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. This matches the CEO's independent
ashburn run bit for bit. No other architecture or accelerator was run.

**Environment-integrity note, resolved: the "contamination" was our own
fix-in-progress for 195451, not a rogue patch.** An earlier pass verified
against a pre-existing `/tmp/pruefer_venv` (torch `2.15.0.dev20260831+cpu`)
and got a GREEN on `alias_slice_scatter_copyback.py` where every unpatched
nightly checked (0824, 0831, 0901) is RED. That venv's
`torch/_inductor/fx_passes/reinplace.py` has a hand-applied
`should_reinplace_scatter()` escape-check guard (function
`_scatter_result_escapes_graph`) -- confirmed absent from official
pytorch/pytorch source at the exact commit that venv's own
`torch.version.git_version` reports, and absent from current
pytorch/pytorch main (`gh api search/code` found zero matches), by diffing
the installed file against `gh api
repos/pytorch/pytorch/contents/...?ref=<commit>`. This is the same bug, the
same author, and the same day as the open fix
[PR #195484](https://github.com/pytorch/pytorch/pull/195484) -- but it is
NOT textually identical to that PR's current diff (different function name,
different control-flow placement inside `should_reinplace_scatter`,
recursive vs. iterative traversal); most likely an earlier local iteration
of the same fix, applied directly to that venv's site-packages rather than
landed as a commit. So the GREEN was real: reinplacing genuinely does not
fire with that guard in the tree. It reflects "the fix, in some form, is
present" rather than "the bug is absent upstream." Every other file spot-
checked in that venv -- the files touched by the fixes behind the other
four cases -- matched official source exactly; only this one file carried a
local patch. All five cases below were re-run in a clean venv (guard not
present) to get the upstream-accurate baseline; only case 1's result
changed there (GREEN to RED, matching every unpatched nightly). Verification
venvs should be created fresh per mission and removed after -- a patched
venv should not outlive the mission that patched it.

| Case | Issue | PR | Status upstream | Oracle | First diverging backend | Signal type | RED on (torch versions run) | GREEN on (torch versions run) |
|---|---|---|---|---|---|---|---|---|
| `alias_slice_scatter_copyback.py` | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [#195484](https://github.com/pytorch/pytorch/pull/195484) (open, unmerged; an earlier local iteration of this fix is what caused the GREEN discussed in the note above) | open, unmerged upstream | alias | not established (only `inductor` run; `--backend aot_eager` available but not separately confirmed against this build) | inductor-only miscompile | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`), plus 0824 and 0831 nightlies per the CEO's and Baumeister's cross-checks | not run on any unpatched build; GREEN only observed with the local fix-iteration guard in the tree (see note) |
| `alias_noop_view_identity.py` | [#191449](https://github.com/pytorch/pytorch/issues/191449) | [#191844](https://github.com/pytorch/pytorch/pull/191844) (merged 2026-09-02T03:45:57Z, commit `a3586f0018`) | merged 2026-09-02 | alias | `inductor` (fix location: AOTAutograd -- see note below on why those differ) | inductor-only miscompile | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`) -- this build predates the 03:45 UTC merge | 2.15.0.dev20260901+cpu, aarch64, CPU (`--backend aot_eager` only) |
| `dtype_int8_matmul_promotion.py` | [#191308](https://github.com/pytorch/pytorch/issues/191308) | none found | open, unfixed | metadata (dtype) | `inductor` | inductor-only miscompile | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`) | 2.15.0.dev20260901+cpu, aarch64, CPU (`--backend aot_eager`) |
| `distributions_validation_branch.py` | [#194593](https://github.com/pytorch/pytorch/issues/194593) (sibling [#194596](https://github.com/pytorch/pytorch/issues/194596)) | none found | open, unfixed | graph (fullgraph capturability) | not established (both `inductor` and `aot_eager` raised under `fullgraph=True`) | fullgraph capturability break, backend-independent | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, `fullgraph=True`, `inductor` and `--backend aot_eager`) | 2.15.0.dev20260901+cpu, aarch64, CPU (default `fullgraph=False` mode, context probe only, not a separate case file) |
| `numerics_cpu_inductor_miscompile.py` | [#190765](https://github.com/pytorch/pytorch/issues/190765) | [#190966](https://github.com/pytorch/pytorch/pull/190966) ("Fixes #190765"; `ModularIndexing` negativity guard) | fixed by #190966 | numerics | not run (matched eager; no divergence observed) | none observed (fixed upstream) | expected on torch <= 2.13.x (pre-fix; not independently re-run here) | 2.15.0.dev20260901+cpu, aarch64, CPU (clean venv, default `inductor`, and `--backend aot_eager`) -- confirmed the guard is present in this build's `torch/utils/_sympy/functions.py`, diffed against official source at the same commit: match |

### Discovery-convention twins (C-2)

The five rows above are the standalone RED/GREEN scripts. Each has a
discovery-convention twin (`cases/README.md`) that runs through
`compile-check` itself; these rows are those runs, real (`tests/test_corpus_twins.py`
re-runs every pair on every test run and asserts the two agree), on torch
`2.14.0+cpu` (git `08187d9e0fba026dc8217405802ab5381dc88d90`), aarch64, CPU,
in a venv created fresh for this slice.

| Twin | Standalone case it mirrors | compile-check invocation | Result | Signal type |
|---|---|---|---|---|
| `alias_noop_view.py` | `alias_noop_view_identity.py` | `compile-check cases/alias_noop_view.py` | exit 1, alias finding (`identity_added`), "inductor returned one object for output[0] and output[1] and eager returned distinct objects that share a storage", first diverges at `inductor` | inductor-only miscompile |
| `distributions_binomial_kl.py` | `distributions_validation_branch.py` | `compile-check cases/distributions_binomial_kl.py` (default) | exit 0, clean | n/a (default mode does not exercise the gap) |
| `distributions_binomial_kl.py` | `distributions_validation_branch.py` | `compile-check cases/distributions_binomial_kl.py --fullgraph` | exit 1 (a raised lane, not a `--fail-on` finding), first diverges at `aot_eager` (checked before `inductor`; both raise `Unsupported: Data-dependent branching` identically) | fullgraph capturability break, backend-independent |
| `numerics_polyjuice_minmax.py` | `numerics_cpu_inductor_miscompile.py` | `compile-check cases/numerics_polyjuice_minmax.py --dynamic` | exit 0, clean (matches the standalone's GREEN on this build) | none observed (fixed upstream) |
| `dtype_promotion.py` | `dtype_int8_matmul_promotion.py` | `compile-check cases/dtype_promotion.py` | exit 1, metadata finding, first diverges at `inductor` | inductor-only miscompile |
| `alias_copyback.py` | `alias_slice_scatter_copyback.py` | `compile-check cases/alias_copyback.py` | exit 1, alias finding, first diverges at `inductor` | inductor-only miscompile |

### Reviewer-reported siblings of 195451 (2026-09-03)

Two more cases, not part of the C-1 slice above: a reviewer on
[PR #195484](https://github.com/pytorch/pytorch/pull/195484) reported these
two additional shapes of the alias bug on 2026-09-03 -- one where the scatter
target is a view of the graph input rather than the input itself, one where
a chained `diagonal_scatter`/`index_put` is reinplaced. Verified 2026-09-03
in this repository's own venv, torch `2.14.0+cpu` (git
`08187d9e0fba026dc8217405802ab5381dc88d90`), aarch64, CPU-only,
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. Neither file has a separate
discovery-convention twin (`cases/README.md`): each exposes the module-level
`fn` and `inputs` PLAN.md's discovery convention looks for in the same file
as its own standalone `build()`/`check()`/`main()` script, so the file below
is both the standalone RED/GREEN script and the `compile-check` run beside
it.

| Case | Issue | PR | Status upstream | Oracle | First diverging backend | Signal type | RED on (torch versions run) | GREEN on (torch versions run) |
|---|---|---|---|---|---|---|---|---|
| `alias_view_slice_scatter_copyback.py` | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [#195484](https://github.com/pytorch/pytorch/pull/195484) (open, unmerged; its diff does not claim to cover this view shape) | open, unmerged upstream | alias | `inductor` (not separately confirmed against `aot_eager` on this build) | inductor-only miscompile | 2.14.0+cpu, aarch64, CPU (this repository's venv, default `inductor`) | not run |
| `alias_diagonal_scatter_index_put_chain.py` | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [#195484](https://github.com/pytorch/pytorch/pull/195484) (open, unmerged; its diff does not claim to cover this chained shape) | open, unmerged upstream | alias | `inductor` (not separately confirmed against `aot_eager` on this build) | inductor-only miscompile | 2.14.0+cpu, aarch64, CPU (this repository's venv, default `inductor`) | not run |

Standalone script RED lines, verbatim from a live run on this box:

```
RED alias_view_slice_scatter_copyback torch=2.14.0+cpu git=08187d9 arch=aarch64 backend=inductor :: compiled output aliases input (data_ptr equal) and mutating the output corrupted the input: before=[[100.0, 2.0, 3.0], [200.0, 5.0, 6.0], [300.0, 8.0, 9.0], [400.0, 11.0, 12.0]] after=[[200.0, 102.0, 103.0], [300.0, 105.0, 106.0], [400.0, 108.0, 109.0], [500.0, 111.0, 112.0]]
RED alias_diagonal_scatter_index_put_chain torch=2.14.0+cpu git=08187d9 arch=aarch64 backend=inductor :: compiled output aliases input (data_ptr equal) and mutating the output corrupted the input: before=[[100.0, 200.0], [3.0, 20.0]] after=[[200.0, 300.0], [103.0, 120.0]]
```

`compile-check` run through the tool itself, same file, verbatim finding message:

| File | compile-check invocation | Result | Signal type |
|---|---|---|---|
| `alias_view_slice_scatter_copyback.py` | `compile-check cases/alias_view_slice_scatter_copyback.py` | exit 1, alias finding (`alias_added`), "inductor output[1] aliases input[0] (same storage, overlapping bytes) and the eager pair does not", first diverges at `inductor` | inductor-only miscompile |
| `alias_diagonal_scatter_index_put_chain.py` | `compile-check cases/alias_diagonal_scatter_index_put_chain.py` | exit 1, alias finding (`identity_added`), "inductor returned input[0] itself as output[0] and eager returned a distinct object", first diverges at `inductor` | inductor-only miscompile |

## Notes on surprises (not predicted by the slice brief)

- **195451 IS RED, as the brief expected, on every unpatched nightly
  checked (0824, 0831, 0901).** The earlier GREEN pass wasn't a false
  negative against an absent bug -- it was a true positive against a
  present, not-yet-landed fix: an earlier local iteration of PR 195484 had
  been applied directly to that venv's `reinplace.py`. See the
  environment-integrity note above. PR 195484 itself is still open,
  unmerged upstream.
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
