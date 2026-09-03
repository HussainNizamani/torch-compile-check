# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **corpus-195451-siblings**: two more regression-corpus cases, reviewer-
  reported on PR #195484 and reported 2026-09-03: `cases/alias_view_slice_scatter_copyback.py`
  (the scatter target is a view of the graph input rather than the input
  itself) and `cases/alias_diagonal_scatter_index_put_chain.py` (a chained
  `diagonal_scatter` then `index_put` before the copy-back). Both reproduce
  RED on inductor, GREEN-clean on `aot_eager`, verified on torch 2.14.0+cpu;
  neither has a separate discovery-convention twin -- each file exposes the
  module-level `fn` and `inputs` the discovery convention looks for, in
  addition to the standalone `build()`/`check()`/`main()` shape, so it is
  its own twin. `cases/markers.py` gets a `CaseMarker` for each, `FINDINGS.md`
  a ground-truth entry for each, and the corpus is seven cases as of this
  entry. README's regression-corpus table extends the #195451 row's
  description to name all three shapes rather than adding new rows for the
  same issue.
- **nightly-hunt-2026-09-03**: documentation of a CUDA-nightly regression
  hunt on torch `2.15.0.dev20260829+cu126` with no new upstream bugs found.
  New "Nightly hunt (2026-09-03)" section in `docs/cross-arch.md` records
  the lanes (baseline, fp16/bf16 numerics, training step fp32, view/in-place
  alias oracle, dynamic shapes probe) and outcomes. `docs/validation.md`
  gains a note that fp16 and bf16 are not validated precisions in v0.1.
  `PLAN.md` v0.2 outlook expanded with precision handling improvements and
  Ampere GPU requirements.

### Changed

- **path-hygiene**: `target.file` -- in the JSON report, the Markdown draft,
  the emitted regression test's header comment, and the reduced repro -- is
  now the path relative to the working directory when the target lives under
  it, or exactly what was given on the command line otherwise, instead of
  always the fully resolved path `discover.py` uses to import the file.
  `resolve()` is still what discovery uses to find and read the target; only
  what a report shows changed, so `schema_version` stays 2 (value format,
  not shape).
- README rewritten in user-facing voice; the regression corpus table gains
  a line noting every bug in it was found by hand before this tool
  existed; prior-art row for `torch.library.opcheck`.

### Fixed

- The committed per-target JSON results under `validation/results/per-target/`
  no longer carry a contributor's absolute home-directory path in
  `target.file` (24 files: 6 from the aarch64 baseline, 18 from the x86 RC
  round) -- a discovery bug always resolved the target path before recording
  it, and this repo went public with it still in the committed data. Every
  value is now `validation/targets/<name>.py`, matching what
  `torch-compile-check` is run with from the repo root; nothing else in
  those files changed, and the cross-architecture parity comparison in
  `docs/cross-arch.md` never read that field (path-hygiene finding, x86 RC
  round).

## [0.1.0] - 2026-09-03

This project was named `compile-check` through M4-3; the `### Changed` entry
below is the M4-5 rename to `torch-compile-check`. Everything else in this
section is written under whichever name was current at the time it landed,
which is `compile-check` for every slice before M4-5.

Everything below is what ships as `0.1.0` — PLAN.md "M4"'s definition of
done for this milestone. The tag, the PyPI upload, the public flip, and the
Marketplace listing are a maintainer's own release step, run from
`docs/release.md`, and this file does not claim any of them has happened
yet. Every entry is grouped by the slice it landed in (PLAN.md's milestone
schedule for the `M*` slices, the office's own numbering for the rest), in
merge order.

### Added

- **M0-1** (PR #1): package scaffold. `pyproject.toml` (hatchling, src
  layout), the `compile_check` package with typed stubs for every module in
  PLAN.md "Package layout", each raising `NotImplementedError` naming the
  milestone it lands in. `compile_check.env`: `collect_environment()` for the
  report's environment block (torch version and git hash, Python version,
  platform, machine architecture, CPU feature summary, CUDA availability,
  inductor cache-disable state), and `probe_apis()` for the torch symbols in
  PLAN.md "Verified API surface". CLI: the full v1 flag surface is parsed
  from M0 so it is fixed and testable; `--version` and `--probe` work, every
  other invocation exits 2. Tooling: ruff lint and format, mypy strict over
  `src/`, pytest, pre-commit, a `Makefile`, and a GitHub Actions matrix over
  Python 3.10 to 3.13 and torch stable and nightly on CPU.
- **M1-1** (PR #2): discovery and the multi-backend runner, the two pieces
  every oracle sits on top of. `compile_check.discover`: `load_target`
  resolves a filesystem path or dotted module against PLAN.md's discovery
  convention (`model` then `fn`; `inputs` then `get_inputs()`), with
  `--entry`/`--inputs` overrides and a `DiscoveryError` naming what it
  looked for; nothing in the module imports torch. `compile_check.runner`:
  `run_all` runs the ablation ladder per backend -- reseed torch/Python/numpy,
  deep-clone the inputs from the original examples (dtype, stride, device,
  `requires_grad` preserved), reset the compiler, force
  `force_disable_caches`, call eager or `torch.compile(..., backend=...)`,
  flatten outputs with pytree keeping both detached clones and live
  references (for the alias oracle to come), time both calls, capture an
  exception rather than propagate it, and run one backward on a fixed scalar
  reduction when anything requires grad. `compile_check.results`:
  `BackendResult`, `RunSet`, `CapturedException`, torch-free. A hidden
  `--run-only` runs discovery and the runner and prints the per-backend rows,
  so the runner is exercisable end to end before M1-3 wires the oracles in.
  Housekeeping: `numpy` joins the `dev` extra; the CI stable lane installs
  torch from the PyTorch CPU index rather than PyPI.
- **M1-2** (PR #4): the numerics and metadata oracles, the two a run can
  actually fail on, plus the fp64 reference PLAN.md's blind-spot section
  promises. `oracles/base.py`: the `Finding` / `OracleConfig` / `Oracle`
  vocabulary, a pytree-spec-or-leaf-count difference as a structural finding,
  and a raised lane is never compared (stage localization already reads the
  exception). `oracles/numerics.py`: pairwise `torch.testing.assert_close`,
  tolerances from `torch.testing._comparison.default_tolerances` or a
  PLAN.md-sourced fallback table kept in sync by its own test, `--rtol`/
  `--atol` overrides, NaN/inf position parity as separate findings.
  `run_all(..., fp64=True)` adds the `eager_fp64` pseudo-backend (a deep-
  copied, `.double()`-d module with widened floating inputs), recorded on
  `RunSet.fp64` and never among the backends, so nothing on the ablation
  ladder can mistake it for a diverging lane; the numerics oracle reports at
  `info` when eager itself is off the fp64 reference. `oracles/metadata.py`:
  dtype, shape, stride, `requires_grad`, device type, `is_contiguous`,
  layout, every difference `fail` except a contiguous-to-contiguous stride
  change, which is `warn`. `cases/dtype_promotion.py`: the 191308 int8
  matmul-promotion bug as a discovery-convention target, so
  `compile-check cases/dtype_promotion.py` catches it through the tool
  itself, and `--run-only` starts printing findings.
- **C-1** (PR #3): the regression corpus, five tiny models, one per known
  `torch.compile` bug class -- `alias_slice_scatter_copyback.py` (195451),
  `alias_noop_view_identity.py` (191449), `dtype_int8_matmul_promotion.py`
  (191308), `distributions_validation_branch.py` (194593/194596), and
  `numerics_cpu_inductor_miscompile.py` (190765, fixed by #190966). Each is a
  standalone script that prints one `RED`/`GREEN` line with the torch
  version and build hash it measured and exits non-zero when the bug
  reproduces; `FINDINGS.md` is the ground-truth table they fill in,
  independently reproduced four times (Baumeister's build, Prüfer's and
  Konrad's fresh-venv verifications, and the CEO's Fable pass) on the same
  canonical nightly.
- **M1-3** (PR #6): the ablation ladder becomes a verdict, the verdict
  becomes a report, and the CLI's main path is wired up --
  `compile-check path/to/model.py` now runs. `localize.py`:
  `localize(runset, findings) -> StageVerdict` applies four rules in order --
  no eager lane is `no reference`; eager raising is `model`; otherwise the
  first ladder lane that raised or drew a `fail` finding names the stage
  (`capture/AOTAutograd/decomposition`, `decomposition/partitioner`,
  `inductor lowering/codegen`); nothing diverging is `clean` -- worded
  "first diverges at &lt;backend&gt;", never "the bug is in", since where a
  divergence appears is not always where the fix belongs. `report/terminal.py`:
  plain ANSI, no dependency, environment block, per-backend table, an
  oracle-by-backend table distinguishing `pass` from `not yet` from `-`,
  findings grouped by oracle under `--max-findings`, the stage verdict, a
  next-step hint. The CLI main path: run, compare, localize, report, exit
  0/1/2; `--fail-on` selects which oracle categories turn a finding into
  exit 1 and never which oracles run; a compiled backend that raised while
  eager did not is always exit 1.
- **A-1** (PR #5): the composite GitHub Action skeleton --
  `action/action.yml` (inputs mirroring the CLI one to one: `targets`,
  `backends`, `fail-on`, `torch`, `python-version`, `baseline`, `budget`,
  `json-out`, `extra-args`, `ref`, `source`, `allow-unimplemented`; outputs
  `exit-code`, `json-path`), `docs/action.md`, and a self-test workflow that
  exercises `--version`/`--probe`/`--run-only` today and turns into a real
  check with no changes needed once M1-3's main path lands. The `source`
  input (`auto`/`local`/`git`) installs from the checked-out source inside
  this repo and falls back to `git+https://...@ref` for external consumers,
  since pip cannot clone a private repo without credentials -- added after
  the self-test's first run failed on exactly that.
- **M2-1** (PR #7): the alias and mutation oracle, the oracle for the 195451
  and 191449 bug classes. Builds a relation over every output-output and
  output-input pair -- object identity, untyped-storage identity, byte-range
  overlap (storage identity alone is not an alias: two disjoint views of one
  buffer share a data pointer) -- plus the input mutation set (values over
  the bytes, and layout via `resize_`/`as_strided_`), and requires the
  compiled relation to equal the eager one entry for entry. Storage sharing
  without overlap and `torch._debug_has_internal_overlap` are recorded as
  context and never fail a run. `cases/alias_copyback.py`: 195451 written to
  the discovery convention, plus the live compile-only-failure exit-code
  test.
- **C-2** (PR #8): corpus discovery twins for the three cases that did not
  have one yet -- `alias_noop_view.py` (191449, the plain identity-collapse
  shape), `distributions_binomial_kl.py` (194593, honest about the two
  answers `--fullgraph` changes), `numerics_polyjuice_minmax.py` (190765,
  fixed upstream, expected clean). `tests/test_corpus_twins.py` runs every
  standalone script and its twin together and asserts they agree on exit
  code and stage line, so the two files cannot drift apart silently.
  `FINDINGS.md` gets a "Signal type" column (`inductor-only miscompile` vs.
  `fullgraph capturability break, backend-independent`) and a table of real
  twin runs.
- **M2-2** (PR #9): the gradients oracle, fourth of the five. Compares, after
  the one backward the runner already ran: the presence set (which inputs
  and parameters received a `.grad`, read off the gradients rather than
  `named_parameters()` so a frozen parameter is not a false mismatch), the
  values (through `numerics.compare_tensors`, lifted out of the numerics
  oracle so a gradient and an output share tolerances), and a backward that
  raised in one lane and not the other (a fail on its own). `requires_grad`
  on the outputs is deliberately not checked here -- that is the metadata
  oracle's field.
- **C-3** (PR #10): the real-world validation runner. Six offline,
  discovery-convention targets under `validation/targets/` -- torchvision
  `resnet18`, `mobilenet_v3_small`, `efficientnet_b0`, a reduced
  `VisionTransformer` (32/8/2/2/32/64 against `vit_b_16`'s
  224/16/12/12/768/3072, so it compiles in seconds on CPU), a tiny random-
  init HF BERT (skipped cleanly, not a tool error, when `transformers` is
  not installed), and a training step (forward, cross-entropy loss, and
  backward under one `nn.Module` target with `requires_grad` inputs).
  `validation/run.py` runs every target through the real CLI, writes
  `validation/results/<date>-<arch>-<torch>.json`, and regenerates
  `docs/validation.md`. `torchvision` and `transformers` are validation-only
  extras, not `pyproject.toml` dependencies. Both stable and nightly torch:
  6 of 6 targets clean, no findings tuned away.
- **M2-3** (PR #12): corpus markers and the summary that makes the regression
  corpus useful in CI. `--seed` is applied before the target module is
  imported, and again before every backend -- a target that builds its model at
  module scope (the shape `model = torchvision.models.resnet18(weights=None)`
  has) draws its weights during discovery, so a seed applied afterwards never
  reached them and two runs of the same command compared two different models.
  `--grad-tol-factor` (default 10): what the grad oracle multiplies the
  numerics tolerances by, measured against a torchvision 0.29.0 resnet18
  backward at 2x3x64x64 on torch 2.14.0+cpu/aarch64 -- 1x in eval mode and
  about 161x in train mode, where batch norm sends every gradient back through
  statistics computed from the batch; a float64 reference puts eager and
  inductor at the same order of error there (3.4e-5 against 3.9e-5), the
  float32 noise floor of a deep backward rather than a miscompile, and the
  report's environment block records the factor. `cases/markers.py`: the
  known-bad version marker PLAN.md "Regression corpus" asks for, as a table
  rather than as prose in five docstrings -- `expected_verdict(case,
  torch_version, git_version)` returns `RED`, `GREEN`, or `UNKNOWN`; a release
  compares by version number, a nightly by its date, and a build that cannot be
  placed against a known fix comes back `UNKNOWN` rather than being guessed at.
  `cases/summary.py` and `python -m cases.summary`: run every standalone case
  and print one Markdown table of observed against expected; CI appends it to
  the job summary on every matrix cell. `tests/test_corpus_markers.py`: the
  marker against the live verdict, per case, with a disagreement raised as a
  warning rather than a failure -- a nightly that fixes a bug upstream must not
  break this repository -- printing `CORPUS <case> observed=... expected=...
  torch=...` for every case. `tests/test_corpus_oracles.py`: every corpus case
  through the runner and every oracle in one parametrized test, graded against
  the case's own `check()`, replacing four hand-written per-case integration
  tests and covering the two cases they never reached.
- **A-2** (PR #11): Action and docs polish, office slice. The self-test
  workflow's harness step moves its torch install to the CPU index; a
  second job, `selftest-git-source`, forces `source: git` against
  `ref: ${{ github.sha }}` with `continue-on-error: true` -- the only job
  that exercises the git-install path an external consumer's workflow
  actually uses, and expected to fail while the repository is private (pip
  cannot clone a private repo without credentials), a red job with a stated
  reason rather than a green one that never ran the path it claims.
  `action/action.yml`: branding, a fuller Marketplace description, and the
  job summary table gains a stage column parsed from the terminal report's
  "first diverges at ..." / "clean: ..." line, plus the compile-check
  version in the heading. `action/README.md`: the Marketplace-facing
  README (an M0 placeholder until now) -- what it does, a minimal workflow,
  and a pointer to `docs/action.md` as the single source for the inputs and
  outputs table, source semantics, and baseline semantics.
  `docs/action.md`: a section cross-referencing what `selftest-git-source`
  does and does not cover. `CHANGELOG.md`: every existing bullet
  re-grouped under its slice, in merge order, PR-numbered -- M1-1 and M1-2
  confirmed already present in substance and labelled, A-1/C-1/C-2/C-3
  added. `CONTRIBUTING.md`: the repo-wide gate as a fresh-venv checklist,
  the no-AI-trailers / "AI assisted." rule, and a pointer to
  `cases/README.md` and `docs/validation.md` for adding a corpus case or a
  validation target.
- **M3-1** (PR #13): the graph health oracle, the fifth and last of PLAN.md
  "Oracles". Every compiled lane is traced once more under
  `torch._dynamo.explain`, and the report gets the graph count, the break
  count, and one finding per break naming the reason and the user line it
  happened on. Graph breaks are `info` by default, because a break is a slower
  plan rather than a wrong answer; three things make one a `fail`:
  `--fullgraph` was asked for and the graph broke anyway, a `--baseline` was
  given and a break appeared that is not in it, or the lane answered the first
  call and raised on the repeat call. A recompile across the repeat call with
  identical inputs -- `counters['stats']['unique_graphs']` moving -- is a
  `warn`. `--fail-on graph` is what turns any of it into exit code 1, as for
  every other oracle. `--baseline FILE` and `--write-baseline FILE`: PLAN.md
  "GitHub Action"'s fail-on-new-breaks-only mode, and the flag that records the
  file to compare against -- the format is `{backend: {graph_break_count,
  break_reasons[]}}`, small enough to review in a pull request; a break already
  listed produces no finding at all, and a baseline that is missing or
  malformed is exit 2 rather than a silently empty comparison. The graph oracle
  also takes over repeat-call health, which M1-3 recorded and left unowned: a
  lane that answers once and then raises is now a fail-severity graph finding.
  Graph findings never move the stage verdict -- `BackendSummary.graph_fail`
  counts them, the stage block says why they are not there, and `--fail-on
  graph` still decides the exit code. The environment block reports the module
  handling that actually happened: a module that refused `copy.deepcopy` now
  reads "shared across every lane (deep copy failed: <reason>)" instead of a
  wrong "deep copied per lane" row, and a plain callable is no longer claimed
  to have been copied either. `tests/fixtures/graph_break.py`: a target with a
  deliberate `print` break and a deliberate data-dependent branch. All five
  oracles now run.
- **M3-2** (PR #14): the three report artifacts of PLAN.md "Reports", off one run.
  `report/json.py`: the CI-consumable artifact, `schema_version` 1, carrying
  the environment block (architecture always, per PLAN.md "Cross-architecture
  parity is a feature"), the run configuration including the module handling
  that actually happened, one record per lane with its timings, its exception,
  its repeat-call exception and its graph health, every finding with its
  details, the verdict with per-lane counts, and the exit code. Validation is
  hand-rolled in `validate()` -- torch stays the only dependency -- and `dump`
  refuses to write a document that does not match the schema. There is no
  timestamp in it, deliberately: parity in v1 is two machines and a diff, and a
  `compile-check compare a.json b.json` subcommand is v0.2. `report/markdown.py`:
  an issue draft in the shape the PyTorch tracker expects -- a title naming the
  lane and what changed, the repro inline, expected versus got per finding, the
  stage line with PLAN.md's observability caveat, the emitted regression test,
  the environment block, and the command that produced it. It writes no
  disclosure line: whether a person discloses tooling on an issue they file
  under their own name is theirs to decide. `report/pytest_case.py`: the top
  fail-severity finding as a regression test in the inductor suite's
  eager-versus-compiled idiom, asserting what the oracle failed on --
  `assertEqual` on `dtype`/`shape`/`stride` for metadata, `assertIsNot` plus a
  `data_ptr` comparison for alias, a backward per lane for grad (and, for a
  parameter gradient, the two lanes run one after the other, since
  `torch.compile(model)` shares the module's parameters), `explain` for graph,
  and `torch.testing.assert_close` as the general form. A clean run writes no
  file and says which of the two reasons it was. `report/repro.py`: the target's
  own source reduced to the statements the entry point and the inputs need,
  shared by the draft and the emitter so the two cannot disagree about what ran;
  a file whose entry point is bound inside a block falls back to the whole file
  and says so. `--json FILE`, `--md FILE` and `--emit-test FILE` write them, each
  saying so on stderr as `--write-baseline` does, and a write that fails is exit
  2 after the report rather than instead of it. `discover.load_target` records
  where the target came from (`results.TargetSource`) so a report can quote it.
- **M3-3** (PR #15): the minimizer of PLAN.md "Minimizer, v1", behind `--minimize`.
  `minimize.py` runs only after a fail-severity finding -- the same one the
  regression-test emitter writes about -- and re-runs exactly two lanes per
  candidate, the eager reference and the one that diverged, judging each by the
  oracle that produced the finding and by the finding's identity (oracle, lane,
  output index, field) rather than by its message, so a shrink that moves the
  first differing element is not read as a different bug. A control re-run of
  the unchanged target comes first and is outside the budget: a finding that
  does not reproduce twice is reported as such instead of being shrunk.
  `shrink_inputs` halves the leading dimension while the finding survives, down
  to one and never below it, taking leaves that share a leading dimension
  together first (a batch of activations and a batch of masks halve as one, and
  halving either alone would only make the model raise) and then offering each
  leaf a halving of its own. `stub_children` is the delta debugging: every child
  is replaced with `torch.nn.Identity()` in turn, parents before their children
  and a stubbed subtree never walked into, the replacement kept when the finding
  survives and reverted with a reason when it does not -- either "a passthrough
  does not fit there", naming the exception, or "it lives in here". The work
  happens on a deep copy, so the target the report describes is never edited.
  `handoff_note` writes PLAN.md's step 4 as a note and never runs it:
  `TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4` for a numerics
  divergence, and for any other oracle the same two variables plus the reason
  the accuracy minifier would not isolate it. `--budget SECONDS` now bounds the
  minimizer (it cannot bound the run: a compile that has started cannot be
  interrupted without killing the process), a ceiling of 100 candidates applies
  when it is not given, and a pass that hits either is reported as **partial**
  in all three formats rather than as a smallest case. The minimized record
  reaches the terminal report as a `minimized` block, the JSON as a top-level
  `minimized` object -- `schema_version` 2, `null` meaning "not run" as against
  a record whose `changed` is false meaning "run, and nothing could be reduced"
  -- the Markdown draft as a `## Minimized` section, and the emitted regression
  test as the shrunk factory and the stub lines that open the test method.
  `tests/fixtures/divergent_child.py`: a three-block model whose middle block is
  the only one a registered perturbing backend keys on, so the delta-debugging
  pass has a target that is a fixture rather than a bug in the installed wheel.
- **M4-1** (PR #16): Action inputs `write-baseline`, `minimize`, `cache` (compile caches
  on plus an `actions/cache` wheel cache) beside the existing `baseline` and
  `budget`; a job summary carrying the graph-break count and a minimized block,
  rendered by `action/summary.sh`; and self-test jobs for the baseline round
  trip and for a seeded regression the action has to fail on.
- **M4-2** (PR #17): README, docs, and CHANGELOG for `0.1.0`; the cross-architecture
  runbook. README: the status banner replaced with "What it does" (one
  paragraph plus the five checks), a real red run against
  `cases/dtype_promotion.py` trimmed to the checks table, the finding, and
  the stage line, a "Bugs it has caught" table built from `FINDINGS.md`
  with every issue and PR linked, the real numbers from
  `docs/validation.md`, and a Quick start that installs from git until the
  package is on PyPI. `docs/usage.md`: every CLI flag with one real,
  executed example each, including the exit-code pair
  (`--fail-on metadata` vs. `--fail-on numerics` against the same finding)
  and the `--write-baseline` / `--baseline` round trip.
  `docs/reports.md`: the JSON schema (v2, pointing at
  `report/json.py`'s own docstring), the Markdown draft's shape, and the
  emitted regression test -- run directly with `python test_case.py`,
  which fails on this torch build because issue #191308 is still open, the
  correct outcome for an assertion that has not been fixed upstream yet.
  `docs/corpus.md`: the two file shapes in `cases/`, the marker table, and
  a real `python -m cases.summary` run, deferring the detailed walkthrough
  to `cases/README.md`. `docs/cross-arch.md`: the runbook PLAN.md
  "Cross-architecture parity is a feature" calls for -- venv setup for an
  x86 CPU or a CUDA box, the corpus health check, per-target JSON via
  `compile-check --json` (`validation/run.py` has no `--json-dir` flag and
  writes one combined summary file, not per-target JSON, so this is the
  documented substitute), a diff script comparing `schema_version`,
  `environment.machine`, `environment.cuda_available`, and `findings`
  between two runs, and what parity does and does not mean. Every command
  on every page above was executed for real in this slice's own venv; the
  `docs/README.md` index, an M0 placeholder that had never been updated, is
  rewritten to link all of it. `CHANGELOG.md`: backfilled the missing
  **A-2** entry and the PR numbers **M3-2** and **M3-3** were merged
  without, and moved the heading to `[Unreleased] → 0.1.0` to say plainly
  what this file is building toward, without claiming the tag or the PyPI
  upload has happened.
- **M4-3** (PR #18): release engineering for `0.1.0`, and the fixes four
  earlier verifiers left behind. `pyproject.toml`: version `0.1.0` (in step
  with `compile_check.__version__`), full PyPI metadata -- PEP 639
  `license = "MIT"` plus `license-files`, keywords, classifiers, a
  `Documentation` URL -- and a `validation` extra (`torchvision`,
  `transformers<5`) so the real-world targets install with one command;
  `build` and `twine` join the `dev` extra. `python -m build` and
  `twine check` are clean, and both the wheel and the sdist were installed
  into a fresh virtual environment and run there. `docs/release.md`: the
  maintainer's runbook for the four steps this milestone prepares and
  deliberately does not take -- the tag, the PyPI upload, the public flip,
  and the Marketplace listing -- in order, with what each one makes visible,
  which of them cannot be undone, and the Marketplace's
  metadata-file-in-the-root requirement that `action/action.yml` does not
  currently meet. `action/run.sh`: the "Run compile-check" step moved out of
  `action.yml` into a file that `tests/test_action_run.py` executes, the same
  split `summary.sh` already had.
- **M4-4** (PR #19): the release workflow that never holds a token.
  `.github/workflows/publish.yml`: on a `v*` tag push (or `workflow_dispatch`
  for a dry-run build), `python -m build` and `twine check dist/*`, then
  TestPyPI and PyPI uploads through PyPI's Trusted Publishing (OIDC) behind
  the `testpypi` and `pypi` GitHub environments -- no API token lives in a
  secret, on disk, or in shell history, and each environment's required
  reviewers gate the corresponding upload before it runs. `docs/release.md`
  gains step 5b: the one-time pending-publisher registration on PyPI and
  TestPyPI (works from a private repository -- GitHub's OIDC token carries
  owner/repository/workflow/environment claims, which is what PyPI checks,
  not the repository's visibility) and the GitHub environment reviewer
  setup, plus the account-scoped-token note in step 5 for whichever upload
  has to happen before the project exists on PyPI to scope a token to it.
- **V-1** (PR #21): the aarch64 baseline for cross-architecture parity. Six
  per-target `torch-compile-check --json` outputs (schema 2, torch
  `2.14.0+cpu` git `08187d9e0fba`, Python 3.10.12, aarch64), committed under
  `validation/results/per-target/aarch64-2.14.0+cpu/`, all exit 0 with zero
  findings -- the reference set every other architecture's per-target run is
  diffed against by `diff_parity.py` (`docs/cross-arch.md`).
- **V-2** (PR #23): the first cross-architecture parity result. ProBook
  x86_64 CPU per-target JSONs (torch `2.14.0+cpu`, `compile-check 0.1.0` @
  `4caf42c`, pre-rename) under
  `validation/results/per-target/x86_64-2.14.0+cpu/`, plus the `run.py`
  summary -- all six hold parity against the committed aarch64 reference
  (`diff_parity.py`, 6/6). `docs/validation.md` gains the cross-architecture
  table, one row per leg; `docs/cross-arch.md` replaces its
  same-architecture placeholder demo with the real aarch64-versus-x86_64
  transcript.
- **V-3** (PR #24): the CUDA leg, and the last of the three. Twelve
  per-target JSONs from the Omen (GTX 1660 Ti, `sm_75`, torch
  `2.14.0+cu126` git `08187d9e0fba`, Python 3.12.13), six run on CPU and six
  on CUDA, under `validation/results/per-target/x86_64-2.14.0+cu126-cpu/`
  and `.../x86_64-2.14.0+cu126-cuda/`, plus the `run.py` summary -- all
  twelve hold parity against the aarch64 reference by the same
  `diff_parity.py`. Cross-architecture validation now stands at 18/18: the
  aarch64 reference held against x86_64 CPU, x86_64+cu126 CPU, and
  x86_64+cu126 CUDA (`sm_75`), six targets on each leg.
  `docs/validation.md` points its rows at the committed files and notes the
  `--fp64-oracle` difference on this leg; `docs/cross-arch.md` gains the
  real CUDA-versus-aarch64 transcript.

### Changed

- **M4-5**: renamed the project, before the `v0.1.0` tag -- distribution,
  repository, console script, and import package are now all
  `torch-compile-check` (formerly `compile-check`), one name for all four.
  `pyproject.toml`'s `name` and `[project.urls]`, the console script entry
  (`torch-compile-check = "torch_compile_check.cli:main"`), the import
  package (`src/compile_check` moved to `src/torch_compile_check`, every
  `import`/`:mod:` reference in `src`, `tests`, `cases`, `validation`,
  `action/run.sh`, and the docs updated with it), the CLI's own `argparse`
  program name and every printed `torch-compile-check: ...` prefix,
  `--version`, the Action's name and description in `action/action.yml` and
  `action/README.md`, the job-summary title in `action/run.sh`, the
  observation-cache environment variable (`COMPILE_CHECK_OBSERVATIONS` to
  `TORCH_COMPILE_CHECK_OBSERVATIONS`) and file prefix in `cases/summary.py`,
  the JSON report's `tool.name`, the Markdown draft's "Drafted by" line, and
  the wheel/sdist names in `docs/release.md`. Nothing about the tool's
  behaviour changed. The GitHub repository itself is renamed separately, by
  the maintainer, after this merges; GitHub redirects the old URL.

### Fixed

- sdist contents pinned. `python -m build` (hatchling) bundled every
  untracked, non-gitignored file sitting in the working tree at build time
  into `dist/*.tar.gz` along with the source -- a stray `results/`, `dp.json`
  or `rc.md` left over from a hand run on the box that built it, found on
  the aarch64 release-candidate leg. `[tool.hatch.build.targets.sdist]`
  now lists exactly what a release needs -- source, tests, the corpus, the
  validation suite, docs, the Action, the CI workflows, and the top-level
  project files -- so the sdist is reproducible from a clean clone
  regardless of what else is lying around the checkout. Reproduced by
  dropping three untracked files into the tree and confirming they are
  absent from the rebuilt sdist while `LICENSE`, `README.md`,
  `pyproject.toml`, `src/`, `tests/` and `docs/` are present; `twine check`
  stays clean on both the wheel and the sdist.
- `cases.summary` no longer caches a crashed corpus observation. An entry whose
  verdict is `UNKNOWN` -- the script exited 2, timed out, or could not be
  placed -- is never written to the observation cache, and an `UNKNOWN` entry
  an older cache file already holds is read back as a miss rather than reused.
  On a box whose environment was broken (missing Python headers made every
  case's inductor compile fail, exit 2), the cache used to keep answering
  `UNKNOWN` for every case after the box was fixed, since nothing about the
  cases' own source had changed -- a crash is environment state, not a
  verdict, and `TORCH_COMPILE_CHECK_OBSERVATIONS=` never covered it because
  the stale entries were sitting under the *default* cache path, not a path
  anyone had overridden. `python -m cases.summary` also takes a `--no-cache`
  flag now (equivalent to `TORCH_COMPILE_CHECK_OBSERVATIONS=`), and the tally
  line always says how many observations were reused versus freshly run,
  including zero (M4-6).
- The Action's "Run compile-check" step no longer dies on the first target it
  cannot run. Its stage-parsing pipeline ran under `set -e`, and `grep` finding
  neither "first diverges at" nor "clean:" -- which is every tool error: a
  missing target, a bad flag, an unknown `--fail-on` category, an unparsable
  `budget` -- aborted the whole step. The result was exit 1, an empty summary
  table, no `exit-code` or `json-path` output, and every later target silently
  unchecked, so one typo in one `targets` line cancelled the check on all the
  others. Each target now gets a row of its own (`2`, `tool error: <the CLI's
  sentence>`), the loop carries on, the worst code across the targets is the
  step's own, and both outputs are written on every path out of it, including
  the ones that refuse an input before anything runs. Pre-existing since A-1;
  found by the M4-1 verifier (M4-3).
- `validation/run.py` reports a target whose extra is installed but will not
  import as **skipped with the reason**, not as a tool error.
  `importlib.util.find_spec` says `transformers` 5 is present and
  `from transformers import BertModel` then raises `ModuleNotFoundError` out of
  its lazy importer, which made `hf_tiny_bert` exit 2 and read as "compile-check
  is broken" rather than "this environment cannot build the target". The
  `validation` extra pins `transformers<5` so the row is real (M4-2 estate run,
  M4-3).
- `minimize.py` no longer says a case is irreducible when a ceiling stopped it.
  Under `--budget 0` the shrink pass reported "every input's leading dimension
  is load-bearing: halving it stopped reproducing", a measurement it had not
  made, and `Minimization.summary` -- the line the Action's job summary prints
  -- said "every input and every child is load-bearing" for the same reason.
  Both now name the ceiling that ran out (M3-3 verifier, M4-3).
- `--budget` below zero is a tool error naming the flag, instead of being read
  as a ceiling that had already expired; so is `nan`, which every comparison
  read as no ceiling at all (M3-3 verifier, M4-3).
- `oracles/graph.py` `read_baseline` refuses a baseline entry that is missing
  `graph_break_count` or `break_reasons`, exit 2 with the field named. They
  defaulted to `0` and `[]`, which turned a truncated or hand-edited entry into
  the strictest baseline there is: every break the lane really had came back as
  a *new* break and failed the job, and nothing in the message mentioned the
  file (M3-1 verifier, M4-3).
- `report/json.py` `validate` checks `schema_version` first and reports it
  alone. A v1 artifact used to be rejected for a missing `minimized` key, which
  described a v1 document as a damaged v2 one; it is now named as version 1
  against the version this build writes (M3-3 verifier, M4-3).
- The corpus runs once per environment instead of twice. `cases.summary.observe`
  keeps its verdicts in a JSON file under the system temporary directory and
  reuses an entry only when the torch build, the interpreter, the machine and
  the case's own source are all unchanged, so CI's job-summary step stops
  re-compiling what the `pytest` step in the same job already measured
  (roughly a minute per matrix cell). The table says how many rows were reused
  (M2-3 note, M4-3).
- `report/repro.py`: the whole-file fallback hands its `from __future__`
  imports out instead of leaving them in the middle of the block. They are
  only legal directly under a module docstring, so every regression test
  emitted for a target whose entry point is bound inside a block -- a `with
  torch.random.fork_rng()`, which is how a seeded model is usually written --
  was a file that did not parse. Found by running an emitted test rather than
  by reading one (M3-3).
- `BackendResult.second_call_exception`, so a lane that answers once and then
  raises is recorded rather than only logged (M1-1).
- `BackendResult.input_meta_before` / `input_meta_after`: shape, stride,
  dtype, storage offset, and the two addresses per input leaf, so a
  `resize_` can be told from a `copy_`, which two clones cannot say (M1-1,
  ahead of M2-1's mutation check).
- Module state is isolated per lane: every backend runs against its own deep
  copy of the `nn.Module`, so a buffer the forward pass writes to (a step
  counter, BatchNorm running statistics in train mode) cannot leak from one
  lane into the next and read as a numerics divergence. `--share-module`
  turns the copy off for a model too large to duplicate (M2-1 review).
- Backend names are validated after the target is imported, not before: a
  target that registers its own backend with
  `torch._dynamo.register_backend` now works from a cold run, where it used
  to be rejected as a typo (M2-1 review).
- The metadata oracle's `requires_grad` comparison reads the runner's record
  rather than the output clone. The clone is detached, so the field answered
  `False` on both sides of every real run and the check was vacuous (M2-2
  review).
- `numerics.compare_tensors`: the value comparison lifted out to one reusable
  call, so a gradient and an output are compared by the same code and the
  same tolerances (M2-2).

[Unreleased]: https://github.com/HussainNizamani/torch-compile-check/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/HussainNizamani/torch-compile-check/releases/tag/v0.1.0
