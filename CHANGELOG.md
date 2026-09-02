# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Every entry below is grouped by the slice it landed in (PLAN.md's milestone
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

### Fixed

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

[Unreleased]: https://github.com/HussainNizamani/compile-check/commits/main
