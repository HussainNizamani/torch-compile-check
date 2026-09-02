# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Package scaffold: `pyproject.toml` (hatchling, src layout), `compile_check`
  package with typed stubs for every module in PLAN.md "Package layout".
- `compile_check.env`: `collect_environment()` for the report's environment
  block, and `probe_apis()` for the torch symbols in PLAN.md "Verified API
  surface".
- CLI: the full v1 flag surface is parsed; `--version` and `--probe` work, every
  other invocation exits 2.
- `compile_check.discover` and `compile_check.runner`: target resolution and the
  per-backend run, with per-backend input clones, a compiler reset between
  lanes, both calls timed, captured exceptions, and the optional `eager_fp64`
  reference behind `--fp64-oracle`.
- The numerics and metadata oracles, and the `Finding` / `OracleConfig` /
  `Oracle` vocabulary they share.
- `compile_check.localize`: the ablation ladder as a decision procedure, with a
  `StageVerdict` that names the first diverging backend and the compilation
  stage it implicates, worded "first diverges at" and never "the bug is in".
- `compile_check.report.terminal`: the ANSI terminal report -- environment block
  with the architecture, per-backend table, oracle-by-backend table, findings
  grouped by oracle under `--max-findings`, the stage verdict, and a next-step
  hint. `--color auto|always|never`, off when stdout is not a terminal or
  `NO_COLOR` is set.
- CLI main path: run, compare, localize, report, exit 0/1/2. `--fail-on` selects
  which oracle categories turn a finding into exit 1 and never which oracles
  run; a compiled backend that raised while eager did not is always exit 1.
  `--no-grad` switches the backward pass off.
- `BackendResult.second_call_exception`, so a lane that answers once and then
  raises is recorded rather than only logged.
- The alias and mutation oracle: object identity, untyped storage identity, and
  byte-range overlap over every output-output and output-input pair, plus the
  set of inputs the call mutated in its values or in its layout. The compiled
  relation must equal the eager one entry for entry; an added alias is the
  195451 shape and an identity collapse the 191449 one. Storage sharing without
  overlap and `torch._debug_has_internal_overlap` are recorded as context and
  never fail a run.
- `BackendResult.input_meta_before` / `input_meta_after`: shape, stride, dtype,
  storage offset, and the two addresses per input leaf, so a `resize_` can be
  told from a `copy_`, which two clones cannot say.
- `cases/alias_copyback.py`: 195451 written to the discovery convention, so
  `compile-check cases/alias_copyback.py` runs it through the tool.
- Module state is isolated per lane: every backend runs against its own deep
  copy of the `nn.Module`, so a buffer the forward pass writes to (a step
  counter, BatchNorm running statistics in train mode) cannot leak from one lane
  into the next and read as a numerics divergence. `--share-module` turns the
  copy off for a model too large to duplicate, and the report's environment
  block says which mode produced it.
- Backend names are validated after the target is imported, not before it: a
  target that registers its own backend with `torch._dynamo.register_backend`
  now works from a cold run, where it used to be rejected as a typo. The
  up-front pass still catches an empty `--backends`, and an unknown name is
  still exit 2 once nothing has registered it.
- `BackendResult.output_requires_grad`, `BackendResult.grads`, and
  `BackendResult.grad_present`: which outputs carried a gradient, and which
  inputs and parameters received one, labelled once so a message and a set
  comparison cannot drift apart. The backward pass reduces at float64, so the
  accumulation order of a float32 sum cannot reach the gradients being compared.
- The metadata oracle's `requires_grad` comparison reads the runner's record
  rather than the output clone. The clone is detached, so the field answered
  `False` on both sides of every real run and the check was vacuous.
- The gradients oracle: one backward per lane on the deterministic scalar
  reduction, then two comparisons. The set of tensors that received a gradient
  must be identical, and a finding names the parameter; the gradients themselves
  go through the numerics rule, so `--rtol` and `--atol` reach them. A
  backward that raised in one lane and not the other is a fail on its own, and
  `--no-grad` reports an `info` line rather than a clean grad row. Registered in
  `ORACLES`, so `--fail-on grad` now decides an exit code.
- `numerics.compare_tensors`: the value comparison as one reusable call, so a
  gradient and an output are compared by the same code and the same tolerances.
- `--seed` is applied before the target module is imported, and again before
  every backend. A target that builds its model at module scope -- the shape
  `model = torchvision.models.resnet18(weights=None)` has -- draws its weights
  during discovery, so a seed applied afterwards never reached them and two runs
  of the same command compared two different models.
- `--grad-tol-factor` (default 10): what the grad oracle multiplies the numerics
  tolerances by. A gradient is a sum over every path that reaches a tensor and
  compilation is free to reassociate that sum, which is why the M2-2
  verification saw a compiled resnet18 gradient about 1.24e-5 from eager's
  against a float32 atol of 1e-5 and the same run flip between clean and
  failing. The tolerances for outputs are unchanged, and the report's
  environment block records the factor the gradients were compared under. Ten
  clears the measured borderline case and not every model: a whole resnet18
  backward at 2x3x64x64 needs about 161x on torch 2.14.0+cpu/aarch64, and a
  float64 reference puts eager and inductor at the same order of error (3.4e-5
  against 3.9e-5), so that is the float32 noise floor rather than a miscompile.
  PLAN.md "Tolerance policy" is what this measurement feeds.
- Tooling: ruff lint and format, mypy strict over `src/`, pytest, pre-commit, a
  `Makefile`, and a GitHub Actions matrix over Python 3.10 to 3.13 and torch
  stable and nightly on CPU.

[Unreleased]: https://github.com/HussainNizamani/compile-check/commits/main
