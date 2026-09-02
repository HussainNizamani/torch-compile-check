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
- Tooling: ruff lint and format, mypy strict over `src/`, pytest, pre-commit, a
  `Makefile`, and a GitHub Actions matrix over Python 3.10 to 3.13 and torch
  stable and nightly on CPU.

[Unreleased]: https://github.com/HussainNizamani/compile-check/commits/main
