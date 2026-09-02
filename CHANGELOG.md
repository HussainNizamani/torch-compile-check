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
- Tooling: ruff lint and format, mypy strict over `src/`, pytest, pre-commit, a
  `Makefile`, and a GitHub Actions matrix over Python 3.10 to 3.13 and torch
  stable and nightly on CPU.

[Unreleased]: https://github.com/HussainNizamani/compile-check/commits/main
