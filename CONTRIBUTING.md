# Contributing

Thanks for looking. The project is at M0, so the most useful contribution today
is a review of [PLAN.md](PLAN.md) rather than code.

## Setup

```console
python -m venv .venv && . .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
pre-commit install
```

## Before you open a pull request

```console
make lint    # ruff check + ruff format --check
make type    # mypy strict over src/
make test    # pytest
```

All three run in CI across Python 3.10 to 3.13 and torch stable and nightly, CPU
only, so a change that passes locally on one version can still fail there.

## House rules

- Every change carries tests. Every oracle gets both positive coverage, where it
  catches a known bug, and negative coverage, where it stays silent on a clean
  model. A checker that fires on everything is worse than no checker.
- Nothing is described as working without the command output that proves it.
- `print` belongs in `cli.py` and `report/` only; everywhere else use
  `logging.getLogger("compile_check")`. ruff enforces this.
- No network access at import time, anywhere in the package.
- mypy runs strict over `src/`, not over tests or cases.
- Any finding on a real model is cross-checked against eager twice before it is
  called a bug.

## Scope

PLAN.md is the scope lock. If a change does not fit a milestone in it, open an
issue first and say which milestone it belongs to.
