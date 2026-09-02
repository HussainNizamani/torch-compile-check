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

The repo-wide gate, run from the repo root, in a venv created fresh for the
change rather than reused from an earlier one -- a venv that outlived the
session it verified is exactly how a stale or hand-patched dependency stays
undetected (see the git history around `/tmp/pruefer_venv`'s quarantine for
what that cost once already):

```console
python -m venv .venv && . .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"

make lint    # ruff check + ruff format --check
make type    # mypy strict over src/
make test    # pytest
```

All three run in CI across Python 3.10 to 3.13 and torch stable and nightly, CPU
only, so a change that passes locally on one version can still fail there.
Paste the real output in the pull request description -- "tests pass" on its
own is not evidence.

Commits carry no AI attribution trailers (no `Co-Authored-By`, no
`Generated with`); the pull request body's last line is "AI assisted."
instead, once, and that is the whole disclosure.

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
- Path discovery mutates the running interpreter on purpose: `discover.py`
  inserts the target file's parent directory at `sys.path[0]` and registers the
  module in `sys.modules` under the file's stem (or
  `_compile_check_target_<stem>` when that name is already taken). Both exist so
  a two-file repro imports its sibling. Keep them; if you change either, say so
  in the README section that documents it, because it is behaviour a user can
  observe.
- Tests point `TORCHINDUCTOR_CACHE_DIR` at a directory `conftest.py` creates and
  deletes. `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` stops inductor *reading* a
  cached artifact, not *writing* generated code, so without this a test run
  leaves hundreds of megabytes behind.

## Adding a regression case or a validation target

Two different things live under `cases/` and `validation/targets/`, for two
different jobs, and a new file almost always belongs to one, not both.

- A **regression case** (`cases/`) is a known `torch.compile` bug, tiny and
  bug-shaped on purpose, that exists to keep a fix from regressing and to
  give the oracles a positive-coverage target. See
  [`cases/README.md`](cases/README.md) for the two file shapes every case
  needs (a standalone RED/GREEN script and a discovery-convention twin), the
  `FINDINGS.md` row it fills in, and the version-marker convention.
- A **validation target** (`validation/targets/`) is a real, public,
  CPU-runnable model that exists to measure the tool's false-positive rate
  against architectures the fixture-sized corpus does not reach. See
  [`docs/validation.md`](docs/validation.md) for how targets are chosen
  (no network access at import time, random init, reduced size where a full
  model does not compile in reasonable time on CI hardware), how
  `validation/run.py` is re-run to regenerate the doc, and how a package a
  target needs beyond `torch` (`torchvision`, `transformers`) stays a
  validation-only extra rather than a `pyproject.toml` dependency.

## Scope

PLAN.md is the scope lock. If a change does not fit a milestone in it, open an
issue first and say which milestone it belongs to.
