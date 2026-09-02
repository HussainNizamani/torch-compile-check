# Using compile-check as a GitHub Action

A composite action published from [`action/`](../action/action.yml) in this
repository. It installs `compile-check`, runs it against the entrypoints you
declare, fails the job on the configured `--fail-on` categories, and writes a
job summary with one row per target.

> **Status:** the CLI's main run path landed in M1-3
> ([PR #6](https://github.com/HussainNizamani/compile-check/pull/6); see
> [PLAN.md](../PLAN.md)). `compile-check <target>` on `main` runs for real:
> exit 0 clean, 1 on a `--fail-on` finding, 2 on a tool error. Only a `ref`
> pinned to a commit before M1-3 still exits 2 with "not implemented" for
> every real target; the `allow-unimplemented` input exists for that case
> (see "Degrading honestly on a pre-M1-3 ref" below), not for `main`.

## Usage

```yaml
name: compile-check

on:
  push:
    branches: [main]
  pull_request:

jobs:
  audit:
    runs-on: ubuntu-latest
    strategy:
      # PLAN.md "GitHub Action": stable and nightly in parallel is what turns
      # this into a nightly-regression tripwire.
      matrix:
        torch: [stable, nightly]
    steps:
      - uses: actions/checkout@v4

      - uses: HussainNizamani/compile-check/action@main
        id: compile-check
        with:
          targets: |
            models/classifier.py
            models/encoder.py:build_model:build_inputs
          torch: ${{ matrix.torch }}
          baseline: .compile-check/baseline.json

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: compile-check-results-${{ matrix.torch }}
          path: ${{ steps.compile-check.outputs.json-path }}*
```

Pin `ref` (not `@main`) once the action stabilizes if your repository wants a
fixed version:

```yaml
- uses: HussainNizamani/compile-check/action@main
  with:
    ref: v0.1.0
    targets: models/classifier.py
```

## Inputs

| Input | Default | Description |
|---|---|---|
| `targets` | *(required)* | Newline-separated list of `path[:entry][:inputs]` targets, one compile-check run per line. See PLAN.md "Discovery convention" for the `path`/`entry`/`inputs` grammar. |
| `backends` | `eager,aot_eager,inductor` | Comma-separated backends, forwarded to `--backends`. |
| `fail-on` | `numerics,alias,metadata,grad` | Comma-separated oracle categories that fail the job, forwarded to `--fail-on`. The correctness categories are on by default; `graph` is informational unless you ask for it (see `baseline` below). |
| `torch` | `stable` | `stable` (PyPI), `nightly` (CPU nightly index), or an explicit pip spec such as `torch==2.5.0`. |
| `python-version` | `3.12` | Passed to `actions/setup-python`. |
| `baseline` | *(unset)* | Path to a stored graph-health baseline JSON, forwarded as `--baseline`. When set, the graph oracle fails on **new** breaks only rather than every break — see "Baseline semantics" below. |
| `budget` | *(unset)* | Wall-clock ceiling in seconds for the minimizer, forwarded to `--budget`. It bounds `--minimize` only; see [Runtime budget](#runtime-budget). |
| `json-out` | `compile-check-results.json` | Base path for the JSON results. With more than one target, each run writes its own file suffixed `.<n>.json` next to this base (`compile-check-results.1.json`, `.2.json`, ...), since one CLI invocation produces one JSON document per PLAN.md "Reports". |
| `extra-args` | *(unset)* | Extra arguments appended verbatim to every invocation, for flags this action does not wrap directly (`--rtol`, `--seed`, `--fullgraph`, ...). |
| `ref` | `main` | Git ref of `HussainNizamani/compile-check` to install from, until the package ships on PyPI. |
| `source` | `auto` | Where to install compile-check from. `auto` installs from the checked-out source when the action runs inside this repo (its parent directory declares `compile-check` in `pyproject.toml`), else falls back to `git`. `local` forces the checked-out-source install. `git` forces `pip install git+https://.../compile-check@ref` — the only option that works for external consumers, since pip cannot clone a private repo without credentials. |
| `allow-unimplemented` | `false` | See "Degrading honestly" below. |

## Outputs

| Output | Description |
|---|---|
| `exit-code` | The worst exit code across all targets: `0` clean, `1` a `--fail-on` finding, `2` a tool error (PLAN.md "CLI surface for v1"). |
| `json-path` | The `json-out` input, echoed back so a following step can locate the per-target result files described above. |

## Baseline semantics

`torch.compile` graph breaks are not, by themselves, wrong answers — a break
just means part of the model fell back to eager. Failing the build on every
break makes the check unusable on day one for any real model with existing
breaks, so it would get disabled immediately, which defeats the point.
Instead: commit a baseline JSON of the graph-health state you consider
acceptable today, pass its path as `baseline`, and the `graph` oracle (when
included in `fail-on`) fails only on breaks that are **new** relative to that
file. The correctness categories (`numerics`, `alias`, `metadata`, `grad`)
always fail regardless of any baseline — there is no such thing as an
acceptable baseline of wrong answers.

Produce the file with the CLI's `--write-baseline`, once per target, and
commit what it writes:

```console
$ compile-check models/classifier.py --write-baseline .compile-check/baseline.json
$ cat .compile-check/baseline.json
{
  "inductor": {
    "break_reasons": [
      "gb0059: Failed to trace builtin operator"
    ],
    "graph_break_count": 1
  }
}
```

It is keyed by backend, so regenerate it when you change `backends`; a run
whose lane is missing from the file reports a warning and falls back to
listing the breaks rather than failing the job on an incomplete baseline.
Break reasons are torch's own wording, so a torch upgrade that rewords one
can show as a new break — keep a baseline per torch version in a matrix, or
regenerate after an upgrade.

## Compile caches

The action disables `torch.compile`'s caches by default (the CLI sets
`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`), matching the plain CLI, so a run
measures the current compiler rather than a cached artifact left over from an
earlier commit or an earlier job in the matrix. Pass `--allow-caches` through
`extra-args` for teams who want the faster, cached run instead; the report
records which mode was used.

## Runtime budget

Compilation is slow, and a matrix multiplies it. `budget` caps the wall-clock
time the *minimizer* spends per target, and nothing else: it is passed to
`--budget`, which bounds what `--minimize` starts. A pass that runs out reports
a partial reduction rather than claiming a smallest case, and the run's verdict
and exit code are unaffected.

It is deliberately not a ceiling on the run. A `torch.compile` call that has
started cannot be interrupted without killing the process, so a flag that
claimed to bound the whole run would either be a lie or a `SIGKILL` with no
report at all. Use the job's own `timeout-minutes` for that, and keep the
target small.

The action does not pass `--minimize` yet (that input lands with M4), so a
workflow that sets `budget` today gets one line on stderr saying the ceiling had
nothing to bound. Add `--minimize` through `extra-args` to turn it on.

## Degrading honestly on a pre-M1-3 `ref`

The CLI's main run path (oracles, localization, the terminal/JSON/Markdown
reports) landed in M1-3 ([PR #6](https://github.com/HussainNizamani/compile-check/pull/6))
and is on `main` today: `compile-check <target>` runs for real, exit 0/1/2
on its own terms. Only a `ref` pinned to a commit before that PR still
exits `2` with a fixed "not implemented" message for anything other than
`--version`, `--probe`, and the hidden `--run-only` developer path. This
action detects exactly that message and, when `allow-unimplemented: true`,
treats it as neutral (the step does not fail) instead of a tool error, with
the summary row noting "not implemented on this ref (pre-M1-3)" — backward
compatibility for a workflow still pinned to such a `ref`, not the current
state of `main`. With the default `allow-unimplemented: false`, that same
pre-M1-3 run is reported as a real failure — a green job that checked
nothing is worse than a red one that says so. A `ref` at or after M1-3
(including the default, `main`) never takes this path at all.

## What the self-test workflow does and does not cover

`.github/workflows/action-selftest.yml` runs the action against this
repository's own checkout on every push and pull request that touches
`action/`. Its `selftest` job uses `source: auto`, which — running inside a
checkout of the action's own repository — always resolves to the
checked-out-source install, never to `git+https://...`, the path an external
consumer's workflow actually exercises. A second job,
`selftest-git-source`, forces `source: git` to test that path directly, and
is expected to fail while this repository is private: pip cannot clone a
private repository without credentials, and the action does not supply one.
That job runs with `continue-on-error: true` so the gap shows up as a red job
with a stated reason in the job list, rather than as a green `selftest` job
that never actually ran the install path a real external user depends on.
Once the repository is public, or `compile-check` ships on PyPI and `source:
auto` resolves external consumers there instead, `selftest-git-source` is
expected to turn green with no other change.
