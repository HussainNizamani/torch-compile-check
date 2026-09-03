# Using torch-compile-check as a GitHub Action

A composite action published from [`action/`](../action/action.yml) in this
repository. It installs `torch-compile-check`, runs it against the entrypoints you
declare, fails the job on the configured `--fail-on` categories, and writes a
job summary with one row per target.

> **Status:** the CLI's main run path landed in M1-3
> ([PR #6](https://github.com/HussainNizamani/torch-compile-check/pull/6); see
> [PLAN.md](../PLAN.md)). `torch-compile-check <target>` on `main` runs for real:
> exit 0 clean, 1 on a `--fail-on` finding, 2 on a tool error. Only a `ref`
> pinned to a commit before M1-3 still exits 2 with "not implemented" for
> every real target; the `allow-unimplemented` input exists for that case
> (see "Degrading honestly on a pre-M1-3 ref" below), not for `main`.

## Usage

```yaml
name: torch-compile-check

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

      - uses: HussainNizamani/torch-compile-check/action@main
        id: torch-compile-check
        with:
          targets: |
            models/classifier.py
            models/encoder.py:build_model:build_inputs
          torch: ${{ matrix.torch }}
          baseline: .torch-compile-check/baseline.json

      - name: Upload results
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: torch-compile-check-results-${{ matrix.torch }}
          path: ${{ steps.torch-compile-check.outputs.json-path }}*
```

Pin `ref` (not `@main`) once the action stabilizes if your repository wants a
fixed version:

```yaml
- uses: HussainNizamani/torch-compile-check/action@main
  with:
    ref: v0.1.0
    targets: models/classifier.py
```

## Inputs

| Input | Default | Description |
|---|---|---|
| `targets` | *(required)* | Newline-separated list of `path[:entry][:inputs]` targets, one torch-compile-check run per line. See PLAN.md "Discovery convention" for the `path`/`entry`/`inputs` grammar. |
| `backends` | `eager,aot_eager,inductor` | Comma-separated backends, forwarded to `--backends`. |
| `fail-on` | `numerics,alias,metadata,grad` | Comma-separated oracle categories that fail the job, forwarded to `--fail-on`. The correctness categories are on by default; `graph` is informational unless you ask for it (see `baseline` below). |
| `torch` | `stable` | `stable` (PyPI), `nightly` (CPU nightly index), or an explicit pip spec such as `torch==2.5.0`. |
| `python-version` | `3.12` | Passed to `actions/setup-python`. |
| `baseline` | *(unset)* | Path to a stored graph-health baseline JSON, forwarded as `--baseline`. When set, the graph oracle fails on **new** breaks only rather than every break — see "Baseline semantics" below. |
| `write-baseline` | *(unset)* | Path to write this run's graph health to, forwarded as `--write-baseline`. For the one-off job that produces the file you commit and then pass back as `baseline`; the run itself still reports its own verdict. Takes a single `targets` line — one baseline file is keyed by backend, not by target, so a second target would silently overwrite the first, and the action refuses instead. |
| `minimize` | `false` | Exactly `"true"` or `"false"`. `"true"` passes `--minimize`, so a finding is shrunk — leading input dimension halved, child modules replaced with a passthrough — before it is reported. Costs one re-run of two lanes per candidate, which is what `budget` bounds. |
| `budget` | *(unset)* | Wall-clock ceiling in seconds for the minimizer, forwarded to `--budget`. It bounds `minimize` only; see [Runtime budget](#runtime-budget). |
| `cache` | `false` | Exactly `"true"` or `"false"`. `"true"` lets the run reuse compiled artifacts: torch's compile caches stay on (via the CLI's `--allow-caches`) and pip's wheel cache is restored and saved with `actions/cache`. The default is what makes a run measure the current compiler — see [Compile caches](#compile-caches). |
| `json-out` | `torch-compile-check-results.json` | Base path for the JSON results. With more than one target, each run writes its own file suffixed `.<n>.json` next to this base (`torch-compile-check-results.1.json`, `.2.json`, ...), since one CLI invocation produces one JSON document per PLAN.md "Reports". |
| `extra-args` | *(unset)* | Extra arguments appended verbatim to every invocation, for flags this action does not wrap directly (`--rtol`, `--seed`, `--fullgraph`, ...). |
| `ref` | `main` | Git ref of `HussainNizamani/torch-compile-check` to install from, until the package ships on PyPI. |
| `source` | `auto` | Where to install torch-compile-check from. `auto` installs from the checked-out source when the action runs inside this repo (its parent directory declares `torch-compile-check` in `pyproject.toml`), else falls back to `git`. `local` forces the checked-out-source install. `git` forces `pip install git+https://.../torch-compile-check@ref` — the only option that works for external consumers, since pip cannot clone a private repo without credentials. |
| `allow-unimplemented` | `false` | Exactly `"true"` or `"false"`. See "Degrading honestly" below. |

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
commit what it writes. In a workflow that is the `write-baseline` input — a
one-off job (`workflow_dispatch` is the usual trigger) whose only purpose is to
produce the file, which you then download and commit:

```yaml
- uses: HussainNizamani/torch-compile-check/action@main
  with:
    targets: models/classifier.py
    write-baseline: .torch-compile-check/baseline.json
- uses: actions/upload-artifact@v4
  with:
    name: baseline
    path: .torch-compile-check/baseline.json
```

By hand, the same thing:

```console
$ torch-compile-check models/classifier.py --write-baseline .torch-compile-check/baseline.json
$ cat .torch-compile-check/baseline.json
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
earlier commit or an earlier job in the matrix. That default is the point of
the check: a cached run can report clean about code the current compiler never
compiled.

`cache: "true"` opts out, for teams who want the faster run. It does two
things, and the report records the first of them in
`environment.inductor_force_disable_caches` so a JSON artifact always says
which mode produced it:

- passes `--allow-caches`, so the CLI leaves `TORCHINDUCTOR_FORCE_DISABLE_CACHES`
  alone and torch reuses whatever it cached earlier in the job;
- restores and saves pip's download cache (`~/.cache/pip`) with
  `actions/cache`, keyed on the runner OS and architecture, `python-version`,
  and the `torch` input. That key is a spec, not a resolved version, so
  `torch: nightly` — whose resolved version moves daily — keeps hitting the
  same entry and simply downloads the wheel that is not in it; the prefix
  restore key is what keeps a *changed* spec partly warm. It is the wheel
  download this saves, minutes of it, not the compile.

Nothing else changes: the same targets run, the same oracles compare, and the
exit code means the same thing.

## Runtime budget

Compilation is slow, and a matrix multiplies it. `budget` caps the wall-clock
time the *minimizer* spends per target, and nothing else: it is passed to
`--budget`, which bounds what `minimize: "true"` starts. A pass that runs out
reports a partial reduction rather than claiming a smallest case — the job
summary marks it **partial** — and the run's verdict and exit code are
unaffected.

It is deliberately not a ceiling on the run. A `torch.compile` call that has
started cannot be interrupted without killing the process, so a flag that
claimed to bound the whole run would either be a lie or a `SIGKILL` with no
report at all. Use the job's own `timeout-minutes` for that, and keep the
target small.

`budget` without `minimize: "true"` has nothing to bound, and the CLI says so
in one line on stderr rather than pretending the ceiling is in force.

## Job summary

Every run writes a table to `$GITHUB_STEP_SUMMARY`, one row per target:

| target | exit code | status | graph breaks | stage |
|---|---|---|---|---|
| `cases/dtype_promotion.py` | 1 | exit 1 | 0 | first diverges at inductor, which implicates inductor lowering/codegen |

`graph breaks` is the graph oracle's break count for the run, read from the
JSON report. It is one number when every compiled lane agrees, `aot_eager 2,
inductor 5` when they do not, `n/a` for a lane whose graph health could not be
measured, and `-` when no lane measured it at all. Breaks stay informational
(PLAN.md "graph"), so this column is the only place a clean run mentions them —
and it is the number that explains a missing speedup.

With `minimize: "true"`, a `### Minimized` section follows the table, one
collapsed block per target, listing the finding that was shrunk, the input
dimensions that were halved, the child modules replaced with
`torch.nn.Identity()`, the ones that had to be kept and why, and the cost in
candidate re-runs. A pass stopped by `budget` is marked **partial** there,
because a partial reduction that reads as a smallest case is worse than no
reduction at all.

The rendering lives in [`action/summary.sh`](../action/summary.sh) rather than
inline in `action.yml`, so it can be run outside a workflow;
`tests/test_action_summary.py` executes that file against reports the CLI
writes during the test. The step that drives it is
[`action/run.sh`](../action/run.sh), split out for the same reason and driven
the same way by `tests/test_action_run.py`.

## The three boolean inputs take exactly `"true"` or `"false"`

`minimize`, `cache` and `allow-unimplemented` are compared against the string
`"true"`. Any other value is refused with an `::error::` line and exit 2,
before any target runs:

```
::error::input cache must be "true" or "false", got "yes"
```

A composite action receives every input as a string, so this is a comparison of
strings and not of booleans. `cache: "true"` and the unquoted YAML boolean
`cache: true` both reach the step as `true` and are accepted; anything that
arrives as some other string — `"yes"`, `"1"`, `"on"` — is refused rather than
read as false, because a job that believed it was reusing its compile cache and
was not is the kind of quiet wrongness this tool exists to complain about (and
for `allow-unimplemented`, a job that believed it was tolerating a pre-M1-3
`ref` and was failing on it). Quoting `"true"` and `"false"` is the habit that
keeps the question from arising.

## A target that cannot run gets a row like every other one

A tool error on one target — a path that does not exist, a module that will not
import, an unknown `--fail-on` category, an unparsable `budget` — produces a row
of its own with exit code `2` and a `tool error: <the CLI's own sentence>`
status, and the loop carries on to the next target. The step's `exit-code`
output is the worst code across all of them, so the job still fails; what does
not happen is the check on every later target being silently cancelled by the
first typo. `exit-code` and `json-path` are written on every path out of the
step, including the ones that refuse an input before running anything, because a
caller with `continue-on-error: true` reads them to decide what to do next.

## Degrading honestly on a pre-M1-3 `ref`

The CLI's main run path (oracles, localization, the terminal/JSON/Markdown
reports) landed in M1-3 ([PR #6](https://github.com/HussainNizamani/torch-compile-check/pull/6))
and is on `main` today: `torch-compile-check <target>` runs for real, exit 0/1/2
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
`action/` or either of the two targets the jobs below run.

Four jobs:

- `selftest` — the action against `tests/fixtures/mlp.py`, plus the direct
  `--version` / `--probe` / `--run-only` CLI smoke checks.
- `selftest-baseline` — the baseline round trip, through the action rather
  than through the CLI. One run writes a baseline from
  `tests/fixtures/graph_break.py` (a target that breaks the graph twice on
  purpose) with `cache: "true"`, and asserts both that the file records the
  breaks and that `environment.inductor_force_disable_caches` came out
  `false`, which is how the run says `cache` reached `--allow-caches`. A
  second run is handed that file back through `baseline` and has to report
  none of those breaks and force the caches off again. Asserting both halves
  is the point: a round trip where the two runs behaved identically would have
  proved nothing.
- `selftest-seeded-regression` — PLAN.md's M4 definition of done, "fails
  correctly on a seeded regression". `cases/dtype_promotion.py` (the 191308
  int8 matmul promotion) runs through the action with `fail-on: metadata`,
  `minimize: "true"` and a `budget`. The action is expected to fail, so the
  step carries `continue-on-error` and the next step asserts that it did, then
  reads the JSON for `exit_code: 1`, a `metadata` failure, and a `minimized`
  object. A green job here without those would be the exact failure the job
  exists to catch, and the assertion says so by name if the seed ever stops
  reproducing on the installed torch.
- `selftest-git-source` — below.

Its `selftest` job uses `source: auto`, which — running inside a
checkout of the action's own repository — always resolves to the
checked-out-source install, never to `git+https://...`, the path an external
consumer's workflow actually exercises. A second job,
`selftest-git-source`, forces `source: git` to test that path directly, and
is expected to fail while this repository is private: pip cannot clone a
private repository without credentials, and the action does not supply one.
That job runs with `continue-on-error: true` so the gap shows up as a red job
with a stated reason in the job list, rather than as a green `selftest` job
that never actually ran the install path a real external user depends on.
Once the repository is public, or `torch-compile-check` ships on PyPI and `source:
auto` resolves external consumers there instead, `selftest-git-source` is
expected to turn green with no other change.
