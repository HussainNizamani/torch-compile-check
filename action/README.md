# compile-check

Differential testing for `torch.compile`, in CI. This composite action installs
`compile-check`, runs it against the entrypoints you declare, fails the job on
the categories you choose, and writes a job-summary row per target with its
stage verdict. See [`docs/action.md`](../docs/action.md) for the full inputs
and outputs reference, baseline semantics, and how the action degrades
honestly today; this file is the Marketplace-facing summary.

## What it does

Point it at one or more discovery-convention targets (PLAN.md "Discovery
convention": a module-level `model` or `fn`, plus `inputs` or `get_inputs()`).
For each target, the action runs `compile-check` under eager and every
compiled backend you list, compares numerics, aliasing, output metadata, and
gradients against eager, and reports the first backend a divergence appears
at. The job fails when a finding lands in one of your `fail-on` categories, or
when a compiled backend raises where eager did not.

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
      # stable and nightly in parallel turns this into a nightly-regression
      # tripwire (PLAN.md "GitHub Action").
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

## Inputs and outputs

The full, current table lives in one place,
[`docs/action.md`](../docs/action.md#inputs) — copied here would drift the
moment one of them changes, so this file links rather than duplicates.
Briefly: `targets` is the only required input; everything else (`backends`,
`fail-on`, `torch`, `python-version`, `baseline`, `write-baseline`,
`minimize`, `budget`, `cache`, `json-out`, `extra-args`, `ref`, `source`,
`allow-unimplemented`) has a default suited to a first run. Two outputs,
`exit-code` and `json-path`.

`minimize`, `cache` and `allow-unimplemented` take exactly the string `"true"`
or the string `"false"` — an action input is always a string, so an unquoted
YAML `true` works too, and anything that reaches the step as some other string
(`"yes"`, `"1"`, `"on"`) is refused with an `::error::` line and exit 2 before
any target runs, rather than read as false: a job that believed it was reusing
its compile cache and was not is the kind of quiet wrongness this tool exists to
complain about.

Three of them decide what a run costs and what it means:

- `cache` (default `false`) keeps torch's compile caches **off**, so the run
  measures the current compiler rather than an artifact an earlier commit left
  behind. `cache: "true"` turns them on and adds an `actions/cache` step for
  pip's wheel cache; the JSON report records which mode was in force.
- `minimize` (default `false`) shrinks a finding before reporting it — leading
  input dimension halved, child modules replaced with a passthrough — and
  `budget` is the wall-clock ceiling on that pass, and on nothing else.
- `write-baseline` is the one-off job that produces the graph-health file you
  commit and pass back as `baseline`.

## Job summary

One table row per target — exit code, status, the graph oracle's break count,
and the stage the divergence first appears at — plus a collapsed **Minimized**
block per target when `minimize` is on, listing what the pass removed and what
it had to keep. A target that could not run at all gets a row too, `2` and
`tool error: <what the CLI said>`, and the loop moves on to the next target
instead of cancelling it; the job still fails, because `exit-code` is the worst
code across all of them.

The step is [`run.sh`](run.sh) and the renderer is [`summary.sh`](summary.sh),
both beside this file and both kept out of `action.yml` so they can be run and
tested outside a workflow (`tests/test_action_run.py`,
`tests/test_action_summary.py`).

## Installing from source: the `source` input

`compile-check` is not on PyPI yet. `source: auto` (the default) installs
from the checked-out repository's own source when the action runs inside this
repo — which is what the self-test workflow in this repository uses — and
falls back to `git+https://github.com/HussainNizamani/compile-check@ref`
otherwise, which is what a consuming repository's workflow actually exercises.
`source: git` forces the git install explicitly; while this repository is
private, that path needs credentials pip does not have, which is a real,
disclosed gap — see the `selftest-git-source` job in
[`../.github/workflows/action-selftest.yml`](../.github/workflows/action-selftest.yml),
run with `continue-on-error: true` for exactly that reason, not hidden behind
a green checkmark.

## Baseline semantics

A `torch.compile` graph break is not, by itself, a wrong answer — failing the
build on every break would make the action unusable on day one for any real
model with existing breaks. Pass `baseline` (a path to a stored graph-health
JSON, produced by the `write-baseline` input or the CLI's `--write-baseline`)
and the graph oracle fails on **new** breaks only. The
correctness categories — numerics, alias, metadata, grad — always fail
regardless of any baseline; there is no such thing as an acceptable baseline
of wrong answers. Full detail in
[`docs/action.md`](../docs/action.md#baseline-semantics).

## Degrading honestly on a pre-M1-3 `ref`

`compile-check`'s main run path — the oracles, stage localization, and the
terminal/JSON/Markdown reports — landed in M1-3 of [PLAN.md](../PLAN.md)
([PR #6](https://github.com/HussainNizamani/compile-check/pull/6)), so
`compile-check <target>` on `main` (the default `ref`) runs for real today:
exit 0 clean, 1 on a `--fail-on` finding, 2 on a tool error. Only a `ref`
pinned to a commit *before* M1-3 still exits 2 with a fixed "not implemented"
message for any real target. This action detects exactly that message and,
only when you set `allow-unimplemented: true`, treats it as neutral instead
of a failure — backward compatibility for a workflow pinned to an old `ref`,
not the current state of `main`. The default, `allow-unimplemented: false`,
reports that same pre-M1-3 run as a real failure: a green job that checked
nothing is worse than a red one that says so.
