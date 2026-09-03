# Cross-architecture runbook

A copy-paste runbook for running torch-compile-check on a second machine — an x86
CPU box, a CUDA box — and comparing what it found against the committed
aarch64 run. PLAN.md ["Cross-architecture parity is a
feature"](../PLAN.md#cross-architecture-parity-is-a-feature): running the
same model on two architectures and diffing the results is a deliberate
capability, not an accident of where this project happens to be developed.
Compile bugs do not present identically across architectures — issue
[#191837](https://github.com/pytorch/pytorch/issues/191837) is the worked
example PLAN.md cites: the same defect aborted the process on x86, three
runs out of three, while on ARM it silently corrupted about 99.5 percent of
the output. A user who tested on one architecture alone would have drawn
the wrong conclusion about the other.

Every command on this page was executed for real, on this project's primary
box (aarch64, torch `2.14.0+cpu`), as the proof the runbook works
mechanically end to end. A second machine is where it earns its keep; there
is not one wired into this repository's CI yet (PLAN.md "Test
infrastructure" lists the boxes this project has access to and their
availability).

## 1. Set up the box

x86 CPU:

```console
$ python -m venv .venv && . .venv/bin/activate
$ python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
$ python -m pip install -e ".[dev]"
```

CUDA: install the `torch` wheel matching the box's CUDA driver from
[pytorch.org's install matrix](https://pytorch.org/get-started/locally/)
(there is no one fixed index URL — it is versioned per CUDA release) instead
of the CPU index above, then the same `pip install -e ".[dev]"`. Everything
below runs unchanged; `torch-compile-check --device cuda` and the `environment.cuda_available`
field are the only things that differ.

Validation targets (torchvision, a tiny HF BERT) need two more packages that
are **not** `pyproject.toml` dependencies — see
[validation.md "Extras"](validation.md#extras) for why:

```console
$ python -m pip install torchvision --index-url https://download.pytorch.org/whl/cpu
$ python -m pip install "transformers<5"
```

Two things about that line are load-bearing. `torchvision` must come from the
**same index as `torch`** (the CPU index here; on a CUDA box the matching
`whl/cuXXX` index for both packages): the PyPI wheel is built against a
different torch ABI and fails at import with `operator torchvision::nms does
not exist`, which turns every `tv_*` target into an exit-2 tool error. And
`transformers` 5.x moved `BertModel` behind a lazy import that fails on this
target, so pin it below 5 until the target catches up (M4-3).

(`pyproject.toml` does not have a `validation` extra yet — see M4-3 in
[PLAN.md "Milestones"](../PLAN.md); once it lands, `pip install -e ".[dev,validation]"`
replaces the line above. This project's own aarch64 venv already carries
`torchvision`; the commands below hit a real `ModuleNotFoundError` for
`transformers`, which was never installed on it — that gap is what
"skipped, not a tool error" in [validation.md](validation.md#extras) means
in practice, and it is the honest state of this specific box, not a claim
about what a second machine needs.)

Confirm the install before running anything that matters:

```console
$ torch-compile-check --version
torch-compile-check 0.1.0
$ torch-compile-check --probe
```

`--probe` (docs/usage.md) is worth reading closely on a new box: it is the
check that a torch upgrade or a different build has not silently removed or
renamed an API an oracle reads.

## 2. Run the corpus health check

`python -m cases.summary` runs every standalone RED/GREEN script in `cases/`
and grades it against `cases/markers.py`'s known-bad table
([corpus.md](corpus.md)). Pass `--no-cache` on a fresh machine's *first* run:
the observation cache lives under the system temporary directory and is keyed
by this checkout's path, not by machine identity, so a box that was ever used
for another checkout — or a container image built from one — could already
have a file sitting there from a run this box never actually did. Real
output, captured here on aarch64:

```console
$ python -m cases.summary --no-cache
### torch-compile-check regression corpus -- torch 2.14.0+cpu (git 08187d9e0fba), python 3.10.12, aarch64

| Case | Issue | Oracle | Observed | Expected | Agrees |
|---|---|---|---|---|---|
| `alias_slice_scatter_copyback` | #195451 | alias | RED | RED | yes |
| `alias_noop_view_identity` | #191449 | alias | RED | RED | yes |
| `dtype_int8_matmul_promotion` | #191308 | metadata | RED | RED | yes |
| `distributions_validation_branch` | #194593 | graph | RED | RED | yes |
| `numerics_cpu_inductor_miscompile` | #190765 | numerics | GREEN | GREEN | yes |

5 cases: 5 agree with the marker, 0 disagree, 0 could not be placed. 0 of the 5 were reused: the observation cache is off.
```

A disagreement here is a note, not a failure (a nightly that fixes a bug
upstream must not turn the box red) — but a disagreement that differs
*between architectures* on the *same* torch version and the *same* git hash
is exactly the kind of fact this runbook exists to surface. Run it on both
boxes and compare the `Observed` column by hand first; it is five lines and
catches the coarse case before any JSON is involved.

The cache never stores a crash: a case that exits 2 is `UNKNOWN`, and an
`UNKNOWN` verdict is never written, so it cannot be replayed once whatever
broke the run is fixed. This matters most on exactly the kind of box this
runbook is about — a second machine, freshly set up, where a missing compiler
or absent Python headers can make every case in the corpus exit 2 the first
time round. Setting up the box (step 1 above) fixes that, but the observation
cache does not know it happened; if a first attempt at this step ran *before*
the box was fully set up, clear the stale file —
`rm /tmp/torch-compile-check-observations-*.json` — or pass `--no-cache`
again, rather than trust a rerun that could otherwise (pre-M4-6) have quietly
replayed the earlier crash as this run's answer.

## 3. Produce comparable JSON

Two ways to get a JSON artifact, and they answer slightly different
questions.

**The validation summary** (`python validation/run.py`, [validation.md](validation.md))
runs the six public-model targets under `validation/targets/` and writes
*one* combined file, `validation/results/<date>-<arch>-<torch version>.json` —
architecture-named by construction, so a run on a second machine with a
different architecture, torch version or date lands next to the committed
aarch64 one. A run that matches all three (this box on the same day)
overwrites the committed file instead: commit or `git checkout --` it
deliberately.

```console
$ python validation/run.py --targets train_step_mlp
wrote validation/results/2026-09-02-aarch64-2.14.0+cpu.json
wrote docs/validation.md
train_step_mlp: clean
```

(`--targets train_step_mlp` above is this page proving the mechanism fast
and without the `torchvision`/`transformers` extras; a real second-machine
run omits `--targets` to run all six. Note for anyone reusing this exact
transcript: `validation/run.py` also regenerates `docs/validation.md` in
place — expected on a machine standing up its own validation table, but not
something this slice's own worktree keeps, since regenerating that table
company-wide is M4-3's job, not this page's.)

That file does not carry `schema_version` or a `findings` array — it is
`validation/run.py`'s own summary shape (`date`, `environment`, one
`{name, status, exit_code, findings_by_oracle, stage, seconds}` row per
target), built for the human-readable table in `docs/validation.md`, not for
a schema-level diff.

**Per-target torch-compile-check JSON** is what actually carries
`schema_version`, `environment.machine`, `environment.cuda_available`, and a
`findings` array per PLAN.md's schema ([reports.md](reports.md)) — call
`torch-compile-check` directly, once per target, with `--json`:

```console
$ mkdir -p results/aarch64
$ for f in validation/targets/*.py; do
    name=$(basename "$f" .py)
    torch-compile-check "$f" --fp64-oracle --json "results/aarch64/${name}.json" || true
  done
```

(`|| true`: a target whose extra package is missing, or that produces a
real finding, exits non-zero, and the loop should still produce every JSON
it can rather than stopping at the first one.) This is "the equivalent that
writes per-target JSON" — `validation/run.py` has no `--json-dir` flag, so
per-target artifacts come from the CLI's own `--json`, run per file, into an
architecture-named directory of your choosing. A real one, captured here:

```console
$ torch-compile-check validation/targets/train_step_mlp.py --json results/aarch64/train_step_mlp.json
torch-compile-check: wrote results/aarch64/train_step_mlp.json (--json)
...
stage
  clean: no backend diverged from eager across 2 lanes
```

## Cross-architecture results (2026-09-03)

The single-host results table in `docs/validation.md` is the aarch64
reference, committed as the per-target set under
`validation/results/per-target/aarch64-2.14.0+cpu/`. The same six
targets were then run on two more machines to confirm a verdict is a
property of the model and compiler, not of the host. The tool name differs
across the rows only because the distribution was renamed from
`compile-check` to `torch-compile-check` mid-round (PR #20) -- that changes
`tool.name`, which the parity comparison does not read, and nothing else.
Parity is `diff_parity.py` (below) against the aarch64
per-target set: equal `schema_version` and identical `findings` sets, with
`environment.machine` and `environment.cuda_available` printed but not
gated. (This section is maintained by hand; `validation/run.py` regenerates
only the single-host table in `docs/validation.md`.) The `target.file` field
in every per-target JSON referenced below was normalised to a repo-relative
path (`validation/targets/<name>.py`) on 2026-09-03, replacing a
contributor's absolute home-directory path that a discovery bug used to
write; nothing else in these files changed, and `diff_parity.py` (below)
never reads that field, so every parity verdict already recorded here still
holds.

| Leg | Machine | torch | Python | Tool @ commit | Result | Per-target JSONs |
|---|---|---|---|---|---|---|
| aarch64 CPU (reference) | estate, aarch64 | `2.14.0+cpu` (git `08187d9e0fba`) | 3.10.12 | `torch-compile-check 0.1.0` @ `570b789` (content-identical to main) | 6/6 clean, exit 0, 0 findings | `per-target/aarch64-2.14.0+cpu/` |
| x86_64 CPU | ProBook, AVX2 | `2.14.0+cpu` | 3.12.14 | `compile-check 0.1.0` @ `4caf42c` | 6/6 parity holds, exit 0, 0 findings | `per-target/x86_64-2.14.0+cpu/` |
| x86_64 CPU (Omen leg) | Omen, Ryzen 4800H | `2.14.0+cu126` | 3.12.13 | `compile-check 0.1.0` @ `dcaf77d` | 6/6 clean, exit 0, 0 findings | `per-target/x86_64-2.14.0+cu126-cpu/` |
| x86_64 + CUDA sm_75 | Omen, GTX 1660 Ti | `2.14.0+cu126` | 3.12.13 | `compile-check 0.1.0` @ `dcaf77d` | 6/6 clean, exit 0, 0 findings | `per-target/x86_64-2.14.0+cu126-cuda/` |

The aarch64 and ProBook legs are diffed from committed JSONs here
(`diff_parity.py`, 6/6 `parity holds`; transcript below).
The two Omen legs (CPU and CUDA, torch `2.14.0+cu126`) were first
transcribed from Turing's 2026-09-03 00:56 UTC report and are now committed
as physical per-target JSONs alongside the other two legs
(`per-target/x86_64-2.14.0+cu126-cpu/` and
`per-target/x86_64-2.14.0+cu126-cuda/`). All 12 pairs (6 targets x CPU and
CUDA) hold parity against the aarch64 reference by the same `diff_parity.py`
used above -- `environment.machine` and `environment.cuda_available` print
`DIFFERENT` on every pair (`aarch64` vs `x86_64`, `False` vs `True`) and are
not gated, same as the ProBook comparison; a second transcript, for a CUDA
pair, is below. The Omen runs also added `--fp64-oracle`
(`run.fp64` is `true` in all twelve of these files) while the aarch64 and
ProBook runs did not; the parity comparison above does not read that
field.

## 4. Diff two JSON results

Compare `schema_version` first — a document written by a different
`torch-compile-check` version may not carry the same field set — then
`environment.machine`, `environment.cuda_available`, and the `findings`
list. A minimal script that does exactly those four things and nothing
else (run wall time and the torch git hash are expected to differ between
two runs and are deliberately not compared):

```python
# diff_parity.py
import json, sys


def load(path):
    with open(path) as f:
        return json.load(f)


def main(path_a, path_b):
    a, b = load(path_a), load(path_b)
    if a["schema_version"] != b["schema_version"]:
        print(f"schema_version differs: {a['schema_version']} vs {b['schema_version']}")
        return 1
    for key in ("machine", "cuda_available"):
        va, vb = a["environment"][key], b["environment"][key]
        print(f"environment.{key}: {va!r} vs {vb!r} ({'same' if va == vb else 'DIFFERENT'})")
    fa = {(f["oracle"], f["backend"], f["output_index"], f["severity"]) for f in a["findings"]}
    fb = {(f["oracle"], f["backend"], f["output_index"], f["severity"]) for f in b["findings"]}
    if fa == fb:
        print(f"findings: {len(fa)} on each side, identical set -- parity holds")
        return 0
    print(f"findings: {len(fa)} vs {len(fb)}, sets differ -- NOT parity")
    for item in sorted(fa - fb):
        print(f"  only in a: {item}")
    for item in sorted(fb - fa):
        print(f"  only in b: {item}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
```

Run against a second machine's `results/<arch>/<target>.json` and this
repository's own committed run for the same target
(`validation/results/2026-09-02-aarch64-2.14.0+cpu.json` is the summary
file, not per-target — regenerate a per-target aarch64 baseline with step 3
above once, commit it, and diff against that). Two real runs of the script
below show what "parity holds" and "NOT parity" each look like. The first
is a genuine cross-architecture pair — the committed aarch64 reference
against the ProBook's x86_64 run of the same target — so
`environment.machine` reads `DIFFERENT`, is printed, and is not gated; the
finding sets are identical, so parity holds:

```console
$ python diff_parity.py validation/results/per-target/aarch64-2.14.0+cpu/train_step_mlp.json validation/results/per-target/x86_64-2.14.0+cpu/train_step_mlp.json
environment.machine: 'aarch64' vs 'x86_64' (DIFFERENT)
environment.cuda_available: False vs False (same)
findings: 0 on each side, identical set -- parity holds
```

All six targets read `parity holds` this way across the aarch64 reference
and the ProBook x86_64 set (2026-09-03; recorded per leg in
`docs/validation.md`). The second is the CUDA leg -- the aarch64 reference
against the Omen's CUDA run of the same target -- so
`environment.cuda_available` reads `False` vs `True`, is printed, and is
not gated, same as `environment.machine` above:

```console
$ python diff_parity.py validation/results/per-target/aarch64-2.14.0+cpu/train_step_mlp.json validation/results/per-target/x86_64-2.14.0+cu126-cuda/train_step_mlp.json
environment.machine: 'aarch64' vs 'x86_64' (DIFFERENT)
environment.cuda_available: False vs True (DIFFERENT)
findings: 0 on each side, identical set -- parity holds
```

All 12 Omen pairs (6 targets x CPU and CUDA) read `parity holds` the same
way against the aarch64 reference (recorded per leg in
`docs/validation.md`). The next run below is two *different* targets on
one machine, kept only to prove the "NOT parity" branch fires on a real
mismatch rather than being untested code:

```console
$ torch-compile-check cases/dtype_promotion.py --json results/aarch64/dtype_promotion.json
$ python diff_parity.py results/aarch64/train_step_mlp.json results/aarch64/dtype_promotion.json
environment.machine: 'aarch64' vs 'aarch64' (same)
environment.cuda_available: False vs False (same)
findings: 0 vs 1, sets differ -- NOT parity
  only in b: ('metadata', 'inductor', 0, 'fail')
```

Comparing the *same* target across two machines is the real use of this
script; the cross-architecture pair above is that use in earnest.

## What "parity" means

Parity, in v1, is a fact about one target on two runs: the same
`schema_version`, and identical `findings` sets once `environment.machine`
and `environment.cuda_available` are read off to know what was actually
compared. It does **not** mean identical timings (`first_call_s`,
`second_call_s` are expected to differ, sometimes by an order of magnitude,
between an aarch64 and an x86 box and are not part of the comparison), and
it does not mean identical `stage` wording if the finding set itself is
identical but the ablation ladder's backend list differs between the two
runs (`--backends` set differently on either side breaks the comparison
before it starts — keep the flag identical on both machines for a
comparison to mean anything).

Two outcomes both matter and are both worth recording, not just the
disagreement:

- **Agreement** is itself the evidence for a claim this project makes and
  that (as PLAN.md notes) nothing else currently answers off the shelf: "the
  same model compiles to the same answers on this architecture as on that
  one."
- **Disagreement** is the more valuable of the two findings. A silent
  divergence that only one architecture shows — the shape issue
  [#191837](https://github.com/pytorch/pytorch/issues/191837) demonstrates —
  is exactly the case a single-architecture CI run cannot catch on its own,
  since there is nothing on that one architecture to compare against.

A first-class `torch-compile-check compare a.json b.json` subcommand that
automates the diff above is PLAN.md's v0.2 outlook; v1 is this runbook and
the two-run-and-`diff`-the-artifacts workflow it walks through.
