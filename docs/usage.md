# Usage

Every flag `torch-compile-check --help` lists, with one real example each. All
output below is verbatim from running the command in a fresh venv on
aarch64 (torch `2.14.0+cpu`), trimmed to the section that shows what the
flag changed — see [PLAN.md "CLI surface for v1"](../PLAN.md) for the
design these flags implement, and run `torch-compile-check --help` for the
one-line form of everything here.

```console
$ torch-compile-check path/to/model.py [options]
```

`path` is a Python file following the [discovery convention](#discovery-overrides):
a module-level `model` or `fn`, and a module-level `inputs` or `get_inputs()`.
Exit codes: `0` clean, `1` a `--fail-on` finding (or a compiled lane that
raised while eager did not), `2` a tool error.

A target whose import raises -- including a missing optional dependency,
such as `transformers` for a target like `validation/targets/hf_tiny_bert.py`
-- is a tool error, exit `2`: `torch-compile-check: importing
hf_tiny_bert.py raised ModuleNotFoundError: No module named 'transformers'`,
because there is nothing to compare without a successful import.
`validation/run.py` runs a whole suite of targets and reports that one as
"skipped" instead (`docs/validation.md` "Extras"), so one missing extra
does not abort the rest; `pip install "torch-compile-check[validation]"`
installs what those targets need.

## `--version`, `--probe`

```console
$ torch-compile-check --version
torch-compile-check 0.1.0
```

`--probe` prints whether every torch API the oracles read (PLAN.md "Verified
API surface") is present on the installed wheel, so a torch upgrade that
removed or renamed one shows up before a run does:

```console
$ torch-compile-check --probe
api                                               status
------------------------------------------------  -------
torch.testing.assert_close                        present
torch._dynamo.explain                             present
torch._inductor.config.repro_after                absent
torch._dynamo.CompileProfiler                     absent
...
```

`torch._inductor.config.repro_after` and `torch._dynamo.CompileProfiler`
report `absent` on every torch version checked so far — PLAN.md records why
next to each: the outline that suggested the first one was wrong (the knob
is on `torch._dynamo.config`, not `torch._inductor.config`), and the second
is the fallback path M3's graph oracle uses instead.

## Discovery overrides

### `--entry module:callable`, `--inputs module:callable`

Override discovery when the file's symbols are not named `model`/`fn` or
`inputs`/`get_inputs()`. The module half names the file itself (its dotted
name or its file stem) or another importable module; the attribute half can
walk dotted attributes, so a callable nested in a class or a namespace
resolves too. An empty module half (just `--entry callable_name`) means
"this file".

```console
$ torch-compile-check tests/fixtures/named_entry.py --entry net --inputs bundle.make_inputs
torch-compile-check 0.1.0   target named_entry:net
...
stage
  clean: no backend diverged from eager across 2 lanes
```

`tests/fixtures/named_entry.py` defines `net` (an `nn.Module`, not `model`)
and a namespace `bundle` whose `make_inputs` staticmethod is the input
factory — neither name is one discovery looks for on its own, which is what
makes this the override fixture.

## Run configuration

### `--backends LIST`

Comma-separated, default `eager,aot_eager,inductor`. `aot_eager_decomp_partition`
is an optional fourth lane, added when a finding needs the extra stage split
between decomposition and partitioning (PLAN.md "Stage localization"):

```console
$ torch-compile-check tests/fixtures/mlp.py --backends eager,inductor
checks
  oracle    fail-on  inductor
  numerics  yes      pass
  alias     yes      pass
  metadata  yes      pass
  grad      yes      pass
  graph     no       pass

stage
  clean: no backend diverged from eager across 1 lane
```

Dropping `aot_eager` from the ladder is also how a divergence gets reported
against fewer lanes than the default three — the "1 lane" above, versus "2
lanes" everywhere else in this page, is `--backends` at work, not a
different tool.

### `--device {cpu,cuda}`

Default `cpu`. `cuda` needs a CUDA-capable box; see
[cross-arch.md](cross-arch.md) for the runbook and what an existing
divergence looked like differently across architectures (issue
[#191837](https://github.com/pytorch/pytorch/issues/191837)).

```console
$ torch-compile-check tests/fixtures/mlp.py --device cpu
stage
  clean: no backend diverged from eager across 2 lanes
```

`cpu` is already the default, so this run is identical to one with no
`--device` flag at all — shown for the syntax, not for a different report.

### `--fullgraph`

Passes `fullgraph=True` to `torch.compile`, so a lane that cannot be traced
as one graph raises instead of falling back to eager for the untraceable
part. `cases/distributions_binomial_kl.py` breaks on data-dependent
branching inside `torch.distributions.kl`'s validation path (issue
[#194593](https://github.com/pytorch/pytorch/issues/194593)):

```console
$ torch-compile-check cases/distributions_binomial_kl.py --fullgraph
backends
  backend    outputs   first call  second call  status
  eager            1      0.0060s      0.0005s  ok
  aot_eager        0      0.2610s            -  raised Unsupported
  inductor         0      0.0449s            -  raised Unsupported

findings
  graph  (2 fail)
    [fail] aot_eager run
        aot_eager broke the graph at .../torch/distributions/kl.py:235
        in _kl_binomial_binomial: gb0170: Data-dependent branching;
        --fullgraph was requested and the graph broke anyway, so this
        lane could not be captured as one graph

stage
  first diverges at aot_eager, which implicates capture/AOTAutograd/decomposition
```
Exit code 1 (a compiled lane raised while eager did not — always a
divergence, regardless of `--fail-on`).

### `--dynamic`

Adds a second full pass per backend with `dynamic=True`, reported
separately, so a dynamic-shape-only divergence is distinguishable from a
static-shape one. `cases/numerics_polyjuice_minmax.py` (issue
[#190765](https://github.com/pytorch/pytorch/issues/190765), fixed upstream
by [PR #190966](https://github.com/pytorch/pytorch/pull/190966)) is clean
here, which is the expected result on this torch build:

```console
$ torch-compile-check cases/numerics_polyjuice_minmax.py --dynamic
findings
  none

stage
  clean: no backend diverged from eager across 2 lanes
```

### `--no-grad`

Skips the backward pass. Without it, one backward runs on a deterministic
scalar reduction whenever any input or parameter requires grad. The grad
oracle reports that it did not run rather than reporting a silent pass, so
a clean grad row never means two different things:

```console
$ torch-compile-check tests/fixtures/mlp.py --no-grad
run       backends eager, aot_eager, inductor   seed 0   fullgraph off   dynamic off   grad off
gradients not compared (--no-grad)

findings
  grad  (2 info)
    [info] aot_eager run
        gradients were not compared: --no-grad switched the backward pass off
        field grad_disabled
```

### `--share-module`

Runs every backend against one module object instead of giving each lane
its own deep copy, trading isolation (a buffer the forward pass writes —
BatchNorm running statistics, a step counter — can leak between lanes) for
the memory of one fewer copy of the weights, on a model too large to
duplicate:

```console
$ torch-compile-check tests/fixtures/mlp.py --share-module
run       ... module    shared across every lane (--share-module)
findings
  none
```

## Tolerances

### `--rtol RTOL`, `--atol ATOL`

Override the numerics relative and absolute tolerance for every dtype,
replacing the per-dtype defaults from `torch.testing._comparison.default_tolerances`
(PLAN.md "Verified API surface"). Widening them (as below) can only ever
turn a finding into a pass, never the reverse:

```console
$ torch-compile-check tests/fixtures/mlp.py --rtol 0.5
$ torch-compile-check tests/fixtures/mlp.py --atol 1.0
```

Both runs stayed clean on this build, since `mlp.py` has nothing to fail
under the default tolerances either; the flags are shown for the syntax, not
for a divergence they masked.

### `--grad-tol-factor N`

What the grad oracle multiplies the numerics tolerances by before comparing
gradients — default `10`, measured (M2-3) against a torchvision resnet18
backward at 1x in eval mode and about 161x in train mode, where BatchNorm
sends every gradient back through batch statistics. `1` compares gradients
exactly as outputs are compared:

```console
$ torch-compile-check tests/fixtures/mlp.py --grad-tol-factor 1
gradients compared at the numerics tolerances (--grad-tol-factor 1)
findings
  none
```

At the default `10` the same line reads "compared at the numerics
tolerances x10 (--grad-tol-factor 10)"; the multiplier is only spelled out
when it changes the comparison.

### `--fp64-oracle`

Adds a third, float64 eager reference (a deep-copied, `.double()`-d module
with widened floating inputs), so the numerics oracle can tell "compiled is
wrong" apart from "both eager and compiled are imprecise at fp32" — PLAN.md
"The oracle blind spot". Reported at `info` when eager itself is off the
fp64 reference, since that is a precision note rather than a compiler
finding:

```console
$ torch-compile-check tests/fixtures/mlp.py --fp64-oracle
backends
  backend                 outputs   first call  second call  status
  eager                         1      0.0002s      0.0001s  ok
  aot_eager                     1      0.9963s      0.0003s  ok
  inductor                      1      3.6018s      0.0003s  ok
  eager_fp64 (reference)        1      0.0011s      0.0001s  ok

findings
  none
```

`eager_fp64` is a reference row, not a lane under test: it never appears on
the ablation ladder and cannot itself be the backend a stage verdict names.

### `--seed SEED`

RNG seed, default `0`. Applied before the target module is imported — so a
model built at module scope draws the same weights every run — and again
before every backend:

```console
$ torch-compile-check tests/fixtures/mlp.py --seed 42
findings
  none
```

### `--allow-caches`

Does not set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`. The run gets faster and
starts measuring whatever an earlier run cached instead of the current
compiler; the report records which mode was in force either way:

```console
$ torch-compile-check tests/fixtures/mlp.py --allow-caches
caches    ENABLED (force_disable_caches=False, --allow-caches)
findings
  none
```

## Artifacts

### `--json OUT.JSON`, `--md REPORT.MD`, `--emit-test TEST.PY`

Three artifacts off one run — see [reports.md](reports.md) for the JSON
schema and the shape of the other two:

```console
$ torch-compile-check cases/dtype_promotion.py --json out.json --md draft.md --emit-test test_case.py
torch-compile-check: wrote out.json (--json)
torch-compile-check: wrote draft.md (--md)
torch-compile-check: wrote test_case.py (--emit-test)
...
findings
  metadata  (1 fail)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
```
Exit 1 (`metadata` is in the default `--fail-on`). A clean run writes no
`--emit-test` file and says which of the two reasons it was — no finding, or
only `warn`/`info`-severity ones. `out.json` validates against the schema
version [reports.md](reports.md) documents; `test_case.py` runs on its own
(it is `unittest`, `pytest test_case.py` also works) and, for this
particular target, currently *fails* when run — the bug it encodes
([#191308](https://github.com/pytorch/pytorch/issues/191308)) is still open
upstream, so the emitted assertion is doing its job.

## Deciding the exit code

### `--fail-on LIST`

Which oracle categories turn a finding into exit code 1, comma-separated
from `numerics,alias,metadata,grad,graph`; default
`numerics,alias,metadata,grad`. It selects which findings can fail the run,
never which oracles run — every oracle's row is in the report either way:

```console
$ torch-compile-check cases/dtype_promotion.py --fail-on metadata
findings
  metadata  (1 fail)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
```
Exit `1`.

```console
$ torch-compile-check cases/dtype_promotion.py --fail-on numerics
findings
  metadata  (1 fail)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
```
Same finding, same report — exit `0`, because `metadata` was left out of
`--fail-on`. This is the pair to run side by side if a flag's effect on the
exit code needs demonstrating.

## The minimizer

### `--minimize`, `--budget SECONDS`

After a finding, re-runs the eager lane and the one that diverged against
smaller and smaller versions of the case (PLAN.md "Minimizer, v1"): halve
the leading input dimension, replace child modules with `torch.nn.Identity()`
one at a time, keep every change the finding survives. `--budget` bounds the
pass — a ceiling of 100 candidates applies when it is not given — and a run
that hits either is reported as **partial**, never as a smallest case: the
report says which ceiling ran out rather than claiming the case would not
shrink. `--budget 0` is allowed and means "start no candidate"; a negative
value, or `nan`, is a tool error naming the flag rather than a ceiling that
is silently already spent (or silently absent).

```console
$ torch-compile-check cases/alias_copyback.py --minimize --budget 60
findings
  alias  (1 fail)
    [fail] inductor output[0]
        inductor returned input[0] itself as output[0] and eager returned a distinct object

minimized
  finding   [fail] alias inductor output[0]   field identity_added
  inputs    leaf 0  (2,) -> (1,)
  notes     the target is a plain callable, so it has no children to replace
  cost      1 candidate re-run in 1.2s
  minifier  torch's accuracy minifier (TORCHDYNAMO_REPRO_AFTER=aot TORCHDYNAMO_REPRO_LEVEL=4)
            compares numbers only, so it would not isolate this alias finding
```
Exit `1`. The `minifier` line is `handoff_note`: the exact environment
variables for torch's own accuracy minifier, handed over as a note and never
run, and an explanation of why it would not have helped here (it is a
numerics-only tool; this is an alias finding).

## Graph baseline

### `--write-baseline FILE`, `--baseline FILE`

`torch.compile` graph breaks are informational by default (PLAN.md
"graph"). `--write-baseline` records this run's graph health; a later run
passing the same file as `--baseline` fails on breaks that are **new**
relative to it, not on every break — the mode
[the GitHub Action](action.md#baseline-semantics) uses. `tests/fixtures/graph_break.py`
has two deliberate breaks:

```console
$ torch-compile-check tests/fixtures/graph_break.py --write-baseline baseline.json --fail-on graph
torch-compile-check: wrote the graph baseline baseline.json (aot_eager, inductor)
findings
  graph  (4 info)
    [info] aot_eager run
        aot_eager broke the graph at tests/fixtures/graph_break.py:31 in fn: gb0059: ...
```
Exit `0` — the breaks are new against no prior baseline, so `--write-baseline`
records them without failing this run.

```console
$ torch-compile-check tests/fixtures/graph_break.py --baseline baseline.json --fail-on graph
run       ... baseline  baseline.json   (the graph oracle reports new breaks only)
findings
  none
```
Exit `0` — same two breaks, now in the baseline, so nothing is new.

## Display

### `--max-findings N`

How many findings to print per oracle group, default `10`; the rest are
counted, never dropped:

```console
$ torch-compile-check tests/fixtures/graph_break.py --max-findings 1 --fail-on graph
findings
  graph  (4 info)
    [info] aot_eager run
        aot_eager broke the graph at tests/fixtures/graph_break.py:31 in fn: gb0059: ...
    3 more graph findings not shown (--max-findings 1)
```

### `--color {auto,always,never}`

Default `auto` — colour when stdout is a terminal and `NO_COLOR` is unset.
`never` is what every example on this page runs with, so the trimmed output
above is exactly what was captured, ANSI codes included or not:

```console
$ torch-compile-check tests/fixtures/mlp.py --color never
$ torch-compile-check tests/fixtures/mlp.py --color always
```

Both exit `0`; `--color always` is the one whose terminal output carries
ANSI escapes even when piped, which this page does not reproduce literally
since a fenced code block cannot show colour.

## Environment variables

Two environment variables the tool itself reads, from `torch-compile-check --help`:

- `TORCHINDUCTOR_FORCE_DISABLE_CACHES` — set to `1` by this tool before torch
  is imported, unless `--allow-caches` was passed.
- `TORCHINDUCTOR_CACHE_DIR` — where inductor writes generated code. Disabling
  the caches stops torch *reading* that directory, not writing to it, and
  nothing prunes it; point this at a scratch directory if disk is tight.
