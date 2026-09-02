# compile-check
Version: v0.1 (draft for Chairman review)
Date: 2026-09-02
Author: Aesop (CEO) for Hussain Nizamani
Status: scope lock pending
Planning document only. No implementation code exists yet.

## Positioning

Bring your own model; compile-check tells you whether torch.compile changed its
answers, and if so hands you a minimal repro and a ready-to-file report.

The tool is a one-command trust auditor for `torch.compile`. The user points it at
a model or a function together with real inputs. It runs the same callable under
eager, `aot_eager`, and `inductor`, then reports whether compilation silently
changed the result.

`torch.compile` is a compiler. compile-check is differential testing for it, in the
same sense that Csmith is differential testing for GCC and LLVM, with the difference
that the program under test is the user's own model rather than a generated one.

### Definitions

A model, for this tool, is the triple of callable, weights, and example inputs that
is handed to `torch.compile`.

The compile contract is that `compile(f)(x)` is observationally equivalent to `f(x)`
except for speed. Concretely: the same numbers, the same aliasing and mutation
behaviour, the same dtype, shape, and stride, the same gradients, and no silent
fallback. Each of the five oracles checks one clause of that contract.

## Why this project, and why us

Each oracle exists because a real correctness bug in that class was found and root
caused by the author while working upstream.

Credential, stated precisely: three merged `torch.compile` fixes (Dynamo, PR 190673;
Inductor, PR 192628; AOTAutograd aliased-output identity, PR 191844, merged
2026-09-02), one open Inductor fix (PR 195484, for the reinplacing alias bug 195451),
and several reproduced and root-caused compile correctness bugs that are not yet
fixed by us (191308, 194593, 194596, 190765). Separately, PR 192667 (linalg
second-order forward AD, approved by lezcano) is an eager autograd fix, not a compile
fix. It is a credential for the author, and it is deliberately not in the table
below, because a tool that uses eager as its oracle could not have found it.

| Bug class | Upstream reference | Status | Oracle that catches it |
|---|---|---|---|
| Inductor reinplacing breaks an output-aliases-input contract | issue 195451, PR 195484 | open | alias |
| AOTAutograd aliased-output identity | issue 191449, PR 191844 | merged 2026-09-02 | alias |
| Silent dtype promotion, int8 matmul becomes int64 on CPU inductor | issue 191308 | reproduced, root caused | metadata |
| Distributions crossed with compile, validation branches diverge | issues 194593, 194596 | reproduced, root caused | numerics, graph |
| CPU inductor miscompile, wrong values with no error | issue 190765 | reproduced, root caused | numerics |

Name availability: `compile-check` is free on PyPI as of 2026-09-02. `torchcheck` is
taken.

## Relationship to PyTorch's built-in tools

PyTorch already tests compile correctness, thoroughly. This section is honest about
what exists, and it is destined for the README as well as this plan. The claim is not
that nothing like this exists; it is that none of it runs on the user's model.

| Existing tool | What it does | Gap for a user |
|---|---|---|
| `test/inductor/test_torchinductor_opinfo.py` | OpInfo-driven eager versus compiled tests | per operator, on PyTorch's commits, not on a composed user model |
| `benchmarks/dynamo/*.py --accuracy` | accuracy runs with an fp64 eager reference over a fixed model zoo | fixed zoo; your model is not in it |
| `dynamo_wrapped` CI shards | reruns the PyTorch test suite under compile | PyTorch's programs, not yours |
| accuracy minifier (`TORCHDYNAMO_REPRO_AFTER=dynamo\|aot`, `TORCHDYNAMO_REPRO_LEVEL=4`) | shrinks an FX graph that already fails | opt-in, after the fact, numerics only, can end with "Input graph did not fail the tester" |
| `torch._dynamo.explain` | graph breaks and guards | diagnostic, not a pass or fail verdict |
| `TORCH_LOGS=graph_breaks,recompiles` | logs breaks and recompiles | logs to read, nothing to assert on |
| backend ablation ladder `eager` to `aot_eager` to `aot_eager_decomp_partition` to `inductor` | stage localization | a manual workflow, done by hand |

The gaps, summarized: these run on PyTorch's commits and PyTorch's programs, not on
your model or your torch upgrade; they are opt-in and after the fact; they are
numerics only; they work at the FX graph level rather than at your source; the
minifier can fail to isolate; there is no unified pass or fail across aliasing, dtype,
stride, and gradients; and none of it is in your CI.

compile-check productizes the workflow the PyTorch team does by hand (ablate backends,
compare, minify, file) into one command with five checks, moved from "after you
noticed" to "before you merged". The adjacent third-party work is random-graph
research fuzzing (NNSmith and similar) and an academic bug-study corpus, which
generate synthetic graphs; compile-check audits the model the user already has.

## Verified API surface

Every torch API named in this document was checked against the installed wheel
before being written down.

Environment used for verification: torch `2.15.0.dev20260824+cpu`, git
`c0577575187a039c482a985e9a594816dc711a4c`, Python 3.10.12, Linux aarch64, CPU only.

| API | Status on 2.15.0.dev20260824 | Note |
|---|---|---|
| `torch.testing.assert_close` | present | kwargs: `rtol`, `atol`, `equal_nan`, `check_device`, `check_dtype`, `check_layout`, `check_stride`, `msg` |
| `torch.testing._comparison.default_tolerances` | present, private | source of the per-dtype default table |
| `torch._dynamo.reset`, `torch.compiler.reset` | both present | per-backend isolation, prefer the public `torch.compiler` one |
| `torch._dynamo.explain` | present | returns `ExplainOutput` |
| `ExplainOutput` fields | `graphs`, `graph_count`, `graph_break_count`, `break_reasons`, `op_count`, `ops_per_graph`, `out_guards`, `compile_times` | all confirmed by dataclass introspection |
| `torch._dynamo.list_backends` | present | signature `(exclude_tags=('debug','experimental'))` |
| backends `eager`, `aot_eager`, `aot_eager_decomp_partition`, `inductor` | all four registered | the full ablation ladder is available by name |
| `TORCH_LOGS` artifacts `graph_breaks`, `recompiles`, `recompiles_verbose` | all registered | confirmed in `torch/_logging/_registrations.py` |
| `torch._dynamo.utils.counters` | present | `counters['stats']` carries `calls_captured` and `unique_graphs` |
| `torch._dynamo.utils.compile_times`, `.same` | both present | timing report, and the internal numeric comparison helper |
| `torch._dynamo.config.repro_after`, `repro_level`, `repro_tolerance` | present, defaults `None`, `2`, `0.001` | read directly from env `TORCHDYNAMO_REPRO_AFTER` and `TORCHDYNAMO_REPRO_LEVEL` at `torch/_dynamo/config.py:332` and `:340`; `repro_dir` is absent, do not reference it |
| `torch._inductor.config.repro_after` | absent | the outline suggested this knob; it does not exist, the dynamo one does |
| `torch._dynamo.repro.after_aot`, `.after_dynamo` | present as modules | `after_aot` exports `AccuracyError`, `ACCURACY_FAILS`, `InputReader`, `InputWriter` |
| `torch._dynamo.debug_utils.same_two_models`, `.backend_accuracy_fails` | both present | |
| `torch._inductor.config.force_disable_caches` | present | reads `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, verified by setting the env var |
| `torch._debug_has_internal_overlap` | present | returns int, `0` for a non-overlapping tensor |
| `Tensor.untyped_storage().data_ptr()`, `.nbytes()` | present | data pointer verified equal for a tensor and its transpose; nbytes needed for the byte-range overlap test |
| `Tensor.storage_offset`, `Tensor._base`, `Tensor._is_view` | present | |
| `torch.compile` keyword arguments | `fullgraph`, `dynamic`, `backend`, `mode`, `options`, `name`, `disable`, `recompile_limit`, `isolate_recompiles`, `dynamic_shapes` | full signature confirmed |
| `torch._dynamo.CompileProfiler` | absent at `torch._dynamo` and `torch._dynamo.utils` | to verify against older and newer torch before relying on it; the counters path is the fallback and is what M3 will use |

Two facts worth carrying into the implementation. First, `eager` and `aot_eager`
carry the `debug` tag, so the default `list_backends()` call does not list them; a
`--backends` validator must call `list_backends(exclude_tags=())`. Second, the
per-dtype defaults behind `assert_close` are, as measured: float16 rtol 1e-3 atol
1e-5, bfloat16 rtol 1.6e-2 atol 1e-5, float32 rtol 1.3e-6 atol 1e-5, float64 rtol
1e-7 atol 1e-7, integer and bool dtypes exact.

## CLI surface for v1

```
compile-check path/to/file.py [options]
```

| Flag | Meaning |
|---|---|
| `--entry module:callable` | override discovery, name the model or function directly |
| `--inputs module:callable` | override discovery, name the input factory |
| `--backends eager,aot_eager,inductor` | which backends to run, default these three; `aot_eager_decomp_partition` is an optional fourth lane for finer stage localization |
| `--device cpu\|cuda` | device to place the model and inputs on, default cpu |
| `--json out.json` | write the versioned JSON result |
| `--md report.md` | write the Markdown issue draft |
| `--emit-test test_case.py` | write the top finding as a regression test in the inductor suite's eager-vs-compiled idiom (M3-2) |
| `--fail-on numerics,alias,metadata,grad,graph` | which oracle categories turn a finding into exit code 1 |
| `--fullgraph` | pass `fullgraph=True` to `torch.compile`, default is False |
| `--dynamic` | add a second pass with `dynamic=True` |
| `--rtol`, `--atol` | override the numerics tolerances for every dtype |
| `--seed` | RNG seed, default fixed |
| `--allow-caches` | do not set `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1` |
| `--fp64-oracle` | add an fp64 eager reference run to the numerics oracle, see the blind spot section |
| `--minimize` | after a finding, shrink the case while it still reproduces: halve the leading input dimension and replace child modules with a passthrough, then report what is left (M3-3) |
| `--budget SECONDS` | wall-clock ceiling for the minimizer, for CI use; v1 bounds what `--minimize` starts, since a compile that has begun cannot be interrupted without killing the process |
| `--baseline FILE` | a stored graph-health baseline, so the graph oracle fails on new breaks rather than on any break |
| `--no-grad` | skip the backward pass and the grad oracle for this run |
| `--max-findings N` | cap the findings printed per oracle group; hidden ones are counted (N >= 0) |
| `--color auto\|always\|never` | colour only on a TTY by default |

Exit codes: `0` clean, `1` at least one finding in a `--fail-on` category, `2` tool
error (import failure, discovery failure, backend unavailable, model raised in
eager).


Exit-code clauses fixed in M1-3: a compiled lane that raises while eager does not is a divergence and exits 1 regardless of `--fail-on` (an exception belongs to no oracle category); a run without an eager lane exits 2, because there is no reference to be clean against. A failing repeat call is recorded on the backend result but does not move the verdict until the graph oracle (M3) owns repeat-call health.

## Discovery convention

Given a file path with no overrides, the tool imports the module and looks for:

1. a module-level `model` (an `nn.Module`) or `fn` (a callable), in that order;
2. a module-level `inputs` (a tuple, list, or dict of tensors) or a callable
   `get_inputs()` returning the same, in that order.

If both an override flag and a discovered symbol exist the override wins. If
neither resolves, the tool exits 2 with a message naming the two symbols it looked
for. Discovery is deliberately narrow in v1: no directory walking, no pytest-style
collection, no config file.

## Runner semantics

The runner establishes that any difference it reports comes from the backend and
not from the harness.

- The RNG is seeded before every run with the same seed, covering
  `torch.manual_seed` and the Python and numpy generators when numpy is present.
- Inputs are deep cloned per backend. The clone preserves dtype, device, stride,
  and `requires_grad`, so a backend never sees a tensor another backend mutated.
- Each backend runs after `torch.compiler.reset()`, so no compiled artifact or
  guard from a previous backend is reused.
- Caches are disabled by default by setting `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`
  before torch does any compiling. Verified: this makes
  `torch._inductor.config.force_disable_caches` read `True`.
- The default is `torch.compile(fullgraph=False)`, matching what users actually
  run. `--fullgraph` switches to strict tracing.
- The eager run is the reference world. If it raises, the tool exits 2 and reports
  the model as broken before compile is involved.
- Each backend is called twice with the same inputs. The first call is the measured
  one; the second exists so the graph oracle can see whether a recompile happened.
- `--dynamic` adds a second full pass per backend with `dynamic=True` and reports
  its findings separately, so a dynamic-shape-only divergence is distinguishable.

## Stage localization

Running more than one backend is not redundancy, it is the diagnosis. Stage
localization is a first-class output: every report names both the failing check and
the compilation stage the divergence first appears in.

| Divergence first seen at | Implicated stage |
|---|---|
| `aot_eager` | Dynamo capture, AOTAutograd, functionalization, or decompositions |
| `aot_eager_decomp_partition` but not `aot_eager` | decomposition or partitioner |
| `inductor` only | lowering, scheduling, or codegen |

The optional `aot_eager_decomp_partition` lane splits the first row and is worth the
extra run when a finding lands there. It is off by default because it adds a full
compile pass for a distinction most runs will not need.

This is the ablation ladder PyTorch maintainers walk by hand when triaging a compile
bug. Doing it automatically and printing the verdict is a large part of what makes a
generated report worth reading.

### Where divergence appears is not always where the fix belongs

The stage verdict names the first backend whose output violates the contract, which
is where the divergence becomes observable, not necessarily where the defect lives.
Worked example, measured on torch 2.15.0.dev20260901 (aarch64, caches disabled): the
191449 no-op view identity bug collapses two outputs into one object under `inductor`
but not under `aot_eager`, because eager kernels return distinct view objects even when
AOTAutograd has misclassified them; the fix (PR 191844) nevertheless lives in
AOTAutograd's metadata analysis. The report therefore says "first diverges at
inductor" and never "the bug is in inductor"; the Markdown issue draft carries the
same wording, and the corpus records the first diverging backend per case so this
distinction stays visible.

## Oracles

Five oracles run against every backend result. Each has a defined comparison, a pass
rule, and a known bug it would have caught.

| Oracle | Compared | Pass rule | Known bug class |
|---|---|---|---|
| numerics | output tensor values, per output | `assert_close` within per-dtype tolerance, plus NaN and inf position parity | 190765 CPU inductor miscompile, 194593/194596 divergent validation branches |
| alias | storage identity and overlap among all outputs and inputs, input mutation set, Python object identity | the compiled relation equals the eager relation exactly | 195451 inductor reinplacing, 191449 and PR 191844 AOTAutograd aliased-output identity |
| metadata | dtype, shape, stride, `requires_grad`, device, contiguity, per output | exact equality on every field | 191308 int8 matmul silently promoted to int64 |
| grad | `.grad` of every input and parameter after one backward on a deterministic scalar reduction, and the set of tensors that received a grad at all | grad values pass the numerics rule, grad presence set is identical | backward-only divergence, partitioner bugs |
| graph | graph break count and reasons, unique graph count across a repeat call, compile wall time | informational unless `--fail-on graph` is set | recompile storms, silent fullgraph regressions |

### numerics

Outputs are flattened with the same tree walk in both worlds and compared pairwise.
Comparison is `torch.testing.assert_close` with `check_dtype=False` and
`check_stride=False`, because dtype and stride are the metadata oracle's job and
reporting the same divergence twice hides which one is the real defect. Tolerances
default to the per-dtype table above and are overridable per run with `--rtol` and
`--atol`.

Inductor fuses operations, and fusion legitimately changes floating point rounding, so
a bitwise equality check would fire on correct compilations. The tool compares within
tolerance and never calls a tolerance-level difference a bug. NaN and inf handling is
separate: the tool compares the boolean masks `isnan(out)` and `isinf(out)` for exact
positional equality, because a NaN appearing or disappearing is a category difference
rather than a rounding difference, and `assert_close` with `equal_nan=False` would
report it as a plain value mismatch.

#### The oracle blind spot

Eager is the reference, so any bug that lives in eager is invisible to this tool. If
eager and compiled agree on the wrong answer, compile-check reports clean. This is
stated plainly here and in the README, because a testing tool that hides its own blind
spot is worse than one that does not exist. Concretely: the author's PR 192667 (linalg
second-order forward AD) is an eager autograd bug, and compile-check could not have
found it.

The partial mitigation, borrowed from `benchmarks/dynamo/common.py`, is an optional
fp64 eager reference. With `--fp64-oracle` the numerics oracle runs a third
computation in float64 eager and compares both the fp32 eager result and the compiled
result against it, which separates two cases the two-way comparison conflates:
compiled is wrong (it is further from the fp64 reference than eager is) versus both
are imprecise (both drift similarly, and the eager-versus-compiled gap is accumulated
rounding). It does not detect a genuine eager correctness bug, only imprecision. Cost
is one extra run at fp64 width, which is why it is a flag and not the default.

### alias

For the eager run the tool builds an alias relation over the union of inputs and
outputs. Two tensors are related when they share an untyped storage
(`untyped_storage().data_ptr()` equality) and their byte ranges, computed from
`storage_offset`, `stride`, `shape`, and element size, overlap. Storage identity
alone is not sufficient: two disjoint views of one buffer share a data pointer and
are not aliases in the sense that matters. Python object identity is recorded
separately, because "output 0 is the same object as input 1" is a stronger contract
than "output 0 aliases input 1", and 191449 lives in that gap.

Input mutation is captured by hashing every input tensor before the call and again
after, recording the set of inputs whose hash changed. The hash is over the tensor
bytes, so a mutation that writes back the same values is deliberately not counted.

The compiled run must reproduce the eager relation and the eager mutation set exactly.
Any added alias, dropped alias, added mutation, or dropped mutation is a finding.
`torch._debug_has_internal_overlap` is recorded per tensor as context, since a
self-overlapping output changes what a downstream mutation means.

### metadata

Per output, the tool records dtype, shape, stride, `requires_grad`, device, and
`is_contiguous()`. Every field must match exactly. This is the oracle that catches
191308: the values were arguably defensible, the dtype was not. Stride is compared
but reported at a lower severity than dtype and shape, because a layout change alone
is usually a performance decision rather than a correctness defect. It still appears
in the report, since a stride change combined with an alias change is how a
reinplacing bug presents.

### grad

The oracle activates when any input or any model parameter has `requires_grad`. It
reduces the outputs to a scalar with a deterministic rule (sum of every floating point
output element, in a fixed traversal order, integer and bool outputs skipped) and
calls backward in both worlds. It then compares the set of tensors that ended up with
a non-None `.grad`, which must be identical, and the grad values themselves, which go
through the numerics oracle's comparison. Grads are zeroed before each backward so a
leaked accumulation cannot be mistaken for a divergence. Only one backward step is
run; multi-step training loop correctness is out of scope for v1.

### graph

The oracle wraps the callable in `torch._dynamo.explain` and reads `graph_count`,
`graph_break_count`, and `break_reasons` from the returned `ExplainOutput`. Recompile
count comes from `torch._dynamo.utils.counters['stats']['unique_graphs']` sampled
around the repeat call: the counter incrementing on a second call with identical
inputs means the model recompiled when it should not have. Compile wall time is
measured around the first compiled call, with `torch._dynamo.utils.compile_times`
output attached as context.

All of it is informational by default. Graph breaks are not bugs; they explain why a
user is not getting the speedup they expect, which is worth reporting without failing
the run. With `--baseline FILE` the oracle compares against a stored break set and
reports only new breaks, which is the mode the GitHub Action uses.

## Minimizer, v1

Granularity is the decision that defines this component, so it is stated explicitly.
v1 works at the module and input level: delta debugging over `nn.Module` children,
plus input shrinking, with the FX graph level handed off to the built-in accuracy
minifier. v2 is user-source-line level output, which is the real differentiator and
is deliberately not attempted first.

The v1 minimizer runs only after a finding, and it is allowed to give up.

1. Submodule delta debugging. Walk `named_children()` and replace subtrees with an
   identity or a shape-preserving stub, keeping any replacement under which the
   finding still reproduces. This is standard delta debugging over the module tree,
   and it is what turns "my 200-layer model is wrong" into "this one block is wrong".
   Not every module is stubbable, so the pass records the subtrees it could not
   replace instead of failing.
2. Input shrinking. Halve the leading batch-like dimension of each input tensor and
   re-run the failing oracle, keeping the smaller input when the finding survives,
   until halving stops reproducing or the dimension reaches one. Other dimensions are
   untouched in v1, because shrinking a feature dimension usually changes which kernel
   is selected.
3. Backend bisection. Re-run the failing oracle down the ablation ladder to produce
   the stage verdict described above. It is nearly free, since the backends already
   ran.
4. Built-in minifier handoff, for the FX level. Where the finding is a numerics
   divergence, re-run with `torch._dynamo.config.repro_after` set (equivalently
   `TORCHDYNAMO_REPRO_AFTER`) and `repro_level` at 4 for accuracy minification, and
   surface whatever repro directory torch produces. Machinery confirmed present:
   `torch._dynamo.repro.after_aot` and `after_dynamo`,
   `torch._dynamo.debug_utils.same_two_models` and `backend_accuracy_fails`, and
   `torch._dynamo.config.repro_tolerance` at 0.001. `torch._inductor.config.repro_after`
   does not exist on the checked version, so the handoff goes through dynamo config
   only. The handoff is best effort by design: the minifier can end with "Input graph
   did not fail the tester", in which case the tool reports the stubbed model, the
   shrunk input, and the stage verdict, and says the minifier declined.

Full FX graph delta debugging of our own, and source-line attribution, are v2.

### Regression test emission

Alongside the repro, M3 emits the same case as a drop-in regression test in the
idiom the inductor suite already uses: a `common`-style eager versus compiled
comparison written as a test method body suitable for
`test/inductor/test_torchinductor.py`. The point is narrow and practical. When a
maintainer accepts a bug report, the next thing they ask for is a test. Handing them
one that is already written in their own house style, in the right file's
conventions, removes the step where a report stalls. The tool does not claim the test
will apply unmodified; it claims the test is half-written instead of unwritten.

## Reports

Three outputs, one model behind them.

Terminal output is plain ANSI with no third-party dependency. One line per backend
per oracle, findings expanded underneath with the first divergent element index, the
two values, and the tolerance that was in force. Every finding names both the failing
check and the implicated stage, since the pair is what a reader acts on.

JSON is versioned with a top-level `schema_version` integer, bumped on any
incompatible field change. It carries the environment block (architecture always
included, see cross-architecture parity), the run configuration, and one record per
backend per oracle with a machine-readable finding list. This is the CI-consumable
artifact, and it is the unit of comparison for cross-architecture parity.

Markdown is an issue draft formatted the way PyTorch issues expect: a short
description, the minimal repro inline as a fenced Python block, expected versus
actual, the stage-localization verdict, the emitted regression test, and an
environment block with torch version and git hash, Python version, OS, architecture,
CPU or GPU model, and the backend configuration that was in force. The tool drafts. The human
reads it, edits it, and files it.

## GitHub Action

A composite action published from `action/` in the same repository. It installs the
package and runs it against declared entrypoints, failing the job on the configured
`--fail-on` categories and uploading the JSON as a build artifact. The documentation
ships a matrix example running torch stable and torch nightly in parallel, which is
the configuration that turns the tool into a nightly-regression tripwire, plus a
README badge.

Three design details decide whether the action is usable in a real repository rather
than only in a demo.

Graph breaks are compared against a committed baseline file, and the action fails on
new breaks only. Failing on any graph break would make the action unusable on day one
for every real model, and a check people turn off immediately has no value. The
correctness categories (numerics, alias, metadata, grad) always fail without a
baseline, because there is no such thing as an acceptable baseline of wrong answers.

Compile caches are disabled inside the action by default, matching the CLI, so a run
measures the current compiler rather than a cached artifact from an earlier commit.
An `allow-caches` input exists for teams who want the faster run, and the report
records which mode was used.

A `budget` input caps wall-clock time and passes through to `--budget`. Compilation is
slow and matrix jobs multiply; a check that occasionally takes forty minutes gets
deleted from CI. On timeout the action reports what it finished and exits with a
distinct status rather than a false clean.

Scope of `--budget` as built in M3-3, because the paragraph above reads wider than
what ships: it bounds the *minimizer*, which is the part of a run that decides for
itself how much work to do. A `torch.compile` call that has started cannot be
interrupted without killing the process, so bounding the run itself is the job's own
`timeout-minutes` and not a flag of ours.

## Regression corpus

`cases/` holds one tiny model per known bug class, five in total, matching the table
in the "why this project" section. Each case is a self-contained file following the
discovery convention so it doubles as a usage example. Note that the five rows there
are five bug classes, not five merged fixes; the corpus is built around what we
reproduced, which is a larger set than what we have landed.

Several of these bugs are fixed on current torch, so a case cannot simply assert
"this fails". Each case carries a known-bad version marker recording the torch
versions where the bug reproduces and the version or commit where it was fixed. The
test suite reads the marker and the running torch version and decides whether the
case is expected to produce a finding or expected to be clean.

Every oracle carries both a positive test (a model where the oracle produces the
expected finding) and a negative test (a clean model where the oracle stays silent).
The negative tests are the ones that keep the tool honest, since a checker that
fires on everything is worse than no checker.

## Real-world validation set

Before v0.1.0 ships, the tool runs against a small set of public CPU-runnable models
to calibrate tolerances and measure the false-positive rate. The candidate set is
torchvision resnet18, torchvision mobilenet_v3_small, a small HuggingFace
transformer encoder, and two of the corpus cases in their fixed state. The result
table, one row per model with per-oracle outcome and runtime, goes in the README.
Any finding on a public model is cross-checked against eager twice before it is
described as a bug.

## Non-goals for v1

Performance benchmarking: compile time is reported as a number, the tool never judges
whether the compiled version is fast enough. GPU-kernel-specific checks: the CUDA
lane exists to smoke test that `--device cuda` works, not to audit kernels.
Distributed and DTensor: out of scope entirely. Export and AOTInductor artifacts: the
tool audits `torch.compile`, not the export path. Training loop correctness beyond
one backward step. Automatic upstream filing: the tool drafts a report, a human reads
it and files it. Eager-side correctness bugs, which the oracle design cannot reach at
all, as stated in the blind spot section.

## Engineering decisions

Pure Python. The only runtime dependency is `torch>=2.4`; numpy is optional and used
only for seeding when present; argument parsing is stdlib `argparse`. No rich, no
click, no colorama. Supported Python: 3.10 through 3.13.

CI on GitHub Actions, CPU only, with a matrix over torch stable and torch nightly.
Tooling: ruff for lint and format, mypy in strict mode over the package (not over
tests or cases), pytest for tests, pre-commit wiring all three.

Repository `HussainNizamani/compile-check`, personal account, public. MIT license
unless the Chairman prefers BSD-3. Semantic versioning with `CHANGELOG.md`,
`CONTRIBUTING.md`, and a code of conduct.

Development happens on ashburn, which is CPU only aarch64. The machines behind the
project are set out in the next section.

## Test infrastructure

Five machines, no new spend for v1. The ProBook was added on 2026-09-02 after the Chairman offered it; it supersedes the earlier caveat that x86 validation depended on Turing being switched on.

| Machine | Hardware | Role | Availability |
|---|---|---|---|
| ashburn | aarch64 ARM Neoverse-N1, 4 OCPU, CPU only, 45 GB disk at 85 percent | primary development box; stock nightly wheel venv plus an editable dev build both present | always on |
| office estate (Konrad) | aarch64 ARM Neoverse-N1, 4 vCPU, 23 GB RAM, 18 GB disk free after prune | ARM build-and-verify parity lane, builder and verifier agents | idle, available; precondition is the disk prune |
| Turing | Chairman's Fedora 44 laptop, x86-64, GTX 1660 Ti (sm_75) | x86 CPU lane and CUDA smoke lane | when the Chairman has it on, so validation runs are scheduled around him |
| GitHub Actions | free x86-64 CPU runners | primary x86 CI, torch stable and nightly matrix | on every push |
| ProBook (HP 640 G2, agent Ritchie) | x86-64 Intel i5-6300U Skylake-U, 2 cores / 4 threads, AVX2, no AVX-512, no GPU, Fedora 44, system Python 3.14 (cp314 x86_64 wheels confirmed on both nightly and stable indexes 2026-09-02), 7.6 GiB RAM (text-mode boot recommended), 115 GB free disk, Wi-Fi (Ethernet recommended) | always-on x86 CPU lane: nightly-wheel testing, small-model validation, ARM to x86 parity partner, general engineering agent for both teams; not for source builds, CUDA, or AVX-512 paths | always on, Chairman's LAN |

Turing's charter widens from GPU-only to a full x86 lane. Konrad pre-cleared that
widening on PLAN.md sign-off.

Decision: no new machine for v1. A dedicated x86 VPS (8 vCPU, 16 to 32 GB, roughly 15
to 30 euro per month) is revisited only if the real-world validation set proves the
need, and not before.

### Cross-architecture parity is a feature

Running the same model on ARM and on x86 and comparing the results is a deliberate
compile-check capability, not an accident of where we happen to develop. Compile bugs
do not present identically across architectures. Issue 191837 is the worked example:
the same defect aborted the process on x86, three runs out of three, while on ARM it
silently corrupted about 99.5 percent of the output. A user who tested on one
architecture would have drawn the wrong conclusion about the other, and the silent
case is the dangerous one.

Two consequences for the design. The report's environment block must always carry the
architecture, alongside the torch version and git hash; a run whose provenance is
ambiguous is not usable as parity evidence. And the JSON schema is what makes the
comparison possible, since parity in v1 means running the tool on two machines and
diffing the two JSON files. A first-class `compile-check compare a.json b.json`
subcommand is v0.2.

This is also a selling point. No existing tool answers "does my model compile to the
same answers on Graviton as on x86", and teams shipping to ARM inference fleets have
that question whether or not they have phrased it.

## Package layout

```
src/compile_check/
  cli.py                 argparse surface, exit codes
  discover.py            entry and input resolution
  runner.py              seeding, cloning, reset, per-backend execution
  oracles/
    numerics.py
    alias.py
    metadata.py
    grad.py
    graph.py
  minimize.py            submodule delta debug, input shrink, minifier handoff
  localize.py            backend ablation ladder, stage verdict
  report/
    terminal.py
    json.py
    markdown.py
    pytest_case.py       inductor-suite regression test emitter
  env.py                 environment block collection
tests/
cases/
action/
docs/
```

## Milestones

Estimates assume one implementer plus one verifier, in working days.

| Milestone | Contents | Definition of done | Days |
|---|---|---|---|
| M0 | scaffold, pyproject, package skeleton, CI workflow, ruff, mypy strict, pre-commit | CI green on the empty package across the full Python and torch matrix; `compile-check --version` runs | 2 |
| M1 | runner, numerics oracle, metadata oracle, stage localization, terminal report | `compile-check cases/dtype_promotion.py` runs all three backends and prints a per-oracle table naming check and stage; a known-bad case reports a finding, a clean case reports none | 5 |
| M2 | alias and mutation oracle, gradients oracle, regression corpus with version markers | all five corpus cases present; every oracle has a positive and a negative test; test suite green on stable and nightly | 6 |
| M3 | graph health oracle with baseline support, minimizer v1 including submodule delta debugging, JSON and Markdown reports, regression test emitter | a finding produces a stubbed model, a shrunk input, a stage verdict, a Markdown draft with a runnable repro, and an inductor-suite-style test method; JSON validates against the committed schema | 6 |
| M4 | GitHub Action with baseline, cache, and budget inputs; docs; README with terminal capture and validation table; v0.1.0 to PyPI | action runs green in a demo repository and fails correctly on a seeded regression; validation table filled with real numbers; `pip install compile-check` works from PyPI | 5 |

Total: 24 working days to v0.1.0. Add one day to M1 if the Chairman puts the fp64
oracle in v0.1 rather than v0.2.

### v0.2 outlook

User-source-line attribution in the minimizer, so a finding points at a line in the
user's file rather than at a node in an FX graph. This is the differentiator against
every existing tool and is why it is not attempted in v1. Our own FX graph delta
debugging, replacing the handoff-only path. A `compile-check compare a.json b.json`
subcommand, making cross-architecture parity a first-class command rather than a
manual diff of two JSON files. A dynamic shapes matrix rather than a single optional
pass. Custom operator and higher-order-operator awareness, since a
custom op with an unregistered fake kernel is a distinct and common failure mode that
v1 detects only indirectly.

## Quality and process gates

Every merged change carries tests. Every oracle has both positive coverage (it
catches a known bug) and negative coverage (it stays silent on a clean model). Before
each milestone closes, an adversarial verification pass is run by a different agent
or model than the one that implemented it; the implementer never grades its own work.
No change is described as working without the command output that proves it. Any
finding on a real model is cross-checked against eager twice before it is called a
bug.

`FINDINGS.md` logs every real upstream bug the tool surfaces, each entry linked to
the issue that was filed for it. This file is the project's argument for its own
existence.

## Release and credibility checklist

The gate for calling v0.1.0 public.

- The README is understandable in thirty seconds, and it contains a real red run
  pasted verbatim, output and all, not a described one.
- A "bugs found" table links each issue and each PR with its current status, kept in
  sync with `FINDINGS.md`.
- CI is green, on real tests, on both the stable and nightly torch lanes.
- Releases are tagged, and the tags match the PyPI versions.
- The Action is published to the GitHub Marketplace, not only present in the repo.
- No hype in any user-facing text. Claims are limited to what the verification table
  shows. The AI-assisted disclosure line is kept.

## Decisions for the Chairman

| Decision | Options | Recommendation |
|---|---|---|
| Final name | `compile-check`, or an alternative | `compile-check`, free on PyPI as of 2026-09-02 |
| License | MIT, BSD-3 | MIT, unless matching PyTorch's BSD-3 matters for upstream optics |
| Repository home | personal account, new organization | personal account, the upstream track record is the credential and it is attached to the person |
| Python floor | 3.10, 3.9 | 3.10, since 3.9 is end of life and torch 2.4 already requires 3.8 or later |
| GitHub Action timing | ship in v0.1, defer to v0.2 | ship in v0.1, it is the difference between a script and a tool teams adopt |
| Public issue numbers in the corpus | reference them, keep them private | reference them, the corpus is more credible when each case names the bug it encodes |
| Tolerance policy | `assert_close` per-dtype defaults, OpInfo-style per-op tolerances | start with the `assert_close` defaults, measure the false-positive rate on the validation set, and only move to per-op tolerances if the defaults prove too tight |
| fp64 oracle timing | ship in v0.1 behind `--fp64-oracle`, defer to v0.2 | ship in v0.1, it costs about one day and it is the answer to the first hard question a reviewer will ask |
