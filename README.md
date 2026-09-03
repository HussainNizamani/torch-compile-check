# torch-compile-check


[![CI](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/ci.yml/badge.svg)](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/ci.yml)
[![Action self-test](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/action-selftest.yml/badge.svg)](https://github.com/HussainNizamani/torch-compile-check/actions/workflows/action-selftest.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

torch-compile-check tells you whether `torch.compile` silently changed your
model's answer. Point it at your own model and it runs the same program
under eager PyTorch and every compiled backend, then reports the first
check that disagrees and the first stage where it does.

```console
$ pip install git+https://github.com/HussainNizamani/torch-compile-check.git
$ torch-compile-check path/to/model.py
```

## A red run

Real, verbatim terminal output, [issue #191308](https://github.com/pytorch/pytorch/issues/191308),
an int8 matmul silently promoted to int64 by CPU inductor:

```console
$ torch-compile-check cases/dtype_promotion.py
checks
  oracle    fail-on  aot_eager         inductor
  numerics  yes      pass              pass
  alias     yes      pass              pass
  metadata  yes      pass              1 fail
  grad      yes      pass              pass
  graph     no       pass              pass

  pass = no finding

findings
  metadata  (1 fail)
    [fail] inductor output[0]
        dtype differs: eager torch.int8, inductor torch.int64
        field dtype   expected torch.int8   got torch.int64

stage
  first diverges at inductor, which implicates inductor lowering/codegen
  that is where the divergence becomes observable, not necessarily where the fix belongs
```

Exit code 1. The stage verdict names the first backend that diverges. That
is where the bug becomes *observable*, not necessarily where the fix
*belongs*, and the report says so every time, because those are different
questions ([PLAN.md "Where divergence appears is not always where the fix
belongs"](PLAN.md)).

## What it does

Point torch-compile-check at your `(callable, weights, example inputs)`
and it runs that program three ways: `eager` (plain PyTorch, the
reference), `aot_eager` (Dynamo capture plus AOTAutograd, no codegen,
"stage 1 only"), and `inductor` (the full pipeline, "stage 1 + 2"). Then
it compares each compiled run against eager with five checks:

1. **Numerics**: outputs match within tolerance.
2. **Alias and mutation**: outputs share storage with inputs, and inputs
   get mutated, only where eager does the same.
3. **Dtype, shape, stride**: identical types and memory layout.
4. **Gradients**: backward signals match.
5. **Graph breaks and recompiles**: where compile gave up or restarted. A
   warning class, not a correctness failure.

Any red result is auto-minimized into the smallest case that still
reproduces, plus a ready-to-file report naming the failing check and the
compilation stage where the divergence first appears, the ablation ladder
PyTorch maintainers walk by hand when triaging a compile bug, run
automatically.

`torch.compile` is a compiler, and torch-compile-check is differential
testing for it, in the sense that Csmith is differential testing for GCC
and LLVM, except the program under test is your own model rather than a
generated one.

## Install and run

Until `v0.1.0` is on PyPI, install from git:

```console
$ pip install git+https://github.com/HussainNizamani/torch-compile-check.git
```

Once it ships:

```console
$ pip install torch-compile-check
```

Then point it at a model. The file needs a module-level `model` (an
`nn.Module`) or `fn` (a callable), and a module-level `inputs` or a
`get_inputs()` that returns them; `--entry` and `--inputs` override both.

```console
$ torch-compile-check path/to/model.py
$ torch-compile-check --probe        # which torch APIs the oracles need are on your install
$ torch-compile-check --version
```

Every flag, with one real example each, is in [docs/usage.md](docs/usage.md).
That page also covers what pointing the tool at a file path does to your
interpreter's `sys.path` and `sys.modules`, see [docs/usage.md "What
pointing the tool at a file does to your
interpreter"](docs/usage.md#what-pointing-the-tool-at-a-file-does-to-your-interpreter).

Three more artifacts come off the same run: `--json` (the versioned,
CI-consumable result), `--md` (an issue draft), `--emit-test` (a
regression test in the inductor suite's own idiom); see
[docs/reports.md](docs/reports.md). After a finding, `--minimize` shrinks
the case while it still reproduces; see
[docs/usage.md#the-minimizer](docs/usage.md#the-minimizer).

## Regression corpus: real bugs each check re-detects

Every oracle is anchored to a real, filed `torch.compile` correctness bug
that lives in `cases/` (ground truth in [FINDINGS.md](FINDINGS.md)).

| Bug | Issue | Fix | Status | Oracle |
|---|---|---|---|---|
| Inductor reinplacing breaks an output-aliases-input contract | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [PR #195484](https://github.com/pytorch/pytorch/pull/195484) | open, unmerged | alias |
| AOTAutograd aliased-output identity | [#191449](https://github.com/pytorch/pytorch/issues/191449) | [PR #191844](https://github.com/pytorch/pytorch/pull/191844) | merged 2026-09-02 | alias |
| Silent dtype promotion: int8 matmul becomes int64 on CPU inductor | [#191308](https://github.com/pytorch/pytorch/issues/191308) | none filed yet | open, unfixed | metadata |
| Distributions crossed with compile: a validation branch diverges under `--fullgraph` | [#194593](https://github.com/pytorch/pytorch/issues/194593) (sibling [#194596](https://github.com/pytorch/pytorch/issues/194596)) | none filed yet | open, unfixed | graph |
| CPU inductor miscompile: wrong values, no error | [#190765](https://github.com/pytorch/pytorch/issues/190765) | [PR #190966](https://github.com/pytorch/pytorch/pull/190966) | fixed, merged | numerics |

Every bug in this table was found and filed by hand before this tool
existed; the tool re-detects each one. Bugs first found by
torch-compile-check rather than by hand: none yet.

Three merged `torch.compile` fixes by the author predate this tool (Dynamo
[PR #190673](https://github.com/pytorch/pytorch/pull/190673), Inductor [PR
#192628](https://github.com/pytorch/pytorch/pull/192628), and the
AOTAutograd fix above), and one eager-autograd fix ([PR
#192667](https://github.com/pytorch/pytorch/pull/192667)) is deliberately
kept out of this table: a tool whose oracle is eager could not have found
a bug that lives in eager. [PLAN.md "Why this project, and why
us"](PLAN.md) has the full credential list.

## Validation against real models

torch-compile-check run against public, CPU-runnable models to measure the
false-positive rate, regenerated by [`validation/run.py`](validation/run.py);
full detail, the tolerance policy, and provenance in
[`docs/validation.md`](docs/validation.md).

| Target | Status | Exit | Findings | Seconds |
|---|---|---|---|---|
| `tv_resnet18` | clean | 0 | none | 28.5 |
| `tv_mobilenet_v3_small` | clean | 0 | none | 61.2 |
| `tv_efficientnet_b0` | clean | 0 | none | 91.1 |
| `tv_vit_b_16_tiny` | clean | 0 | none | 18.9 |
| `hf_tiny_bert` | clean | 0 | none | 19.4 |
| `train_step_mlp` | clean | 0 | none | 10.7 |

This is the aarch64 CPU host run: 2026-09-02, torch `2.14.0+cpu` (git
`08187d9e0fba`), aarch64, CPU only, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`,
backends `eager,aot_eager,inductor`; the four vision/ViT targets add
`--fp64-oracle`. 6 of 6 clean on this host, no finding tuned away to get
there. The same six targets also ran clean on x86_64 CPU and on x86_64
with CUDA sm_75, 18 of 18 parity checks against this aarch64 reference;
see [docs/cross-arch.md "Cross-architecture results
(2026-09-03)"](docs/cross-arch.md#cross-architecture-results-2026-09-03).

## Use it in CI

A composite GitHub Action lives in [`action/`](action/action.yml):

```yaml
- uses: HussainNizamani/torch-compile-check/action@main
  with:
    targets: models/classifier.py
```

It installs `torch-compile-check`, runs it against the entrypoints you
declare, and fails the job on the configured `--fail-on` categories. The
`source` input (default `auto`) installs from the checked-out source when
the action runs inside this repo, and from `git+https://...@ref`
otherwise, the git path is what external consumers use once the repo is
public or the package is on PyPI. See [docs/action.md](docs/action.md) for
the full inputs and outputs reference, baseline semantics, and how it
degrades honestly on an old `ref`: on any current ref,
`torch-compile-check <target>` runs for real and exits 0, 1, or 2; a ref
pinned before the CLI's run path existed exits 2 with "not implemented"
instead.

## Blind spot, stated up front

Eager is the reference, so any bug that lives in eager is invisible to
this tool. If eager and compiled agree on the wrong answer,
torch-compile-check reports clean. A testing tool that hides its own blind
spot is worse than one that does not exist. `--fp64-oracle` is a partial
mitigation for numerics: a third, float64 eager reference that separates
"compiled is wrong" from "both eager and compiled are imprecise at fp32."
It narrows imprecision. It does not detect a genuine eager correctness
bug.

By default v0.1 runs static shapes (`dynamic=False`, recorded in the JSON as
`run.dynamic`); `--dynamic` adds a single dynamic-shapes pass, and a full
dynamic-shapes matrix is planned.

## Relationship to PyTorch's built-in tools

PyTorch already tests compile correctness, thoroughly. This section is
honest about what exists; the claim is not that nothing like this exists,
it is that none of it runs on your model, in your CI.

| Existing tool | What it does | Gap for a user |
|---|---|---|
| `torch.library.opcheck` | checks one custom operator's schema, autograd registration, and fake-tensor kernel, then compares it against AOTAutograd under static and dynamic shapes | one operator, not a composed model; does not exercise inductor lowering; returns a pass/fail dict, not a diff report with stage localization; no minimizer; no CI report |
| `test/inductor/test_torchinductor_opinfo.py` | OpInfo-driven eager versus compiled tests | per operator, on PyTorch's commits, not on a composed user model |
| `benchmarks/dynamo/*.py --accuracy` | accuracy runs with an fp64 eager reference over a fixed model zoo | fixed zoo; your model is not in it |
| `dynamo_wrapped` CI shards | reruns the PyTorch test suite under compile | PyTorch's programs, not yours |
| accuracy minifier (`TORCHDYNAMO_REPRO_AFTER=dynamo\|aot`, `TORCHDYNAMO_REPRO_LEVEL=4`) | shrinks an FX graph that already fails | opt-in, after the fact, numerics only, can end with "Input graph did not fail the tester" |
| `torch._dynamo.explain` | graph breaks and guards | diagnostic, not a pass or fail verdict |
| `TORCH_LOGS=graph_breaks,recompiles` | logs breaks and recompiles | logs to read, nothing to assert on |
| backend ablation ladder `eager` to `aot_eager` to `aot_eager_decomp_partition` to `inductor` | stage localization | a manual workflow, done by hand |

The gaps, summarized: these run on PyTorch's commits and PyTorch's
programs, or on one operator at a time, not on your composed model or your
torch upgrade; they are opt-in and after the fact; they are numerics only,
or skip inductor entirely; they work at the FX graph level rather than at
your source; the minifier can fail to isolate; there is no unified pass or
fail across aliasing, dtype, stride, and gradients; and none of it is in
your CI.

torch-compile-check productizes the workflow the PyTorch team does by hand
(ablate backends, compare, minify, file) into one command with five
checks, moved from "after you noticed" to "before you merged". The
adjacent third-party work is random-graph research fuzzing (NNSmith and
similar) and an academic bug-study corpus, which generate synthetic
graphs; torch-compile-check audits the model the user already has.

## Cross-architecture parity

Running the same model on ARM, x86, and CUDA and diffing the JSON results
is a supported workflow ([PLAN.md "Cross-architecture parity is a
feature"](PLAN.md)). [docs/cross-arch.md](docs/cross-arch.md) is the
copy-paste runbook for a second machine and what "parity" means for a JSON
diff.

## Documentation

[docs/](docs/README.md) indexes every page: flags, report shapes, the
regression corpus, real-model validation, the GitHub Action, and the
cross-architecture runbook. [PLAN.md](PLAN.md) is the full design, and
[CHANGELOG.md](CHANGELOG.md) records what shipped in each release.

## Development

```console
python -m venv .venv && . .venv/bin/activate
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
make lint type test
```

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, see [LICENSE](LICENSE).

AI assisted.
