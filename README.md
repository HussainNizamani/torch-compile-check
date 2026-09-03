# torch-compile-check

*(formerly `compile-check`, renamed before the `v0.1.0` tag)*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!--
The CI, PyPI and Python-version badges are held back on purpose: a badge for a
private repository or an unpublished package renders as a broken image, which
is worse than no badge. The exact lines to paste here after the public flip and
the upload are in docs/release.md, step 8.
-->

## What it does

Bring your own model. Point torch-compile-check at your `(callable, weights,
example inputs)` and it runs that program three ways — `eager` (plain
PyTorch, the reference), `aot_eager` (Dynamo capture plus AOTAutograd, no
codegen — "stage 1 only"), and `inductor` (the full pipeline — "stage 1 +
2") — and reports whether compiling silently changed the answer.
`torch.compile` is a compiler; torch-compile-check is differential testing for it,
in the sense that Csmith is differential testing for GCC and LLVM, except
the program under test is your own model rather than a generated one. Any
red result is auto-minimized into a minimal repro plus a ready-to-file
report naming both the failing check and the compilation stage the
divergence first appears in — the ablation ladder PyTorch maintainers walk
by hand when triaging a compile bug, run automatically.

Five checks, each compiled lane against eager:

1. **Numerics** — outputs match within tolerance.
2. **Alias and mutation** — outputs share storage with inputs, and inputs
   get mutated, only where eager does the same.
3. **Dtype, shape, stride** — identical types and memory layout.
4. **Gradients** — backward signals match.
5. **Graph breaks and recompiles** — where compile gave up or restarted; a
   warning class, not a correctness failure.

## A red run

Real, verbatim terminal output — [issue #191308](https://github.com/pytorch/pytorch/issues/191308),
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

Exit code 1. The stage verdict names the first backend that diverges, which
is where the divergence becomes *observable*, not necessarily where the fix
*belongs* — the report says so every time, because the two are not the same
question ([PLAN.md "Where divergence appears is not always where the fix
belongs"](PLAN.md)).

(A terminal-recording `docs/demo.gif` was evaluated for this run and
skipped: `asciinema` installs from PyPI, but `agg`, its gif renderer, is a
Rust binary distributed via `cargo install`, not pip — the PyPI package
named `agg` is an unrelated CSV tool. Recording a `.cast` file and finding
another way to render it stays a follow-up rather than something this slice
does with a mismatched tool.)

## Quick start

Until `v0.1.0` is on PyPI (the tag and the upload are a maintainer's own
release step, not something that happens automatically once the code for a
milestone lands):

```console
$ pip install git+https://github.com/HussainNizamani/torch-compile-check.git
```

That command works: this repository has been public since 2026-09-03. The
PyPI upload is what's still pending, which is why it is the first install
path here rather than the block below. Once torch-compile-check ships to
PyPI:

```console
$ pip install torch-compile-check
```

Then point it at a model:

```console
$ torch-compile-check path/to/model.py
```

The file needs a module-level `model` (an `nn.Module`) or `fn` (a
callable), and a module-level `inputs` or a `get_inputs()` that returns
them; `--entry` and `--inputs` override both. Every flag, with one real
example each, is in [docs/usage.md](docs/usage.md).

```console
$ torch-compile-check --probe        # which torch APIs the oracles need are on your install
$ torch-compile-check --version
```

Three more artifacts come off the same run — `--json` (the versioned,
CI-consumable result), `--md` (an issue draft), `--emit-test` (a regression
test in the inductor suite's own idiom) — see [docs/reports.md](docs/reports.md).
After a finding, `--minimize` shrinks the case while it still reproduces;
see [docs/usage.md](docs/usage.md#the-minimizer).

## Bugs it has caught

Real, reproduced `torch.compile` correctness bugs, one per oracle in the
regression corpus (`cases/`; ground truth in [FINDINGS.md](FINDINGS.md)).

| Bug | Issue | Fix | Status | Oracle |
|---|---|---|---|---|
| Inductor reinplacing breaks an output-aliases-input contract | [#195451](https://github.com/pytorch/pytorch/issues/195451) | [PR #195484](https://github.com/pytorch/pytorch/pull/195484) | open, unmerged | alias |
| AOTAutograd aliased-output identity | [#191449](https://github.com/pytorch/pytorch/issues/191449) | [PR #191844](https://github.com/pytorch/pytorch/pull/191844) | merged 2026-09-02 | alias |
| Silent dtype promotion: int8 matmul becomes int64 on CPU inductor | [#191308](https://github.com/pytorch/pytorch/issues/191308) | none filed yet | open, unfixed | metadata |
| Distributions crossed with compile: a validation branch diverges under `--fullgraph` | [#194593](https://github.com/pytorch/pytorch/issues/194593) (sibling [#194596](https://github.com/pytorch/pytorch/issues/194596)) | none filed yet | open, unfixed | graph |
| CPU inductor miscompile: wrong values, no error | [#190765](https://github.com/pytorch/pytorch/issues/190765) | [PR #190966](https://github.com/pytorch/pytorch/pull/190966) | fixed, merged | numerics |

Separately: three merged `torch.compile` fixes by the author predate this
tool (Dynamo [PR #190673](https://github.com/pytorch/pytorch/pull/190673),
Inductor [PR #192628](https://github.com/pytorch/pytorch/pull/192628), and
the AOTAutograd fix above), and one eager-autograd fix
([PR #192667](https://github.com/pytorch/pytorch/pull/192667)) is
deliberately kept out of this table — a tool whose oracle is eager could not
have found a bug that lives in eager. [PLAN.md "Why this project, and why
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

Run 2026-09-02 on torch `2.14.0+cpu` (git `08187d9e0fba`), aarch64, CPU
only, `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`, backends
`eager,aot_eager,inductor`; the four vision/ViT targets add `--fp64-oracle`.
6 of 6 clean — no finding tuned away to get there.

## What pointing the tool at a file does to your interpreter

A target given as a file path is imported by executing it, which is the
only way to get at the model it defines. Two side effects of that are
deliberate and worth knowing before you point the tool at something:

- the file's parent directory is inserted at `sys.path[0]`, so a target that
  imports a sibling module works — that is what a two-file repro needs;
- the module is registered in `sys.modules` under the file's stem, so a
  dataclass, a pickle, or a self-import inside the target resolves. If that
  name is already taken by an unrelated module the target is registered as
  `_torch_compile_check_target_<stem>` instead, rather than shadowing it.

Both happen in the `torch-compile-check` process only. Nothing is written to your
project, and nothing outlives the command.

## Use it in CI

A composite GitHub Action lives in [`action/`](action/action.yml):

```yaml
- uses: HussainNizamani/torch-compile-check/action@main
  with:
    targets: models/classifier.py
```

It installs `torch-compile-check`, runs it against the entrypoints you declare,
and fails the job on the configured `--fail-on` categories. The `source`
input (default `auto`) installs from the checked-out source when the action
runs inside this repo, and from `git+https://...@ref` otherwise — the git
path is what external consumers use once the repo is public or the package
is on PyPI. See [docs/action.md](docs/action.md) for the full inputs and
outputs reference, baseline semantics, and how it degrades honestly on an
old `ref`: the CLI's main run path landed in M1-3, so `torch-compile-check
<target>` runs for real on any current ref and exits 0, 1, or 2, and only a
ref pinned before M1-3 exits 2 with "not implemented".

## Blind spot, stated up front

Eager is the reference, so any bug that lives in eager is invisible to this
tool. If eager and compiled agree on the wrong answer, torch-compile-check
reports clean. A testing tool that hides its own blind spot is worse than
one that does not exist. `--fp64-oracle` is a partial mitigation for
numerics: a third, float64 eager reference that separates "compiled is
wrong" from "both eager and compiled are imprecise at fp32" — it narrows
imprecision, it does not detect a genuine eager correctness bug.

## Relationship to PyTorch's built-in tools

PyTorch already tests compile correctness, thoroughly. This section is
honest about what exists, and it is destined for the README as well as
PLAN.md. The claim is not that nothing like this exists; it is that none of
it runs on the user's model.

| Existing tool | What it does | Gap for a user |
|---|---|---|
| `test/inductor/test_torchinductor_opinfo.py` | OpInfo-driven eager versus compiled tests | per operator, on PyTorch's commits, not on a composed user model |
| `benchmarks/dynamo/*.py --accuracy` | accuracy runs with an fp64 eager reference over a fixed model zoo | fixed zoo; your model is not in it |
| `dynamo_wrapped` CI shards | reruns the PyTorch test suite under compile | PyTorch's programs, not yours |
| accuracy minifier (`TORCHDYNAMO_REPRO_AFTER=dynamo\|aot`, `TORCHDYNAMO_REPRO_LEVEL=4`) | shrinks an FX graph that already fails | opt-in, after the fact, numerics only, can end with "Input graph did not fail the tester" |
| `torch._dynamo.explain` | graph breaks and guards | diagnostic, not a pass or fail verdict |
| `TORCH_LOGS=graph_breaks,recompiles` | logs breaks and recompiles | logs to read, nothing to assert on |
| backend ablation ladder `eager` to `aot_eager` to `aot_eager_decomp_partition` to `inductor` | stage localization | a manual workflow, done by hand |

The gaps, summarized: these run on PyTorch's commits and PyTorch's
programs, not on your model or your torch upgrade; they are opt-in and
after the fact; they are numerics only; they work at the FX graph level
rather than at your source; the minifier can fail to isolate; there is no
unified pass or fail across aliasing, dtype, stride, and gradients; and none
of it is in your CI.

torch-compile-check productizes the workflow the PyTorch team does by hand
(ablate backends, compare, minify, file) into one command with five checks,
moved from "after you noticed" to "before you merged". The adjacent
third-party work is random-graph research fuzzing (NNSmith and similar) and
an academic bug-study corpus, which generate synthetic graphs;
torch-compile-check audits the model the user already has.

## Cross-architecture parity

Running the same model on ARM and on x86 (and CUDA) and diffing the results
is a deliberate capability, not an accident of where this project happens to
be developed — [PLAN.md "Cross-architecture parity is a
feature"](PLAN.md). [docs/cross-arch.md](docs/cross-arch.md) is the
copy-paste runbook for a second machine and what "parity" means for a JSON
diff.

## Documentation

[docs/](docs/README.md) indexes every page: flags, report shapes, the
regression corpus, real-model validation, the GitHub Action, and the
cross-architecture runbook. [PLAN.md](PLAN.md) is the full design, and
[CHANGELOG.md](CHANGELOG.md) says what shipped in which slice.

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
