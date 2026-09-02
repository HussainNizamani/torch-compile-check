# compile-check

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
<!-- badge placeholders: CI, PyPI, and the torch matrix land with M4 -->

> **Status: M0 scaffold, nothing works yet.** The package installs, `compile-check
> --version` and `compile-check --probe` run, and every other module is a typed
> stub that raises `NotImplementedError`. Do not use this to check anything yet.

Bring your own model; compile-check tells you whether `torch.compile` changed its
answers, and if so hands you a minimal repro and a ready-to-file report. The tool is a
one-command trust auditor for `torch.compile`: point it at a model or a function
together with real inputs, and it runs the same callable under eager, `aot_eager`, and
`inductor`, then reports whether compilation silently changed the result.
`torch.compile` is a compiler, and compile-check is differential testing for it, in the
same sense that Csmith is differential testing for GCC and LLVM, with the difference
that the program under test is your own model rather than a generated one.

## What works today

```console
$ compile-check --version
compile-check 0.0.1.dev0

$ compile-check --probe        # which torch APIs the oracles need are on your install
```

Everything else exits 2 with "not implemented in M0". The plan, including the full
oracle design and the milestone schedule, is in [PLAN.md](PLAN.md).

## Use it in CI

A composite GitHub Action lives in [`action/`](action/action.yml):

```yaml
- uses: HussainNizamani/compile-check/action@main
  with:
    targets: models/classifier.py
```

It installs `compile-check`, runs it against the entrypoints you declare, and
fails the job on the configured `--fail-on` categories. See
[docs/action.md](docs/action.md) for the full inputs/outputs reference,
baseline semantics, and the note on how it degrades honestly before M1-3
lands the CLI's main run path.

## Blind spot, stated up front

Eager is the reference, so any bug that lives in eager is invisible to this tool. If
eager and compiled agree on the wrong answer, compile-check reports clean. A testing
tool that hides its own blind spot is worse than one that does not exist.

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
