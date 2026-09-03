"""Inductor reinplacing turns a chain of two functional scatter-shaped ops
into an alias of the input, where the *first* op in the chain is the one
reinplaced even though its own copy-back is not what the pass computed
profitability from.

Issue: https://github.com/pytorch/pytorch/issues/195451
PR (open, not merged as of 2026-09-02): https://github.com/pytorch/pytorch/pull/195484

Reviewer-reported sibling of `cases/alias_slice_scatter_copyback.py`,
reported 2026-09-03 on PR #195484. Where the sibling case is a single
`slice_scatter` followed directly by a copy-back, this case chains two
functional scatter-shaped ops -- `a = torch.diagonal_scatter(x, diag)` then
`b = torch.index_put(a, (idx,), val)` -- before `x.copy_(b)` writes the
final result back into the graph input. The reviewer's concern was that the
reinplace pass, having decided `b` can be written in place because of the
trailing copy-back, reinplaces the chain all the way back into `x` even
though only `b` (not `a`) is returned.

Contract violated: aliasing. Both `torch.diagonal_scatter` and
`torch.index_put` (the out-of-place form used here) are functional: in
eager, `b` is an independent tensor. Under Inductor, the returned `b`
shares `x`'s storage -- it is not merely equal to `x`, it *is* `x` -- so
mutating `b` afterwards corrupts `x`.

Oracle: alias (identity / data_ptr comparison after the call, plus a mutate
side-effect check) -- same oracle as the sibling case.

First diverging backend: inductor. `aot_eager` matches eager here; run with
`--backend aot_eager` for comparison, as the sibling case does.

Known-bad torch versions: RED reproduced on torch 2.14.0+cpu (git
08187d9e0fba026dc8217405802ab5381dc88d90), aarch64, CPU, in this
repository's own venv, 2026-09-03. PR 195484 (the fix for the
`slice_scatter` sibling case) was still open and unmerged as of 2026-09-02,
and its diff does not claim to cover this chained shape; no fix is known to
exist for it.

Unlike the five C-1 cases (`cases/README.md` "Two shapes of file"), this
file has no separate discovery-convention twin: it exposes the module-level
`fn` and `inputs` PLAN.md's discovery convention looks for *in addition to*
the standalone `build()`/`check()`/`main()` shape, so `torch-compile-check
cases/alias_diagonal_scatter_index_put_chain.py` runs it through the tool
directly and `tests/test_corpus_twins.py` names this same file as its own
twin.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import sys

import torch


def fn(x: torch.Tensor, diag: torch.Tensor, idx: torch.Tensor, val: torch.Tensor):
    """Two chained functional scatter-shaped ops, then a copy-back into the
    graph input.

    `a` is a functional result of scattering `diag` onto the diagonal of (a
    copy of) `x`. `b` is a functional result of `index_put` onto `a` at rows
    `idx`. `x.copy_(b)` is the copy-back that makes reinplacing `b` (and, on
    this build, `a` underneath it) look profitable. `b` is what is returned
    and what the alias oracle inspects against `x`.
    """
    a = torch.diagonal_scatter(x, diag)  # (2, 2), functional
    b = torch.index_put(a, (idx,), val)  # (2, 2), functional, out-of-place
    x.copy_(b)  # write back into x
    return b


# Fixed literal values, on the pattern of alias_slice_scatter_copyback.py.
inputs = (
    torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    torch.tensor([10.0, 20.0]),
    torch.tensor([0]),
    torch.tensor([[100.0, 200.0]]),
)


def build():
    return fn, inputs


def _git_hash():
    # The provenance the RED/GREEN line needs is torch's own build commit,
    # not this repo's -- a case can be RED/GREEN purely because of which
    # torch checkout it ran against, and that is what the parity table in
    # FINDINGS.md keys on.
    git_version = getattr(torch.version, "git_version", None)
    if not git_version:
        return "unknown"
    return git_version[:7]


def check(eager_out, compiled_out, inputs):
    """inputs is the *original* (x, diag, idx, val) tuple passed to fn for
    the eager run.

    Runs its own fresh input tuple for the compiled call internally via
    main(), so `check` here only inspects the already-produced outputs plus
    a fresh mutate-and-observe probe done by the caller (main). This
    function reports the alias comparison; main() drives the mutate probe
    because it needs a second, unmutated copy of x to compare against.
    """
    eager_x, eager_out_t = eager_out
    compiled_x, compiled_out_t = compiled_out

    eager_aliases = eager_x.data_ptr() == eager_out_t.data_ptr()
    compiled_aliases = compiled_x.data_ptr() == compiled_out_t.data_ptr()

    if eager_aliases:
        return False, "eager itself aliases input and output; case assumption violated"

    if not compiled_aliases:
        return (
            False,
            "compiled output does not alias input; matches eager (no bug on this backend/version)",
        )

    # Confirm the alias is load-bearing: mutate the returned tensor and check
    # that the "input" (already consumed) tensor moved too.
    before = compiled_x.clone()
    compiled_out_t.add_(100.0)
    after = compiled_x
    corrupted = not torch.equal(before, after)

    if corrupted:
        return True, (
            f"compiled output aliases input (data_ptr equal) and mutating the "
            f"output corrupted the input: before={before.tolist()} "
            f"after={after.tolist()}"
        )
    return (
        True,
        "compiled output aliases input (data_ptr equal) but mutate-probe did not corrupt it",
    )


def _run_variant(backend_name):
    x, diag, idx, val = (t.clone() for t in inputs)
    if backend_name == "eager":
        out = fn(x, diag, idx, val)
    else:
        compiled_fn = torch.compile(fn, backend=backend_name)
        out = compiled_fn(x, diag, idx, val)
    return x, out


def _report(case, backend_name, eager_result, other_result):
    is_red, message = check(eager_result, other_result, None)
    status = "RED" if is_red else "GREEN"
    print(
        f"{status} {case} torch={torch.__version__} git={_git_hash()} "
        f"arch={platform.machine()} backend={backend_name} :: {message}"
    )
    return is_red


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backend",
        default=None,
        help="extra backend to also check, e.g. aot_eager",
    )
    args = parser.parse_args()

    case = "alias_diagonal_scatter_index_put_chain"
    try:
        eager_result = _run_variant("eager")
        inductor_result = _run_variant("inductor")
    except Exception as exc:
        print(f"CRASH {case} :: {type(exc).__name__}: {exc}")
        sys.exit(2)

    is_red = _report(case, "inductor", eager_result, inductor_result)
    exit_code = 1 if is_red else 0

    if args.backend:
        extra_result = _run_variant(args.backend)
        _report(case, args.backend, eager_result, extra_result)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
