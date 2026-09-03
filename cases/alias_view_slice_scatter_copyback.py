"""Inductor reinplacing turns a non-aliasing functional result into an alias
of the input, when the scatter target is a *view* of the input rather than
the input itself.

Issue: https://github.com/pytorch/pytorch/issues/195451
PR (open, not merged as of 2026-09-02): https://github.com/pytorch/pytorch/pull/195484

Reviewer-reported sibling of `cases/alias_slice_scatter_copyback.py`,
reported 2026-09-03 on PR #195484: the fix under review handles a direct
`slice_scatter(x, ...)` on the graph input `x` itself, but a reviewer
pointed out that `should_reinplace_scatter()` also has to trace through a
*view* of `x` to see that reinplacing would corrupt something the caller
can still observe through `x`. This case scatters into `x.permute(1, 0)`
instead of `x`, and copies back through the same view.

Contract violated: aliasing. `torch.slice_scatter(view, src, ...)` is
functional: in eager, `updated` is an independent tensor that happens to
hold the same values as `view` (and therefore `x`) after the copy-back, and
`before` (the view read out before the scatter) is independent of it too.
Under Inductor, reinplacing fires through the view -- but only once
`before` is also read out of the graph, which is what makes the pass see
`view` as something worth eliminating in place of `x` -- and the compiled
function returns `updated` sharing `x`'s storage, so mutating `updated`
afterwards corrupts `x`.

Oracle: alias (identity / data_ptr comparison after the call, plus a mutate
side-effect check) -- same oracle as the sibling case.

First diverging backend: inductor. `aot_eager` matches eager here; run with
`--backend aot_eager` for comparison, as the sibling case does.

Known-bad torch versions: RED reproduced on torch 2.14.0+cpu (git
08187d9e0fba026dc8217405802ab5381dc88d90), aarch64, CPU, in this
repository's own venv, 2026-09-03. PR 195484 (the fix for the direct-input
sibling case) was still open and unmerged as of 2026-09-02, and its diff
does not claim to cover this view shape; no fix is known to exist for it.

Unlike the five C-1 cases (`cases/README.md` "Two shapes of file"), this
file has no separate discovery-convention twin: it exposes the module-level
`fn` and `inputs` PLAN.md's discovery convention looks for *in addition to*
the standalone `build()`/`check()`/`main()` shape, so `torch-compile-check
cases/alias_view_slice_scatter_copyback.py` runs it through the tool
directly and `tests/test_corpus_twins.py` names this same file as its own
twin.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import sys

import torch


def fn(x: torch.Tensor, weight: torch.Tensor, src: torch.Tensor):
    """A view of the graph input is scattered into, then copied back through
    another view of the same input.

    ``before`` is an ordinary read of the view before the scatter; returning
    it alongside ``updated`` is what makes the reinplace pass treat the view
    as eliminable. ``updated`` is the value the alias oracle cares about: in
    eager it must be an independent tensor.
    """
    view = x.permute(1, 0)  # (3, 4), a view of x -- no copy
    before = view @ weight  # (3, 4) @ (4, 2) -> (3, 2)
    updated = torch.slice_scatter(view, src, 0, 0, 1)  # (3, 4), functional
    x.copy_(updated.permute(1, 0))  # write back into x through a view
    return before, updated


# Fixed literal values, on the pattern of alias_slice_scatter_copyback.py:
# small enough to read at a glance, and this exact shape (x permuted to
# (3, 4), weight (4, 2), src one row of width 4) is what reproduces the
# reinplace on this torch build.
inputs = (
    torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]]),
    torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]]),
    torch.tensor([[100.0, 200.0, 300.0, 400.0]]),
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
    """inputs is the *original* (x, weight, src) triple passed to fn for the
    eager run.

    Runs its own fresh input triple for the compiled call internally via
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
    x, weight, src = (t.clone() for t in inputs)
    if backend_name == "eager":
        _before, updated = fn(x, weight, src)
    else:
        compiled_fn = torch.compile(fn, backend=backend_name)
        _before, updated = compiled_fn(x, weight, src)
    return x, updated


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

    case = "alias_view_slice_scatter_copyback"
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
