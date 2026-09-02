"""Inductor reinplacing turns a non-aliasing functional result into an alias
of the input.

Issue: https://github.com/pytorch/pytorch/issues/195451
PR (open, not merged as of 2026-09-02): https://github.com/pytorch/pytorch/pull/195484

Contract violated: aliasing. `torch.slice_scatter(x, src, ...)` followed by
`x.copy_(updated)` returns `updated`. In eager, `updated` is an independent
tensor that happens to hold the same values as `x` after the copy; mutating
`updated` afterwards must not change `x`. Under Inductor,
`should_reinplace_scatter()` treats the direct copy-back as profitable and
reinplaces the scatter, so the compiled function returns the mutated input
itself -- `updated` and `x` become the same object. Mutating the returned
tensor then corrupts `x`.

Oracle: alias (identity / data_ptr comparison after the call, plus a mutate
side-effect check).

First diverging backend: the issue reports `aot_eager` clean and `inductor`
diverging. Run here with `--backend aot_eager` for comparison.

Known-bad torch versions: RED reproduced on torch 2.15.0.dev20260901+cpu
(git 279f79e09c3f3ef458061013bda2d2f483c02cae), aarch64, CPU, in a freshly
installed venv (`pip install --index-url
https://download.pytorch.org/whl/nightly/cpu torch` into a clean venv,
nothing else on top). PR 195484 (the fix) was still open, unmerged, as of
2026-09-02.

An earlier verification pass on this same box, against a torch
2.15.0.dev20260831+cpu install in a pre-existing venv
(`/tmp/pruefer_venv`), came back GREEN on this exact case, where every
unpatched nightly checked (0824, 0831, 0901) is RED. Root cause: that
venv's `torch/_inductor/fx_passes/reinplace.py` had a hand-applied
`should_reinplace_scatter()` escape-check guard
(`_scatter_result_escapes_graph`), confirmed absent from official
pytorch/pytorch source at the exact commit that venv's own
`torch.version.git_version` reported (diffed against `gh api
repos/pytorch/pytorch/contents/...?ref=<that commit>`) and absent from
current pytorch/pytorch main (`gh api search/code`, zero matches). This
turned out to be the mission's own fix-in-progress for this issue -- same
bug, same author, same day as the open PR 195484 -- but not textually
identical to that PR's current diff (different function name, different
control-flow placement, recursive vs. iterative). Most likely an earlier
local iteration of the fix applied directly to that venv's site-packages
rather than landed as a commit. So the GREEN was real in the sense that
reinplacing genuinely did not fire with that guard present; it reflects
"the fix, in some form, is in the tree" rather than "the bug is absent
upstream." Every other file spot-checked in that venv (the files touched
by the other four cases' relevant fixes) matched official source exactly;
only this one file carried a local patch. Re-running in a clean venv (guard
not present), with no other change, reproduces RED and matches the issue
and the CEO's independent ashburn run on the same build
(2.15.0.dev20260901+cpu, git 279f79e) bit for bit. Verification venvs
should be created fresh per mission and removed after; a patched venv
should not outlive the mission that patched it.

An in-region `.clone()` on the returned value does not protect against this;
the bug is that the returned *object* aliases the input at the Python level,
so any caller-side mutation of the returned tensor reaches through, not just
in-graph mutation.
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import sys

import torch


def build():
    def fn(x, src):
        updated = torch.slice_scatter(x, src, 0, 0, 1)
        x.copy_(updated)
        return updated

    example_inputs = (torch.tensor([1.0, 2.0]), torch.tensor([10.0]))
    return fn, example_inputs


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
    """inputs is the *original* (x, src) pair passed to fn for the eager run.

    Runs its own fresh input pair for the compiled call internally via main(),
    so `check` here only inspects the already-produced outputs plus a
    fresh mutate-and-observe probe done by the caller (main). This function
    reports the alias comparison; main() drives the mutate probe because it
    needs a second, unmutated copy of x to compare against.
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
    fn, _ = build()
    x = torch.tensor([1.0, 2.0])
    src = torch.tensor([10.0])
    if backend_name == "eager":
        out = fn(x, src)
    else:
        compiled_fn = torch.compile(fn, backend=backend_name)
        out = compiled_fn(x, src)
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

    case = "alias_slice_scatter_copyback"
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
