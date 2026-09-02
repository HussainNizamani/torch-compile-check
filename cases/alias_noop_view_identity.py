"""A no-op view returned alongside its base must stay a distinct Python
tensor object sharing storage, exactly as in eager. Before the fix, Inductor
collapses the two into a single returned object.

Issue: https://github.com/pytorch/pytorch/issues/191449
Fix PR: https://github.com/pytorch/pytorch/pull/191844 (merged 2026-09-02,
commit a3586f00181a395b066cdf3a8933c2b47b7a6890)

Contract violated: aliasing / object identity. `base = x + 1; return base,
base.view(-1)` gives two distinct tensor objects in eager, sharing one
storage. Without the fix, `collect_metadata_analysis.py` only classifies a
view as an alias-needing-regeneration when both it and its base require
grad; a no-grad no-op view falls into `non_alias`, so nothing stops the
backend from returning one Python object for both outputs (Inductor's
`remove_noop_ops` / `pointless_view` do exactly that: `return (buf0, buf0,)`).
That is directly observable as `out_base is out_alias`, and it has a real
consequence: `resize_()` graph-breaks (documented as gb0126), so it runs in
eager between two compiled graphs. If the alias and the base are one object,
resizing the alias resizes the base too, and the base is fed into the next
compiled graph with the wrong shape.

Oracle: alias (object identity + downstream shape corruption after a
graph-break).

First diverging backend: the issue frames the root cause as an AOTAutograd
metadata-classification bug (`collect_metadata_analysis.py`), upstream of
backend codegen, so it would be expected to affect `aot_eager` too. Run
here with `--backend aot_eager`: RED on `inductor`, GREEN on `aot_eager` --
`aot_eager` did NOT reproduce the shape corruption on this box, so the
observed split contradicts the "same root cause, both backends" framing for
this specific resize_-after-graph-break repro; not investigated further
here.

Known-bad torch versions: RED reproduced here on torch
2.15.0.dev20260831+cpu (git cbf102a9aec0f6f83466e0584e66d9a96ab613f6),
aarch64, CPU, on the default `inductor` backend -- this build predates the
fix, which merged at 2026-09-02T03:45:57Z. Expected GREEN on any torch
nightly built from a checkout that includes commit
a3586f00181a395b066cdf3a8933c2b47b7a6890 or later; not yet verified against
such a build (none was available in this environment at the time this case
was written).
"""

import os

os.environ.setdefault("TORCHINDUCTOR_FORCE_DISABLE_CACHES", "1")

import platform
import subprocess
import sys

import torch


def build():
    def fn(x):
        base = x + 1
        alias = base.view(-1)
        alias.resize_(12)
        return base + 1

    example_inputs = (torch.zeros(1),)
    return fn, example_inputs


def _git_hash():
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            capture_output=True,
            text=True,
            timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def check(eager_out, compiled_out, inputs):
    if not torch.equal(eager_out, compiled_out):
        return True, (
            f"compiled output diverges from eager after resize_() on a "
            f"no-op view: eager={eager_out.tolist()} "
            f"compiled={compiled_out.tolist()}"
        )
    return False, f"compiled output matches eager: {compiled_out.tolist()}"


def _identity_probe():
    """Cheaper, no-crash probe of the underlying object-identity bug: does
    the compiled function return the same Python object for a base and its
    no-op view? Reported alongside the RED/GREEN line as extra context.
    """

    def fn_view(x):
        base = x + 1
        return base, base.view(-1)

    base, alias = torch.compile(fn_view)(torch.zeros(1))
    return base is alias


def _report(case, backend_name, eager_out, compiled_out, extra=""):
    is_red, message = check(eager_out, compiled_out, None)
    status = "RED" if is_red else "GREEN"
    print(
        f"{status} {case} torch={torch.__version__} git={_git_hash()} "
        f"arch={platform.machine()} backend={backend_name} :: {message}{extra}"
    )
    return is_red


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default=None, help="extra backend to also check, e.g. aot_eager")
    args = parser.parse_args()

    case = "alias_noop_view_identity"
    fn, example_inputs = build()

    try:
        eager_out = fn(*example_inputs)
        compiled_out = torch.compile(fn, backend="inductor")(*example_inputs)
    except Exception as exc:  # noqa: BLE001 - a crash in the case itself, not a RED finding
        print(f"CRASH {case} :: {type(exc).__name__}: {exc}")
        sys.exit(2)

    identity_collapsed = _identity_probe()
    is_red = _report(case, "inductor", eager_out, compiled_out, extra=f" (base_is_alias={identity_collapsed})")
    exit_code = 1 if is_red else 0

    if args.backend:
        extra_out = torch.compile(fn, backend=args.backend)(*example_inputs)
        _report(case, args.backend, eager_out, extra_out)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
