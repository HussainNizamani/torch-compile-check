"""Markdown report.

PLAN.md "Reports": Markdown is an issue draft formatted the way PyTorch issues
expect -- a short description, the minimal repro inline as a fenced Python
block, expected versus actual, the stage-localization verdict, the emitted
regression test, and an environment block with torch version and git hash,
Python version, OS, architecture, CPU or GPU model, and the backend
configuration that was in force. The tool drafts; the human reads it, edits it,
and files it.

Three decisions about what this draft does *not* do, because each is a way a
generated report could do harm.

It adds no disclosure line. "AI assisted." belongs on the project's own commits
and pull requests, where the project decided to put it; whether and how a person
discloses tooling on an issue they file under their own name is theirs to
decide, and a tool that wrote it into the draft would be making that choice for
them.

It never claims the bug is in a stage. The stage section carries PLAN.md "Where
divergence appears is not always where the fix belongs" verbatim from the
verdict, because a maintainer reading "the bug is in inductor" from a tool that
ran three backends would be right to stop reading.

It says whether it minimized. The repro block is the target's own source,
reduced to the statements the entry point and the inputs need
(:mod:`torch_compile_check.report.repro`), which is a *shorter* file and not a
*smaller case*. ``--minimize`` is what makes the case smaller, and the draft
either carries the "Minimized" section that says what was removed or says the
flag was not passed, so a maintainer is never left to assume a reproducer is
minimal because a tool produced it.

Nothing here imports torch.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from torch_compile_check import __version__
from torch_compile_check.localize import MODEL, NO_REFERENCE, StageVerdict
from torch_compile_check.minimize import Minimization
from torch_compile_check.oracles import DEFAULT_GRAD_TOL_FACTOR, ORACLE_NAMES, Finding
from torch_compile_check.report import repro as repro_source
from torch_compile_check.report.pytest_case import emit, select
from torch_compile_check.results import RunSet

__all__ = ["render", "title"]

# Per severity group, as in the terminal report: a draft with two hundred
# numerics findings is not a better issue than one with ten and a count.
DEFAULT_MAX_FINDINGS = 10

# What each oracle's finding changed, in the words an issue title uses. The
# metadata row is filled in from the finding's own field, which is the one that
# reads best in a title ("changes the output dtype").
_SUBJECT: dict[str, str] = {
    "numerics": "the output values",
    "alias": "the aliasing of the outputs",
    "metadata": "the output metadata",
    "grad": "the gradients",
    "graph": "the graph health",
}


def render(
    runset: RunSet,
    findings: Sequence[Finding],
    verdict: StageVerdict,
    *,
    fail_on: Sequence[str] = (),
    grad_tol_factor: float = DEFAULT_GRAD_TOL_FACTOR,
    baseline: str | None = None,
    max_findings: int = DEFAULT_MAX_FINDINGS,
    minimized: Minimization | None = None,
) -> str:
    """Render a run as a Markdown issue draft.

    Args:
        runset: the run, from :func:`torch_compile_check.runner.run_all`.
        findings: every finding the oracles produced.
        verdict: the stage verdict, from :func:`torch_compile_check.localize.localize`.
        fail_on: the ``--fail-on`` categories, for the command line the draft
            quotes: an issue whose reader cannot re-run the tool the same way is
            an issue with a repro that does not reproduce.
        grad_tol_factor: ``--grad-tol-factor``, for the environment block.
        baseline: the ``--baseline`` path, when one was in force.
        max_findings: how many findings to list. The rest are counted.
        minimized: what ``--minimize`` did, or ``None`` when it was not asked
            for. It changes two things: the repro block says whether the case
            was reduced, and a "Minimized" section lists what was removed.

    Returns:
        The draft, ready to write. The first line is the issue title.
    """
    repro = repro_source.extract(runset.target_source)
    blocks = [
        f"# {title(runset, findings, verdict)}",
        _preamble(runset, verdict, findings),
        _repro(runset, repro, minimized),
        _minimized(minimized),
        _findings(findings, max_findings),
        _stage(verdict),
        _test(runset, findings, verdict, minimized),
        _environment(runset, grad_tol_factor, baseline),
        _command(runset, fail_on, baseline, minimized),
    ]
    return "\n\n".join(block for block in blocks if block) + "\n"


def title(runset: RunSet, findings: Sequence[Finding], verdict: StageVerdict) -> str:
    """The issue title, derived from the finding a reader would file about.

    Shaped like the titles the PyTorch tracker already carries -- the lane in
    brackets, then what changed, then where -- because a title is the only part
    of a report most people read, and one written in the tracker's own idiom is
    read as a bug report rather than as tool output.
    """
    target = _target_label(runset)
    top = select(findings)
    if top is not None:
        return f"[{top.backend}] torch.compile changes {_subject(top)} of {target}"

    if not verdict.compared:
        # No working eager run to have diverged from -- MODEL (eager itself
        # raised) or NO_REFERENCE (no eager lane ran at all) alike. A compiled
        # lane that also raised in either case did not diverge from anything,
        # so the title names what actually happened rather than naming a
        # compiled lane as though torch.compile were the one that broke it.
        # NO_REFERENCE reuses _preamble's own phrase for the same fact.
        if verdict.eager_exception is not None:
            return (
                f"torch-compile-check could not compare {target}: "
                f"the eager reference raised {verdict.eager_exception.type}"
            )
        return (
            f"torch-compile-check could not compare {target}: "
            "the run had no eager lane to compare against"
        )

    raised = next(
        (
            entry
            for entry in verdict.backends
            if entry.raised is not None and entry.backend != "eager"
        ),
        None,
    )
    if raised is not None and raised.raised is not None:
        return f"[{raised.backend}] torch.compile raises {raised.raised.type} on {target}"
    return f"torch-compile-check found no divergence on {target}"


def _subject(finding: Finding) -> str:
    """What one finding changed, as the phrase a title uses."""
    if finding.oracle == "metadata":
        field = finding.details.get("field")
        if isinstance(field, str) and field not in {"type", "output_spec", "output_count"}:
            return f"the output {field}"
    return _SUBJECT.get(finding.oracle, f"the {finding.oracle} behaviour")


def _preamble(runset: RunSet, verdict: StageVerdict, findings: Sequence[Finding]) -> str:
    """The paragraph under the title: what ran, and what came back."""
    lanes = [name for name in runset.backends if name != "eager"]
    counted = len([finding for finding in findings if finding.severity == "fail"])
    lines = [
        "> Drafted by "
        "[torch-compile-check](https://github.com/HussainNizamani/torch-compile-check) "
        f"{__version__}. "
        "The line above is the issue title; everything below is the body. "
        "Read it, check it, and edit it before filing -- the tool drafts, a person files.",
        "",
    ]
    if not verdict.compared:
        why = (
            "the model raised under eager, so nothing was compiled"
            if verdict.stage == MODEL
            else "the run had no eager lane to compare against"
        )
        lines.append(
            f"`{runset.target_name}` was not compared: {why}. There is nothing to file here "
            "until that is fixed."
        )
        return "\n".join(lines)

    ran = ", ".join(f"`{name}`" for name in lanes) if lanes else "no compiled lane"
    if not findings:
        lines.append(
            f"`{runset.target_name}` was run under eager and {ran} and no oracle reported a "
            "divergence. This draft is a record of a clean run rather than a bug report."
        )
        return "\n".join(lines)

    lines.append(
        f"`{runset.target_name}` was run under eager and {ran} on the environment at the bottom "
        f"of this report. {_count_phrase(counted, len(findings))} {_sentence(verdict.summary)}"
    )
    return "\n".join(lines)


def _sentence(text: str) -> str:
    """One of the verdict's own phrases, as a sentence in a paragraph.

    ``str.capitalize`` would lowercase the rest, and the rest carries backend
    names the report has to keep as they are.
    """
    stripped = text.strip()
    if not stripped:
        return ""
    ended = stripped if stripped.endswith((".", "!", "?")) else f"{stripped}."
    return ended[0].upper() + ended[1:]


def _count_phrase(fails: int, total: int) -> str:
    """ "Two of five findings break the contract", in one sentence."""
    findings = f"{total} finding{'' if total == 1 else 's'}"
    if fails == total:
        return f"The oracles reported {findings}, all of them fail-severity."
    if fails == 0:
        return f"The oracles reported {findings}, none of them fail-severity."
    return f"The oracles reported {findings}, {fails} of them fail-severity."


def _repro(
    runset: RunSet,
    repro: repro_source.Repro | None,
    minimized: Minimization | None = None,
) -> str:
    """The reproducer: the target's own source, and how the two lanes were run."""
    if repro is None:
        return (
            "## Repro\n\n"
            f"The source of `{runset.target_name}` was not available when this draft was "
            "written, so there is nothing to inline here. Attach the target file before filing."
        )

    note = (
        "The target's own source, reduced to the statements the entry point and the inputs need."
        if repro.complete
        else f"The whole of `{_short(repro.file) or runset.target_name}`: the entry point or the "
        "inputs are not bound by a plain top-level statement, so the tool did not try to reduce it."
    )
    code = [*repro.future_imports, ""] if repro.future_imports else []
    code.append(repro.source)
    driver = _driver(runset, repro)
    if driver:
        code += ["", driver]
    return "\n".join(
        [
            "## Repro",
            "",
            f"{note} {_minimality(minimized)}",
            "",
            "```python",
            *code,
            "```",
        ]
    )


def _minimality(minimized: Minimization | None) -> str:
    """Whether the block above is a minimized case, in one sentence.

    The distinction the section's docstring is about: this block is always the
    *whole* target, because a reader has to be able to run it. What ``--minimize``
    established is which parts of it the finding does not need, and that is the
    section below rather than an edit to the code above.
    """
    if minimized is None:
        return "It is the whole case: run torch-compile-check again with `--minimize` to shrink it."
    if minimized.finding is None or not minimized.reproduced or not minimized.changed:
        return f"It is the whole case: `--minimize` {minimized.summary}."
    return (
        "It is the whole case; the **Minimized** section below says which parts of it the "
        "finding does not need."
    )


def _minimized(minimized: Minimization | None) -> str:
    """What ``--minimize`` reduced the case to, for the maintainer who triages it.

    Written as prose and a list rather than as a diff, because the reduction is
    two statements about the original -- these children are not needed, these
    inputs can be this small -- and a maintainer acts on those before they act
    on a patch.
    """
    if minimized is None or minimized.finding is None:
        return ""
    lines = ["## Minimized", "", _sentence(minimized.summary)]
    if not minimized.reproduced:
        lines += ["", *[f"- {note}" for note in minimized.notes]]
        return "\n".join(lines)

    bullets = [f"- input leaf {s.index}: `{s.before}` -> `{s.after}`" for s in minimized.shrinks]
    bullets += [
        f"- `{stub.path}` ({stub.module}) replaced with `torch.nn.Identity()`"
        for stub in minimized.stubs
    ]
    bullets += [f"- kept `{k.path}` ({k.module}): {k.reason}" for k in minimized.kept]
    bullets += [f"- {note}" for note in minimized.notes]
    if bullets:
        lines += ["", *bullets]
    if minimized.partial and minimized.partial_reason is not None:
        lines += [
            "",
            f"This reduction is **partial**: {minimized.partial_reason}. What is left still "
            "reproduces, and there may be more to remove.",
        ]
    lines += ["", minimized.handoff]
    return "\n".join(lines)


def _driver(runset: RunSet, repro: repro_source.Repro) -> str:
    """The two lines that ran the target under both worlds.

    Written against the names the block above already binds, so the two stay one
    runnable script. The comment about the inputs is not a formality: the runner
    clones them per lane, and a reader who ran these two lines against a target
    that mutates its inputs would see something the tool did not (PLAN.md
    "Runner semantics").
    """
    if repro.entry is None:
        return ""
    lane = _compiled_lane(runset)
    call = _call(repro)
    return "\n".join(
        [
            "# torch-compile-check ran it like this. It gives each lane its own clone of the "
            "inputs,",
            "# so rebuild them between the two calls if the target mutates what it is given.",
            f"expected = {repro.entry}{call}",
            f'actual = torch.compile({repro.entry}, backend="{lane}"){call}',
        ]
    )


def _call(repro: repro_source.Repro) -> str:
    """The argument list the entry point is called with."""
    if repro.inputs_ref is None:
        return "()"
    return f"({'**' if repro.keyword_inputs else '*'}{repro.inputs_ref})"


def _findings(findings: Sequence[Finding], max_findings: int) -> str:
    """Expected versus got, one bullet per finding, strongest severity first."""
    if not findings:
        return ""
    ordered = sorted(findings, key=_order)
    shown = ordered[: max(0, max_findings)]
    lines = ["## Expected versus got", ""]
    for finding in shown:
        where = "the run" if finding.output_index is None else f"output[{finding.output_index}]"
        lines.append(
            f"- **[{finding.severity}] {finding.oracle} · {finding.backend} · {where}** — "
            f"{finding.message}"
        )
        detail = _detail(finding)
        if detail:
            lines.append(f"  - {detail}")
    hidden = len(ordered) - len(shown)
    if hidden > 0:
        lines += ["", f"{hidden} further finding{'' if hidden == 1 else 's'} are not listed here."]
    return "\n".join(lines)


def _detail(finding: Finding) -> str:
    """One finding's expected/got pair and the rule it was decided under."""
    details = finding.details
    parts = []
    if "expected" in details or "got" in details:
        parts.append(f"expected `{_show(details.get('expected'))}`")
        parts.append(f"got `{_show(details.get('got'))}`")
    if "rtol" in details and "atol" in details:
        parts.append(f"tolerances rtol `{_show(details['rtol'])}`, atol `{_show(details['atol'])}`")
    if "field" in details:
        parts.append(f"field `{_show(details['field'])}`")
    return ", ".join(parts)


def _stage(verdict: StageVerdict) -> str:
    """The stage-localization verdict and the caveat that belongs with it."""
    lines = ["## Stage", "", _sentence(verdict.summary)]
    if verdict.note and verdict.stage not in (MODEL, NO_REFERENCE):
        lines += ["", _sentence(verdict.note)]
    return "\n".join(lines)


def _test(
    runset: RunSet,
    findings: Sequence[Finding],
    verdict: StageVerdict,
    minimized: Minimization | None = None,
) -> str:
    """The regression test, if there is a finding to write one from.

    PLAN.md "Regression test emission": when a maintainer accepts a bug report,
    the next thing they ask for is a test, and handing them one already written
    in their own house style removes the step where a report stalls.
    """
    # Not the standalone file: the target is already quoted in the repro block
    # above, and an issue that carried it twice would leave a reader diffing one
    # copy against the other.
    case = emit(runset, findings, verdict, standalone=False, minimized=minimized)
    if case is None:
        return ""
    return "\n".join(
        [
            "## Regression test",
            "",
            "A starting point in the idiom `test/inductor/test_torchinductor.py` uses, against "
            "the target above. It is half-written rather than ready: check the assertion says "
            "what you mean before you paste it in.",
            "",
            "```python",
            case.rstrip("\n"),
            "```",
        ]
    )


def _environment(runset: RunSet, grad_tol_factor: float, baseline: str | None) -> str:
    """The environment block PLAN.md "Reports" asks every draft to carry."""
    env = runset.env
    torch_line = str(env.get("torch_version"))
    git_hash = env.get("torch_git_version")
    if git_hash:
        torch_line += f" (git `{str(git_hash)[:12]}`)"
    machine = str(env.get("machine"))
    flags = env.get("cpu_flags")
    if flags:
        machine += f", cpu flags `{flags}`"

    rows = [
        ("torch", torch_line),
        ("python", str(env.get("python_version"))),
        ("os", str(env.get("platform"))),
        # PLAN.md "Cross-architecture parity is a feature": always carried, on
        # every report, because a run whose provenance is ambiguous is not
        # usable as parity evidence.
        ("architecture", machine),
        ("device", f"{runset.device} (cuda available: {_yes_no(env.get('cuda_available'))})"),
        ("backends", ", ".join(f"`{name}`" for name in runset.backends)),
        (
            "compile flags",
            f"fullgraph={runset.fullgraph}, dynamic={runset.dynamic}, "
            f"backward={'on' if runset.grad else 'off'}, seed={runset.seed}",
        ),
        ("module", runset.module_handling),
        ("gradient tolerance", f"the numerics tolerances x{grad_tol_factor:g}"),
        ("inductor caches", _caches(env.get("inductor_force_disable_caches"))),
    ]
    if baseline is not None:
        rows.append(("graph baseline", f"`{baseline}` (new breaks only)"))
    lines = ["## Environment", ""]
    lines += [f"- **{name}**: {value}" for name, value in rows]
    return "\n".join(lines)


def _command(
    runset: RunSet,
    fail_on: Sequence[str],
    baseline: str | None,
    minimized: Minimization | None = None,
) -> str:
    """The command that produced this, so a reader can run it themselves."""
    source = runset.target_source
    target = _short(source.file) if source is not None else None
    parts = ["torch-compile-check", target or runset.target_name]
    parts += ["--backends", ",".join(runset.backends)]
    if runset.device != "cpu":
        parts += ["--device", runset.device]
    if runset.fullgraph:
        parts.append("--fullgraph")
    if runset.dynamic:
        parts.append("--dynamic")
    if not runset.grad:
        parts.append("--no-grad")
    if runset.share_module:
        parts.append("--share-module")
    if fail_on:
        parts += ["--fail-on", ",".join(fail_on)]
    if baseline is not None:
        parts += ["--baseline", baseline]
    if minimized is not None:
        parts.append("--minimize")
    if runset.seed != 0:
        parts += ["--seed", str(runset.seed)]
    return "\n".join(["## How this was produced", "", "```console", f"$ {' '.join(parts)}", "```"])


def _order(finding: Finding) -> tuple[int, int]:
    """Sort key: fail before warn before info, then in oracle order."""
    severity = {"fail": 0, "warn": 1, "info": 2}.get(finding.severity, 3)
    oracle = (
        ORACLE_NAMES.index(finding.oracle) if finding.oracle in ORACLE_NAMES else len(ORACLE_NAMES)
    )
    return severity, oracle


def _compiled_lane(runset: RunSet) -> str:
    """The lane a repro should compile with: the last one on the ladder that ran."""
    lanes = [name for name in runset.backends if name != "eager"]
    return lanes[-1] if lanes else "inductor"


def _target_label(runset: RunSet) -> str:
    """How the draft names the target: its file if it has one, else its symbol."""
    source = runset.target_source
    short = _short(source.file) if source is not None else None
    return short or runset.target_name


def _short(path: str | None) -> str | None:
    """A path as a reader would type it: relative to the working directory.

    A path that is already relative is already that, and one that leads
    somewhere else entirely is named by its file: a draft is read on GitHub, not
    on the machine the run happened on, and an absolute path from a stranger's
    home directory is noise in an issue.
    """
    if not path:
        return None
    location = Path(path)
    if not location.is_absolute():
        return str(location)
    try:
        return str(location.relative_to(Path.cwd()))
    except ValueError:
        return location.name


def _show(value: Any) -> str:
    """One details value, rendered the way a draft reads it."""
    if isinstance(value, Mapping):
        return ", ".join(f"{key}={item}" for key, item in value.items())
    if isinstance(value, list | tuple):
        return ", ".join(str(item) for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _caches(disabled: bool | None) -> str:
    """What the cache setting was, in a sentence a reader can act on."""
    if disabled is None:
        return "unknown: torch did not report `force_disable_caches`"
    if disabled:
        return "disabled (`TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`)"
    return "enabled (`--allow-caches`), so the run may have measured a cached artifact"


def _yes_no(value: bool | None) -> str:
    """``yes``/``no``, or ``unknown`` when the fact could not be established."""
    return "unknown" if value is None else ("yes" if value else "no")
