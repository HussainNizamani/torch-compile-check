"""The target's own source, reduced to the lines a repro needs.

PLAN.md "Package layout" does not name this module, for the same reason it does
not name :mod:`torch_compile_check.results`: it is shared vocabulary. PLAN.md
"Reports" wants "the minimal repro inline as a fenced Python block" in the
Markdown draft and PLAN.md "Regression test emission" wants the same case as a
test method, so two reports need the same answer to one question -- which lines
of the user's file are the reproducer -- and deriving it twice would let the
draft and the test disagree about what was run.

What "reduced" means here, and what it deliberately does not mean. This is not
the minimizer: M3-3 shrinks the *run* by re-running it, and everything below is
a read of the file's syntax that runs nothing. It keeps the statements the
entry point and the inputs are built from, transitively, and drops the module
docstring and any statement nothing in that closure refers to. A file whose
entry point is not bound by a plain top-level statement -- built inside a
``with`` block, say -- cannot be reduced that way at all, and then the whole
file is the repro and :attr:`Repro.complete` says so. Guessing would be worse:
a repro that does not run is worse than a long one that does.

Nothing here imports torch. It parses text.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass

from torch_compile_check.results import TargetSource

__all__ = ["Repro", "extract"]

log = logging.getLogger("torch_compile_check")

# The issue link a corpus case carries in its own docstring, e.g.
# "Issue: https://github.com/pytorch/pytorch/issues/191308 -- ...". Matched
# against the docstring only: a URL further down the file is as likely to be a
# reference in a comment as it is to be the bug this file reproduces.
_ISSUE_URL = re.compile(r"https?://(?:www\.)?github\.com/[\w.-]+/[\w.-]+/issues/\d+")


@dataclass(frozen=True)
class Repro:
    """One target file, reduced to the statements a reproducer needs."""

    source: str
    """The statements, in file order, ready for a fenced block or a test file.

    Never empty: a target whose pieces could not be isolated falls back to its
    whole file (see :attr:`complete`).
    """

    future_imports: tuple[str, ...] = ()
    """``from __future__ import ...`` lines, kept out of :attr:`source`.

    They are only legal directly after a module docstring, so the caller that
    writes a file decides where they go rather than finding them in the middle
    of the block it is pasting.
    """

    entry: str | None = None
    """The expression the repro calls: ``fn``, ``model``, ``model.net``."""

    inputs_expr: str | None = None
    """The expression that builds one *fresh* set of inputs.

    Taken from the right-hand side of the ``inputs`` assignment, or written as a
    call to the discovered factory (``get_inputs()``). Its point is that
    evaluating it twice makes two independent sets of inputs, which is what a
    test comparing two lanes of a target that mutates them needs. ``None`` when
    the inputs are not bound by a top-level assignment or function.
    """

    imports: tuple[str, ...] = ()
    """The import statements of :attr:`source`, one per entry.

    Separated for the caller that writes a file of its own: an emitted test has
    imports of its own to group these with, and PEP 8 wants one blank line
    between the groups rather than the two that separate definitions. Empty in
    the whole-file fallback, where the statements were never taken apart.
    """

    body: str = ""
    """:attr:`source` without :attr:`imports`, for the same caller."""

    body_without_inputs: str = ""
    """:attr:`body` minus the module-level ``inputs`` assignment.

    The test emitter turns :attr:`inputs_expr` into a factory, so keeping the
    assignment as well would put the same tuple in the generated file twice,
    once under a name nothing calls. Equal to :attr:`body` when there is no such
    assignment to drop.
    """

    inputs_ref: str | None = None
    """The shortest expression for the inputs *inside* :attr:`source`.

    The bound name (``inputs``) where the file binds one, and the factory call
    (``get_inputs()``) where it binds a factory instead. The difference from
    :attr:`inputs_expr` is what each is for: a block that already contains the
    assignment refers to it by name, and a file that has to build a second
    independent set repeats the expression.
    """

    keyword_inputs: tuple[str, ...] = ()
    """Copied from :class:`~torch_compile_check.results.TargetSource`, so a caller
    that writes a call knows whether it is ``fn(*inputs)`` or ``fn(**inputs)``."""

    complete: bool = True
    """Whether the reduction found the pieces, or fell back to the whole file.

    ``False`` is not a failure -- the source is still there and still runs --
    but a caller that emits code has to say so, because a whole file may carry
    top-level work the reduced form would have dropped.
    """

    issue: str | None = None
    """The issue URL the module docstring names, when it names one."""

    file: str | None = None
    """The path the source came from, for a report to cite.

    Copied straight from :attr:`~torch_compile_check.results.TargetSource.file`,
    which is already the display form -- relative to the working directory or
    exactly as given, never a resolved path -- so nothing here shortens it
    again."""


def extract(source: TargetSource | None) -> Repro | None:
    """Reduce a target file to its reproducer.

    Args:
        source: what discovery recorded about the target, from
            :attr:`torch_compile_check.results.RunSet.target_source`.

    Returns:
        The reduced repro, or ``None`` when there is no source to reduce --
        a hand-built run, or a target file that could not be read. ``None``
        means "quote nothing", which is what a report must do rather than
        describing code it has not seen.
    """
    if source is None or not source.text:
        return None

    text = source.text
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        # The module imported, so it parsed for the interpreter; reaching this
        # means the file on disk is not the file that ran any more.
        log.debug("cannot parse the target source at %s: %s", source.file, exc)
        return Repro(
            source=text,
            body=text,
            body_without_inputs=text,
            complete=False,
            entry=source.entry,
            keyword_inputs=source.keyword_inputs,
            file=source.file,
        )

    issue = _issue(tree)
    future, statements = _split_futures(tree.body)
    bound = _bindings(statements)
    wanted = [name for name in (source.entry, source.inputs) if name is not None]
    kept = _closure(statements, bound, wanted)
    # Answered before the fallback and not after it: how the inputs are built is
    # a fact about the file, and it holds whether or not the entry point could
    # be isolated. A whole-file repro still wants a factory.
    expression, reference = _inputs_expression(text, statements, bound, source.inputs)

    if kept is None:
        # The whole file, minus its ``__future__`` imports, which are carried
        # out the same way the reduced form carries them. They are only legal
        # directly under a module docstring, so a caller that pastes this block
        # into the middle of a generated file -- which is exactly what
        # report/pytest_case.py does -- would otherwise produce a file that does
        # not parse. Everything else is left verbatim: the point of the fallback
        # is that the tool did not understand the file well enough to edit it.
        whole = _drop_lines(text, future)
        return Repro(
            source=whole,
            body=whole,
            body_without_inputs=whole,
            future_imports=tuple(_segment(text, node) for node in future),
            entry=source.entry,
            inputs_expr=expression,
            inputs_ref=reference,
            keyword_inputs=source.keyword_inputs,
            complete=False,
            issue=issue,
            file=source.file,
        )

    # The assignment the emitter drops, and only when it has the expression to
    # replace it with: dropping it otherwise would leave a file naming inputs
    # nothing builds. `reference == source.inputs` is what says the binding was
    # an assignment rather than the `get_inputs()` factory, which stays.
    dropped = kept
    if expression is not None and source.inputs is not None and reference == source.inputs:
        dropped = kept - {bound[source.inputs]}
    imports = _imports(text, statements, kept)
    body = _body(text, statements, kept)
    return Repro(
        source=_render(imports, body),
        body_without_inputs=_body(text, statements, dropped),
        imports=imports,
        body=body,
        future_imports=tuple(_segment(text, node) for node in future),
        entry=source.entry,
        inputs_expr=expression,
        inputs_ref=reference,
        keyword_inputs=source.keyword_inputs,
        complete=True,
        issue=issue,
        file=source.file,
    )


def _issue(tree: ast.Module) -> str | None:
    """The issue URL the module docstring names, or ``None``."""
    docstring = ast.get_docstring(tree)
    if not docstring:
        return None
    found = _ISSUE_URL.search(docstring)
    return found.group(0) if found else None


def _split_futures(body: list[ast.stmt]) -> tuple[list[ast.stmt], list[ast.stmt]]:
    """Split the module body into ``__future__`` imports and everything else.

    The module docstring goes with neither: it is prose about the bug, and a
    repro block that opened with three paragraphs of it would bury the code.
    """
    futures: list[ast.stmt] = [
        node for node in body if isinstance(node, ast.ImportFrom) and node.module == "__future__"
    ]
    rest = [node for node in body if node not in futures and not _is_docstring(node, body)]
    return futures, rest


def _is_docstring(node: ast.stmt, body: list[ast.stmt]) -> bool:
    """Whether ``node`` is this module's docstring."""
    return (
        bool(body)
        and node is body[0]
        and isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    )


def _bindings(statements: list[ast.stmt]) -> dict[str, int]:
    """Map every name a top-level statement binds to that statement's index.

    First binding wins, which matches how a reader reads the file: a name
    rebound later is the same object's story, and pulling in both statements
    would put the rebinding in a repro that never asked for it.

    Only the statement kinds that bind a name *at module level and in one
    statement* are read. A name bound inside an ``if``, a ``with``, or a ``try``
    is deliberately not found: the binding depends on control flow the reduction
    would have to reproduce, and :func:`_closure` falls back to the whole file
    rather than emitting something that does not run.
    """
    bound: dict[str, int] = {}
    for index, node in enumerate(statements):
        for name in _binds(node):
            bound.setdefault(name, index)
    return bound


def _binds(node: ast.stmt) -> list[str]:
    """The names one top-level statement binds."""
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [name for target in node.targets for name in _target_names(target)]
    if isinstance(node, ast.AnnAssign | ast.AugAssign):
        return _target_names(node.target)
    if isinstance(node, ast.Import | ast.ImportFrom):
        return [
            alias.asname or alias.name.partition(".")[0]
            for alias in node.names
            if alias.name != "*"
        ]
    return []


def _target_names(target: ast.expr) -> list[str]:
    """The names one assignment target binds, unpacking tuples and lists."""
    if isinstance(target, ast.Name):
        return [target.id]
    if isinstance(target, ast.Tuple | ast.List):
        return [name for element in target.elts for name in _target_names(element)]
    return []


def _closure(
    statements: list[ast.stmt],
    bound: dict[str, int],
    wanted: list[str],
) -> set[int] | None:
    """The statements the wanted names need, transitively.

    Every import is kept whatever it is used by: an import is cheap, deciding
    which one a name inside a function body came from means resolving names
    this module deliberately does not resolve, and a repro missing an import is
    a repro that does not run.

    Returns:
        The indices to keep, or ``None`` when a wanted name is not bound by a
        top-level statement at all, which is the signal to fall back.
    """
    if not wanted:
        return None

    keep = {
        index
        for index, node in enumerate(statements)
        if isinstance(node, ast.Import | ast.ImportFrom)
    }
    queue: list[int] = []
    for name in wanted:
        index = bound.get(name)
        if index is None:
            log.debug("cannot isolate %r in the target source; using the whole file", name)
            return None
        queue.append(index)

    while queue:
        index = queue.pop()
        if index in keep:
            continue
        keep.add(index)
        queue.extend(
            bound[name]
            for name in _names_used(statements[index])
            if name in bound and bound[name] not in keep
        )
    return keep


def _names_used(node: ast.stmt) -> set[str]:
    """Every name read anywhere inside one statement, function bodies included.

    Bodies are walked rather than skipped because that is where a repro's real
    dependencies are: ``def fn(a, b): return helper(a, b)`` needs ``helper``,
    and a class needs whatever its methods call. The cost of walking too far is
    a statement kept that was not needed; the cost of walking too little is a
    ``NameError`` in code the tool handed a user.
    """
    return {
        child.id
        for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }


def _inputs_expression(
    text: str,
    statements: list[ast.stmt],
    bound: dict[str, int],
    inputs: str | None,
) -> tuple[str | None, str | None]:
    """How to build a fresh set of inputs, and how to name the existing one.

    Fresh matters: a test that compares two lanes of a target that mutates its
    inputs has to build them twice, so an assignment's right-hand side is what
    is wanted there and the assigned *name* is not. A block that already carries
    the assignment wants the opposite, which is the second value.
    """
    if inputs is None:
        return None, None
    index = bound.get(inputs)
    if index is None:
        return None, None
    node = statements[index]
    if isinstance(node, ast.Assign):
        return _segment(text, node.value), inputs
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        # The `get_inputs()` half of the discovery convention: the factory is
        # already in the repro, so both answers are a call to it.
        return f"{node.name}()", f"{node.name}()"
    return None, None


def _imports(text: str, statements: list[ast.stmt], kept: set[int]) -> tuple[str, ...]:
    """The kept import statements, in file order, one string each."""
    return tuple(
        _segment(text, node)
        for index, node in enumerate(statements)
        if index in kept and isinstance(node, ast.Import | ast.ImportFrom)
    )


def _body(text: str, statements: list[ast.stmt], kept: set[int]) -> str:
    """The kept statements that are not imports, two blank lines apart.

    Two, because that is what PEP 8 puts between top-level definitions and what
    a formatter would put back; a caller writing a file rather than quoting one
    wants the imports separately, which is why they are not here.
    """
    return "\n\n\n".join(
        _segment(text, node)
        for index, node in enumerate(statements)
        if index in kept and not isinstance(node, ast.Import | ast.ImportFrom)
    )


def _render(imports: tuple[str, ...], body: str) -> str:
    """The two halves as one block, spaced the way Python is written."""
    if not imports:
        return body
    head = "\n".join(imports)
    return f"{head}\n\n\n{body}" if body else head


def _drop_lines(text: str, nodes: Sequence[ast.stmt]) -> str:
    """``text`` without the lines ``nodes`` occupy, blank edges trimmed.

    Line-based rather than statement-based on purpose: the whole-file fallback
    keeps the user's own formatting, comments included, and rebuilding it from
    the statements it parsed would silently reformat a file the tool has just
    admitted it could not take apart.
    """
    if not nodes:
        return text
    removed = {
        line for node in nodes for line in range(node.lineno, (node.end_lineno or node.lineno) + 1)
    }
    lines = text.splitlines()
    kept = [line for number, line in enumerate(lines, start=1) if number not in removed]
    return "\n".join(kept).strip("\n")


def _segment(text: str, node: ast.AST) -> str:
    """One node's own source, verbatim.

    ``ast.get_source_segment`` answers ``None`` for a node with no position,
    which nothing parsed from a file has; the fallback is ``ast.unparse``, which
    is correct code but not the user's own text, and losing their formatting is
    a smaller cost than losing the statement.
    """
    segment = ast.get_source_segment(text, node)
    return segment if segment is not None else ast.unparse(node)
