"""Entry and input resolution.

PLAN.md "Package layout": ``discover.py`` -- entry and input resolution.

PLAN.md "Discovery convention": given a file path with no overrides, import the
module and look for a module-level ``model`` (an ``nn.Module``) or ``fn``, in
that order, then for a module-level ``inputs`` or a callable ``get_inputs()``,
in that order. An override flag wins over a discovered symbol. If neither
resolves, the tool exits 2 naming the two symbols it looked for.

Discovery is deliberately narrow in v1: no directory walking, no pytest-style
collection, no config file. Two things are deliberately *not* checked here.
Nothing in this module imports torch, so ``model`` is accepted on being callable
rather than on being an ``nn.Module``; the runner is where torch enters. And a
target module is ordinary Python that the tool executes on the user's say-so, so
its import errors are reported as discovery failures rather than caught and
hidden.
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any

__all__ = ["DiscoveryError", "Target", "load_target"]

log = logging.getLogger("compile_check")

_MISSING = object()

# PLAN.md "Discovery convention", in the order they are tried.
ENTRY_ATTRS: tuple[str, ...] = ("model", "fn")
INPUT_ATTRS: tuple[str, ...] = ("inputs", "get_inputs")


class DiscoveryError(Exception):
    """Nothing to test, or nothing to test it with.

    The CLI turns this into exit code 2, PLAN.md's "tool error (import failure,
    discovery failure, backend unavailable, model raised in eager)". The message
    is user-facing and names what was looked for.
    """


@dataclass(frozen=True)
class Target:
    """The callable under test and the inputs to call it with."""

    fn: Any
    """The ``nn.Module`` or plain callable to run under every backend."""

    example_inputs: tuple[Any, ...] = ()
    """Positional arguments, exactly as the target will be called with them."""

    kwargs: dict[str, Any] = field(default_factory=dict)
    """Keyword arguments, from an ``inputs`` mapping."""

    name: str = "<target>"
    """``module:attribute``, for the report header."""


def load_target(
    path_or_module: str,
    entry: str | None = None,
    inputs: str | None = None,
) -> Target:
    """Resolve a target file or module into a :class:`Target`.

    Args:
        path_or_module: a filesystem path to a ``.py`` file, or a dotted module
            name that is importable from the current interpreter.
        entry: ``module:callable`` override for the model or function. The
            module half may be omitted (``":model"`` or plain ``"model"``), in
            which case the attribute is looked up on the target module.
        inputs: ``module:callable`` override for the input factory. Resolves the
            same way; if what it names is callable, it is called.

    Returns:
        The target, with its inputs already materialised.

    Raises:
        DiscoveryError: the module would not import, or no entry point or inputs
            could be resolved.
    """
    module = import_target_module(path_or_module)
    fn, name = _resolve_entry(module, entry)
    args, kwargs = _resolve_inputs(module, inputs)
    log.debug(
        "discovered %s with %d positional and %d keyword inputs",
        name,
        len(args),
        len(kwargs),
    )
    return Target(fn=fn, example_inputs=args, kwargs=kwargs, name=name)


def import_target_module(path_or_module: str) -> ModuleType:
    """Import a target given as a filesystem path or as a dotted module name.

    A ``.py`` suffix or an existing path means the file loader; anything else is
    tried as a dotted import.
    """
    path = Path(path_or_module).expanduser()
    if path.suffix == ".py" or path.exists():
        return _import_from_path(path)
    return _import_dotted(path_or_module)


def _import_from_path(path: Path) -> ModuleType:
    """Import a ``.py`` file with ``importlib.util.spec_from_file_location``."""
    if not path.is_file():
        raise DiscoveryError(f"no such file: {path}")
    resolved = path.resolve()

    name = resolved.stem
    existing = sys.modules.get(name)
    if existing is not None:
        if _module_file(existing) == resolved:
            return existing
        # Some unrelated module already owns that name; do not shadow it.
        name = f"_compile_check_target_{name}"

    spec = importlib.util.spec_from_file_location(name, resolved)
    if spec is None or spec.loader is None:
        raise DiscoveryError(f"cannot import {path} as a python module")
    module = importlib.util.module_from_spec(spec)

    # Registered before execution so a dataclass, a pickle, or a self-import
    # inside the target file resolves, and so an --entry naming the same module
    # by name finds this object rather than importing the file twice.
    sys.modules[name] = module
    # The file's own directory goes on the path so a target that imports a
    # sibling module works, which is what a user with a two-file repro has.
    parent = str(resolved.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        del sys.modules[name]
        raise DiscoveryError(f"importing {path} raised {type(exc).__name__}: {exc}") from exc
    return module


def _import_dotted(dotted: str) -> ModuleType:
    """Import a dotted module name, reporting a failure as a discovery error."""
    try:
        return importlib.import_module(dotted)
    except Exception as exc:
        raise DiscoveryError(
            f"could not import {dotted!r}: it is neither an existing file nor an "
            f"importable module ({type(exc).__name__}: {exc})"
        ) from exc


def _where(module: ModuleType) -> str:
    """Name a module the way the user named it, for an error message."""
    filename = getattr(module, "__file__", None)
    return str(filename) if filename else module.__name__


def _module_file(module: ModuleType) -> Path | None:
    """Return the resolved ``__file__`` of ``module``, or ``None``."""
    filename = getattr(module, "__file__", None)
    if not filename:
        return None
    return Path(filename).resolve()


def _resolve_entry(module: ModuleType, entry: str | None) -> tuple[Any, str]:
    """Return the ``(callable, name)`` to test."""
    if entry is not None:
        obj, name = _resolve_spec(entry, module, "--entry")
    else:
        obj, name = _MISSING, ""
        for attr in ENTRY_ATTRS:
            candidate = getattr(module, attr, _MISSING)
            if candidate is not _MISSING:
                obj, name = candidate, f"{module.__name__}:{attr}"
                break
        if obj is _MISSING:
            raise DiscoveryError(
                f"{_where(module)}: no entry point found. Looked for a module-level "
                f"{' or '.join(repr(a) for a in ENTRY_ATTRS)}, in that order "
                "('model' an nn.Module, 'fn' a callable). Name one explicitly with "
                "--entry module:callable."
            )

    if not callable(obj):
        raise DiscoveryError(f"{name} is not callable (it is a {type(obj).__name__})")
    return obj, name


def _resolve_inputs(
    module: ModuleType, inputs: str | None
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Return the ``(args, kwargs)`` to call the entry point with."""
    if inputs is not None:
        obj, name = _resolve_spec(inputs, module, "--inputs")
    else:
        obj, name = _MISSING, ""
        for attr in INPUT_ATTRS:
            candidate = getattr(module, attr, _MISSING)
            if candidate is not _MISSING:
                obj, name = candidate, f"{module.__name__}:{attr}"
                break
        if obj is _MISSING:
            raise DiscoveryError(
                f"{_where(module)}: no example inputs found. Looked for a module-level "
                f"{' or '.join(repr(a) for a in INPUT_ATTRS)}, in that order ('inputs' a "
                "tuple, list, or dict of tensors, 'get_inputs()' a callable returning "
                "the same). Name one explicitly with --inputs module:callable."
            )

    # An input factory is called; a materialised 'inputs' value is used as it is.
    if callable(obj):
        try:
            obj = obj()
        except Exception as exc:
            raise DiscoveryError(f"calling {name} raised {type(exc).__name__}: {exc}") from exc

    return _normalise_inputs(obj, name)


def _normalise_inputs(obj: Any, name: str) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Split an inputs value into positional and keyword arguments.

    A mapping is keyword arguments, a tuple or list is the positional argument
    sequence, and anything else is a single positional argument, which is the
    one-tensor case users write most often.
    """
    if isinstance(obj, dict):
        bad = sorted(repr(key) for key in obj if not isinstance(key, str))
        if bad:
            raise DiscoveryError(
                f"{name} is a dict used as keyword arguments, so every key must be a "
                f"string; these are not: {', '.join(bad)}"
            )
        return (), dict(obj)
    if isinstance(obj, tuple):
        return obj, {}
    if isinstance(obj, list):
        return tuple(obj), {}
    return (obj,), {}


def _resolve_spec(spec: str, module: ModuleType, flag: str) -> tuple[Any, str]:
    """Resolve a ``module:attribute`` override against ``module`` as the default."""
    module_part, sep, attr_part = spec.partition(":")
    if sep:
        if not attr_part:
            raise DiscoveryError(
                f"{flag} {spec!r}: nothing after the colon, expected module:callable"
            )
        target = module if _names_module(module_part, module) else import_target_module(module_part)
        attr_path = attr_part
    else:
        target, attr_path = module, spec
    if not attr_path:
        raise DiscoveryError(f"{flag} {spec!r}: expected module:callable")

    obj: Any = target
    walked = target.__name__
    for attr in attr_path.split("."):
        candidate = getattr(obj, attr, _MISSING)
        if candidate is _MISSING:
            raise DiscoveryError(f"{flag} {spec!r}: {walked} has no attribute {attr!r}")
        obj, walked = candidate, f"{walked}.{attr}"
    return obj, f"{target.__name__}:{attr_path}"


def _names_module(module_part: str, module: ModuleType) -> bool:
    """Whether ``module_part`` refers to the already-imported target module.

    An empty module half means "this module". Otherwise the dotted name and the
    file stem both count, so ``--entry mlp:model`` finds the module loaded from
    ``mlp.py`` instead of importing that file a second time under another name.
    """
    if not module_part:
        return True
    if module_part == module.__name__:
        return True
    filename = _module_file(module)
    return filename is not None and module_part == filename.stem
