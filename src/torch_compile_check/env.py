"""Environment block collection.

PLAN.md "Package layout": ``env.py`` -- environment block collection.

Two responsibilities. :func:`collect_environment` builds the block that travels
with every report; PLAN.md "Cross-architecture parity is a feature" requires the
architecture to be in it, alongside the torch version and git hash, because a run
whose provenance is ambiguous is not usable as parity evidence.
:func:`probe_apis` answers the other question the project has to keep answering:
does the torch we are running against still have the private APIs the oracles
lean on.

Both functions import torch lazily, so importing this module (and therefore
``torch-compile-check --version``) does not pay for the torch import. Nothing here
touches the network.
"""

from __future__ import annotations

import importlib
import logging
import platform
from pathlib import Path
from typing import Any

log = logging.getLogger("torch_compile_check")

_MISSING = object()

# Every dotted symbol named in the API column of PLAN.md "Verified API surface",
# in table order. Rows of that table that do not name an importable symbol are
# not probed here and are checked by the oracle that uses them, from M1 on:
# the ``ExplainOutput`` field list, the backend names (`eager`, `aot_eager`,
# `aot_eager_decomp_partition`, `inductor`), the ``TORCH_LOGS`` artifact names,
# and the ``torch.compile`` keyword arguments.
#
# Three entries are expected to probe absent on the torch the plan was written
# against; they are listed because "still absent" is as much a fact worth
# recording as "still present".
PROBED_APIS: tuple[str, ...] = (
    "torch.testing.assert_close",
    "torch.testing._comparison.default_tolerances",
    "torch._dynamo.reset",
    "torch.compiler.reset",
    "torch._dynamo.explain",
    "torch._dynamo.list_backends",
    "torch._dynamo.utils.counters",
    "torch._dynamo.utils.compile_times",
    "torch._dynamo.utils.same",
    "torch._dynamo.config.repro_after",
    "torch._dynamo.config.repro_level",
    "torch._dynamo.config.repro_tolerance",
    "torch._inductor.config.repro_after",  # expected absent
    "torch._dynamo.repro.after_aot",
    "torch._dynamo.repro.after_dynamo",
    "torch._dynamo.debug_utils.same_two_models",
    "torch._dynamo.debug_utils.backend_accuracy_fails",
    "torch._inductor.config.force_disable_caches",
    "torch._debug_has_internal_overlap",
    # `Tensor.untyped_storage().data_ptr()` and `.nbytes()`: the method is on
    # the tensor, the two the alias oracle calls are on the storage it returns.
    "torch.Tensor.untyped_storage",
    "torch.UntypedStorage.data_ptr",
    "torch.UntypedStorage.nbytes",
    "torch.Tensor.storage_offset",
    "torch.Tensor._base",
    "torch.Tensor._is_view",
    "torch.compile",
    "torch._dynamo.CompileProfiler",  # expected absent
    "torch._dynamo.utils.CompileProfiler",  # expected absent
)

# CPU features that change what a compiled kernel does, so the ones worth
# carrying in a parity report. x86 names first, then aarch64.
_CPU_FLAG_WATCHLIST: tuple[str, ...] = (
    "avx2",
    "avx512f",
    "avx512bw",
    "avx512_bf16",
    "amx_tile",
    "amx_bf16",
    "f16c",
    "fma",
    "asimd",
    "asimdhp",
    "asimddp",
    "bf16",
    "i8mm",
    "sve",
    "sve2",
)

_CPUINFO = Path("/proc/cpuinfo")


def collect_environment() -> dict[str, Any]:
    """Return the environment block for a report.

    Keys are stable and always present; a value is ``None`` when the fact could
    not be established on this machine (no torch, no ``/proc/cpuinfo``).
    """
    torch = _import_torch()
    env: dict[str, Any] = {
        "torch_version": None,
        "torch_git_version": None,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu_flags": _cpu_flags(),
        "cuda_available": None,
        "inductor_force_disable_caches": None,
    }
    if torch is None:
        return env

    env["torch_version"] = str(torch.__version__)
    env["torch_git_version"] = getattr(torch.version, "git_version", None)
    try:
        env["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:  # pragma: no cover - driver-dependent
        log.warning("could not query CUDA availability: %s", exc)
    env["inductor_force_disable_caches"] = _inductor_caches_disabled()
    return env


def probe_apis() -> dict[str, bool]:
    """Return ``{dotted name: present}`` for every symbol in :data:`PROBED_APIS`."""
    results = {name: _resolves(name) for name in PROBED_APIS}
    absent = [name for name, present in results.items() if not present]
    if absent:
        log.debug("torch APIs absent on this install: %s", ", ".join(absent))
    return results


def _import_torch() -> Any:
    """Import torch, or return ``None`` and log why not."""
    try:
        return importlib.import_module("torch")
    except Exception as exc:  # pragma: no cover - torch is a hard dependency
        log.warning("torch is not importable: %s", exc)
        return None


def _resolves(dotted: str) -> bool:
    """Return whether ``dotted`` resolves on this install.

    The longest importable prefix is imported as a module, and the remainder is
    walked with ``getattr``. That handles the three shapes in the table at once:
    a module (``torch._dynamo.repro.after_aot``), an attribute of a module
    (``torch._dynamo.utils.same``), and an attribute of a class
    (``torch.Tensor._base``). Presence, not truth, is what is being measured:
    ``torch._dynamo.config.repro_after`` defaults to ``None`` and is present.
    """
    parts = dotted.split(".")
    obj: Any = None
    rest: list[str] = []
    for split in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:split]))
        except Exception:
            continue
        rest = parts[split:]
        break
    else:
        return False

    for attr in rest:
        try:
            obj = getattr(obj, attr, _MISSING)
        except Exception:
            # torch's config modules raise for unknown names rather than
            # returning the default.
            return False
        if obj is _MISSING:
            return False
    return True


def _cpu_flags() -> str | None:
    """Return the watchlist CPU features this machine reports, or ``None``."""
    try:
        text = _CPUINFO.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None

    reported: set[str] = set()
    for line in text.splitlines():
        key, sep, value = line.partition(":")
        if not sep:
            continue
        if key.strip().lower() in {"flags", "features"}:
            reported.update(value.split())
    hits = [flag for flag in _CPU_FLAG_WATCHLIST if flag in reported]
    return " ".join(hits) if hits else None


def _inductor_caches_disabled() -> bool | None:
    """Return ``torch._inductor.config.force_disable_caches``, or ``None``.

    The config value is read from ``TORCHINDUCTOR_FORCE_DISABLE_CACHES`` when
    torch is imported, so what this reports is the state of the process that is
    about to compile, which is the state worth putting in a report.
    """
    try:
        config = importlib.import_module("torch._inductor.config")
    except Exception as exc:
        log.warning("could not read the inductor cache configuration: %s", exc)
        return None
    value = getattr(config, "force_disable_caches", _MISSING)
    if value is _MISSING:
        return None
    return bool(value)
