"""Tests for the environment block and the torch API probe."""

from __future__ import annotations

import platform

import torch

from torch_compile_check.env import PROBED_APIS, _resolves, collect_environment, probe_apis

EXPECTED_KEYS = {
    "torch_version",
    "torch_git_version",
    "python_version",
    "platform",
    "machine",
    "cpu_flags",
    "cuda_available",
    "inductor_force_disable_caches",
}


def test_environment_keys():
    assert set(collect_environment()) == EXPECTED_KEYS


def test_environment_reports_this_machine():
    env = collect_environment()
    assert env["python_version"] == platform.python_version()
    assert env["platform"] == platform.platform()
    # PLAN.md "Cross-architecture parity is a feature": the architecture is what
    # makes a run usable as parity evidence, so it is never allowed to be empty.
    assert isinstance(env["machine"], str)
    assert env["machine"]


def test_environment_reports_torch():
    env = collect_environment()
    assert env["torch_version"] == torch.__version__
    assert isinstance(env["torch_git_version"], str)
    assert isinstance(env["cuda_available"], bool)
    assert isinstance(env["inductor_force_disable_caches"], bool)


def test_cpu_flags_are_a_summary_or_none():
    flags = collect_environment()["cpu_flags"]
    assert flags is None or (isinstance(flags, str) and flags.strip() == flags and flags)


def test_probe_returns_a_boolean_per_probed_api():
    probe = probe_apis()
    assert list(probe) == list(PROBED_APIS)
    assert all(isinstance(value, bool) for value in probe.values())


def test_probe_finds_the_apis_every_oracle_needs():
    probe = probe_apis()
    for name in (
        "torch.testing.assert_close",
        "torch.testing._comparison.default_tolerances",
        "torch.compiler.reset",
        "torch._dynamo.explain",
        "torch._dynamo.list_backends",
        "torch._dynamo.utils.counters",
        "torch._inductor.config.force_disable_caches",
        "torch._debug_has_internal_overlap",
        "torch.Tensor.untyped_storage",
        "torch.compile",
    ):
        assert probe[name] is True, f"{name} is absent on torch {torch.__version__}"


def test_resolver_says_no_to_symbols_that_are_not_there():
    assert _resolves("torch.no_such_attribute_exists") is False
    assert _resolves("torch.no_such_module.no_such_attribute") is False
    assert _resolves("no_such_top_level_module") is False


def test_resolver_says_yes_to_a_present_but_falsy_value():
    # torch._dynamo.config.repro_after defaults to None; presence, not truth.
    assert _resolves("torch._dynamo.config.repro_after") is True
