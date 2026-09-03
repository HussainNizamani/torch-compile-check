"""Tests for the discovery convention and its error messages."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from torch_compile_check.discover import DiscoveryError, Target, load_target

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_path_finds_model_and_inputs():
    target = load_target(str(FIXTURES / "mlp.py"))
    assert isinstance(target, Target)
    assert isinstance(target.fn, torch.nn.Module)
    assert target.name == "mlp:model"
    assert len(target.example_inputs) == 1
    assert target.example_inputs[0].shape == (4, 8)
    assert target.kwargs == {}


def test_dotted_module_finds_fn_and_calls_get_inputs(monkeypatch):
    monkeypatch.syspath_prepend(str(FIXTURES))
    target = load_target("fn_target")
    assert target.name == "fn_target:fn"
    assert callable(target.fn)
    # No module-level `inputs`, so get_inputs() was called for these.
    assert len(target.example_inputs) == 2
    assert target.example_inputs[0].shape == (3, 5)


def test_entry_and_inputs_overrides_win():
    target = load_target(
        str(FIXTURES / "named_entry.py"),
        entry="named_entry:net",
        inputs="named_entry:make_inputs",
    )
    assert target.name == "named_entry:net"
    assert torch.equal(target.example_inputs[0], torch.ones(2, 3))


def test_override_without_a_module_half():
    target = load_target(str(FIXTURES / "named_entry.py"), entry="net", inputs="make_inputs")
    assert target.name == "named_entry:net"
    assert len(target.example_inputs) == 1


def test_override_walks_a_dotted_attribute_path():
    target = load_target(
        str(FIXTURES / "named_entry.py"),
        entry="bundle.net",
        inputs="bundle.make_inputs",
    )
    assert target.name == "named_entry:bundle.net"
    assert torch.equal(target.example_inputs[0], torch.ones(2, 3))


def test_override_beats_the_convention():
    # mlp.py has both `model` and `inputs`; the flags must still win.
    target = load_target(
        str(FIXTURES / "mlp.py"),
        entry="mlp:MLP",
        inputs="mlp:get_inputs",
    )
    assert target.name == "mlp:MLP"


def test_mapping_inputs_become_keyword_arguments():
    target = load_target(str(FIXTURES / "kwargs_target.py"))
    assert target.example_inputs == ()
    assert set(target.kwargs) == {"x", "scale"}
    assert target.kwargs["scale"] == pytest.approx(2.0)


def test_missing_entry_point_names_what_was_looked_for():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "empty_target.py"))
    message = str(excinfo.value)
    assert "no entry point found" in message
    assert "'model'" in message
    assert "'fn'" in message
    assert "--entry" in message


def test_missing_inputs_names_what_was_looked_for():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "no_inputs.py"))
    message = str(excinfo.value)
    assert "no example inputs found" in message
    assert "'inputs'" in message
    assert "'get_inputs'" in message
    assert "--inputs" in message


def test_missing_override_attribute_names_the_attribute():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "mlp.py"), entry="mlp:nope")
    assert "has no attribute 'nope'" in str(excinfo.value)


def test_entry_that_is_not_callable_is_rejected():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "empty_target.py"), entry="WHAT_THIS_IS_NOT")
    assert "is not callable" in str(excinfo.value)


def test_missing_file_is_a_discovery_error():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "not_here.py"))
    assert "no such file" in str(excinfo.value)


def test_unimportable_module_name_is_a_discovery_error():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target("torch_compile_check_no_such_module")
    assert "neither an existing file nor an importable module" in str(excinfo.value)


def test_target_that_raises_on_import_is_a_discovery_error():
    with pytest.raises(DiscoveryError) as excinfo:
        load_target(str(FIXTURES / "broken_import.py"))
    message = str(excinfo.value)
    assert "raised ValueError" in message
    assert "refuses to import" in message
