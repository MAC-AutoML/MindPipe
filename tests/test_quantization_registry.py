import subprocess
import sys
from types import SimpleNamespace

import pytest

from algorithm.quantization import registry


def test_registry_import_does_not_eagerly_import_qwen3_vl() -> None:
    code = """
import sys
from algorithm.quantization import registry
assert registry.METHOD_REGISTRY
assert all(callable(factory) for factory in registry.METHOD_REGISTRY.values())
assert not any(name.startswith('transformers.models.qwen3_vl') for name in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_get_method_imports_only_selected_implementation(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyMethod:
        pass

    imported = []

    def fake_import_module(name: str):
        imported.append(name)
        return SimpleNamespace(DummyMethod=DummyMethod)

    lazy_method = registry._LazyMethod("example.quant:DummyMethod")
    monkeypatch.setitem(registry.METHOD_REGISTRY, "dummy", lazy_method)
    monkeypatch.setattr(registry, "import_module", fake_import_module)

    assert imported == []
    assert isinstance(registry.METHOD_REGISTRY["dummy"](), DummyMethod)
    assert isinstance(registry.get_method("dummy"), DummyMethod)
    assert imported == ["example.quant"]


def test_get_method_reports_available_methods() -> None:
    with pytest.raises(KeyError, match="Available methods"):
        registry.get_method("does-not-exist")
