"""
Tests for the get_tool_manager singleton accessor.
"""

import tool_library
from tool_library import get_tool_manager


def test_singleton_is_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_library, "_manager", None)
    monkeypatch.setattr("rayforge.config.CONFIG_DIR", tmp_path, raising=False)

    first = get_tool_manager()
    second = get_tool_manager()
    assert first is second


def test_singleton_reset_reconstructs(monkeypatch, tmp_path):
    monkeypatch.setattr(tool_library, "_manager", None)
    monkeypatch.setattr("rayforge.config.CONFIG_DIR", tmp_path, raising=False)

    first = get_tool_manager()
    monkeypatch.setattr(tool_library, "_manager", None)
    second = get_tool_manager()
    assert first is not second
