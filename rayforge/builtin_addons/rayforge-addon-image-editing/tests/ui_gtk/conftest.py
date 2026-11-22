"""UI fixtures for image_editing dialog tests."""

import asyncio

import pytest

from rayforge import config as config_module
from rayforge import context as context_module
from rayforge.context import get_context


@pytest.fixture
def ui_context(monkeypatch, tmp_path):
    """An isolated application context, as the unit rows need one."""
    config_dir = tmp_path / "config"
    monkeypatch.setattr(config_module, "CONFIG_DIR", config_dir)
    monkeypatch.setattr(config_module, "DIALECT_DIR", config_dir / "dialects")
    monkeypatch.setattr(config_module, "MACHINE_DIR", config_dir / "machines")
    monkeypatch.setattr(config_module, "ADDONS_DIR", config_dir / "addons")
    monkeypatch.setattr(
        config_module, "CONFIG_FILE", config_dir / "config.yaml"
    )
    monkeypatch.setattr(
        config_module, "AI_CONFIG_FILE", config_dir / "ai.yaml"
    )
    context = get_context()
    yield context
    asyncio.run(context.shutdown())
    context_module._context_instance = None
