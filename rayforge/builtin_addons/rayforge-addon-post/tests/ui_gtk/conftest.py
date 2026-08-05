"""UI fixtures for post_processors addon UI tests."""

import asyncio

import pytest

from rayforge import config as config_module
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.shared import tasker
from rayforge.shared.tasker.manager import TaskManager
from rayforge.shared.util.glib import idle_add


@pytest.fixture
def ui_task_mgr():
    """A test-isolated TaskManager for sync UI tests."""
    tm = TaskManager(main_thread_scheduler=idle_add)
    yield tm
    if tm.has_tasks():
        tm.shutdown()


@pytest.fixture
def ui_context(ui_task_mgr, monkeypatch, tmp_path):
    """A UI context for post_processors addon tests."""
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    temp_addons_dir = temp_config_dir / "addons"
    monkeypatch.setattr(config_module, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config_module, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config_module, "MACHINE_DIR", temp_machine_dir)
    monkeypatch.setattr(config_module, "ADDONS_DIR", temp_addons_dir)
    monkeypatch.setattr(
        config_module, "CONFIG_FILE", temp_config_dir / "config.yaml"
    )
    monkeypatch.setattr(
        config_module, "AI_CONFIG_FILE", temp_config_dir / "ai.yaml"
    )
    monkeypatch.setattr(tasker.task_mgr, "_instance", ui_task_mgr)

    context = get_context()
    yield context

    asyncio.run(context.shutdown())
    context_module._context_instance = None
