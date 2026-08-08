"""UI fixtures for tool_library settings-page tests."""

import asyncio
import logging

import pytest

from rayforge import config as config_module
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.shared import tasker
from rayforge.shared.tasker.manager import TaskManager
from rayforge.shared.util.glib import idle_add

logger = logging.getLogger(__name__)


@pytest.fixture
def ui_task_mgr():
    tm = TaskManager(main_thread_scheduler=idle_add)
    yield tm
    if tm.has_tasks():
        logger.warning("Task manager had pending tasks at teardown.")
    tm.shutdown()


@pytest.fixture
def ui_context(ui_task_mgr, monkeypatch, tmp_path):
    temp_config_dir = tmp_path / "config"
    temp_machine_dir = temp_config_dir / "machines"
    temp_addons_dir = temp_config_dir / "addons"
    monkeypatch.setattr(config_module, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config_module, "MACHINE_DIR", temp_machine_dir)
    monkeypatch.setattr(config_module, "ADDONS_DIR", temp_addons_dir)
    monkeypatch.setattr(tasker.task_mgr, "_instance", ui_task_mgr)

    # Reset the tool_library singleton so each test gets a fresh manager
    # rooted at the test's CONFIG_DIR.
    import tool_library

    monkeypatch.setattr(tool_library, "_manager", None)

    context = get_context()
    yield context

    asyncio.run(context.shutdown())
    context_module._context_instance = None
