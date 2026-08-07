"""UI fixtures for cnc_essentials page tests."""

import asyncio
import logging

import pytest

from rayforge import config as config_module
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.doceditor.editor import DocEditor
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.spindle import SpindleHead
from rayforge.shared import tasker
from rayforge.shared.tasker.manager import TaskManager
from rayforge.shared.util.glib import idle_add

logger = logging.getLogger(__name__)


@pytest.fixture
def ui_task_mgr():
    """A test-isolated TaskManager for sync UI tests."""
    tm = TaskManager(main_thread_scheduler=idle_add)
    yield tm
    if tm.has_tasks():
        logger.warning(
            "Task manager still has tasks at end of test. Shutting down."
        )
    tm.shutdown()


@pytest.fixture
def ui_context(ui_task_mgr, monkeypatch, tmp_path):
    """A UI context for CNC addon tests."""
    temp_config_dir = tmp_path / "config"
    temp_dialect_dir = temp_config_dir / "dialects"
    temp_machine_dir = temp_config_dir / "machines"
    temp_addons_dir = temp_config_dir / "addons"
    monkeypatch.setattr(config_module, "CONFIG_DIR", temp_config_dir)
    monkeypatch.setattr(config_module, "DIALECT_DIR", temp_dialect_dir)
    monkeypatch.setattr(config_module, "MACHINE_DIR", temp_machine_dir)
    monkeypatch.setattr(config_module, "ADDONS_DIR", temp_addons_dir)
    monkeypatch.setattr(tasker.task_mgr, "_instance", ui_task_mgr)

    context = get_context()
    yield context

    asyncio.run(context.shutdown())
    context_module._context_instance = None


@pytest.fixture
def editor(ui_context, ui_task_mgr):
    editor = DocEditor(task_manager=ui_task_mgr, context=ui_context)
    yield editor
    editor.cleanup()


@pytest.fixture
def cnc_machine(ui_context):
    """A machine with a spindle head, set as the active machine."""
    machine = Machine(ui_context)
    machine.set_axis_extents(200, 150)
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000

    spindle = SpindleHead()
    spindle.name = "Spindle 1"
    machine.heads.clear()
    machine.add_head(spindle)

    ui_context.machine_mgr.machines.clear()
    ui_context.machine_mgr.add_machine(machine)
    ui_context.config.set_machine(machine)
    return machine
