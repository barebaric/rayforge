"""Fixtures for core step settings UI tests (no addons)."""

import asyncio

import pytest

from rayforge import config as config_module
from rayforge import context as context_module
from rayforge.context import get_context
from rayforge.core.step import Step
from rayforge.doceditor.editor import DocEditor
from rayforge.machine.models.laser import Laser
from rayforge.machine.models.machine import Machine
from rayforge.shared import tasker


class FakeStep(Step):
    """A minimal concrete step for exercising the core settings UI."""

    ASSEMBLER_NAME = "fake"

    def __init__(self, name="fake"):
        super().__init__(typelabel="Fake", name=name)
        self.power = 0.5
        self.mode = "a"
        self.enabled = True
        self.count = 3
        self.weight = 1.0

    @classmethod
    def recipe_keys(cls) -> tuple[str, ...]:
        return (*super().recipe_keys(), "power", "count")

    def set_power(self, value):
        if self.power != value:
            self.power = value
            self.updated.send(self)

    def set_mode(self, value):
        if self.mode != value:
            self.mode = value
            self.updated.send(self)

    def set_enabled(self, value):
        if self.enabled != value:
            self.enabled = bool(value)
            self.updated.send(self)

    def set_count(self, value):
        if self.count != value:
            self.count = int(value)
            self.updated.send(self)


@pytest.fixture
def ui_context(ui_task_mgr, monkeypatch, tmp_path):
    """A UI context without any addon loading."""
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


@pytest.fixture
def editor(ui_context, ui_task_mgr):
    editor = DocEditor(task_manager=ui_task_mgr, context=ui_context)
    yield editor
    editor.cleanup()


@pytest.fixture
def step():
    return FakeStep()


@pytest.fixture
def machine(ui_context):
    """A machine with one laser head, set as the active machine."""
    machine = Machine(ui_context)
    machine.set_axis_extents(200, 150)
    machine.max_cut_speed = 5000
    machine.max_travel_speed = 10000

    head = Laser()
    head.name = "Laser 1"
    head.spot_size_mm = (0.1, 0.2)
    machine.heads.clear()
    machine.add_head(head)

    ui_context.machine_mgr.machines.clear()
    ui_context.machine_mgr.add_machine(machine)
    ui_context.config.set_machine(machine)
    return machine
