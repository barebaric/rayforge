import pytest
from unittest.mock import MagicMock

from rayforge.doceditor.step_cmd import StepCmd
from rayforge.doceditor.editor import DocEditor
from rayforge.shared.tasker.manager import TaskManager
from rayforge.config import ConfigManager


@pytest.fixture
def mock_editor():
    """Provides a DocEditor instance with mocked dependencies."""
    task_manager = MagicMock(spec=TaskManager)
    config_manager = MagicMock(spec=ConfigManager)
    editor = DocEditor(task_manager, config_manager)
    return editor


@pytest.fixture
def step_cmd(mock_editor):
    """Provides a StepCmd instance."""
    return StepCmd(mock_editor)


def test_set_step_param(step_cmd):
    """Test setting a step parameter."""
    target_dict = {}
    key = "test_key"
    new_value = "test_value"
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == new_value


def test_set_step_param_no_change(step_cmd):
    """Test that setting the same value does nothing."""
    target_dict = {"test_key": "test_value"}
    key = "test_key"
    new_value = "test_value"
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == new_value


def test_set_step_param_float_tolerance(step_cmd):
    """Test that setting a float value within tolerance does nothing."""
    target_dict = {"test_key": 1.0}
    key = "test_key"
    new_value = 1.0000001  # Within 1e-6 tolerance
    name = "Test Command"

    step_cmd.set_step_param(target_dict, key, new_value, name)

    assert target_dict[key] == 1.0
