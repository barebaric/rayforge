from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict
from ..undo import DictItemCommand

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class StepCmd:
    """Handles commands related to step settings."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def set_step_param(
        self,
        target_dict: Dict[str, Any],
        key: str,
        new_value: Any,
        name: str,
        on_change_callback: Any = None,
    ):
        """
        Sets a parameter in a step's dictionary with an undoable command.

        Args:
            target_dict: The dictionary to modify.
            key: The key of the parameter to set.
            new_value: The new value for the parameter.
            name: The name of the command for the undo stack.
            on_change_callback: A callback to execute after the command.
        """
        # Check if the value is a float and compare with a tolerance
        if isinstance(new_value, float):
            old_value = target_dict.get(key, 0.0)
            if abs(new_value - old_value) < 1e-6:
                return
        elif new_value == target_dict.get(key):
            return

        command = DictItemCommand(
            target_dict=target_dict,
            key=key,
            new_value=new_value,
            name=name,
            on_change_callback=on_change_callback,
        )
        self._editor.history_manager.execute(command)
