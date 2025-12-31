from __future__ import annotations
import logging
from typing import TYPE_CHECKING, Any, Dict
from ..core.undo import DictItemCommand

if TYPE_CHECKING:
    from ..core.step import Step
    from .editor import DocEditor


logger = logging.getLogger(__name__)


class StepCmd:
    """Handles commands related to step settings."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor
        self._doc = editor.doc
        self._context = editor.context

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

    def apply_best_recipe_to_step(self, step: "Step"):
        """
        Finds the best matching recipe for a given step and applies its
        settings. This modifies the step object directly and is not undoable
        by itself; it should be called before the step is added to the
        document via an undoable command.
        """
        # Get the context from the active layer and machine
        active_layer = self._doc.active_layer
        stock_item = active_layer.stock_item if active_layer else None
        machine = self._context.machine

        # Query the RecipeManager for the best match for ANY supported
        # capability
        matching_recipes = []
        if step.capabilities:
            recipe_mgr = self._context.recipe_mgr
            matching_recipes = recipe_mgr.find_recipes(
                stock_item=stock_item,
                capabilities=step.capabilities,
                machine=machine,
            )

        # If matching_recipes is not empty, apply the best one
        if matching_recipes:
            best_recipe = matching_recipes[0]
            logger.info(
                f"Applying best recipe '{best_recipe.name}' to new step."
            )
            # Apply the settings to the step object
            for key, value in best_recipe.settings.items():
                if hasattr(step, key):
                    setattr(step, key, value)

            # Store a reference to the applied recipe
            step.applied_recipe_uid = best_recipe.uid
