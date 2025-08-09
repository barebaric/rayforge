from __future__ import annotations
import math
from copy import deepcopy
from typing import Optional, Dict, Any

from .base import OpsTransformer
from ...core.ops import Ops
from ...shared.tasker.proxy import BaseExecutionContext


class MultiPassTransformer(OpsTransformer):
    """
    Repeats the sequence of operations multiple times.

    This transformer is typically used in the "post-assembly" phase of a Step
    to create multiple cutting or engraving passes over the entire assembled
    geometry. It can also apply a Z-axis step-down for each subsequent pass,
    which is useful for cutting through thick materials.
    """

    def __init__(
        self, enabled: bool = True, passes: int = 1, z_step_down: float = 0.0
    ):
        """
        Initializes the MultiPassTransformer.

        Args:
            enabled: Whether the transformer is active.
            passes: The total number of passes to perform. Must be >= 1.
            z_step_down: The distance to move down the Z-axis after each
                         pass. A positive value indicates downward movement.
        """
        super().__init__(enabled=enabled)
        self._passes: int = 1
        self._z_step_down: float = 0.0

        # Use property setters to ensure validation logic is applied
        self.passes = passes
        self.z_step_down = z_step_down

    @property
    def passes(self) -> int:
        """The total number of passes to perform (e.g., 3 means 3 total)."""
        return self._passes

    @passes.setter
    def passes(self, value: int):
        """Sets the total number of passes, ensuring it's at least 1."""
        new_value = max(1, int(value))
        if self._passes != new_value:
            self._passes = new_value
            self.changed.send(self)

    @property
    def z_step_down(self) -> float:
        """The amount to step down in Z for each pass after the first."""
        return self._z_step_down

    @z_step_down.setter
    def z_step_down(self, value: float):
        """Sets the Z step-down value."""
        new_value = float(value)
        if not math.isclose(self._z_step_down, new_value):
            self._z_step_down = new_value
            self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Multi-Pass")

    @property
    def description(self) -> str:
        return _(
            "Repeats the path multiple times, optionally stepping down in Z."
        )

    def run(
        self, ops: Ops, context: Optional[BaseExecutionContext] = None
    ) -> None:
        """
        Executes the multi-pass transformation on the Ops object.

        Args:
            ops: The Ops object to transform in-place.
            context: Execution context for cancellation (not used here).
        """
        # No-op if only one pass and no Z movement is needed.
        if self.passes <= 1 and self._z_step_down == 0.0:
            return

        # No-op if there are no commands to duplicate.
        if not ops.commands:
            return

        # Make a pristine copy of the original commands for the first pass.
        # This is crucial because `ops.translate` modifies commands in-place.
        original_ops = deepcopy(ops)
        final_commands = list(ops.commands)  # Start with the first pass

        # Generate subsequent passes
        for i in range(1, self.passes):
            # Create a fresh copy for this pass
            pass_ops = deepcopy(original_ops)

            # Apply Z step-down if configured
            if self._z_step_down != 0.0:
                z_offset = i * self._z_step_down
                # Translate by a negative amount to move down the Z axis
                pass_ops.translate(0, 0, -abs(z_offset))

            final_commands.extend(pass_ops.commands)

        # Replace the original commands with the full multi-pass sequence
        ops.commands = final_commands

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the transformer's configuration to a dictionary."""
        return {
            **super().to_dict(),
            "passes": self.passes,
            "z_step_down": self.z_step_down,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MultiPassTransformer":
        """Creates a MultiPassTransformer instance from a dictionary."""
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(
            enabled=data.get("enabled", True),
            passes=data.get("passes", 1),
            z_step_down=data.get("z_step_down", 0.0),
        )
