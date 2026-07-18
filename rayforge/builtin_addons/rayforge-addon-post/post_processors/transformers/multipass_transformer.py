from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops.transform.multipass import MultiPassSpec

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry


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
    def execution_phase(self) -> ExecutionPhase:
        """Multi-pass duplicates the final path, so it runs late."""
        return ExecutionPhase.POST_PROCESSING

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

    def to_spec(
        self,
        workpiece: Optional[WorkPiece],
        stock_geometries: Optional[List["Geometry"]],
        settings: Optional[Dict[str, Any]],
    ) -> MultiPassSpec:
        return MultiPassSpec(passes=self.passes, z_step_down=self.z_step_down)

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
