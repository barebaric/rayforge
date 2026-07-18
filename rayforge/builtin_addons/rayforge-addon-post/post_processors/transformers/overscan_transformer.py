from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops.transform.overscan import OverscanSpec

from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class OverscanTransformer(OpsTransformer):
    """
    Intelligently rewrites raster line patterns to include overscan for
    machine acceleration and deceleration, ensuring constant engraving
    velocity.

    This transformer operates only on commands within a `RASTER_FILL` section.
    It identifies a raster line (a MoveTo followed by cutting commands) and
    replaces it with a physically correct toolpath that includes lead-in and
    lead-out moves at zero power.
    """

    def __init__(
        self, enabled: bool = True, distance_mm: float = 2.0, auto: bool = True
    ):
        super().__init__(enabled=enabled)
        self._distance_mm: float = 0.0
        self.distance_mm = distance_mm
        self._auto: bool = auto

    @staticmethod
    def calculate_auto_distance(
        step_speed: int, max_acceleration: int
    ) -> float:
        """
        Calculate the optimal overscan distance based on step speed and machine
        acceleration with a safety factor of 2.

        Formula: distance = (speed²) / (2 * acceleration * safety_factor)
        Where safety_factor = 2 for additional safety margin

        Args:
            step_speed: The cutting speed in mm/min
            max_acceleration: The maximum machine acceleration in mm/s²

        Returns:
            The calculated overscan distance in millimeters
        """
        # Convert speed from mm/min to mm/s for the calculation
        speed_mm_per_sec = step_speed / 60.0

        # Safety factor of 2 as specified in requirements
        safety_factor = 2.0

        # Calculate distance using physics formula with safety factor
        # d = v² / (2 * a * safety_factor)
        distance_mm = (speed_mm_per_sec**2) / (
            2 * max_acceleration * safety_factor
        )

        # Ensure minimum distance for practical purposes
        return max(0.5, distance_mm)

    @property
    def execution_phase(self) -> ExecutionPhase:
        """
        Overscan must run before path optimization to ensure travel moves
        are planned between the final, extended endpoints of the toolpaths.
        """
        return ExecutionPhase.POST_PROCESSING

    @property
    def distance_mm(self) -> float:
        return self._distance_mm

    @distance_mm.setter
    def distance_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._distance_mm, new_value):
            self._distance_mm = new_value
            self.changed.send(self)

    @property
    def auto(self) -> bool:
        return self._auto

    @auto.setter
    def auto(self, value: bool):
        if self._auto != bool(value):
            self._auto = bool(value)
            self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Overscan")

    @property
    def description(self) -> str:
        return _("Extends raster lines to ensure constant engraving speed.")

    def to_spec(
        self,
        workpiece: Optional[WorkPiece],
        stock_geometries: Optional[List["Geometry"]],
        settings: Optional[Dict[str, Any]],
    ) -> OverscanSpec:
        if settings and settings.get("driver_native_overscan"):
            return OverscanSpec(distance_mm=0.0)
        return OverscanSpec(distance_mm=self.distance_mm)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "distance_mm": self.distance_mm,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "OverscanTransformer":
        return cls(
            enabled=data.get("enabled", True),
            distance_mm=data.get("distance_mm", 2.0),
            auto=data.get("auto", True),
        )
