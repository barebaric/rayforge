import logging
import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.ops.transform.lead_in_out import LeadInOutSpec

from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from rayforge.core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class LeadInOutTransformer(OpsTransformer):
    """
    Adds zero-power lead-in and lead-out moves to vector contour paths.

    For each contour within a VECTOR_OUTLINE section, this transformer
    computes the tangent direction at the start and end of the path using
    the geometry module, then extends the toolpath with lead-in (before
    the cut starts) and lead-out (after the cut ends) segments at zero
    laser power. This allows the laser head to reach constant velocity
    before the actual cut begins and to decelerate after the cut ends,
    improving cut quality at start/end points.
    """

    SPEC_NAME = "lead_in_out"

    def __init__(
        self,
        enabled: bool = True,
        lead_in_mm: float = 2.0,
        lead_out_mm: float = 2.0,
        auto: bool = True,
    ):
        super().__init__(enabled=enabled)
        self._lead_in_mm: float = 0.0
        self.lead_in_mm = lead_in_mm
        self._lead_out_mm: float = 0.0
        self.lead_out_mm = lead_out_mm
        self._auto: bool = auto

    @staticmethod
    def calculate_auto_distance(
        step_speed: int, max_acceleration: int
    ) -> float:
        """
        Calculate the optimal lead-in/out distance based on step speed
        and machine acceleration with a safety factor of 2.

        Formula: distance = (speed^2) / (2 * acceleration * safety_factor)
        Where safety_factor = 2 for additional safety margin.

        Args:
            step_speed: The cutting speed in mm/min
            max_acceleration: The maximum machine acceleration in mm/s^2

        Returns:
            The calculated distance in millimeters
        """
        speed_mm_per_sec = step_speed / 60.0
        safety_factor = 2.0
        distance_mm = (speed_mm_per_sec**2) / (
            2 * max_acceleration * safety_factor
        )
        return distance_mm

    @property
    def lead_in_mm(self) -> float:
        return self._lead_in_mm

    @lead_in_mm.setter
    def lead_in_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._lead_in_mm, new_value):
            self._lead_in_mm = new_value
            self.changed.send(self)

    @property
    def lead_out_mm(self) -> float:
        return self._lead_out_mm

    @lead_out_mm.setter
    def lead_out_mm(self, value: float):
        new_value = max(0.0, float(value))
        if not math.isclose(self._lead_out_mm, new_value):
            self._lead_out_mm = new_value
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
        return _("Lead-In/Out")

    @property
    def description(self) -> str:
        return _(
            "Adds zero-power lead-in and lead-out moves to vector contours."
        )

    def to_spec(
        self,
        workpiece: "WorkPiece | None",
        stock_geometries: "list[Geometry] | None",
        settings: dict[str, Any] | None,
    ) -> LeadInOutSpec:
        return LeadInOutSpec(
            lead_in_mm=self.lead_in_mm, lead_out_mm=self.lead_out_mm
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "lead_in_mm": self.lead_in_mm,
            "lead_out_mm": self.lead_out_mm,
            "auto": self.auto,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LeadInOutTransformer":
        return cls(
            enabled=data.get("enabled", True),
            lead_in_mm=data.get("lead_in_mm", 2.0),
            lead_out_mm=data.get("lead_out_mm", 2.0),
            auto=data.get("auto", True),
        )
