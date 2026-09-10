from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry
    from raygeo.ops.transform.tangential_knife import (  # pyright: ignore
        TangentialKnifeSpec,
    )

    from rayforge.core.workpiece import WorkPiece


class TangentialKnifeTransformer(OpsTransformer):
    """
    Adds tangential-knife A-axis rotation to vector contours.

    Attaches the path heading in degrees as an ``A`` extra axis to
    every contour command so a rotary blade stays tangent to the
    path. Heading changes beyond the angle tolerance - or arcs
    tighter than the radius tolerance - lift the knife to a safe Z,
    rotate in the air, and plunge back in; smaller changes rotate the
    blade in place with the knife down.
    """

    SPEC_NAME = "tangential_knife"

    def __init__(
        self,
        enabled: bool = True,
        angle_tolerance_deg: float = 30.0,
        radius_tolerance_mm: float = 1.0,
        safe_z: float = 2.0,
    ):
        super().__init__(enabled=enabled)
        self._angle_tolerance_deg: float = 30.0
        self.angle_tolerance_deg = angle_tolerance_deg
        self._radius_tolerance_mm: float = 1.0
        self.radius_tolerance_mm = radius_tolerance_mm
        self._safe_z: float = 2.0
        self.safe_z = safe_z

    @property
    def angle_tolerance_deg(self) -> float:
        return self._angle_tolerance_deg

    @angle_tolerance_deg.setter
    def angle_tolerance_deg(self, value: float):
        new_value = min(180.0, max(0.0, float(value)))
        if self._angle_tolerance_deg != new_value:
            self._angle_tolerance_deg = new_value
            self.changed.send(self)

    @property
    def radius_tolerance_mm(self) -> float:
        return self._radius_tolerance_mm

    @radius_tolerance_mm.setter
    def radius_tolerance_mm(self, value: float):
        new_value = max(0.0, float(value))
        if self._radius_tolerance_mm != new_value:
            self._radius_tolerance_mm = new_value
            self.changed.send(self)

    @property
    def safe_z(self) -> float:
        return self._safe_z

    @safe_z.setter
    def safe_z(self, value: float):
        new_value = max(0.0, float(value))
        if self._safe_z != new_value:
            self._safe_z = new_value
            self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Tangential Knife")

    @property
    def description(self) -> str:
        return _(
            "Rotates a tangential knife to stay tangent to the path "
            "and lifts it at sharp corners."
        )

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: list[Geometry] | None,
        settings: dict[str, Any] | None,
    ) -> TangentialKnifeSpec:
        from raygeo.ops.transform.tangential_knife import (  # pyright: ignore
            TangentialKnifeSpec,
        )

        return TangentialKnifeSpec(
            angle_tolerance_deg=self.angle_tolerance_deg,
            radius_tolerance_mm=self.radius_tolerance_mm,
            safe_z=self.safe_z,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "angle_tolerance_deg": self.angle_tolerance_deg,
            "radius_tolerance_mm": self.radius_tolerance_mm,
            "safe_z": self.safe_z,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TangentialKnifeTransformer:
        return cls(
            enabled=data.get("enabled", True),
            angle_tolerance_deg=data.get("angle_tolerance_deg", 30.0),
            radius_tolerance_mm=data.get("radius_tolerance_mm", 1.0),
            safe_z=data.get("safe_z", 2.0),
        )
