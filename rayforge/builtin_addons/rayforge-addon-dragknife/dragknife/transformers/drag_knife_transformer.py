from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry
    from raygeo.ops.transform.drag_knife import (  # pyright: ignore
        DragKnifeSpec,
    )

    from rayforge.core.workpiece import WorkPiece


class DragKnifeTransformer(OpsTransformer):
    """
    Compensates vector contours for a trailing-blade drag knife.

    Rewrites each contour into the pivot path the machine must
    follow: a half-circle arc-in aligns the blade with the initial
    cut direction, straight segments are shifted forward by the blade
    offset, corners within the swivel tolerance are pivoted around
    with the blade down, sharper corners lift the knife before
    pivoting, and a half-circle arc-out overcuts the path end.
    """

    SPEC_NAME = "drag_knife"

    def __init__(
        self,
        enabled: bool = True,
        offset_mm: float = 0.5,
        swivel_angle_deg: float = 45.0,
    ):
        super().__init__(enabled=enabled)
        self._offset_mm: float = 0.5
        self.offset_mm = offset_mm
        self._swivel_angle_deg: float = 45.0
        self.swivel_angle_deg = swivel_angle_deg

    @property
    def offset_mm(self) -> float:
        return self._offset_mm

    @offset_mm.setter
    def offset_mm(self, value: float):
        new_value = max(0.0, float(value))
        if self._offset_mm != new_value:
            self._offset_mm = new_value
            self.changed.send(self)

    @property
    def swivel_angle_deg(self) -> float:
        return self._swivel_angle_deg

    @swivel_angle_deg.setter
    def swivel_angle_deg(self, value: float):
        new_value = min(180.0, max(0.0, float(value)))
        if self._swivel_angle_deg != new_value:
            self._swivel_angle_deg = new_value
            self.changed.send(self)

    @property
    def label(self) -> str:
        return _("Drag Knife Compensation")

    @property
    def description(self) -> str:
        return _(
            "Compensates contours for the blade offset of a drag "
            "knife and pivots the blade around corners."
        )

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: list[Geometry] | None,
        settings: dict[str, Any] | None,
    ) -> DragKnifeSpec:
        from raygeo.ops.transform.drag_knife import (  # pyright: ignore
            DragKnifeSpec,
        )

        return DragKnifeSpec(
            offset_mm=self.offset_mm,
            swivel_angle_deg=self.swivel_angle_deg,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **super().to_dict(),
            "offset_mm": self.offset_mm,
            "swivel_angle_deg": self.swivel_angle_deg,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DragKnifeTransformer:
        return cls(
            enabled=data.get("enabled", True),
            offset_mm=data.get("offset_mm", 0.5),
            swivel_angle_deg=data.get("swivel_angle_deg", 45.0),
        )
