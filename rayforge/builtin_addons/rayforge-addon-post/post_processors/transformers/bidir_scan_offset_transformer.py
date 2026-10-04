from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.ops.transform.bidir_scan_offset import BidirScanOffsetSpec

from rayforge.pipeline.transformer.base import OpsTransformer

if TYPE_CHECKING:
    from raygeo.geo import Geometry

    from rayforge.core.workpiece import WorkPiece


class BidirScanOffsetTransformer(OpsTransformer):
    """
    Corrects the X misalignment between left-to-right and right-to-left
    raster passes seen on machines with a fixed mechanical/firmware skew
    between scan directions.

    For every raster pass (a MoveTo immediately followed by a ScanLine),
    if the pass runs right-to-left, both its entry MoveTo and its ScanLine
    endpoint are shifted along X by the configured offset. Left-to-right
    passes are left untouched. Running after overscan means any lead-in/
    lead-out already baked into the pass is shifted along with it.
    """

    SPEC_NAME = "bidir_scan_offset"

    def __init__(self, enabled: bool = True, offset_mm: float = 0.0):
        super().__init__(enabled=enabled)
        self.offset_mm = float(offset_mm)

    @property
    def label(self) -> str:
        return _("Bidirectional Scan Offset")

    @property
    def description(self) -> str:
        return _(
            "Shifts right-to-left raster passes along X to correct "
            "scan-direction skew."
        )

    def to_spec(
        self,
        workpiece: WorkPiece | None,
        stock_geometries: list[Geometry] | None,
        machine=None,
    ) -> BidirScanOffsetSpec:
        return BidirScanOffsetSpec(offset_mm=self.offset_mm)

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "offset_mm": self.offset_mm}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BidirScanOffsetTransformer:
        return cls(
            enabled=data.get("enabled", True),
            offset_mm=data.get("offset_mm", 0.0),
        )
