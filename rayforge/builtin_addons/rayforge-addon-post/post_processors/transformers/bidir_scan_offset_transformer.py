from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops.transform.bidir_scan_offset import BidirScanOffsetSpec

from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer

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

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

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
        workpiece: Optional["WorkPiece"],
        stock_geometries: Optional[List["Geometry"]],
        settings: Optional[Dict[str, Any]],
    ) -> BidirScanOffsetSpec:
        offset = settings.get("bidir_x_offset_mm", 0.0) if settings else 0.0
        return BidirScanOffsetSpec(offset_mm=offset)

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BidirScanOffsetTransformer":
        return cls(enabled=data.get("enabled", True))
