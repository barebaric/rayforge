from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.ops.transform.bidir_scan_offset import BidirScanOffsetSpec

from rayforge.machine.models.bidir_offset import interpolate_bidir_offset
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

    The step's own ``bidir_x_offset_mm`` wins when it is non-zero;
    otherwise the offset comes from the machine's speed table
    (``bidir_offset_table``), looked up at the step's ``cut_speed``.
    """

    SPEC_NAME = "bidir_scan_offset"

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

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
        settings: dict[str, Any] | None,
    ) -> BidirScanOffsetSpec:
        return BidirScanOffsetSpec(offset_mm=_resolve_offset(settings))

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict()}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BidirScanOffsetTransformer:
        return cls(enabled=data.get("enabled", True))


def _resolve_offset(settings: dict[str, Any] | None) -> float:
    """Per-step offset, else the machine table at the step's speed."""
    if not settings:
        return 0.0
    offset = settings.get("bidir_x_offset_mm") or 0.0
    if offset:
        return offset
    speed = settings.get("cut_speed")
    table = settings.get("bidir_offset_table")
    if speed is None or not table:
        return 0.0
    rows = sorted((float(s), float(o)) for s, o in table)
    return interpolate_bidir_offset(rows, float(speed))
