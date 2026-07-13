from __future__ import annotations

import math
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from raygeo.ops import Ops
from raygeo.ops.types import CommandCategory, CommandType

from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

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

    def run(
        self,
        ops: Ops,
        workpiece: Optional["WorkPiece"] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict[str, Any]] = None,
    ) -> None:
        offset_mm = settings.get("bidir_x_offset_mm", 0.0) if settings else 0.0
        if not self.enabled or math.isclose(offset_mm, 0.0):
            return

        source = ops.copy()
        ops.clear()
        n = source.len()
        idx = 0
        while idx < n:
            if source.command_type(idx) == CommandType.MOVE_TO:
                move_end = source.endpoint(idx)
                j = idx + 1
                while j < n and source.category(j) == CommandCategory.STATE:
                    j += 1
                if j < n and source.is_scanline(j):
                    scan_end = source.endpoint(j)
                    if scan_end[0] < move_end[0]:
                        ops.move_to(
                            move_end[0] + offset_mm,
                            move_end[1],
                            move_end[2],
                            extra=source.extra_axes(idx),
                        )
                        for k in range(idx + 1, j):
                            ops.transfer_command_from(source, k)
                        ops.scan_to(
                            scan_end[0] + offset_mm,
                            scan_end[1],
                            scan_end[2],
                            power_values=list(source.scanline_data(j)),
                            extra=source.extra_axes(j),
                        )
                    else:
                        for k in range(idx, j + 1):
                            ops.transfer_command_from(source, k)
                    idx = j + 1
                    continue
            ops.transfer_command_from(source, idx)
            idx += 1

    def to_dict(self) -> Dict[str, Any]:
        return {**super().to_dict()}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BidirScanOffsetTransformer":
        return cls(enabled=data.get("enabled", True))
