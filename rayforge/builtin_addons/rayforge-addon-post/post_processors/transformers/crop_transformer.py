from __future__ import annotations
import logging
from typing import Dict, Any, List, Optional, TYPE_CHECKING
from gettext import gettext as _

from rayforge.pipeline.transformer.base import OpsTransformer, ExecutionPhase
from rayforge.core.matrix import Matrix
from rayforge.core.ops import Ops
from rayforge.core.workpiece import WorkPiece
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.geo import Geometry

logger = logging.getLogger(__name__)


class CropTransformer(OpsTransformer):
    """
    Crops cutting lines to stock boundary.

    This removes any toolpath that extends beyond the stock material,
    keeping only the parts that lie inside the stock boundary.
    """

    POSITION_SENSITIVE = True

    def __init__(
        self,
        enabled: bool = True,
        tolerance: float = 0.03,
        offset: float = 0.0,
    ):
        super().__init__(enabled=enabled)
        self._tolerance = tolerance
        self._offset = offset
        logger.debug(f"CropTransformer enabled={enabled}")

    @property
    def execution_phase(self) -> ExecutionPhase:
        return ExecutionPhase.POST_PROCESSING

    @property
    def label(self) -> str:
        return _("Crop to Stock")

    @property
    def description(self) -> str:
        return _("Crops cutting lines to stock boundary.")

    @property
    def tolerance(self) -> float:
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float):
        if self._tolerance != value:
            self._tolerance = value
            self.changed.send(self)

    @property
    def offset(self) -> float:
        return self._offset

    @offset.setter
    def offset(self, value: float):
        if self._offset != value:
            self._offset = value
            self.changed.send(self)

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
    ) -> None:
        if not self.enabled:
            return

        if not stock_geometries:
            return

        if workpiece is None:
            return

        world_to_local = workpiece.get_world_transform().invert()
        transform_matrix = world_to_local.to_4x4_numpy()
        regions = []

        wp_size = workpiece.size
        scale_x, scale_y = wp_size if wp_size else (1.0, 1.0)

        for stock_geo in stock_geometries:
            if self._offset != 0.0:
                stock_geo = stock_geo.grow(self._offset)
            local_geo = stock_geo.transform(transform_matrix)
            if wp_size:
                scale_matrix = Matrix.scale(scale_x, scale_y).to_4x4_numpy()
                local_geo = local_geo.transform(scale_matrix)
            polygons = local_geo.to_polygons(self._tolerance)
            regions.extend(p for p in polygons if len(p) >= 3)

        if not regions:
            return

        ops.clip_to_regions(regions, tolerance=self._tolerance)

    def to_dict(self) -> Dict[str, Any]:
        return {
            **super().to_dict(),
            "tolerance": self.tolerance,
            "offset": self.offset,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CropTransformer":
        return cls(
            enabled=data.get("enabled", True),
            tolerance=data.get("tolerance", 0.03),
            offset=data.get("offset", 0.0),
        )
