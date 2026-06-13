from __future__ import annotations

import logging
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    NamedTuple,
    Optional,
)

from raygeo.ops import Ops

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo.geo import Geometry

logger = logging.getLogger(__name__)


class _ClipPoint(NamedTuple):
    x: float
    y: float
    width: float


class TabOpsTransformer(OpsTransformer):
    """
    Creates gaps in toolpaths by finding the closest point on the path for
    each tab and creating a precise cut. This is robust against prior ops
    transformations and avoids clipping unrelated paths that may be nearby.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    @property
    def execution_phase(self) -> ExecutionPhase:
        """Tabs intentionally break paths, so they must run after smoothing."""
        return ExecutionPhase.PATH_INTERRUPTION

    @property
    def label(self) -> str:
        return _("Tabs")

    @property
    def description(self) -> str:
        return _(
            "Creates holding tabs by adding gaps or reducing power "
            "on cut paths"
        )

    def _generate_tab_clip_data(
        self, workpiece: WorkPiece
    ) -> List[_ClipPoint]:
        if not workpiece.boundaries or workpiece.boundaries.is_empty():
            logger.debug(
                "TabOps: workpiece has no vectors, cannot generate clip data."
            )
            return []

        clip_data: List[_ClipPoint] = []
        vectors = workpiece.boundaries

        logger.debug(
            "TabOps: Generating clip data in LOCAL space for workpiece "
            f"'{workpiece.name}'"
        )
        logger.debug(f"TabOps: Workpiece vectors bbox: {vectors.rect()}")

        for tab in workpiece.tabs:
            if tab.segment_index >= len(vectors):
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            cmd = vectors.get_typed_command_at(tab.segment_index)
            if cmd is None:
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            from raygeo.geo import Move

            if isinstance(cmd, Move):
                continue

            point = vectors.get_point_at(tab.segment_index, tab.pos)
            if point is None:
                logger.warning(
                    f"Tab {tab.uid}: could not evaluate point on "
                    f"segment {tab.segment_index} at t={tab.pos}, skipping."
                )
                continue

            center_x, center_y = point[0], point[1]

            logger.debug(
                f"Local space tab center (from normalized vectors): "
                f"({center_x:.4f}, {center_y:.4f}), "
                f"width: {tab.width:.2f}mm"
            )
            clip_data.append(_ClipPoint(center_x, center_y, tab.width))

        logger.debug(f"TabOps: Finished generating clip data: {clip_data}")
        return clip_data

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[ProgressContext] = None,
        stock_geometries: Optional[List["Geometry"]] = None,
        settings: Optional[Dict] = None,
    ) -> None:
        if not self.enabled:
            return
        if not workpiece:
            logger.debug("TabOpsTransformer: No workpiece provided, skipping.")
            return
        if not workpiece.tabs_enabled or not workpiece.tabs:
            logger.debug(
                "TabOpsTransformer: Tabs disabled or no tabs on workpiece "
                f"'{workpiece.name}', skipping."
            )
            return

        tab_power = 0.0
        original_power = 1.0
        if settings:
            tab_power = settings.get("tab_power", 0.0)
            original_power = settings.get("power", 1.0)

        logger.debug(
            f"TabOpsTransformer running for workpiece '{workpiece.name}' "
            f"with {len(workpiece.tabs)} tabs, tab_power={tab_power}."
        )

        tab_clip_data = self._generate_tab_clip_data(workpiece)
        if not tab_clip_data:
            logger.debug("No tab clip data was generated. Skipping clipping.")
            return

        logger.debug(
            f"Generated {len(tab_clip_data)} tab clip points for clipping."
        )

        processed_clip_data = tab_clip_data
        final_w, final_h = workpiece.size
        if final_w > 1e-6 and final_h > 1e-6:
            logger.debug(
                "TabOps: Scaling tab clip points from vector "
                "space to final ops space. "
                f"Scale=({final_w:.3f}, {final_h:.3f})"
            )
            processed_clip_data = [
                (cp.x * final_w, cp.y * final_h, cp.width)
                for cp in tab_clip_data
            ]

        logger.debug(
            f"TabOps: Clipping points to be used: {processed_clip_data}"
        )

        if tab_power > 0:
            actual_tab_power = tab_power * original_power
            ops.apply_tab_power(
                processed_clip_data,
                actual_tab_power,
                original_power,
            )
        else:
            ops.apply_tab_gaps(processed_clip_data)

    @classmethod
    def from_dict(cls, data: Dict) -> "TabOpsTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(enabled=data.get("enabled", True))
