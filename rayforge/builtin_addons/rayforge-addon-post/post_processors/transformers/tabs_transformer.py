from __future__ import annotations

import logging
import math
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Dict,
    List,
    NamedTuple,
    Optional,
)

from raygeo.geo.types import Point3D
from raygeo.geo import Arc, Bezier, Line, Move
from raygeo.ops import Ops

from rayforge.core.workpiece import WorkPiece
from rayforge.pipeline.transformer.base import ExecutionPhase, OpsTransformer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from raygeo import Geometry

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
        """
        Generates clip data (center point and width) for each tab in the
        workpiece's local coordinate space. This matches the coordinate space
        of the incoming Ops object during the generation phase.
        """
        if not workpiece.boundaries or workpiece.boundaries.is_empty():
            logger.debug(
                "TabOps: workpiece has no vectors, cannot generate clip data."
            )
            return []

        clip_data: List[_ClipPoint] = []

        logger.debug(
            "TabOps: Generating clip data in LOCAL space for workpiece "
            f"'{workpiece.name}'"
        )
        logger.debug(
            f"TabOps: Workpiece vectors bbox: {workpiece.boundaries.rect()}"
        )

        for tab in workpiece.tabs:
            cmd = workpiece.boundaries.get_typed_command_at(tab.segment_index)
            if cmd is None:
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            end_point = cmd.end

            if isinstance(cmd, Move):
                continue

            p_start_3d: Point3D = (0.0, 0.0, 0.0)
            if tab.segment_index > 0:
                prev_cmd = workpiece.boundaries.get_typed_command_at(
                    tab.segment_index - 1
                )
                if prev_cmd:
                    p_start_3d = prev_cmd.end

            logger.debug(
                f"Processing Tab UID {tab.uid} on segment "
                f"{tab.segment_index} "
                f"(type: {type(cmd).__name__}) starting from "
                f"{p_start_3d}"
            )

            center_x, center_y = 0.0, 0.0

            if isinstance(cmd, Line):
                p_start, p_end = p_start_3d[:2], end_point[:2]
                center_x = p_start[0] + (p_end[0] - p_start[0]) * tab.pos
                center_y = p_start[1] + (p_end[1] - p_start[1]) * tab.pos

            elif isinstance(cmd, Arc):
                center_offset = cmd.center_offset
                clockwise = cmd.clockwise
                center = (
                    p_start_3d[0] + center_offset[0],
                    p_start_3d[1] + center_offset[1],
                )
                radius = math.dist(p_start_3d[:2], center)
                if radius < 1e-9:
                    continue

                start_angle = math.atan2(
                    p_start_3d[1] - center[1],
                    p_start_3d[0] - center[0],
                )
                end_angle = math.atan2(
                    end_point[1] - center[1],
                    end_point[0] - center[0],
                )
                angle_range = end_angle - start_angle
                if clockwise:
                    if angle_range > 0:
                        angle_range -= 2 * math.pi
                else:
                    if angle_range < 0:
                        angle_range += 2 * math.pi

                tab_angle = start_angle + angle_range * tab.pos
                center_x = center[0] + radius * math.cos(tab_angle)
                center_y = center[1] + radius * math.sin(tab_angle)

            elif isinstance(cmd, Bezier):
                c1x, c1y = cmd.control1
                c2x, c2y = cmd.control2
                t = tab.pos
                t2 = t * t
                t3 = t2 * t
                mt = 1.0 - t
                mt2 = mt * mt
                mt3 = mt2 * mt
                center_x = (
                    mt3 * p_start_3d[0]
                    + 3.0 * mt2 * t * c1x
                    + 3.0 * mt * t2 * c2x
                    + t3 * end_point[0]
                )
                center_y = (
                    mt3 * p_start_3d[1]
                    + 3.0 * mt2 * t * c1y
                    + 3.0 * mt * t2 * c2y
                    + t3 * end_point[1]
                )

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
        if workpiece.boundaries and not workpiece.boundaries.is_empty():
            vector_rect = workpiece.boundaries.rect()
            if vector_rect:
                final_w, final_h = workpiece.size
                _vx, _vy, vector_w, vector_h = vector_rect
                if vector_w > 1e-6 and vector_h > 1e-6:
                    scale_x = final_w / vector_w
                    scale_y = final_h / vector_h
                    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                        logger.debug(
                            "TabOps: Scaling tab clip points from vector "
                            "space to final ops space. "
                            f"Scale=({scale_x:.3f}, {scale_y:.3f})"
                        )
                        processed_clip_data = [
                            (cp.x * scale_x, cp.y * scale_y, cp.width)
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
