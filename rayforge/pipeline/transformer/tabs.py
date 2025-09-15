from __future__ import annotations
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
from .base import OpsTransformer
from ...core.ops import Ops
from ...core.geometry import (
    LineToCommand as GeoLineToCommand,
    ArcToCommand as GeoArcToCommand,
    MovingCommand as GeoMovingCommand,
)
from ...core.workpiece import WorkPiece
from ...shared.tasker.proxy import BaseExecutionContext

logger = logging.getLogger(__name__)


class TabOpsTransformer(OpsTransformer):
    """
    Creates gaps in toolpaths by generating geometric regions for each tab
    and using the Ops object's internal subtraction method. This is robust
    against prior ops transformations like offsetting or smoothing.
    """

    def __init__(self, enabled: bool = True):
        super().__init__(enabled=enabled)

    @property
    def label(self) -> str:
        return _("Tabs")

    @property
    def description(self) -> str:
        return _("Creates holding tabs by adding gaps to cut paths")

    def _generate_tab_polygons(
        self, workpiece: WorkPiece
    ) -> List[List[Tuple[float, float]]]:
        """
        Generates rectangular clipping polygons for each tab in the workpiece's
        local coordinate space, which matches the coordinate space of the
        incoming Ops object during the generation phase.
        """
        if not workpiece.vectors:
            return []

        local_polygons = []

        # The Ops object at this stage is in local coordinates, so we
        # generate polygons in the same space. No world transform is needed.
        logger.debug(
            "TabOps: Generating polygons in LOCAL space for workpiece "
            f"'{workpiece.name}'"
        )

        for tab in workpiece.tabs:
            if tab.segment_index >= len(workpiece.vectors.commands):
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            p_start_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            for i in range(tab.segment_index - 1, -1, -1):
                prev_cmd = workpiece.vectors.commands[i]
                if isinstance(prev_cmd, GeoMovingCommand):
                    p_start_3d = prev_cmd.end
                    break

            cmd = workpiece.vectors.commands[tab.segment_index]
            if (
                not isinstance(cmd, (GeoLineToCommand, GeoArcToCommand))
                or not cmd.end
            ):
                continue

            logger.debug(
                f"Processing Tab UID {tab.uid} on segment {tab.segment_index} "
                f"(type: {cmd.__class__.__name__}) starting from {p_start_3d}"
            )

            center_x, center_y, angle_rad = 0.0, 0.0, 0.0

            if isinstance(cmd, GeoLineToCommand):
                p_start, p_end = p_start_3d[:2], cmd.end[:2]
                center_x = p_start[0] + (p_end[0] - p_start[0]) * tab.t
                center_y = p_start[1] + (p_end[1] - p_start[1]) * tab.t
                angle_rad = math.atan2(
                    p_end[1] - p_start[1], p_end[0] - p_start[0]
                )

            elif isinstance(cmd, GeoArcToCommand):
                center = (
                    p_start_3d[0] + cmd.center_offset[0],
                    p_start_3d[1] + cmd.center_offset[1],
                )
                radius = math.dist(p_start_3d[:2], center)
                if radius < 1e-9:
                    continue

                start_angle = math.atan2(
                    p_start_3d[1] - center[1], p_start_3d[0] - center[0]
                )
                end_angle = math.atan2(
                    cmd.end[1] - center[1], cmd.end[0] - center[0]
                )
                angle_range = end_angle - start_angle
                if cmd.clockwise:
                    if angle_range > 0:
                        angle_range -= 2 * math.pi
                else:
                    if angle_range < 0:
                        angle_range += 2 * math.pi

                tab_angle = start_angle + angle_range * tab.t
                center_x = center[0] + radius * math.cos(tab_angle)
                center_y = center[1] + radius * math.sin(tab_angle)
                angle_rad = tab_angle + (
                    math.pi / 2.0 if not cmd.clockwise else -math.pi / 2.0
                )

            logger.debug(
                f"Local space tab center: ({center_x:.2f}, {center_y:.2f}), "
                f"angle: {math.degrees(angle_rad):.1f} deg"
            )

            w, d = tab.width / 2.0, tab.length / 2.0
            cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)
            points = [(-w, -d), (w, -d), (w, d), (-w, d)]

            rotated_translated_points = [
                (
                    center_x + p[0] * cos_a - p[1] * sin_a,
                    center_y + p[0] * sin_a + p[1] * cos_a,
                )
                for p in points
            ]
            local_polygons.append(rotated_translated_points)
            logger.debug(
                f"Local polygon vertices: {rotated_translated_points}"
            )

        return local_polygons

    def run(
        self,
        ops: Ops,
        workpiece: Optional[WorkPiece] = None,
        context: Optional[BaseExecutionContext] = None,
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

        logger.debug(
            f"TabOpsTransformer running for workpiece '{workpiece.name}' "
            f"with {len(workpiece.tabs)} tabs."
        )

        tab_polygons = self._generate_tab_polygons(workpiece)
        if not tab_polygons:
            logger.debug(
                "No tab polygons were generated. Skipping subtraction."
            )
            return

        logger.debug(
            f"Generated {len(tab_polygons)} tab polygons for subtraction."
        )

        ops_bounds_before = ops.rect()
        num_commands_before = len(ops.commands)
        logger.debug(
            f"Ops state before subtraction: bounds={ops_bounds_before}, "
            f"commands={num_commands_before}"
        )

        ops.subtract_regions(tab_polygons)

        num_commands_after = len(ops.commands)
        logger.debug(
            f"Ops command count after subtraction: {num_commands_after}"
        )

        if (
            num_commands_before == num_commands_after
            and num_commands_before > 0
        ):
            logger.warning(
                "TabOpsTransformer: Subtraction did not change the number of "
                "commands. The polygons and paths likely do not overlap."
            )
        else:
            logger.info(
                "TabOpsTransformer successfully applied tabs. Command count "
                f"changed from {num_commands_before} to {num_commands_after}."
            )

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabOpsTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(enabled=data.get("enabled", True))
