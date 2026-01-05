from __future__ import annotations
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
from ...core.geo.constants import (
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
)
from ...core.ops import (
    Ops,
    Command,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
)
from ...core.workpiece import WorkPiece
from ...shared.tasker.proxy import BaseExecutionContext
from .base import OpsTransformer, ExecutionPhase

logger = logging.getLogger(__name__)


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
        return _("Creates holding tabs by adding gaps to cut paths")

    def _generate_tab_clip_data(
        self, workpiece: WorkPiece
    ) -> List[Tuple[float, float, float]]:
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

        clip_data = []

        # The Ops object at this stage is in local coordinates, so we
        # generate clip points in the same space. No world transform needed.
        logger.debug(
            "TabOps: Generating clip data in LOCAL space for workpiece "
            f"'{workpiece.name}'"
        )
        logger.debug(
            f"TabOps: Workpiece vectors bbox: {workpiece.boundaries.rect()}"
        )

        for tab in workpiece.tabs:
            cmd = workpiece.boundaries.get_command_at(tab.segment_index)
            if cmd is None:
                logger.warning(
                    f"Tab {tab.uid} has invalid segment_index "
                    f"{tab.segment_index}, skipping."
                )
                continue

            cmd_type, x, y, z, i, j, cw = cmd
            end_point = (x, y, z)

            if cmd_type not in (CMD_TYPE_LINE, CMD_TYPE_ARC):
                continue

            p_start_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
            if tab.segment_index > 0:
                prev_cmd = workpiece.boundaries.get_command_at(
                    tab.segment_index - 1
                )
                if prev_cmd:
                    _, prev_x, prev_y, prev_z, _, _, _ = prev_cmd
                    p_start_3d = (prev_x, prev_y, prev_z)

            logger.debug(
                f"Processing Tab UID {tab.uid} on segment {tab.segment_index} "
                f"(type: {cmd_type}) starting from {p_start_3d}"
            )

            center_x, center_y = 0.0, 0.0

            if cmd_type == CMD_TYPE_LINE:
                p_start, p_end = p_start_3d[:2], end_point[:2]
                center_x = p_start[0] + (p_end[0] - p_start[0]) * tab.pos
                center_y = p_start[1] + (p_end[1] - p_start[1]) * tab.pos

            elif cmd_type == CMD_TYPE_ARC:
                center_offset = (i, j)
                clockwise = bool(cw)
                center = (
                    p_start_3d[0] + center_offset[0],
                    p_start_3d[1] + center_offset[1],
                )
                radius = math.dist(p_start_3d[:2], center)
                if radius < 1e-9:
                    continue

                start_angle = math.atan2(
                    p_start_3d[1] - center[1], p_start_3d[0] - center[0]
                )
                end_angle = math.atan2(
                    end_point[1] - center[1], end_point[0] - center[0]
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

            logger.debug(
                f"Local space tab center (from normalized vectors): "
                f"({center_x:.4f}, {center_y:.4f}), "
                f"width: {tab.width:.2f}mm"
            )
            clip_data.append((center_x, center_y, tab.width))

        logger.debug(f"TabOps: Finished generating clip data: {clip_data}")
        return clip_data

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

        tab_clip_data = self._generate_tab_clip_data(workpiece)
        if not tab_clip_data:
            logger.debug("No tab clip data was generated. Skipping clipping.")
            return

        logger.debug(
            f"Generated {len(tab_clip_data)} tab clip points for clipping."
        )

        processed_clip_data = tab_clip_data
        # Check if the coordinate space of the workpiece vectors (where tabs
        # are defined) is different from the final workpiece size (which the
        # incoming Ops object should represent). If so, we must scale the tab
        # points.
        # This handles cases like FrameProducer where Ops are generated at
        # final size, but the workpiece.boundaries are still normalized to a
        # 1x1 box.
        if workpiece.boundaries and not workpiece.boundaries.is_empty():
            vector_rect = workpiece.boundaries.rect()
            if vector_rect:
                final_w, final_h = workpiece.size
                _vx, _vy, vector_w, vector_h = vector_rect

                # Avoid division by zero for empty or linear geometry
                if vector_w > 1e-6 and vector_h > 1e-6:
                    scale_x = final_w / vector_w
                    scale_y = final_h / vector_h

                    # Only apply scaling if it's significant, to avoid
                    # float errors
                    if abs(scale_x - 1.0) > 1e-3 or abs(scale_y - 1.0) > 1e-3:
                        logger.debug(
                            "TabOps: Scaling tab clip points from vector space"
                            " to final ops space. "
                            f"Scale=({scale_x:.3f}, {scale_y:.3f})"
                        )
                        processed_clip_data = [
                            (x * scale_x, y * scale_y, width)
                            for x, y, width in tab_clip_data
                        ]

        logger.debug(
            f"TabOps: Clipping points to be used: {processed_clip_data}"
        )

        new_commands: List[Command] = []
        active_section_type: Optional[SectionType] = None
        section_buffer: List[Command] = []

        def _process_buffer():
            if not section_buffer:
                return

            if active_section_type == SectionType.VECTOR_OUTLINE:
                logger.debug(
                    "Processing buffered VECTOR_OUTLINE section for tabs."
                )
                temp_ops = Ops()
                temp_ops.commands = section_buffer
                num_before = len(temp_ops)

                for x, y, width in processed_clip_data:
                    logger.debug(
                        f"TabOps: Clipping at ({x:.4f}, {y:.4f}) "
                        f"with width {width:.2f}"
                    )
                    temp_ops.clip_at(x, y, width)

                num_after = len(temp_ops)
                logger.debug(
                    "Tab clipping changed command count from "
                    f"{num_before} to {num_after}."
                )
                new_commands.extend(temp_ops)
            else:
                # For any other section type, or commands outside a section,
                # pass them through unmodified.
                logger.debug(
                    "Passing through buffered section of type "
                    f"{active_section_type}."
                )
                new_commands.extend(section_buffer)

        for cmd in ops:
            if isinstance(cmd, OpsSectionStartCommand):
                _process_buffer()  # Process the previous section
                section_buffer = []
                active_section_type = cmd.section_type
                new_commands.append(cmd)  # Preserve the marker
            elif isinstance(cmd, OpsSectionEndCommand):
                _process_buffer()  # Process the current section
                section_buffer = []
                active_section_type = None
                new_commands.append(cmd)  # Preserve the marker
            else:
                section_buffer.append(cmd)

        _process_buffer()  # Process any commands in the final buffer

        ops.commands = new_commands

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TabOpsTransformer":
        if data.get("name") != cls.__name__:
            raise ValueError(
                f"Mismatched transformer name: expected {cls.__name__},"
                f" got {data.get('name')}"
            )
        return cls(enabled=data.get("enabled", True))
