from __future__ import annotations
import math
import logging
from typing import Optional, List, Tuple, Dict, Any
from .base import OpsTransformer, ExecutionPhase
from ...core.ops import (
    Ops,
    Command,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
)
from ...core.geo import (
    LineToCommand as GeoLineToCommand,
    ArcToCommand as GeoArcToCommand,
    MovingCommand as GeoMovingCommand,
)
from ...core.workpiece import WorkPiece
from ...shared.tasker.proxy import BaseExecutionContext

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
        if not workpiece.vectors:
            return []

        clip_data = []

        # The Ops object at this stage is in local coordinates, so we
        # generate clip points in the same space. No world transform needed.
        logger.debug(
            "TabOps: Generating clip data in LOCAL space for workpiece "
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
            # Find the start point of the command segment by looking backwards
            # for the last command that had an endpoint.
            for i in range(tab.segment_index - 1, -1, -1):
                prev_cmd = workpiece.vectors.commands[i]
                if isinstance(prev_cmd, GeoMovingCommand) and prev_cmd.end:
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

            center_x, center_y = 0.0, 0.0

            if isinstance(cmd, GeoLineToCommand):
                p_start, p_end = p_start_3d[:2], cmd.end[:2]
                center_x = p_start[0] + (p_end[0] - p_start[0]) * tab.pos
                center_y = p_start[1] + (p_end[1] - p_start[1]) * tab.pos

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

                tab_angle = start_angle + angle_range * tab.pos
                center_x = center[0] + radius * math.cos(tab_angle)
                center_y = center[1] + radius * math.sin(tab_angle)

            logger.debug(
                f"Local space tab center: ({center_x:.2f}, {center_y:.2f}), "
                f"width: {tab.width:.2f}mm"
            )
            clip_data.append((center_x, center_y, tab.width))

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

                for x, y, width in tab_clip_data:
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
