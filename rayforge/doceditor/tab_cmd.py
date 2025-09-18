from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, List, Dict, Any, Tuple, Optional
from copy import deepcopy

from ..core.tab import Tab
from ..core.geometry import (
    Geometry,
    LineToCommand,
    ArcToCommand,
    MoveToCommand,
)
from ..undo import Command
from ..core.step import Step
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class ConfigureTabsCommand(Command):
    """
    An undoable command to set tabs and the required transformer on a step.
    """

    def __init__(
        self,
        editor: DocEditor,
        workpiece: WorkPiece,
        step: Step,
        new_tabs: List[Tab],
        transformer_config: Dict[str, Any],
        name: str = "Configure Tabs",
    ):
        super().__init__(name=name)
        self.editor = editor
        self.workpiece_uid = workpiece.uid
        self.step_uid = step.uid
        self.new_tabs = new_tabs
        self.transformer_config = transformer_config
        self.old_tabs = deepcopy(workpiece.tabs)
        self.old_opstransformers_dicts = deepcopy(step.opstransformers_dicts)

    def _get_targets(self) -> Optional[Tuple[WorkPiece, Step]]:
        """Helper to find the model objects from stored UIDs."""
        workpiece = self.editor.doc.find_descendant_by_uid(self.workpiece_uid)
        step = self.editor.doc.find_descendant_by_uid(self.step_uid)
        if not isinstance(workpiece, WorkPiece) or not isinstance(step, Step):
            logger.error(
                "Could not find target WorkPiece or Step for command."
            )
            return None
        return workpiece, step

    def execute(self) -> None:
        """Applies the new tab configuration."""
        targets = self._get_targets()
        if not targets:
            return

        workpiece, step = targets
        workpiece.tabs = self.new_tabs
        # Remove any existing tab transformer to avoid duplicates
        new_transformers = [
            t
            for t in step.opstransformers_dicts
            if t.get("name") != "TabOpsTransformer"
        ]
        new_transformers.append(self.transformer_config)
        step.opstransformers_dicts = new_transformers

        workpiece.updated.send(workpiece)
        step.updated.send(step)

    def undo(self) -> None:
        """Reverts to the previous tab configuration."""
        targets = self._get_targets()
        if not targets:
            return

        workpiece, step = targets
        workpiece.tabs = self.old_tabs
        step.opstransformers_dicts = self.old_opstransformers_dicts

        workpiece.updated.send(workpiece)
        step.updated.send(step)


class TabCmd:
    """Handles commands related to creating and managing workpiece tabs."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def _calculate_equidistant_tabs(
        self, geometry: Geometry, count: int, width: float
    ) -> List[Tab]:
        """Calculates positions for a number of equally spaced tabs."""
        if not geometry.commands or count <= 0:
            return []

        # 1. Calculate total perimeter and individual segment lengths
        total_length = 0.0
        segment_lengths: List[Tuple[int, float]] = []
        last_point = (0.0, 0.0, 0.0)

        for i, cmd in enumerate(geometry.commands):
            # MoveTo just updates the pen position for the next drawable
            # command. It has no length and cannot contain a tab.
            if isinstance(cmd, MoveToCommand):
                if cmd.end:
                    last_point = cmd.end
                continue

            # Only process drawable commands (LineTo, ArcTo)
            if (
                not isinstance(cmd, (LineToCommand, ArcToCommand))
                or cmd.end is None
            ):
                continue

            length = 0.0
            if isinstance(cmd, LineToCommand):
                length = math.dist(last_point[:2], cmd.end[:2])
            elif isinstance(cmd, ArcToCommand):
                # Use analytical arc length for accuracy.
                p0 = last_point
                center = (
                    p0[0] + cmd.center_offset[0],
                    p0[1] + cmd.center_offset[1],
                )
                radius = math.dist(p0[:2], center)
                if radius > 1e-9:
                    start_angle = math.atan2(
                        p0[1] - center[1], p0[0] - center[0]
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
                    length = radius * abs(angle_range)
                else:
                    length = math.dist(last_point[:2], cmd.end[:2])

            if length > 1e-6:
                segment_lengths.append((i, length))
                total_length += length

            # Update last_point for the next segment
            last_point = cmd.end

        if total_length == 0:
            return []

        # 2. Determine target positions and find them on the path
        tabs: List[Tab] = []
        spacing = total_length / count
        for i in range(count):
            target_dist = (i + 0.5) * spacing
            cumulative_dist = 0.0
            for segment_index, seg_len in segment_lengths:
                if cumulative_dist + seg_len >= target_dist:
                    dist_into_segment = target_dist - cumulative_dist
                    t = dist_into_segment / seg_len
                    tabs.append(
                        Tab(
                            width=width,
                            segment_index=segment_index,
                            t=min(1.0, max(0.0, t)),
                        )
                    )
                    break
                cumulative_dist += seg_len
        return tabs

    def _calculate_cardinal_tabs(
        self, geometry: Geometry, width: float
    ) -> List[Tab]:
        """Calculates positions for 4 tabs at the cardinal points."""
        if not geometry.commands:
            return []

        # 1. Get bounding box of the geometry
        min_x, min_y, max_x, max_y = geometry.rect()
        width_bbox = max_x - min_x
        height_bbox = max_y - min_y

        if width_bbox < 1e-6 or height_bbox < 1e-6:
            return []

        # 2. Define the 4 cardinal points on the bounding box
        mid_x = min_x + width_bbox / 2
        mid_y = min_y + height_bbox / 2
        cardinal_points = [
            (mid_x, max_y),  # North
            (mid_x, min_y),  # South
            (max_x, mid_y),  # East
            (min_x, mid_y),  # West
        ]

        # 3. For each point, find the closest location on the geometry path
        tabs: List[Tab] = []
        for x, y in cardinal_points:
            closest = geometry.find_closest_point(x, y)
            if closest:
                segment_index, t, _ = closest
                tabs.append(
                    Tab(
                        width=width,
                        segment_index=segment_index,
                        t=min(1.0, max(0.0, t)),
                    )
                )

        # 4. Deduplicate tabs that might land on the same spot (e.g., corners)
        unique_tabs: List[Tab] = []
        seen: set[Tuple[int, int]] = set()
        for tab in tabs:
            # Round `t` to avoid floating point inaccuracies causing missed
            # duplicates
            key = (tab.segment_index, round(tab.t * 1e5))
            if key not in seen:
                unique_tabs.append(tab)
                seen.add(key)

        return unique_tabs

    def add_tabs(
        self,
        workpiece: WorkPiece,
        step: Step,
        count: int,
        width: float,
        strategy: str = "equidistant",
    ):
        """
        Creates and applies tabs to a workpiece for a given step using a
        specified strategy. This is an undoable action.

        Args:
            workpiece: The WorkPiece to add tabs to.
            step: The Step whose pipeline will process the tabs.
            count: The number of tabs to add.
            width: The width of each tab in millimeters.
            strategy: The placement strategy (currently only 'equidistant').
        """
        if not workpiece.vectors:
            logger.warning(
                f"Cannot add tabs to workpiece '{workpiece.name}' "
                "because it has no vector geometry."
            )
            return

        if strategy == "equidistant":
            new_tabs = self._calculate_equidistant_tabs(
                workpiece.vectors, count, width
            )
        else:
            raise NotImplementedError(
                f"Tabbing strategy '{strategy}' not implemented."
            )

        transformer_config = {"name": "TabOpsTransformer", "enabled": True}

        cmd = ConfigureTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            step=step,
            new_tabs=new_tabs,
            transformer_config=transformer_config,
            name=_("Add Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def add_cardinal_tabs(
        self, workpiece: WorkPiece, step: Step, width: float
    ):
        """
        Creates and applies 4 tabs to a workpiece at the cardinal points
        (North, South, East, West). This is an undoable action.

        Args:
            workpiece: The WorkPiece to add tabs to.
            step: The Step whose pipeline will process the tabs.
            width: The width of each tab in millimeters.
        """
        if not workpiece.vectors:
            logger.warning(
                f"Cannot add tabs to workpiece '{workpiece.name}' "
                "because it has no vector geometry."
            )
            return

        new_tabs = self._calculate_cardinal_tabs(workpiece.vectors, width)

        transformer_config = {"name": "TabOpsTransformer", "enabled": True}

        cmd = ConfigureTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            step=step,
            new_tabs=new_tabs,
            transformer_config=transformer_config,
            name=_("Add Cardinal Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def clear_tabs(self, workpiece: WorkPiece, step: Step):
        """Removes all tabs from a workpiece for a given step."""
        transformer_config = {"name": "TabOpsTransformer", "enabled": True}
        cmd = ConfigureTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            step=step,
            new_tabs=[],
            transformer_config=transformer_config,
            name=_("Clear Tabs"),
        )
        self._editor.history_manager.execute(cmd)
