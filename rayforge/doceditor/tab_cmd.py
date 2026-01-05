from __future__ import annotations
import logging
import math
from typing import TYPE_CHECKING, List, Tuple, Optional
from copy import deepcopy
from dataclasses import replace

from ..core.geo import Geometry
from ..core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
)
from ..core.tab import Tab
from ..core.undo import Command
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class SetWorkpieceTabsCommand(Command):
    """An undoable command that sets the list of tabs for a workpiece."""

    def __init__(
        self,
        editor: DocEditor,
        workpiece: WorkPiece,
        new_tabs: List[Tab],
        name: str = "Set Tabs",
    ):
        super().__init__(name=name)
        self.editor = editor
        self.workpiece_uid = workpiece.uid
        self.new_tabs = new_tabs
        self.old_tabs = deepcopy(workpiece.tabs)

    def _get_workpiece(self) -> Optional[WorkPiece]:
        """Helper to find the model object from the stored UID."""
        workpiece = self.editor.doc.find_descendant_by_uid(self.workpiece_uid)
        if isinstance(workpiece, WorkPiece):
            return workpiece
        logger.error("Could not find target WorkPiece for command.")
        return None

    def execute(self) -> None:
        """Applies the new list of tabs."""
        workpiece = self._get_workpiece()
        if workpiece:
            workpiece.tabs = self.new_tabs

    def undo(self) -> None:
        """Reverts to the previous list of tabs."""
        workpiece = self._get_workpiece()
        if workpiece:
            workpiece.tabs = self.old_tabs


class TabCmd:
    """Handles commands related to creating and managing workpiece tabs."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def _calculate_equidistant_tabs(
        self, geometry: Geometry, count: int, width: float
    ) -> List[Tab]:
        """Calculates positions for a number of equally spaced tabs."""
        if count <= 0:
            return []

        # 1. Calculate total perimeter and individual segment lengths
        total_length = 0.0
        segment_lengths: List[Tuple[int, float]] = []
        last_point = (0.0, 0.0, 0.0)

        for segment_idx, (cmd_type, x, y, z, i, j, cw) in enumerate(
            geometry.iter_commands()
        ):
            end_point = (x, y, z)

            # MoveTo just updates the pen position for the next drawable
            # command. It has no length and cannot contain a tab.
            if cmd_type == CMD_TYPE_MOVE:
                last_point = end_point
                continue

            if cmd_type not in (CMD_TYPE_LINE, CMD_TYPE_ARC):
                continue

            length = 0.0
            if cmd_type == CMD_TYPE_LINE:
                length = math.dist(last_point[:2], end_point[:2])
            elif cmd_type == CMD_TYPE_ARC:
                center_offset = (i, j)
                clockwise = bool(cw)

                p0 = last_point
                center = (
                    p0[0] + center_offset[0],
                    p0[1] + center_offset[1],
                )
                radius = math.dist(p0[:2], center)
                if radius > 1e-9:
                    start_angle = math.atan2(
                        p0[1] - center[1], p0[0] - center[0]
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
                    length = radius * abs(angle_range)
                else:
                    length = math.dist(last_point[:2], end_point[:2])

            if length > 1e-6:
                segment_lengths.append((segment_idx, length))
                total_length += length

            # Update last_point for the next segment
            last_point = end_point

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
                            pos=min(1.0, max(0.0, t)),
                        )
                    )
                    break
                cumulative_dist += seg_len
        return tabs

    def _calculate_cardinal_tabs(
        self, geometry: Geometry, width: float
    ) -> List[Tab]:
        """Calculates positions for 4 tabs at the cardinal points."""
        if geometry.is_empty():
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
                        pos=min(1.0, max(0.0, t)),
                    )
                )

        # 4. Deduplicate tabs that might land on the same spot (e.g., corners)
        unique_tabs: List[Tab] = []
        seen: set[Tuple[int, int]] = set()
        for tab in tabs:
            # Round `t` to avoid floating point inaccuracies causing missed
            # duplicates
            key = (tab.segment_index, round(tab.pos * 1e5))
            if key not in seen:
                unique_tabs.append(tab)
                seen.add(key)

        return unique_tabs

    def add_tabs(
        self,
        workpiece: WorkPiece,
        count: int,
        width: float,
        strategy: str = "equidistant",
    ):
        """
        Creates and applies tabs to a workpiece. This is an undoable action.

        Args:
            workpiece: The WorkPiece to add tabs to.
            count: The number of tabs to add.
            width: The width of each tab in millimeters.
            strategy: The placement strategy (currently only 'equidistant').
        """
        if not workpiece.boundaries:
            logger.warning(
                f"Cannot add tabs to workpiece '{workpiece.name}' "
                "because it has no vector geometry."
            )
            return

        if strategy == "equidistant":
            new_tabs = self._calculate_equidistant_tabs(
                workpiece.boundaries, count, width
            )
        else:
            raise NotImplementedError(
                f"Tabbing strategy '{strategy}' not implemented."
            )

        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=new_tabs,
            name=_("Add Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def add_cardinal_tabs(self, workpiece: WorkPiece, width: float):
        """
        Creates and applies 4 tabs to a workpiece at the cardinal points. This
        is an undoable action.

        Args:
            workpiece: The WorkPiece to add tabs to.
            width: The width of each tab in millimeters.
        """
        if not workpiece.boundaries:
            logger.warning(
                f"Cannot add tabs to workpiece '{workpiece.name}' "
                "because it has no vector geometry."
            )
            return

        new_tabs = self._calculate_cardinal_tabs(workpiece.boundaries, width)

        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=new_tabs,
            name=_("Add Cardinal Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def add_single_tab(
        self,
        workpiece: WorkPiece,
        segment_index: int,
        pos: float,
        width: float = 2.0,
        length: float = 1.0,
    ):
        """Adds a single new tab to a workpiece. Undoable."""
        new_tab = Tab(width=width, segment_index=segment_index, pos=pos)

        # Create a new list with the added tab
        new_tabs_list = deepcopy(workpiece.tabs)
        new_tabs_list.append(new_tab)

        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=new_tabs_list,
            name=_("Add Tab"),
        )
        self._editor.history_manager.execute(cmd)

    def remove_single_tab(self, workpiece: WorkPiece, tab_to_remove: Tab):
        """Removes a single tab from a workpiece. Undoable."""
        new_tabs_list = [
            t for t in workpiece.tabs if t.uid != tab_to_remove.uid
        ]

        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=new_tabs_list,
            name=_("Remove Tab"),
        )
        self._editor.history_manager.execute(cmd)

    def clear_tabs(self, workpiece: WorkPiece):
        """Removes all tabs from a workpiece."""
        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=[],
            name=_("Clear Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workpiece_tabs_enabled(self, workpiece: WorkPiece, enabled: bool):
        """Enables or disables tabs for a workpiece."""
        if workpiece.tabs_enabled == enabled:
            return

        old_value = workpiece.tabs_enabled
        workpiece.tabs_enabled = enabled

        # This is a simple property change, so we can use a generic command
        from ..core.undo import ChangePropertyCommand

        cmd = ChangePropertyCommand(
            target=workpiece,
            property_name="tabs_enabled",
            new_value=enabled,
            old_value=old_value,
            name=_("Toggle Tabs"),
        )
        self._editor.history_manager.execute(cmd)

    def set_workpiece_tab_width(self, workpiece: WorkPiece, width: float):
        """Sets the width of all tabs on a workpiece."""
        if not workpiece.tabs:
            return

        old_tabs = deepcopy(workpiece.tabs)
        # Check if any change is actually needed to avoid empty undo commands
        if all(tab.width == width for tab in old_tabs):
            return

        new_tabs = [replace(tab, width=width) for tab in old_tabs]

        cmd = SetWorkpieceTabsCommand(
            editor=self._editor,
            workpiece=workpiece,
            new_tabs=new_tabs,
            name=_("Change Tab Width"),
        )
        self._editor.history_manager.execute(cmd)
