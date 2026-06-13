from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import replace
from gettext import gettext as _
from typing import TYPE_CHECKING, List, Optional, Tuple

from raygeo.geo import Geometry

from ..core.tab import Tab
from ..core.undo import Command
from ..core.workpiece import WorkPiece
from ..usage import get_usage_tracker

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

        total_length = geometry.distance()
        if total_length == 0:
            return []

        spacing = total_length / count
        targets = [(i + 0.5) * spacing for i in range(count)]
        positions = geometry.get_positions_at_distances(targets)
        return [
            Tab(width=width, segment_index=si, pos=min(1.0, max(0.0, t)))
            for si, t, _ in positions
        ]

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
        get_usage_tracker().track_page_view(
            "/doc/add-tabs/equidistant", "Add Equidistant Tabs"
        )

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
        get_usage_tracker().track_page_view(
            "/doc/add-tabs/cardinal", "Add Cardinal Tabs"
        )

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
