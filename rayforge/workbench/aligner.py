import math
from typing import Optional, Tuple, TYPE_CHECKING
from ..core.workpiece import WorkPiece
from ..undo import SetterCommand

# Use TYPE_CHECKING to avoid circular imports at runtime.
# The WorkSurface will import this Aligner, so we can't have a runtime
# import of WorkSurface here.
if TYPE_CHECKING:
    from .surface import WorkSurface


class Aligner:
    """
    Handles alignment and distribution operations for workpieces on a
    WorkSurface.

    This class centralizes the logic for calculating bounding boxes and
    creating undoable commands for alignment actions.
    """

    def __init__(self, surface: "WorkSurface"):
        """
        Initializes the Aligner.

        Args:
            surface: The WorkSurface instance to operate on.
        """
        self.surface = surface
        self.doc = surface.doc

    def _get_workpiece_bbox_mm(
        self, wp: WorkPiece
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the axis-aligned bounding box (min_x, min_y, max_x, max_y)
        of a single workpiece in world (mm) coordinates, accounting for its
        rotation.
        """
        if not wp.pos or not wp.size:
            return None

        pos_x, pos_y = wp.pos
        w, h = wp.size
        angle_rad = math.radians(wp.angle)

        center_x, center_y = pos_x + w / 2, pos_y + h / 2

        # Corner coordinates relative to the workpiece's center
        corners_rel = [
            (-w / 2, -h / 2),
            (w / 2, -h / 2),
            (w / 2, h / 2),
            (-w / 2, h / 2),
        ]
        cos_a, sin_a = math.cos(angle_rad), math.sin(angle_rad)

        min_x, max_x = float("inf"), float("-inf")
        min_y, max_y = float("inf"), float("-inf")

        for rel_x, rel_y in corners_rel:
            rot_x = rel_x * cos_a - rel_y * sin_a
            rot_y = rel_x * sin_a + rel_y * cos_a
            abs_corner_x, abs_corner_y = center_x + rot_x, center_y + rot_y
            min_x, max_x = min(min_x, abs_corner_x), max(max_x, abs_corner_x)
            min_y, max_y = min(min_y, abs_corner_y), max(max_y, abs_corner_y)

        if math.isinf(min_x):
            return None

        return (min_x, min_y, max_x, max_y)

    def _get_selection_bbox_mm(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Calculates the collective axis-aligned bounding box (x, y, width,
        height) of all selected workpieces.
        """
        workpieces = self.surface.get_selected_workpieces()
        if not workpieces:
            return None

        overall_min_x, overall_max_x = float("inf"), float("-inf")
        overall_min_y, overall_max_y = float("inf"), float("-inf")

        for wp in workpieces:
            bbox = self._get_workpiece_bbox_mm(wp)
            if not bbox:
                continue
            min_x, min_y, max_x, max_y = bbox
            overall_min_x = min(overall_min_x, min_x)
            overall_max_x = max(overall_max_x, max_x)
            overall_min_y = min(overall_min_y, min_y)
            overall_max_y = max(overall_max_y, max_y)

        if math.isinf(overall_min_x):
            return None

        return (
            overall_min_x,
            overall_min_y,
            overall_max_x - overall_min_x,
            overall_max_y - overall_min_y,
        )

    def center_horizontally(self):
        """
        Horizontally aligns selected workpieces.
        - One item: centered on the canvas.
        - Multiple items: centered on their collective bounding box center.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_center_x: float
        if len(selected_wps) == 1:
            target_center_x = self.surface.width_mm / 2
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_center_x = bbox[0] + bbox[2] / 2

        with self.doc.history_manager.transaction(
            _("Center Horizontally")
        ) as t:
            for wp in selected_wps:
                if not wp.pos or not wp.size:
                    continue
                wp_center_x = wp.pos[0] + wp.size[0] / 2
                delta_x = target_center_x - wp_center_x
                if abs(delta_x) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0] + delta_x, old_pos[1])
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))

    def center_vertically(self):
        """
        Vertically aligns selected workpieces.
        - One item: centered on the canvas.
        - Multiple items: centered on their collective bounding box center.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_center_y: float
        if len(selected_wps) == 1:
            target_center_y = self.surface.height_mm / 2
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_center_y = bbox[1] + bbox[3] / 2

        with self.doc.history_manager.transaction(_("Center Vertically")) as t:
            for wp in selected_wps:
                if not wp.pos or not wp.size:
                    continue
                wp_center_y = wp.pos[1] + wp.size[1] / 2
                delta_y = target_center_y - wp_center_y
                if abs(delta_y) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0], old_pos[1] + delta_y)
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))

    def align_left(self):
        """
        Aligns the left edge of selected items.
        - One item: aligned to the left edge of the canvas.
        - Multiple items: aligned to the left edge of their collective bbox.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_x: float
        if len(selected_wps) == 1:
            target_x = 0.0
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_x = bbox[0]  # Left edge of the collective box

        with self.doc.history_manager.transaction(_("Align Left")) as t:
            for wp in selected_wps:
                wp_bbox = self._get_workpiece_bbox_mm(wp)
                if not wp_bbox or not wp.pos:
                    continue
                delta_x = target_x - wp_bbox[0]
                if abs(delta_x) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0] + delta_x, old_pos[1])
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))

    def align_right(self):
        """
        Aligns the right edge of selected items.
        - One item: aligned to the right edge of the canvas.
        - Multiple items: aligned to the right edge of their collective bbox.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_x: float
        if len(selected_wps) == 1:
            target_x = self.surface.width_mm
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_x = bbox[0] + bbox[2]  # Right edge of collective box

        with self.doc.history_manager.transaction(_("Align Right")) as t:
            for wp in selected_wps:
                wp_bbox = self._get_workpiece_bbox_mm(wp)
                if not wp_bbox or not wp.pos:
                    continue
                delta_x = target_x - wp_bbox[2]
                if abs(delta_x) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0] + delta_x, old_pos[1])
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))

    def align_top(self):
        """
        Aligns the top edge of selected items.
        - One item: aligned to the top edge of the canvas.
        - Multiple items: aligned to the top edge of their collective bbox.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_y: float
        if len(selected_wps) == 1:
            target_y = self.surface.height_mm
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_y = bbox[1] + bbox[3]  # Top edge of collective box

        with self.doc.history_manager.transaction(_("Align Top")) as t:
            for wp in selected_wps:
                wp_bbox = self._get_workpiece_bbox_mm(wp)
                if not wp_bbox or not wp.pos:
                    continue
                delta_y = target_y - wp_bbox[3]
                if abs(delta_y) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0], old_pos[1] + delta_y)
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))

    def align_bottom(self):
        """
        Aligns the bottom edge of selected items.
        - One item: aligned to the bottom edge of the canvas.
        - Multiple items: aligned to the bottom edge of their collective bbox.
        """
        selected_wps = self.surface.get_selected_workpieces()
        if not selected_wps:
            return

        target_y: float
        if len(selected_wps) == 1:
            target_y = 0.0
        else:
            bbox = self._get_selection_bbox_mm()
            if not bbox:
                return
            target_y = bbox[1]  # Bottom edge of collective box

        with self.doc.history_manager.transaction(_("Align Bottom")) as t:
            for wp in selected_wps:
                wp_bbox = self._get_workpiece_bbox_mm(wp)
                if not wp_bbox or not wp.pos:
                    continue
                delta_y = target_y - wp_bbox[1]
                if abs(delta_y) > 1e-6:
                    old_pos = wp.pos
                    new_pos = (old_pos[0], old_pos[1] + delta_y)
                    t.execute(SetterCommand(wp, "set_pos", new_pos, old_pos))
