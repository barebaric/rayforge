import math
from typing import Optional, cast
from .base import SketchTool


def _remove_point_if_unused(element, pid: Optional[int]):
    """Removes a point from the registry if it's not part of any entity."""
    if pid is None:
        return
    # Call the new backend method
    if not element.sketch.registry.is_point_used(pid):
        # Directly manipulate the list to remove the point
        element.sketch.registry.points = [
            p for p in element.sketch.registry.points if p.id != pid
        ]
        element.mark_dirty()


class LineTool(SketchTool):
    """Handles creating lines between points."""

    def __init__(self, element):
        super().__init__(element)
        self.line_start_id: Optional[int] = None

    def on_deactivate(self):
        """Clean up the starting point if a line was not finished."""
        _remove_point_if_unused(self.element, self.line_start_id)
        self.line_start_id = None

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        # Use screen_to_model for coordinate entry
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        # Check if we hit an existing point to snap to
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None

        return self._handle_click(pid_hit, mx, my)

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def _handle_click(self, pid_hit, mx, my) -> bool:
        if self.line_start_id is not None:
            try:
                self.element.sketch.registry.get_point(self.line_start_id)
            except IndexError:
                # Start point was deleted, reset the tool
                self.line_start_id = None

        if pid_hit is None:
            pid_hit = self.element.sketch.add_point(mx, my)
            self.element.update_bounds_from_sketch()

        if self.line_start_id is None:
            self.line_start_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)
        else:
            if self.line_start_id != pid_hit:
                self.element.sketch.add_line(
                    self.line_start_id, cast(int, pid_hit)
                )

            # Start a new line segment from this point
            self.line_start_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True


class ArcTool(SketchTool):
    """Handles creating arcs (Center -> Start -> End)."""

    def __init__(self, element):
        super().__init__(element)
        self.center_id: Optional[int] = None
        self.start_id: Optional[int] = None

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        _remove_point_if_unused(self.element, self.start_id)
        _remove_point_if_unused(self.element, self.center_id)
        self.center_id = None
        self.start_id = None

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None
        return self._handle_click(pid_hit, mx, my)

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Start -> End

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool completely
                self.center_id = None
                self.start_id = None

        if self.start_id is not None:
            try:
                self.element.sketch.registry.get_point(self.start_id)
            except IndexError:
                # Start point was deleted, reset to expecting start point
                self.start_id = None

        if self.center_id is None:
            # Step 1: Center Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.element.update_bounds_from_sketch()

            self.center_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        elif self.start_id is None:
            # Step 2: Start Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.element.update_bounds_from_sketch()

            # Cannot start where center is
            if pid_hit != self.center_id:
                self.start_id = pid_hit
                self.element.selection.select_point(pid_hit, True)

        else:
            # Step 3: End Point
            if pid_hit is None:
                c = self.element.sketch.registry.get_point(self.center_id)
                s = self.element.sketch.registry.get_point(self.start_id)

                radius = math.hypot(s.x - c.x, s.y - c.y)
                curr_dist = math.hypot(mx - c.x, my - c.y)

                if curr_dist > 1e-9:
                    # Project onto circle
                    scale = radius / curr_dist
                    final_x = c.x + (mx - c.x) * scale
                    final_y = c.y + (my - c.y) * scale
                else:
                    final_x, final_y = mx, my

                pid_hit = self.element.sketch.add_point(final_x, final_y)
                self.element.update_bounds_from_sketch()

            # Cannot end at start or center
            if pid_hit != self.start_id and pid_hit != self.center_id:
                self.element.sketch.add_arc(
                    start=self.start_id, end=pid_hit, center=self.center_id
                )

                # ENFORCE ARC GEOMETRY: dist(C, S) == dist(C, E)
                self.element.sketch.constrain_equal_distance(
                    self.center_id, self.start_id, self.center_id, pid_hit
                )

                self.element.sketch.solve()

                # Reset tool state
                self.center_id = None
                self.start_id = None

                # Select the last point
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True


class CircleTool(SketchTool):
    """Handles creating circles (Center -> Radius Point)."""

    def __init__(self, element):
        super().__init__(element)
        self.center_id: Optional[int] = None

    def on_deactivate(self):
        """Clean up the center point if a circle was not finished."""
        _remove_point_if_unused(self.element, self.center_id)
        self.center_id = None

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        hit_type, hit_obj = self.element.hittester.get_hit_data(
            world_x, world_y, self.element
        )
        pid_hit = hit_obj if hit_type == "point" else None
        return self._handle_click(pid_hit, mx, my)

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Radius Point

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool
                self.center_id = None

        if self.center_id is None:
            # Step 1: Center Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.element.update_bounds_from_sketch()

            self.center_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        else:
            # Step 2: Radius Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.element.update_bounds_from_sketch()

            # Cannot have radius point at center
            if pid_hit != self.center_id:
                self.element.sketch.add_circle(self.center_id, pid_hit)
                self.element.sketch.solve()

                # Reset for next circle
                self.center_id = None
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True
