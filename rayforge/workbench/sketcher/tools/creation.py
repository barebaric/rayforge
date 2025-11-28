import math
from typing import Optional
from rayforge.core.sketcher.entities import Point, Line, Arc, Circle
from rayforge.core.sketcher.constraints import EqualDistanceConstraint
from ..sketch_cmd import AddItemsCommand
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

        new_point = None
        if pid_hit is None:
            # Create a point temporarily, but don't add to registry yet.
            # Give it a temporary ID that the AddItemsCommand will replace.
            temp_id = self.element.sketch.registry._id_counter
            pid_hit = temp_id
            new_point = Point(temp_id, mx, my)

        if self.line_start_id is None:
            if new_point:
                # This is the first point of a new line, add it for preview.
                # This is not undoable, but is cleaned up by on_deactivate.
                self.line_start_id = self.element.sketch.add_point(mx, my)
                self.element.update_bounds_from_sketch()
            else:
                self.line_start_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(self.line_start_id, False)
        else:
            if self.line_start_id != pid_hit:
                # Create the line entity with a temporary ID.
                temp_line_id = self.element.sketch.registry._id_counter + (
                    1 if new_point else 0
                )
                new_line = Line(temp_line_id, self.line_start_id, pid_hit)

                # Create command
                points_to_add = [new_point] if new_point else []
                cmd = AddItemsCommand(
                    self.element,
                    "Add Line",
                    points=points_to_add,
                    entities=[new_line],
                )
                if self.element.sketch_canvas:
                    self.element.sketch_canvas.history_manager.execute(cmd)

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
            new_point = None
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

                temp_id = self.element.sketch.registry._id_counter
                pid_hit = temp_id
                new_point = Point(temp_id, final_x, final_y)

            # Cannot end at start or center
            if pid_hit != self.start_id and pid_hit != self.center_id:
                temp_arc_id = self.element.sketch.registry._id_counter + (
                    1 if new_point else 0
                )
                new_arc = Arc(
                    temp_arc_id, self.start_id, pid_hit, self.center_id
                )

                # ENFORCE ARC GEOMETRY: dist(C, S) == dist(C, E)
                geom_constr = EqualDistanceConstraint(
                    self.center_id, self.start_id, self.center_id, pid_hit
                )

                cmd = AddItemsCommand(
                    self.element,
                    "Add Arc",
                    points=[new_point] if new_point else [],
                    entities=[new_arc],
                    constraints=[geom_constr],
                )
                if self.element.sketch_canvas:
                    self.element.sketch_canvas.history_manager.execute(cmd)

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
            new_point = None
            if pid_hit is None:
                temp_id = self.element.sketch.registry._id_counter
                pid_hit = temp_id
                new_point = Point(temp_id, mx, my)

            # Cannot have radius point at center
            if pid_hit != self.center_id:
                temp_circle_id = self.element.sketch.registry._id_counter + (
                    1 if new_point else 0
                )
                new_circle = Circle(temp_circle_id, self.center_id, pid_hit)
                cmd = AddItemsCommand(
                    self.element,
                    "Add Circle",
                    points=[new_point] if new_point else [],
                    entities=[new_circle],
                )
                if self.element.sketch_canvas:
                    self.element.sketch_canvas.history_manager.execute(cmd)

                # Reset for next circle
                self.center_id = None
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True
