from typing import Optional
from ..commands import AddItemsCommand
from ..entities import Point, Line
from .base import SketchTool


class LineTool(SketchTool):
    """Handles creating lines between points."""

    def __init__(self, element):
        super().__init__(element)
        self.line_start_id: Optional[int] = None
        self.start_point_temp: bool = False

    def on_deactivate(self):
        """Clean up the starting point if a line was not finished."""
        if self.start_point_temp:
            self.element.remove_point_if_unused(self.line_start_id)
        self.line_start_id = None
        self.start_point_temp = False

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
                self.start_point_temp = False

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
                self.start_point_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.line_start_id = pid_hit
                self.start_point_temp = False

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

                # Adopt start point if it was temp
                if self.start_point_temp:
                    try:
                        p_start = self.element.sketch.registry.get_point(
                            self.line_start_id
                        )
                        # Remove from registry so Command can add it properly
                        self.element.sketch.registry.points.remove(p_start)
                        points_to_add.insert(0, p_start)
                    except (IndexError, ValueError):
                        pass

                cmd = AddItemsCommand(
                    self.element.sketch,
                    _("Add Line"),
                    points=points_to_add,
                    entities=[new_line],
                )
                self.element.execute_command(cmd)

            # Start a new line segment from this point
            self.line_start_id = pid_hit
            # The new start point is either existing or just committed,
            # so it's not temp
            self.start_point_temp = False
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True
