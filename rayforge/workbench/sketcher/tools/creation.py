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
        self.start_point_temp: bool = False

    def on_deactivate(self):
        """Clean up the starting point if a line was not finished."""
        if self.start_point_temp:
            _remove_point_if_unused(self.element, self.line_start_id)
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
                    self.element,
                    _("Add Line"),
                    points=points_to_add,
                    entities=[new_line],
                )
                if self.element.editor:
                    self.element.editor.history_manager.execute(cmd)

            # Start a new line segment from this point
            self.line_start_id = pid_hit
            # The new start point is either existing or just committed,
            # so it's not temp
            self.start_point_temp = False
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
        self.center_temp: bool = False
        self.start_temp: bool = False

        # Live Preview State
        self.temp_end_id: Optional[int] = None
        self.temp_entity_id: Optional[int] = None

    def _cleanup_temps(self):
        """Removes temporary preview entities from the registry."""
        if self.temp_entity_id is not None:
            # Remove entity
            self.element.sketch.registry.entities = [
                e
                for e in self.element.sketch.registry.entities
                if e.id != self.temp_entity_id
            ]
            # Rebuild map
            reg = self.element.sketch.registry
            reg._entity_map = {e.id: e for e in reg.entities}
            self.temp_entity_id = None

        if self.temp_end_id is not None:
            _remove_point_if_unused(self.element, self.temp_end_id)
            self.temp_end_id = None

    def on_deactivate(self):
        """Clean up any intermediate points if the arc was not finished."""
        self._cleanup_temps()

        if self.start_temp:
            _remove_point_if_unused(self.element, self.start_id)
        if self.center_temp:
            _remove_point_if_unused(self.element, self.center_id)

        self.center_id = None
        self.start_id = None
        self.center_temp = False
        self.start_temp = False
        self.element.mark_dirty()

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

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the arc."""
        # Only update if we are in the final stage (drawing the arc curve)
        if (
            self.center_id is None
            or self.start_id is None
            or self.temp_entity_id is None
            or self.temp_end_id is None
        ):
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            c = self.element.sketch.registry.get_point(self.center_id)
            s = self.element.sketch.registry.get_point(self.start_id)
            e = self.element.sketch.registry.get_point(self.temp_end_id)
            arc_ent = self.element.sketch.registry.get_entity(
                self.temp_entity_id
            )
        except IndexError:
            return

        if not isinstance(arc_ent, Arc):
            return

        # 1. Project mouse position onto the circle defined by Center-Start
        radius = math.hypot(s.x - c.x, s.y - c.y)
        curr_dist = math.hypot(mx - c.x, my - c.y)

        if curr_dist > 1e-9:
            scale = radius / curr_dist
            final_x = c.x + (mx - c.x) * scale
            final_y = c.y + (my - c.y) * scale
        else:
            final_x, final_y = mx, my

        e.x = final_x
        e.y = final_y

        # 2. Determine Winding (Clockwise/Counter-Clockwise)
        # We determine direction based on the cross product of
        # Vector(Center->Start) and Vector(Center->Mouse).
        # This allows reversing direction by passing back through the start
        # line.
        vec_s_x, vec_s_y = s.x - c.x, s.y - c.y
        vec_m_x, vec_m_y = mx - c.x, my - c.y

        # 2D Cross Product: A_x * B_y - A_y * B_x
        det = vec_s_x * vec_m_y - vec_s_y * vec_m_x

        # In standard Y-Up math, det < 0 is CW.
        # In Y-Down (Screen), det > 0 is CW.
        # The Sketcher coordinate system behavior depends on transforms,
        # but typically det < 0 maps to "Clockwise" property for arcs in CAD.
        arc_ent.clockwise = bool(det < 0)

        self.element.mark_dirty()

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Start -> End

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool completely
                self.on_deactivate()

        if self.start_id is not None:
            try:
                self.element.sketch.registry.get_point(self.start_id)
            except IndexError:
                # Start point was deleted, reset to expecting start point
                self.start_id = None
                self.start_temp = False
                self._cleanup_temps()

        if self.center_id is None:
            # Step 1: Center Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.center_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.center_temp = False

            self.center_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

        elif self.start_id is None:
            # Step 2: Start Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.start_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.start_temp = False

            # Cannot start where center is
            if pid_hit != self.center_id:
                self.start_id = pid_hit
                self.element.selection.select_point(pid_hit, True)

                # Create a temporary End point and Arc entity to visualize
                # dragging
                self.temp_end_id = self.element.sketch.add_point(mx, my)
                self.temp_entity_id = self.element.sketch.add_arc(
                    self.start_id, self.temp_end_id, self.center_id
                )

        else:
            # Step 3: End Point (Finalize)

            # Determine logic from the Preview State
            is_clockwise = False
            if self.temp_entity_id is not None:
                temp_ent = self.element.sketch.registry.get_entity(
                    self.temp_entity_id
                )
                if isinstance(temp_ent, Arc):
                    is_clockwise = temp_ent.clockwise

            # If we hit our own preview point, ignore the hit so we create a
            # new, real point
            if pid_hit == self.temp_end_id:
                pid_hit = None

            # Clean up the preview geometry before adding the real command
            self._cleanup_temps()

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
                    temp_arc_id,
                    self.start_id,
                    pid_hit,
                    self.center_id,
                    clockwise=bool(is_clockwise),
                )

                # ENFORCE ARC GEOMETRY: dist(C, S) == dist(C, E)
                geom_constr = EqualDistanceConstraint(
                    self.center_id, self.start_id, self.center_id, pid_hit
                )

                points_to_add = [new_point] if new_point else []

                # Adopt temp points
                if self.center_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.center_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                if self.start_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.start_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                cmd = AddItemsCommand(
                    self.element,
                    _("Add Arc"),
                    points=points_to_add,
                    entities=[new_arc],
                    constraints=[geom_constr],
                )
                if self.element.editor:
                    self.element.editor.history_manager.execute(cmd)

                # Reset tool state
                self.center_id = None
                self.start_id = None
                self.center_temp = False
                self.start_temp = False

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
        self.center_temp: bool = False

        # Live Preview State
        self.temp_radius_id: Optional[int] = None
        self.temp_entity_id: Optional[int] = None

    def _cleanup_temps(self):
        """Removes temporary preview entities from the registry."""
        if self.temp_entity_id is not None:
            # Remove entity
            self.element.sketch.registry.entities = [
                e
                for e in self.element.sketch.registry.entities
                if e.id != self.temp_entity_id
            ]
            # Rebuild map
            reg = self.element.sketch.registry
            reg._entity_map = {e.id: e for e in reg.entities}
            self.temp_entity_id = None

        if self.temp_radius_id is not None:
            _remove_point_if_unused(self.element, self.temp_radius_id)
            self.temp_radius_id = None

    def on_deactivate(self):
        """Clean up the center point if a circle was not finished."""
        self._cleanup_temps()
        if self.center_temp:
            _remove_point_if_unused(self.element, self.center_id)
        self.center_id = None
        self.center_temp = False
        self.element.mark_dirty()

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

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the circle."""
        if (
            self.center_id is None
            or self.temp_entity_id is None
            or self.temp_radius_id is None
        ):
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            r_pt = self.element.sketch.registry.get_point(self.temp_radius_id)
            r_pt.x = mx
            r_pt.y = my
            self.element.mark_dirty()
        except IndexError:
            pass

    def _handle_click(self, pid_hit, mx, my) -> bool:
        # State machine: Center -> Radius Point

        if self.center_id is not None:
            try:
                self.element.sketch.registry.get_point(self.center_id)
            except IndexError:
                # Center point was deleted, reset the tool
                self.on_deactivate()

        if self.center_id is None:
            # Step 1: Center Point
            if pid_hit is None:
                pid_hit = self.element.sketch.add_point(mx, my)
                self.center_temp = True
                self.element.update_bounds_from_sketch()
            else:
                self.center_temp = False

            self.center_id = pid_hit
            self.element.selection.clear()
            self.element.selection.select_point(pid_hit, False)

            # Create a temporary Radius point and Circle entity
            self.temp_radius_id = self.element.sketch.add_point(mx, my)
            self.temp_entity_id = self.element.sketch.add_circle(
                self.center_id, self.temp_radius_id
            )

        else:
            # Step 2: Radius Point (Finalize)

            # If we hit our own preview point, ignore the hit so we create a
            # new, real point
            if pid_hit == self.temp_radius_id:
                pid_hit = None

            # Clean up the preview geometry before adding the real command
            self._cleanup_temps()

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

                points_to_add = [new_point] if new_point else []

                if self.center_temp:
                    try:
                        p = self.element.sketch.registry.get_point(
                            self.center_id
                        )
                        self.element.sketch.registry.points.remove(p)
                        points_to_add.append(p)
                    except (IndexError, ValueError):
                        pass

                cmd = AddItemsCommand(
                    self.element,
                    _("Add Circle"),
                    points=points_to_add,
                    entities=[new_circle],
                )
                if self.element.editor:
                    self.element.editor.history_manager.execute(cmd)

                # Reset for next circle
                self.center_id = None
                self.center_temp = False
                self.element.selection.clear()
                self.element.selection.select_point(pid_hit, False)

        self.element.mark_dirty()
        return True
