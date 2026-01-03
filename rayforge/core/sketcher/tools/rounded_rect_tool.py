from typing import Optional, Dict
from ..entities import Point, Line, Arc
from ..commands import AddItemsCommand
from ..constraints import (
    HorizontalConstraint,
    VerticalConstraint,
    TangentConstraint,
    EqualLengthConstraint,
    EqualDistanceConstraint,
)
from .base import SketchTool


class RoundedRectTool(SketchTool):
    """Handles creating rounded rectangles."""

    DEFAULT_RADIUS = 10.0

    def __init__(self, element):
        super().__init__(element)
        self.start_id: Optional[int] = None
        self.start_temp: bool = False

        # Live Preview State
        self._preview_ids: Dict[str, int] = {}
        self._is_previewing = False

    def _cleanup_temps(self):
        """Removes temporary preview entities and points from the registry."""
        if not self._is_previewing:
            return

        registry = self.element.sketch.registry
        ids_to_remove = list(self._preview_ids.values())

        # Remove entities that use these points
        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in ids_to_remove for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        # Remove points
        registry.points = [
            p for p in registry.points if p.id not in ids_to_remove
        ]

        self._preview_ids.clear()
        self._is_previewing = False

    def on_deactivate(self):
        """Clean up if the tool is deactivated mid-creation."""
        self._cleanup_temps()
        if self.start_temp:
            self.element.remove_point_if_unused(self.start_id)
        self.start_id = None
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

    def on_hover_motion(self, world_x: float, world_y: float):
        """Updates the live preview of the rounded rectangle."""
        if not self._is_previewing or self.start_id is None:
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        try:
            # Update the opposite corner's position
            p_end_temp = self.element.sketch.registry.get_point(
                self._preview_ids["p_end"]
            )
            p_end_temp.x = mx
            p_end_temp.y = my
            self._update_preview_geometry()
            self.element.mark_dirty()
        except (IndexError, KeyError):
            self.on_deactivate()  # something went wrong, reset

    def _handle_click(
        self, pid_hit: Optional[int], mx: float, my: float
    ) -> bool:
        if self.start_id is None:
            # --- First Click: Define the start corner ---
            if pid_hit is None:
                self.start_id = self.element.sketch.add_point(mx, my)
                self.start_temp = True
            else:
                self.start_id = pid_hit
                self.start_temp = False

            # Initialize preview state
            self._is_previewing = True
            p_end_id = self.element.sketch.add_point(mx, my)
            self._preview_ids["p_end"] = p_end_id
            self._update_preview_geometry(is_creation=True)
        else:
            # --- Second Click: Finalize the rectangle ---
            final_mx, final_my = mx, my
            # If a point is hit, snap to it
            if pid_hit is not None:
                final_p = self.element.sketch.registry.get_point(pid_hit)
                final_mx, final_my = final_p.x, final_p.y

            start_p = self.element.sketch.registry.get_point(self.start_id)

            # Cleanup preview geometry before creating the final command
            self._cleanup_temps()

            # Generate geometry and constraints for the command
            points, entities, constraints = self._generate_geometry(
                start_p.x, start_p.y, final_mx, final_my, start_p.id
            )

            # Abort if geometry is degenerate (zero size)
            if not points:
                # If the start point was temporary, it needs to be cleaned up.
                if self.start_temp:
                    self.element.remove_point_if_unused(self.start_id)
                self.start_id = None
                self.start_temp = False
                self.element.mark_dirty()
                return True  # We handled the click by aborting.

            # Determine the points to add. We intentionally exclude p1 and p3
            # (the sharp corners) as they are virtual construction points
            # for the rounded rect and shouldn't persist as dangling points.
            points_to_add = []

            # The 'points' dict contains p1 (start), p3 (end), and all t/c
            # points. We filter out p1 and p3.
            for name, p_obj in points.items():
                if name not in ("p1", "p3"):
                    points_to_add.append(p_obj)

            # Special handling for the start point if it was temporary
            if self.start_temp:
                # Remove the temporary start point from the registry.
                # Since we are NOT adding it to 'points_to_add', it will
                # simply cease to exist, effectively "consuming" the click.
                self.element.sketch.registry.points.remove(start_p)

            # We don't need logic for pid_hit (snapping end) regarding p3,
            # because p3 is never added to the sketch anyway.

            cmd = AddItemsCommand(
                self.element.sketch,
                _("Add Rounded Rectangle"),
                points=points_to_add,
                entities=entities,
                constraints=constraints,
            )
            self.element.execute_command(cmd)

            # Reset tool for the next rectangle
            self.start_id = None
            self.start_temp = False
            self._is_previewing = False

        self.element.mark_dirty()
        return True

    def _update_preview_geometry(self, is_creation: bool = False):
        """Calculates and creates/updates preview geometry."""
        registry = self.element.sketch.registry
        p1 = registry.get_point(self.start_id)
        p3 = registry.get_point(self._preview_ids["p_end"])

        width = abs(p3.x - p1.x)
        height = abs(p3.y - p1.y)

        # On hover, if size is zero, do nothing. On creation, we must proceed
        # to create the coincident points for the hover to update later.
        if not is_creation and (width < 1e-6 or height < 1e-6):
            return

        radius = 0.0
        if width > 1e-6 and height > 1e-6:
            radius = min(self.DEFAULT_RADIUS, width / 2.0, height / 2.0)

        sx = 1 if p3.x > p1.x else -1
        sy = 1 if p3.y > p1.y else -1

        # Point coordinates
        coords = {
            "t1": (p1.x + sx * radius, p1.y),
            "t2": (p3.x - sx * radius, p1.y),
            "t3": (p3.x, p1.y + sy * radius),
            "t4": (p3.x, p3.y - sy * radius),
            "t5": (p3.x - sx * radius, p3.y),
            "t6": (p1.x + sx * radius, p3.y),
            "t7": (p1.x, p3.y - sy * radius),
            "t8": (p1.x, p1.y + sy * radius),
            "c1": (p1.x + sx * radius, p1.y + sy * radius),
            "c2": (p3.x - sx * radius, p1.y + sy * radius),
            "c3": (p3.x - sx * radius, p3.y - sy * radius),
            "c4": (p1.x + sx * radius, p3.y - sy * radius),
        }

        # Correct logic for convex rounded corners
        is_cw = sx * sy < 0

        if is_creation:
            # Create all points and entities
            for name, (px, py) in coords.items():
                self._preview_ids[name] = registry.add_point(px, py)

            # Lines
            registry.add_line(self._preview_ids["t1"], self._preview_ids["t2"])
            registry.add_line(self._preview_ids["t3"], self._preview_ids["t4"])
            registry.add_line(self._preview_ids["t5"], self._preview_ids["t6"])
            registry.add_line(self._preview_ids["t7"], self._preview_ids["t8"])

            # Arcs
            # registry.add_arc(start, end, center, cw)
            self._preview_ids["arc1"] = registry.add_arc(
                self._preview_ids["t8"],
                self._preview_ids["t1"],
                self._preview_ids["c1"],
                cw=is_cw,
            )
            self._preview_ids["arc2"] = registry.add_arc(
                self._preview_ids["t2"],
                self._preview_ids["t3"],
                self._preview_ids["c2"],
                cw=is_cw,
            )
            self._preview_ids["arc3"] = registry.add_arc(
                self._preview_ids["t4"],
                self._preview_ids["t5"],
                self._preview_ids["c3"],
                cw=is_cw,
            )
            self._preview_ids["arc4"] = registry.add_arc(
                self._preview_ids["t6"],
                self._preview_ids["t7"],
                self._preview_ids["c4"],
                cw=is_cw,
            )
        else:
            # Update existing points
            for name, (px, py) in coords.items():
                p = registry.get_point(self._preview_ids[name])
                p.x, p.y = px, py

            # Update arc directions
            arc_keys = ["arc1", "arc2", "arc3", "arc4"]
            for key in arc_keys:
                arc_entity = registry.get_entity(self._preview_ids[key])
                if isinstance(arc_entity, Arc):
                    arc_entity.clockwise = is_cw

    def _generate_geometry(self, x1, y1, x2, y2, start_id: int):
        """Generates final points, entities, and constraints."""
        width = abs(x2 - x1)
        height = abs(y2 - y1)

        if width < 1e-6 or height < 1e-6:
            return {}, [], []

        radius = min(self.DEFAULT_RADIUS, width / 2.0, height / 2.0)
        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        temp_id_counter = -1

        def next_temp_id():
            nonlocal temp_id_counter
            temp_id_counter -= 1
            return temp_id_counter

        # Create points
        points = {
            "p1": Point(start_id, x1, y1),
            "p3": Point(next_temp_id(), x2, y2),
            "t1": Point(next_temp_id(), x1 + sx * radius, y1),
            "t2": Point(next_temp_id(), x2 - sx * radius, y1),
            "t3": Point(next_temp_id(), x2, y1 + sy * radius),
            "t4": Point(next_temp_id(), x2, y2 - sy * radius),
            "t5": Point(next_temp_id(), x2 - sx * radius, y2),
            "t6": Point(next_temp_id(), x1 + sx * radius, y2),
            "t7": Point(next_temp_id(), x1, y2 - sy * radius),
            "t8": Point(next_temp_id(), x1, y1 + sy * radius),
            "c1": Point(next_temp_id(), x1 + sx * radius, y1 + sy * radius),
            "c2": Point(next_temp_id(), x2 - sx * radius, y1 + sy * radius),
            "c3": Point(next_temp_id(), x2 - sx * radius, y2 - sy * radius),
            "c4": Point(next_temp_id(), x1 + sx * radius, y2 - sy * radius),
        }

        # Correct logic for convex rounded corners
        is_cw = sx * sy < 0

        # Create entities
        # Arc(id, start, end, center, cw)
        entities = [
            Line(next_temp_id(), points["t1"].id, points["t2"].id),
            Line(next_temp_id(), points["t3"].id, points["t4"].id),
            Line(next_temp_id(), points["t5"].id, points["t6"].id),
            Line(next_temp_id(), points["t7"].id, points["t8"].id),
            Arc(
                next_temp_id(),
                points["t8"].id,
                points["t1"].id,
                points["c1"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t2"].id,
                points["t3"].id,
                points["c2"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t4"].id,
                points["t5"].id,
                points["c3"].id,
                clockwise=is_cw,
            ),
            Arc(
                next_temp_id(),
                points["t6"].id,
                points["t7"].id,
                points["c4"].id,
                clockwise=is_cw,
            ),
        ]

        # Create constraints
        constraints = [
            HorizontalConstraint(points["t1"].id, points["t2"].id),
            VerticalConstraint(points["t3"].id, points["t4"].id),
            HorizontalConstraint(points["t5"].id, points["t6"].id),
            VerticalConstraint(points["t7"].id, points["t8"].id),
            # top line -> top-left arc
            TangentConstraint(entities[0].id, entities[4].id),
            # left line -> top-left arc
            TangentConstraint(entities[3].id, entities[4].id),
            # top line -> top-right arc
            TangentConstraint(entities[0].id, entities[5].id),
            # right line -> top-right arc
            TangentConstraint(entities[1].id, entities[5].id),
            # right line -> bottom-right arc
            TangentConstraint(entities[1].id, entities[6].id),
            # bottom line -> bottom-right arc
            TangentConstraint(entities[2].id, entities[6].id),
            # bottom line -> bottom-left arc
            TangentConstraint(entities[2].id, entities[7].id),
            # left line -> bottom-left arc
            TangentConstraint(entities[3].id, entities[7].id),
            # all arcs equal radius
            EqualLengthConstraint([e.id for e in entities[4:]]),
            # Ensure arc endpoints are equidistant from center (force
            # circularity)
            EqualDistanceConstraint(
                points["c1"].id,
                points["t8"].id,
                points["c1"].id,
                points["t1"].id,
            ),
            EqualDistanceConstraint(
                points["c2"].id,
                points["t2"].id,
                points["c2"].id,
                points["t3"].id,
            ),
            EqualDistanceConstraint(
                points["c3"].id,
                points["t4"].id,
                points["c3"].id,
                points["t5"].id,
            ),
            EqualDistanceConstraint(
                points["c4"].id,
                points["t6"].id,
                points["c4"].id,
                points["t7"].id,
            ),
        ]

        return points, entities, constraints

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass
