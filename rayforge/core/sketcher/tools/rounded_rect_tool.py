from typing import Optional, Dict
from ..entities import Arc
from ..commands import RoundedRectCommand
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
            self._cleanup_temps()

            cmd = RoundedRectCommand(
                self.element.sketch,
                self.start_id,
                (mx, my),
                self.DEFAULT_RADIUS,
                is_start_temp=self.start_temp,
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

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass
