from typing import Optional, Dict
from ..commands import RectangleCommand
from .base import SketchTool


class RectangleTool(SketchTool):
    """Handles creating rectangles."""

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
        point_ids_to_remove = {
            self._preview_ids.get("p_end"),
            self._preview_ids.get("p2"),
            self._preview_ids.get("p4"),
        }
        point_ids_to_remove.discard(None)

        entity_ids_to_remove = {
            e.id
            for e in registry.entities
            if any(pid in point_ids_to_remove for pid in e.get_point_ids())
        }
        registry.remove_entities_by_id(list(entity_ids_to_remove))

        # Remove points
        registry.points = [
            p for p in registry.points if p.id not in point_ids_to_remove
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
        """Updates the live preview of the rectangle."""
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
            final_pid_hit = pid_hit
            # Check if we hit a preview point BEFORE cleaning up temps.
            # If we hit our own ghost geometry, treat it as a miss
            # (coord input).
            if pid_hit in self._preview_ids.values():
                final_pid_hit = None

            self._cleanup_temps()

            # Create the command to generate the rectangle
            cmd = RectangleCommand(
                self.element.sketch,
                self.start_id,
                (mx, my),
                end_pid=final_pid_hit,
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

        coords = {
            "p2": (p3.x, p1.y),
            "p4": (p1.x, p3.y),
        }

        if is_creation:
            # Create points
            for name, (px, py) in coords.items():
                self._preview_ids[name] = registry.add_point(px, py)

            # Create lines
            registry.add_line(self.start_id, self._preview_ids["p2"])
            registry.add_line(
                self._preview_ids["p2"], self._preview_ids["p_end"]
            )
            registry.add_line(
                self._preview_ids["p_end"], self._preview_ids["p4"]
            )
            registry.add_line(self._preview_ids["p4"], self.start_id)
        else:
            # Update points
            for name, (px, py) in coords.items():
                p = registry.get_point(self._preview_ids[name])
                p.x, p.y = px, py

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass
