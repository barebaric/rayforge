import logging
from typing import Optional, cast
from enum import Enum, auto
import cairo
from blinker import Signal
from ..commands import TextBoxCommand
from ..entities import Line, Point, TextBoxEntity
from .base import SketchTool, SketcherKey
from ...geo import primitives

logger = logging.getLogger(__name__)


class TextBoxState(Enum):
    """Defines the state of the TextBoxTool."""

    IDLE = auto()
    EDITING = auto()


class TextBoxTool(SketchTool):
    def __init__(self, element):
        super().__init__(element)
        self.state = TextBoxState.IDLE
        self.editing_entity_id: Optional[int] = None
        self.text_buffer = ""
        self.cursor_pos: int = 0
        self.cursor_visible = True
        self.is_hovering = False

        # Signals for the UI layer to manage timers, etc.
        self.editing_started = Signal()
        self.editing_finished = Signal()
        self.cursor_moved = Signal()

    def start_editing(self, entity_id: int):
        """Public method to begin editing an existing text box."""
        from ..entities import TextBoxEntity

        entity = self.element.sketch.registry.get_entity(entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        self.editing_entity_id = entity_id
        self.text_buffer = entity.content
        self.cursor_pos = len(self.text_buffer)  # Cursor at the end
        self.state = TextBoxState.EDITING
        self.cursor_visible = True
        self.element.mark_dirty()
        self.editing_started.send(self)

    def on_deactivate(self):
        if self.state == TextBoxState.EDITING:
            self.editing_finished.send(self)
            self._finalize_edit()

        self.state = TextBoxState.IDLE
        self.editing_entity_id = None
        self.text_buffer = ""
        self.cursor_pos = 0
        self.is_hovering = False

    def toggle_cursor_visibility(self):
        """Called by the UI timer to toggle the cursor's visual state."""
        if self.state == TextBoxState.EDITING:
            self.cursor_visible = not self.cursor_visible
            self.element.mark_dirty()

    def on_press(self, world_x: float, world_y: float, n_press: int) -> bool:
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )

        if self.state == TextBoxState.IDLE:
            return self._handle_idle_press(mx, my)
        elif self.state == TextBoxState.EDITING:
            return self._handle_editing_press(mx, my)
        return False

    def _handle_idle_press(self, mx: float, my: float) -> bool:
        cmd = TextBoxCommand(self.element.sketch, origin=(mx, my))
        self.element.execute_command(cmd)

        if cmd.text_box_id is not None:
            self.start_editing(cmd.text_box_id)
        return True

    def _handle_editing_press(self, mx: float, my: float) -> bool:
        if not self._is_point_inside_box(mx, my):
            self.on_deactivate()
            # Allow the click to fall through to the SelectTool
            return False

        # If inside, update cursor position
        self._update_cursor_from_click(mx, my)
        return True

    def _is_point_inside_box(self, mx: float, my: float) -> bool:
        if self.editing_entity_id is None:
            return False

        entity = self.element.sketch.registry.get_entity(
            self.editing_entity_id
        )
        if not isinstance(entity, TextBoxEntity):
            return False

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        p4_id = entity.get_fourth_corner_id(self.element.sketch.registry)
        if not p4_id:
            return False
        p4 = self.element.sketch.registry.get_point(p4_id)

        # Define the polygon for the text box
        polygon = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4.x, p4.y),
            (p_height.x, p_height.y),
        ]

        # Use point-in-polygon check for accurate hit testing (handles
        # rotation)
        return primitives.is_point_in_polygon((mx, my), polygon)

    def _finalize_edit(self):
        from ..commands.text_property import ModifyTextPropertyCommand

        if self.editing_entity_id is not None:
            entity = self.element.sketch.registry.get_entity(
                self.editing_entity_id
            )
            if entity:
                # Use a command to make the final text and size change undoable
                cmd = ModifyTextPropertyCommand(
                    self.element.sketch,
                    self.editing_entity_id,
                    self.text_buffer,
                    entity.font_params,
                )
                self.element.execute_command(cmd)

    def on_drag(self, world_dx: float, world_dy: float):
        pass

    def on_release(self, world_x: float, world_y: float):
        pass

    def on_hover_motion(self, world_x: float, world_y: float):
        if self.state != TextBoxState.EDITING:
            self.is_hovering = False
            return

        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        self.is_hovering = self._is_point_inside_box(mx, my)

    def handle_text_input(self, text: str) -> bool:
        if self.state != TextBoxState.EDITING:
            return False

        self.cursor_visible = True
        self.text_buffer = (
            self.text_buffer[: self.cursor_pos]
            + text
            + self.text_buffer[self.cursor_pos :]
        )
        self.cursor_pos += 1
        self._resize_box_to_fit_text()
        self.cursor_moved.send(self)
        return True

    def handle_key_event(self, key: SketcherKey) -> bool:
        if self.state != TextBoxState.EDITING:
            return False

        self.cursor_visible = True  # Make cursor visible on keypress

        if key == SketcherKey.BACKSPACE:
            if self.cursor_pos > 0:
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos - 1]
                    + self.text_buffer[self.cursor_pos :]
                )
                self.cursor_pos -= 1
                self._resize_box_to_fit_text()
                self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.DELETE:
            if self.cursor_pos < len(self.text_buffer):
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos]
                    + self.text_buffer[self.cursor_pos + 1 :]
                )
                self._resize_box_to_fit_text()
                self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.ARROW_LEFT:
            self.cursor_pos = max(0, self.cursor_pos - 1)
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.ARROW_RIGHT:
            self.cursor_pos = min(len(self.text_buffer), self.cursor_pos + 1)
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.HOME:
            self.cursor_pos = 0
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.END:
            self.cursor_pos = len(self.text_buffer)
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.RETURN or key == SketcherKey.ESCAPE:
            self.on_deactivate()
            return True

        return False

    def _find_opposite_corner(
        self, text_entity: TextBoxEntity
    ) -> Optional[Point]:
        """Finds the 4th point of the bounding box parallelogram."""
        p_w = text_entity.width_id
        for eid in text_entity.construction_line_ids:
            line = self.element.sketch.registry.get_entity(eid)
            if isinstance(line, Line):
                if line.p1_idx == p_w and line.p2_idx != text_entity.origin_id:
                    return self.element.sketch.registry.get_point(line.p2_idx)
                if line.p2_idx == p_w and line.p1_idx != text_entity.origin_id:
                    return self.element.sketch.registry.get_point(line.p1_idx)
        return None

    def _resize_box_to_fit_text(self):
        """Live-updates the box points to match the current text buffer."""
        from ....core.geo.geometry import Geometry

        if self.editing_entity_id is None:
            return

        entity = self.element.sketch.registry.get_entity(
            self.editing_entity_id
        )
        if not isinstance(entity, TextBoxEntity):
            return

        if not self.text_buffer:
            natural_width = 10.0
            natural_height = entity.font_params.get("size", 10.0)
        else:
            natural_geo = Geometry.from_text(
                self.text_buffer,
                font_family=entity.font_params.get("family", "sans-serif"),
                font_size=entity.font_params.get("size", 10.0),
                is_bold=entity.font_params.get("bold", False),
                is_italic=entity.font_params.get("italic", False),
            )
            natural_geo.flip_y()
            min_x, min_y, max_x, max_y = natural_geo.rect()
            natural_width = max(max_x - min_x, 1.0)
            natural_height = max(max_y - min_y, 1.0)

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        # Update point positions based on origin and natural size
        p_width.x = p_origin.x + natural_width
        p_width.y = p_origin.y
        p_height.x = p_origin.x
        p_height.y = p_origin.y + natural_height

        # Manually update the 4th corner to keep the box rectangular
        p4_id = entity.get_fourth_corner_id(self.element.sketch.registry)
        if p4_id:
            p4 = self.element.sketch.registry.get_point(p4_id)
            p4.x = p_width.x
            p4.y = p_height.y

        self.element.mark_dirty()

    def _update_cursor_from_click(self, mx: float, my: float):
        """Finds the best cursor position based on a click in model space."""
        from ....core.geo.geometry import Geometry

        if self.editing_entity_id is None:
            return
        entity = self.element.sketch.registry.get_entity(
            self.editing_entity_id
        )
        if not isinstance(entity, TextBoxEntity):
            return

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        # 1. Project the click point onto the width vector (u) of the box.
        # This handles rotated and scaled text boxes correctly.
        u_vec = (p_width.x - p_origin.x, p_width.y - p_origin.y)
        v_vec = (p_height.x - p_origin.x, p_height.y - p_origin.y)
        det = u_vec[0] * v_vec[1] - u_vec[1] * v_vec[0]
        if abs(det) < 1e-9:
            return

        inv_det = 1.0 / det
        click_vec = (mx - p_origin.x, my - p_origin.y)

        # alpha is the normalized coordinate (0..1) along the width axis
        alpha = (click_vec[0] * v_vec[1] - click_vec[1] * v_vec[0]) * inv_det

        font_params = {
            "font_family": entity.font_params.get("family", "sans-serif"),
            "font_size": entity.font_params.get("size", 10.0),
            "is_bold": entity.font_params.get("bold", False),
            "is_italic": entity.font_params.get("italic", False),
        }

        # 2. Get bounds of full text to determine coordinate space range
        natural_geo = Geometry.from_text(self.text_buffer, **font_params)
        natural_geo.flip_y()
        min_x, _, max_x, _ = natural_geo.rect()

        if not self.text_buffer:
            self.cursor_pos = 0
            self.cursor_visible = True
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return

        src_width = max_x - min_x
        if src_width < 1e-9:
            src_width = 1.0

        # Map normalized alpha to geometry x-coordinate
        target_x_natural = min_x + alpha * src_width

        # 3. Find closest character break
        best_i, min_dist = 0, float("inf")

        # Iterate through all possible cursor positions
        # (before first char ... after last char)
        for i in range(len(self.text_buffer) + 1):
            if i == 0:
                # The start of the text corresponds to min_x
                sub_max_x = min_x
            else:
                # Measure width of substring to find character boundary
                sub_geo = Geometry.from_text(
                    self.text_buffer[:i], **font_params
                )
                sub_geo.flip_y()
                _, _, sub_max_x, _ = sub_geo.rect()

            dist = abs(sub_max_x - target_x_natural)
            if dist < min_dist:
                min_dist = dist
                best_i = i

        self.cursor_pos = best_i
        self.cursor_visible = True
        self.element.mark_dirty()
        self.cursor_moved.send(self)

    def draw_overlay(self, ctx: cairo.Context):
        if (
            self.state != TextBoxState.EDITING
            or self.editing_entity_id is None
        ):
            return

        entity = cast(
            TextBoxEntity,
            self.element.sketch.registry.get_entity(self.editing_entity_id),
        )
        if not entity:
            return

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        from ....core.geo.geometry import Geometry

        font_params = {
            "font_family": entity.font_params.get("family", "sans-serif"),
            "font_size": entity.font_params.get("size", 10.0),
            "is_bold": entity.font_params.get("bold", False),
            "is_italic": entity.font_params.get("italic", False),
        }
        natural_geo = Geometry.from_text(self.text_buffer, **font_params)
        natural_geo.flip_y()
        logger.debug(f"Natural geometry: {natural_geo.rect()}")

        nat_min_x, nat_min_y, nat_max_x, nat_max_y = natural_geo.rect()

        # Handle empty text case for frame mapping logic
        if not self.text_buffer:
            nat_min_x, nat_min_y = 0.0, 0.0
            nat_max_x = 10.0
            nat_max_y = font_params["font_size"]

        transformed_geo = natural_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
        )
        logger.debug(f"Transformed text geometry: {transformed_geo.rect()}")

        ctx.save()
        model_to_screen_matrix = (
            self.element.hittester.get_model_to_screen_transform(self.element)
        )
        cairo_mat = cairo.Matrix(*model_to_screen_matrix.for_cairo())
        ctx.transform(cairo_mat)

        transformed_geo.to_cairo(ctx)
        ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
        ctx.fill()

        if self.cursor_visible:
            # Calculate view scale for consistent cursor size
            scale = 1.0
            if self.element.canvas:
                scale_x, _ = self.element.canvas.get_view_scale()
                scale = scale_x if scale_x > 1e-13 else 1.0
            cursor_width = 3.0 / scale

            # Calculate cursor geometry in Natural Space
            sub_geo = Geometry.from_text(
                self.text_buffer[: self.cursor_pos], **font_params
            )
            sub_geo.flip_y()
            _, _, sub_max_x, _ = sub_geo.rect()

            if self.cursor_pos == 0:
                sub_max_x = nat_min_x
            else:
                # Add a small margin to "unglue" the cursor from the last
                # character. Margin = half cursor width + 1 visual pixel gap
                sub_max_x += (cursor_width / 2) + (3.0 / scale)

            cursor_height = nat_max_y - nat_min_y
            if cursor_height <= 0:
                cursor_height = font_params["font_size"]

            c_center_y = (nat_min_y + nat_max_y) / 2

            c_half_w = cursor_width / 2
            c_half_h = cursor_height / 2

            # Cursor corners in natural space
            pts_nat = [
                (sub_max_x - c_half_w, c_center_y - c_half_h),
                (sub_max_x + c_half_w, c_center_y - c_half_h),
                (sub_max_x + c_half_w, c_center_y + c_half_h),
                (sub_max_x - c_half_w, c_center_y + c_half_h),
            ]

            # Prepare transformation to Model Space
            src_w = nat_max_x - nat_min_x
            src_h = nat_max_y - nat_min_y
            if abs(src_w) < 1e-9:
                src_w = 1.0
            if abs(src_h) < 1e-9:
                src_h = 1.0

            u = (p_width.x - p_origin.x, p_width.y - p_origin.y)
            v = (p_height.x - p_origin.x, p_height.y - p_origin.y)
            origin = (p_origin.x, p_origin.y)

            def trans(px, py):
                xn = (px - nat_min_x) / src_w
                yn = (py - nat_min_y) / src_h
                return (
                    origin[0] + xn * u[0] + yn * v[0],
                    origin[1] + xn * u[1] + yn * v[1],
                )

            pts_model = [trans(*p) for p in pts_nat]

            ctx.move_to(*pts_model[0])
            for p in pts_model[1:]:
                ctx.line_to(*p)
            ctx.close_path()
            ctx.set_source_rgba(0.0, 0.0, 0.0, 1.0)
            ctx.fill()

        ctx.restore()
