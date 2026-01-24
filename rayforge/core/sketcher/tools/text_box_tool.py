import logging
from typing import Optional, cast
from enum import Enum, auto
import cairo
from blinker import Signal
from ...geo import Geometry, primitives
from ...geo.text import get_text_width
from ..commands import TextBoxCommand
from ..commands.live_text_edit import LiveTextEditCommand
from ..entities import Line, Point, TextBoxEntity
from .base import SketchTool, SketcherKey

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
        self.live_edit_cmd: Optional[LiveTextEditCommand] = None

        # Text selection state
        self.selection_start: Optional[int] = None
        self.selection_end: Optional[int] = None
        self.is_drag_selecting = False
        self.drag_start_pos: int = 0
        self._drag_start_world_x: float = 0.0
        self._drag_start_world_y: float = 0.0

        # Signals for the UI layer to manage timers, etc.
        self.editing_started = Signal()
        self.editing_finished = Signal()
        self.cursor_moved = Signal()

    def _get_selection_range(self) -> tuple[int, int]:
        """Returns normalized selection range (start, end)."""
        if self.selection_start is None or self.selection_end is None:
            return 0, 0
        return (
            min(self.selection_start, self.selection_end),
            max(self.selection_start, self.selection_end),
        )

    def get_selected_text(self) -> str:
        """Returns the currently selected text."""
        start, end = self._get_selection_range()
        if start == end:
            return ""
        return self.text_buffer[start:end]

    def clear_selection(self):
        """Clears the current selection."""
        self.selection_start = None
        self.selection_end = None

    def set_selection(self, start: int, end: int):
        """Sets the selection range."""
        self.selection_start = start
        self.selection_end = end

    def extend_selection(self, new_pos: int):
        """Extends the selection to a new position."""
        if self.selection_start is None:
            self.selection_start = self.cursor_pos
        self.selection_end = new_pos

    def start_editing(self, entity_id: int):
        """Public method to begin editing an existing text box."""
        from ..entities import TextBoxEntity

        entity = self.element.sketch.registry.get_entity(entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        self.editing_entity_id = entity_id
        self.text_buffer = entity.content
        self.cursor_pos = len(self.text_buffer)  # Cursor at the end
        self.clear_selection()
        self.state = TextBoxState.EDITING
        self.cursor_visible = True
        self.element.mark_dirty()
        self.editing_started.send(self)

        self.live_edit_cmd = LiveTextEditCommand(
            self.element.sketch, entity_id
        )
        self.live_edit_cmd.capture_state(self.text_buffer, self.cursor_pos)

    def on_deactivate(self):
        if self.state == TextBoxState.EDITING:
            self.editing_finished.send(self)

            if self.live_edit_cmd:
                self.element.execute_command(self.live_edit_cmd)
                self.live_edit_cmd = None

            self._finalize_edit()

        self.state = TextBoxState.IDLE
        self.editing_entity_id = None
        self.text_buffer = ""
        self.cursor_pos = 0
        self.clear_selection()
        self.is_drag_selecting = False
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
            return self._handle_idle_press(mx, my, world_x, world_y)
        elif self.state == TextBoxState.EDITING:
            return self._handle_editing_press(mx, my, world_x, world_y)
        return False

    def _handle_idle_press(
        self, mx: float, my: float, world_x: float, world_y: float
    ) -> bool:
        clicked_entity_id = self._find_text_box_at_point(mx, my)
        if clicked_entity_id is not None:
            self.start_editing(clicked_entity_id)
            # Initialize drag selection state to allow immediate selection
            # without requiring a second click.
            self._update_cursor_from_click(mx, my)
            self.is_drag_selecting = True
            self.drag_start_pos = self.cursor_pos
            self.selection_start = self.cursor_pos
            self.selection_end = self.cursor_pos
            self._drag_start_world_x = world_x
            self._drag_start_world_y = world_y
            return False  # Don't claim gesture, allow drag events

        cmd = TextBoxCommand(self.element.sketch, origin=(mx, my))
        self.element.execute_command(cmd)

        if cmd.text_box_id is not None:
            self.start_editing(cmd.text_box_id)
        return True

    def _handle_editing_press(
        self, mx: float, my: float, world_x: float, world_y: float
    ) -> bool:
        if self._is_point_inside_box(mx, my):
            self._update_cursor_from_click(mx, my)
            self.is_drag_selecting = True
            self.drag_start_pos = self.cursor_pos
            self.selection_start = self.cursor_pos
            self.selection_end = self.cursor_pos
            self._drag_start_world_x = world_x
            self._drag_start_world_y = world_y
            logger.debug(
                f"_handle_editing_press: is_drag_selecting=True, "
                f"cursor_pos={self.cursor_pos}, "
                f"start=({world_x}, {world_y})"
            )
            return False  # Don't claim gesture, allow drag events

        clicked_entity_id = self._find_text_box_at_point(mx, my)
        if clicked_entity_id is not None:
            self.on_deactivate()
            self.start_editing(clicked_entity_id)
            self._update_cursor_from_click(mx, my)
            # Initialize drag state and return False to allow drag to start
            self.is_drag_selecting = True
            self.drag_start_pos = self.cursor_pos
            self.selection_start = self.cursor_pos
            self.selection_end = self.cursor_pos
            self._drag_start_world_x = world_x
            self._drag_start_world_y = world_y
            return False

        self.on_deactivate()
        return self._handle_idle_press(mx, my, world_x, world_y)

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
        if p4_id:
            p4 = self.element.sketch.registry.get_point(p4_id)
            p4_x, p4_y = p4.x, p4.y
        else:
            # Calculate fourth corner from origin, width, and height points
            p4_x = p_width.x + p_height.x - p_origin.x
            p4_y = p_width.y + p_height.y - p_origin.y

        # Define the polygon for the text box
        polygon = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

        # Use point-in-polygon check for accurate hit testing (handles
        # rotation)
        return primitives.is_point_in_polygon((mx, my), polygon)

    def _is_point_inside_any_text_box(self, mx: float, my: float) -> bool:
        for entity in self.element.sketch.registry.entities:
            if isinstance(entity, TextBoxEntity):
                if self._is_point_inside_entity_box(entity, mx, my):
                    return True
        return False

    def _find_text_box_at_point(self, mx: float, my: float) -> Optional[int]:
        for entity in reversed(self.element.sketch.registry.entities):
            if isinstance(entity, TextBoxEntity):
                if self._is_point_inside_entity_box(entity, mx, my):
                    return entity.id
        return None

    def _is_point_inside_entity_box(
        self, entity: TextBoxEntity, mx: float, my: float
    ) -> bool:
        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        p4_id = entity.get_fourth_corner_id(self.element.sketch.registry)
        if p4_id:
            p4 = self.element.sketch.registry.get_point(p4_id)
            p4_x, p4_y = p4.x, p4.y
        else:
            p4_x = p_width.x + p_height.x - p_origin.x
            p4_y = p_width.y + p_height.y - p_origin.y

        polygon = [
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p4_x, p4_y),
            (p_height.x, p_height.y),
        ]

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
        if self.state == TextBoxState.EDITING and self.is_drag_selecting:
            logger.debug(
                f"on_drag: state=EDITING, is_drag_selecting=True, "
                f"dx={world_dx}, dy={world_dy}"
            )
            current_world_x = self._drag_start_world_x + world_dx
            current_world_y = self._drag_start_world_y + world_dy
            mx, my = self.element.hittester.screen_to_model(
                current_world_x, current_world_y, self.element
            )
            self._update_cursor_from_click(mx, my)
            self.selection_end = self.cursor_pos
            logger.debug(
                f"on_drag: cursor_pos={self.cursor_pos}, "
                f"selection_start={self.selection_start}, "
                f"selection_end={self.selection_end}"
            )
            self.element.mark_dirty()
        else:
            logger.debug(
                f"on_drag: state={self.state}, "
                f"is_drag_selecting={self.is_drag_selecting}"
            )

    def on_release(self, world_x: float, world_y: float):
        if self.is_drag_selecting:
            self.is_drag_selecting = False
            start, end = self._get_selection_range()
            logger.debug(
                f"on_release: start={start}, end={end}, "
                f"is_drag_selecting=False"
            )
            if start == end:
                self.clear_selection()
            self.element.mark_dirty()

    def on_hover_motion(self, world_x: float, world_y: float):
        mx, my = self.element.hittester.screen_to_model(
            world_x, world_y, self.element
        )
        self.is_hovering = self._is_point_inside_any_text_box(mx, my)

    def handle_text_input(self, text: str) -> bool:
        if self.state != TextBoxState.EDITING:
            return False

        self.cursor_visible = True

        start, end = self._get_selection_range()
        if start != end:
            self.text_buffer = (
                self.text_buffer[:start] + text + self.text_buffer[end:]
            )
            self.cursor_pos = start + 1
            self.clear_selection()
        else:
            self.text_buffer = (
                self.text_buffer[: self.cursor_pos]
                + text
                + self.text_buffer[self.cursor_pos :]
            )
            self.cursor_pos += 1

        self._resize_box_to_fit_text()
        self.cursor_moved.send(self)

        if self.live_edit_cmd:
            self.live_edit_cmd.capture_state(self.text_buffer, self.cursor_pos)

        return True

    def handle_key_event(
        self, key: SketcherKey, shift: bool = False, ctrl: bool = False
    ) -> bool:
        if self.state != TextBoxState.EDITING:
            return False

        self.cursor_visible = True  # Make cursor visible on keypress

        if key == SketcherKey.UNDO:
            if self.live_edit_cmd:
                self.live_edit_cmd.undo()
                self.text_buffer = self.live_edit_cmd.get_current_content()
                self.cursor_pos = self.live_edit_cmd.get_current_cursor_pos()
                self._resize_box_to_fit_text()
                self.element.mark_dirty()
                self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.REDO:
            if self.live_edit_cmd:
                self.live_edit_cmd.redo()
                self.text_buffer = self.live_edit_cmd.get_current_content()
                self.cursor_pos = self.live_edit_cmd.get_current_cursor_pos()
                self._resize_box_to_fit_text()
                self.element.mark_dirty()
                self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.BACKSPACE:
            start, end = self._get_selection_range()
            if start != end:
                self.text_buffer = (
                    self.text_buffer[:start] + self.text_buffer[end:]
                )
                self.cursor_pos = start
                self.clear_selection()
            elif self.cursor_pos > 0:
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos - 1]
                    + self.text_buffer[self.cursor_pos :]
                )
                self.cursor_pos -= 1

            self._resize_box_to_fit_text()
            self.cursor_moved.send(self)

            if self.live_edit_cmd:
                self.live_edit_cmd.capture_state(
                    self.text_buffer, self.cursor_pos
                )

            return True
        elif key == SketcherKey.DELETE:
            start, end = self._get_selection_range()
            if start != end:
                self.text_buffer = (
                    self.text_buffer[:start] + self.text_buffer[end:]
                )
                self.cursor_pos = start
                self.clear_selection()
            elif self.cursor_pos < len(self.text_buffer):
                self.text_buffer = (
                    self.text_buffer[: self.cursor_pos]
                    + self.text_buffer[self.cursor_pos + 1 :]
                )

            self._resize_box_to_fit_text()
            self.cursor_moved.send(self)

            if self.live_edit_cmd:
                self.live_edit_cmd.capture_state(
                    self.text_buffer, self.cursor_pos
                )

            return True
        elif key == SketcherKey.ARROW_LEFT:
            new_pos = max(0, self.cursor_pos - 1)
            if shift:
                self.extend_selection(new_pos)
            else:
                self.clear_selection()
            self.cursor_pos = new_pos
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.ARROW_RIGHT:
            new_pos = min(len(self.text_buffer), self.cursor_pos + 1)
            if shift:
                self.extend_selection(new_pos)
            else:
                self.clear_selection()
            self.cursor_pos = new_pos
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.HOME:
            new_pos = 0
            if shift:
                self.extend_selection(new_pos)
            else:
                self.clear_selection()
            self.cursor_pos = new_pos
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.END:
            new_pos = len(self.text_buffer)
            if shift:
                self.extend_selection(new_pos)
            else:
                self.clear_selection()
            self.cursor_pos = new_pos
            self.element.mark_dirty()
            self.cursor_moved.send(self)
            return True
        elif key == SketcherKey.SELECT_ALL:
            self.set_selection(0, len(self.text_buffer))
            self.cursor_pos = len(self.text_buffer)
            self.element.mark_dirty()
            return True
        elif key == SketcherKey.COPY and ctrl:
            from gi.repository import Gdk

            display = Gdk.Display.get_default()
            if display is None:
                return True
            clipboard = display.get_clipboard()
            clipboard.set(self.get_selected_text())
            return True
        elif key == SketcherKey.CUT and ctrl:
            from gi.repository import Gdk

            display = Gdk.Display.get_default()
            if display is None:
                return True
            clipboard = display.get_clipboard()
            clipboard.set(self.get_selected_text())

            start, end = self._get_selection_range()
            if start != end:
                self.text_buffer = (
                    self.text_buffer[:start] + self.text_buffer[end:]
                )
                self.cursor_pos = start
                self.clear_selection()
                self._resize_box_to_fit_text()
                self.element.mark_dirty()
                self.cursor_moved.send(self)

                if self.live_edit_cmd:
                    self.live_edit_cmd.capture_state(
                        self.text_buffer, self.cursor_pos
                    )
            return True
        elif key == SketcherKey.PASTE and ctrl:
            from gi.repository import Gdk

            display = Gdk.Display.get_default()
            if display is None:
                return True
            clipboard = display.get_clipboard()

            def on_paste_ready(clipboard, result):
                try:
                    text = clipboard.read_text_finish(result)
                    if text:
                        start, end = self._get_selection_range()
                        if start != end:
                            self.text_buffer = (
                                self.text_buffer[:start]
                                + text
                                + self.text_buffer[end:]
                            )
                            self.cursor_pos = start + len(text)
                            self.clear_selection()
                        else:
                            self.text_buffer = (
                                self.text_buffer[: self.cursor_pos]
                                + text
                                + self.text_buffer[self.cursor_pos :]
                            )
                            self.cursor_pos += len(text)

                        self._resize_box_to_fit_text()
                        self.element.mark_dirty()
                        self.cursor_moved.send(self)

                        if self.live_edit_cmd:
                            self.live_edit_cmd.capture_state(
                                self.text_buffer, self.cursor_pos
                            )
                except Exception:
                    pass

            clipboard.read_text_async(None, on_paste_ready)
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
        if self.editing_entity_id is None:
            return

        entity = self.element.sketch.registry.get_entity(
            self.editing_entity_id
        )
        if not isinstance(entity, TextBoxEntity):
            return

        ascent, descent, font_height = entity.get_font_metrics()

        if not self.text_buffer:
            natural_width = 10.0
        else:
            natural_geo = Geometry.from_text(
                self.text_buffer, **entity.font_params
            )
            natural_geo.flip_y()
            min_x, _, max_x, _ = natural_geo.rect()
            natural_width = max(max_x - min_x, 1.0)

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        # Update point positions based on origin and natural size
        # Origin is at bottom-left corner of box
        # Text baseline is offset by descent within the box
        # Box extends from origin.y to origin.y + font_height
        p_width.x = p_origin.x + natural_width
        p_width.y = p_origin.y
        p_height.x = p_origin.x
        p_height.y = p_origin.y + font_height

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

        # 2. Get bounds of full text to determine coordinate space range
        natural_geo = Geometry.from_text(
            self.text_buffer, **entity.font_params
        )
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
                # Measure width of substring including spaces
                sub_max_x = get_text_width(
                    self.text_buffer[:i], **entity.font_params
                )

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

        natural_geo = Geometry.from_text(
            self.text_buffer, **entity.font_params
        )
        natural_geo.flip_y()
        logger.debug(f"Natural geometry: {natural_geo.rect()}")

        nat_min_x, nat_min_y, nat_max_x, nat_max_y = natural_geo.rect()

        # Handle empty text case for frame mapping logic
        if not self.text_buffer:
            nat_min_x, nat_min_y = 0.0, 0.0
            nat_max_x = 10.0
            nat_max_y = entity.font_params.get("font_size", 10.0)

        _, descent, font_height = entity.get_font_metrics()

        transformed_geo = natural_geo.map_to_frame(
            (p_origin.x, p_origin.y),
            (p_width.x, p_width.y),
            (p_height.x, p_height.y),
            anchor_y=-descent,
            stable_src_height=font_height,
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

        start, end = self._get_selection_range()
        if start != end:
            self._draw_selection_highlight(
                ctx, nat_min_x, nat_min_y, nat_max_x, nat_max_y
            )

        if self.cursor_visible:
            # Calculate view scale for consistent cursor size
            scale = 1.0
            if self.element.canvas:
                scale_x, _ = self.element.canvas.get_view_scale()
                scale = scale_x if scale_x > 1e-13 else 1.0
            cursor_width = 3.0 / scale

            # Calculate cursor position using text width (includes spaces)
            if self.cursor_pos == 0:
                sub_max_x = nat_min_x
            else:
                sub_max_x = get_text_width(
                    self.text_buffer[: self.cursor_pos], **entity.font_params
                )

            cursor_height = nat_max_y - nat_min_y
            if cursor_height <= 0:
                cursor_height = entity.font_params.get("font_size", 10.0)

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

    def _draw_selection_highlight(
        self,
        ctx: cairo.Context,
        nat_min_x: float,
        nat_min_y: float,
        nat_max_x: float,
        nat_max_y: float,
    ):
        """Draws the selection highlight for selected text."""
        if self.editing_entity_id is None:
            return

        entity = self.element.sketch.registry.get_entity(
            self.editing_entity_id
        )
        if not entity:
            return

        p_origin = self.element.sketch.registry.get_point(entity.origin_id)
        p_width = self.element.sketch.registry.get_point(entity.width_id)
        p_height = self.element.sketch.registry.get_point(entity.height_id)

        _, descent, font_height = entity.get_font_metrics()

        start, end = self._get_selection_range()
        if start == end:
            return

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

        def get_char_x(pos: int) -> float:
            """Get the x-coordinate of a cursor position."""
            if pos == 0:
                return nat_min_x
            return get_text_width(self.text_buffer[:pos], **entity.font_params)

        start_x = get_char_x(start)
        end_x = get_char_x(end)

        # Draw selection highlight as a rectangle
        sel_nat_pts = [
            (start_x, nat_min_y),
            (end_x, nat_min_y),
            (end_x, nat_max_y),
            (start_x, nat_max_y),
        ]

        sel_model_pts = [trans(*p) for p in sel_nat_pts]

        ctx.save()
        ctx.move_to(*sel_model_pts[0])
        for p in sel_model_pts[1:]:
            ctx.line_to(*p)
        ctx.close_path()
        ctx.set_source_rgba(0.2, 0.6, 1.0, 0.3)
        ctx.fill()
        ctx.restore()
