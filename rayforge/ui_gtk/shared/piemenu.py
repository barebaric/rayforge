import math
import cairo
import logging
from typing import List, Any
from gi.repository import Gtk, Gdk
from blinker import Signal
from ..icons import get_icon_pixbuf
from .gtk import apply_css

logger = logging.getLogger(__name__)

css = """
.pie-menu > contents {
    background-color: transparent;
    box-shadow: none;
    border: none;
}
"""


class PieMenuItem:
    def __init__(self, icon_name: str, label: str, data: Any = None):
        self.icon_name = icon_name
        self.label = label
        self.data = data
        self.visible = True
        # Signal emitted when item is activated. argument: sender (PieMenuItem)
        self.on_click = Signal()


class PieMenu(Gtk.Popover):
    """
    A radial menu implemented as a Gtk.Popover.
    It is transparent (custom CSS) and centers itself over the cursor.
    """

    def __init__(self, parent_widget: Gtk.Widget):
        super().__init__()
        self.set_parent(parent_widget)
        self.set_has_arrow(False)

        # Signal emitted when user right-clicks the menu, to request
        # repositioning.
        # arguments: sender(PieMenu), gesture, n_press, x, y
        self.right_clicked = Signal()

        # Disable autohide to prevent Gtk from aggressively closing the popover
        # on clicks it thinks are "outside".
        # We will manually handle closing in _on_release and _on_key_press.
        self.set_autohide(False)

        self.radius_outer = 75
        self.radius_inner = 30
        self.icon_size = 24

        # Margin to allow text to be drawn outside the pie without clipping
        self.text_margin = 120
        # Total radius including margins for calculating the widget size
        self.total_radius = self.radius_outer + self.text_margin

        self.add_css_class("pie-menu")
        apply_css(css)

        self.items: List[PieMenuItem] = []
        self._active_index: int = -1

        self.drawing_area = Gtk.DrawingArea()
        # Size needs to cover diameter + margins on both sides
        size = int(self.total_radius * 2)
        self.drawing_area.set_content_width(size)
        self.drawing_area.set_content_height(size)

        # Make drawing area focusable to help with event state accounting
        self.drawing_area.set_draw_func(self._draw_func)
        self.drawing_area.set_focusable(True)
        self.set_child(self.drawing_area)

        motion = Gtk.EventControllerMotion()
        motion.connect("motion", self._on_motion)
        motion.connect("leave", self._on_leave)
        self.drawing_area.add_controller(motion)

        # Click: Execute action
        click = Gtk.GestureClick()
        click.connect("pressed", self._on_press)
        click.connect("released", self._on_release)
        self.drawing_area.add_controller(click)

        # Key: Escape to close
        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key_press)
        self.add_controller(key)

        # Handle right-clicks on the menu itself to allow repositioning.
        right_click = Gtk.GestureClick()
        right_click.set_button(3)
        right_click.connect("pressed", self._on_right_press)
        self.drawing_area.add_controller(right_click)

    def add_item(self, item: PieMenuItem):
        self.items.append(item)
        self.drawing_area.queue_draw()

    def set_items(self, items: List[PieMenuItem]):
        self.items = items
        self._active_index = -1
        self.drawing_area.queue_draw()

    def popup_at_location(self, widget_x: float, widget_y: float):
        """
        Opens the menu centered at the specific widget coordinates.
        """
        rect = Gdk.Rectangle()
        rect.x = int(widget_x)
        rect.y = int(widget_y)
        rect.width = 0
        rect.height = 0

        self.set_pointing_to(rect)
        self.set_position(Gtk.PositionType.BOTTOM)

        # Offset must account for the larger drawing area size due to text
        # margins. We shift up/left by the center coordinate to align the
        # pie center with the target rect.
        self.set_offset(0, -int(self.total_radius))

        logger.debug(f"Popup at {widget_x}, {widget_y}")
        self.popup()
        self._active_index = -1
        self.drawing_area.grab_focus()

    def _get_index_at(self, x, y):
        """Calculates which slice index is under the coordinates."""
        items = [i for i in self.items if i.visible]
        dx = x - self.total_radius
        dy = y - self.total_radius
        dist = math.hypot(dx, dy)

        # Allow interaction only within the visible pie slices.
        if dist < self.radius_inner or dist > self.radius_outer or not items:
            return -1

        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        slice_angle = (2 * math.pi) / len(items)
        return int(angle / slice_angle) % len(items)

    def _on_motion(self, controller, x, y):
        new_index = self._get_index_at(x, y)
        if new_index != self._active_index:
            self._active_index = new_index
            self.drawing_area.queue_draw()

    def _on_leave(self, controller):
        self._active_index = -1
        self.drawing_area.queue_draw()

    def _on_press(self, gesture, n_press, x, y):
        """
        Handle press. CRITICAL: We must CLAIM the event sequence here.
        """
        logger.debug(f"Press at {x:.1f}, {y:.1f}")
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _on_release(self, gesture, n_press, x, y):
        """Handle click release to trigger action."""
        logger.debug(f"Release at {x:.1f}, {y:.1f}")
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

        triggered_index = self._get_index_at(x, y)
        items = [i for i in self.items if i.visible]

        self.popdown()

        if triggered_index >= 0 and triggered_index < len(items):
            item = items[triggered_index]
            logger.debug(f"Activating '{item.label}' with data '{item.data}'")
            item.on_click.send(item)
        else:
            logger.debug("Release on background/nothing")

    def _on_key_press(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_Escape:
            self.popdown()
            return True
        return False

    def _on_right_press(self, gesture, n_press, x, y):
        """Fires a signal to let the parent handle repositioning."""
        self.right_clicked.send(
            self, gesture=gesture, n_press=n_press, x=x, y=y
        )
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)

    def _draw_func(self, drawing_area, ctx, width, height):
        items = [i for i in self.items if i.visible]
        if not items:
            return

        # Fetch theme colors from the style context
        style = drawing_area.get_style_context()
        fg = style.get_color()

        # Base color components (0-1)
        r, g, b = fg.red, fg.green, fg.blue

        # Create palette based on theme foreground
        color_fg = (r, g, b, 1.0)
        # Slices use the FG color but with low opacity
        color_slice_normal = (r, g, b, 0.1)
        color_slice_active = (r, g, b, 0.3)
        color_border = (r, g, b, 0.2)

        # Use the actual center of the drawing area for robustness
        cx, cy = width / 2, height / 2
        count = len(items)
        step = (2 * math.pi) / count

        # 1. Draw Slices and Icons
        for i, item in enumerate(items):
            start_angle = i * step
            end_angle = (i + 1) * step
            mid_angle = start_angle + (step / 2)

            is_active = i == self._active_index

            # Slice Shape
            ctx.new_path()
            ctx.arc(cx, cy, self.radius_outer, start_angle, end_angle)
            ctx.arc_negative(cx, cy, self.radius_inner, end_angle, start_angle)
            ctx.close_path()

            if is_active:
                ctx.set_source_rgba(*color_slice_active)
            else:
                ctx.set_source_rgba(*color_slice_normal)

            ctx.fill_preserve()

            # Border
            ctx.set_source_rgba(*color_border)
            ctx.set_line_width(1)
            ctx.stroke()

            # Icon
            icon_dist = (self.radius_inner + self.radius_outer) / 2
            ix = cx + math.cos(mid_angle) * icon_dist
            iy = cy + math.sin(mid_angle) * icon_dist

            if item.icon_name:
                icon_pixbuf = get_icon_pixbuf(item.icon_name, self.icon_size)
                if icon_pixbuf:
                    icon_x = ix - (icon_pixbuf.get_width() / 2)
                    icon_y = iy - (icon_pixbuf.get_height() / 2)

                    ctx.save()
                    # 1. Place the icon in the source
                    Gdk.cairo_set_source_pixbuf(
                        ctx, icon_pixbuf, icon_x, icon_y
                    )
                    # 2. Paint it (creates the shape)
                    ctx.paint()
                    # 3. Use operator IN to keep only the intersection of the
                    #    next paint with the previously drawn icon shape
                    ctx.set_operator(cairo.OPERATOR_IN)
                    # 4. Set source to theme foreground and paint
                    ctx.set_source_rgba(*color_fg)
                    ctx.paint()
                    ctx.restore()

        # 2. Draw Active Label (External)
        if self._active_index >= 0 and self._active_index < len(items):
            active_item = items[self._active_index]

            # Calculate angle again for the active item
            start_angle = self._active_index * step
            mid_angle = start_angle + (step / 2)

            # Determine position outside the ring
            label_dist = self.radius_outer + 15
            lx = cx + math.cos(mid_angle) * label_dist
            ly = cy + math.sin(mid_angle) * label_dist

            ctx.save()
            ctx.set_source_rgba(*color_fg)
            ctx.select_font_face(
                "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
            )
            ctx.set_font_size(13)
            extents = ctx.text_extents(active_item.label)

            # Determine Alignment based on angle (cos)
            cos_a = math.cos(mid_angle)

            text_x = 0.0
            text_y = ly - (extents.height / 2) - extents.y_bearing

            if cos_a > 0.3:
                # Right side: Text starts at lx
                text_x = lx
            elif cos_a < -0.3:
                # Left side: Text ends at lx
                text_x = lx - extents.width - extents.x_bearing
            else:
                # Top/Bottom: Text centered on lx
                text_x = lx - (extents.width / 2) - extents.x_bearing

            ctx.move_to(text_x, text_y)
            ctx.show_text(active_item.label)
            ctx.restore()
