import logging
import math
from typing import Any

import cairo
from blinker import Signal
from gi.repository import Gdk, Gtk

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

# Free arc length (px) an icon needs inside a slice, on top of the
# icon size itself.
ICON_CLEARANCE = 8.0
# Corridor slack (px of arc) around the parent wedge borders inside
# which an open submenu does not switch to a neighboring entry.
CORRIDOR_ARC = 8.0


class PieMenuItem:
    def __init__(
        self,
        icon_name: str,
        label: str,
        data: Any = None,
        children: list["PieMenuItem"] | None = None,
    ):
        self.icon_name = icon_name
        self.label = label
        self.data = data
        self.visible = True
        self.children: list[PieMenuItem] = children or []
        # Signal emitted when item is activated. argument: sender (PieMenuItem)
        self.on_click = Signal()

    @property
    def has_children(self) -> bool:
        return bool(self.children)


def get_visible_children(item: PieMenuItem) -> list[PieMenuItem]:
    return [child for child in item.children if child.visible]


def normalize_angle(angle: float) -> float:
    """Wraps an angle into [0, 2*pi)."""
    return angle % (2 * math.pi)


def angle_in_span(angle: float, start: float, end: float) -> bool:
    """
    True if the angle lies within [start, end], tolerating spans that
    wrap around the 2*pi boundary.
    """
    angle = normalize_angle(angle)
    start = normalize_angle(start)
    end = normalize_angle(end)
    if start <= end:
        return start <= angle <= end
    return angle >= start or angle <= end


def compute_pie_layout(
    items: list[PieMenuItem], max_inner_items: int
) -> tuple[bool, list[PieMenuItem]]:
    """
    Resolves the arrangement of the inner ring.

    Returns (collapsed, inner_items). While the total number of visible
    entries fits into max_inner_items, children of group items are
    hoisted into the inner ring (flat mode). Otherwise groups stay
    intact to be expanded into the outer ring (collapsed mode), except
    that groups without any visible child are dropped and groups with a
    single visible child are hoisted: an outer ring pays off only with
    at least two entries to cluster.
    """
    visible = [item for item in items if item.visible]
    total = sum(1 + len(get_visible_children(item)) for item in visible)
    if total <= max_inner_items:
        return False, _flatten_items(visible)

    inner = []
    for item in visible:
        if not item.has_children:
            inner.append(item)
            continue
        children = get_visible_children(item)
        if not children:
            continue
        if len(children) == 1:
            inner.append(children[0])
        else:
            inner.append(item)
    return True, inner


def _flatten_items(items: list[PieMenuItem]) -> list[PieMenuItem]:
    flat = []
    for item in items:
        if item.has_children:
            flat.extend(get_visible_children(item))
        else:
            flat.append(item)
    return flat


def submenu_span(
    mid_angle: float,
    half_wedge: float,
    child_count: int,
    mid_radius: float,
    min_arc: float,
) -> tuple[float, float, float]:
    """
    Start angle, end angle, and per-child step of a submenu ring
    centered on mid_angle. The span never covers less than the parent
    wedge and widens symmetrically when the children would otherwise
    not fit their icons.
    """
    if child_count <= 0:
        return mid_angle, mid_angle, 0.0
    min_step = min_arc / max(mid_radius, 1.0)
    span = max(2.0 * half_wedge, child_count * min_step)
    step = span / child_count
    start = mid_angle - span / 2.0
    return start, start + span, step


def index_in_span(angle: float, start: float, step: float, count: int) -> int:
    """
    Maps an angle to its slice index within [start, start + count *
    step]. Angles beyond the span clamp to the outermost slice; callers
    gate on angle_in_span() first.
    """
    if count <= 0 or step <= 0:
        return -1
    offset = normalize_angle(angle - start)
    return min(int(offset / step), count - 1)


class PieMenu(Gtk.Popover):
    """
    A radial menu implemented as a Gtk.Popover.
    It is transparent (custom CSS) and centers itself over the cursor.

    Items may declare children. While the visible entries fit into the
    inner ring, children are flattened into it. Otherwise group items
    collapse into a single entry that shows its children in a partial
    outer ring while hovered.
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
        self.label_gap = 15
        self.label_font_size = 13
        self.label_outline_width = 3.0
        self.background_opacity = 0.95

        # Dead band between the rings; crossing it keeps the open
        # submenu alive instead of dropping the hover state.
        self.sub_ring_gap = 5
        self.sub_ring_width = self.radius_outer - self.radius_inner

        # Margin to allow text to be drawn outside the pie without clipping
        self.text_margin = 120
        # Total radius including margins for calculating the widget size
        self.total_radius = self.radius_outer + self.text_margin

        self.add_css_class("pie-menu")
        apply_css(css)

        self.items: list[PieMenuItem] = []
        self._collapsed: bool = False
        self._inner_items: list[PieMenuItem] = []
        self._active_index: int = -1
        self._active_child_index: int = -1

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

    @property
    def sub_radius_inner(self) -> float:
        return self.radius_outer + self.sub_ring_gap

    @property
    def sub_radius_outer(self) -> float:
        return self.sub_radius_inner + self.sub_ring_width

    def add_item(self, item: PieMenuItem):
        self.items.append(item)
        self._refresh_layout()
        self.drawing_area.queue_draw()

    def set_items(self, items: list[PieMenuItem]):
        self.items = items
        self._refresh_layout()
        self.drawing_area.queue_draw()

    def _refresh_layout(self):
        self._collapsed, self._inner_items = compute_pie_layout(
            self.items, self._get_max_inner_items()
        )
        self._active_index = -1
        self._active_child_index = -1

    def _get_max_inner_items(self) -> int:
        """
        Number of entries the inner ring can hold before icons start
        crowding each other.
        """
        mid_radius = (self.radius_inner + self.radius_outer) / 2
        capacity = (2 * math.pi * mid_radius) / (
            self.icon_size + ICON_CLEARANCE
        )
        return max(4, int(capacity))

    def _apply_label_font(self, ctx):
        ctx.select_font_face(
            "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD
        )
        ctx.set_font_size(self.label_font_size)

    def _get_max_label_width(self) -> float:
        """Measures the widest item label using the label font."""
        items = list(self._inner_items)
        if self._collapsed:
            for item in self._inner_items:
                items.extend(get_visible_children(item))
        if not items:
            return 0.0
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
        ctx = cairo.Context(surface)
        self._apply_label_font(ctx)
        return max(ctx.text_extents(i.label).width for i in items)

    def _get_background_color(
        self, style: Gtk.StyleContext, fg: tuple
    ) -> tuple:
        """
        Returns the theme background color as an RGB tuple. Falls back
        to a contrasting value derived from the foreground if the theme
        does not provide one.
        """
        found, bg = style.lookup_color("theme_bg_color")
        if not found:
            found, bg = style.lookup_color("view_bg_color")
        if found:
            return (bg.red, bg.green, bg.blue)
        luminance = 0.299 * fg[0] + 0.587 * fg[1] + 0.114 * fg[2]
        return (1.0, 1.0, 1.0) if luminance < 0.5 else (0.0, 0.0, 0.0)

    def _get_colors(self, fg: Gdk.RGBA) -> dict:
        """Creates the palette based on the theme foreground color."""
        r, g, b = fg.red, fg.green, fg.blue
        return {
            "fg": (r, g, b, 1.0),
            # Outline is the inverted fill color, keeping the label
            # readable on any canvas background
            "outline": (1 - r, 1 - g, 1 - b, 1.0),
            # Slices use the FG color but with low opacity
            "slice_normal": (r, g, b, 0.1),
            "slice_active": (r, g, b, 0.3),
            "border": (r, g, b, 0.2),
        }

    def _paint_icon(self, ctx, pixbuf, x: float, y: float, fg: tuple):
        """
        Recolors the icon to the foreground color and composites it.

        The recoloring is done in an isolated group: the mask operation
        only affects the icon shape, not the rest of the surface.
        """
        ctx.save()
        ctx.push_group()
        Gdk.cairo_set_source_pixbuf(ctx, pixbuf, x, y)
        ctx.paint()
        ctx.set_operator(cairo.OPERATOR_IN)
        ctx.set_source_rgba(*fg)
        ctx.paint()
        ctx.pop_group_to_source()
        ctx.set_operator(cairo.OPERATOR_OVER)
        ctx.paint()
        ctx.restore()

    def _update_size(self):
        """
        Enlarges the drawing area so the label of the active item fits
        between the outermost ring and the widget edge without being
        clipped. When groups are collapsed, the submenu ring is
        reserved up front: resizing while a submenu opens would shift
        the popover, since it is positioned via total_radius.
        """
        ring_outer = (
            self.sub_radius_outer if self._collapsed else self.radius_outer
        )
        needed = ring_outer + self.label_gap + self._get_max_label_width()
        minimum = ring_outer + self.text_margin
        self.total_radius = max(minimum, needed)
        size = int(self.total_radius * 2)
        self.drawing_area.set_content_width(size)
        self.drawing_area.set_content_height(size)

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
        self._active_child_index = -1
        self.drawing_area.grab_focus()

    def _wedge_step(self) -> float:
        return (2 * math.pi) / max(len(self._inner_items), 1)

    def _wedge_mid_angle(self, index: int) -> float:
        return (index + 0.5) * self._wedge_step()

    def _active_item(self) -> PieMenuItem | None:
        if 0 <= self._active_index < len(self._inner_items):
            return self._inner_items[self._active_index]
        return None

    def _active_children(self) -> list[PieMenuItem]:
        item = self._active_item()
        if item is None:
            return []
        return get_visible_children(item)

    def _is_submenu_open(self) -> bool:
        return bool(self._active_children())

    def _submenu_geometry(self) -> tuple[float, float, float]:
        mid_radius = (self.sub_radius_inner + self.sub_radius_outer) / 2
        return submenu_span(
            self._wedge_mid_angle(self._active_index),
            self._wedge_step() / 2,
            len(self._active_children()),
            mid_radius,
            self.icon_size + ICON_CLEARANCE,
        )

    def _corridor_slack(self) -> float:
        mid_radius = (self.radius_inner + self.radius_outer) / 2
        return CORRIDOR_ARC / max(mid_radius, 1.0)

    def _angle_in_corridor(self, angle: float) -> bool:
        step = self._wedge_step()
        start = self._active_index * step - self._corridor_slack()
        end = (self._active_index + 1) * step + self._corridor_slack()
        return angle_in_span(angle, start, end)

    def _get_inner_index_at(self, x, y) -> int:
        """Calculates which inner slice index is under the coordinates."""
        dx = x - self.total_radius
        dy = y - self.total_radius
        dist = math.hypot(dx, dy)

        # Allow interaction only within the visible pie slices.
        if dist < self.radius_inner or dist > self.radius_outer:
            return -1
        if not self._inner_items:
            return -1

        angle = math.atan2(dy, dx)
        if angle < 0:
            angle += 2 * math.pi

        step = (2 * math.pi) / len(self._inner_items)
        return int(angle / step) % len(self._inner_items)

    def _get_child_index_at(self, angle: float, dist: float) -> int:
        """Calculates which submenu child is under the polar coords."""
        if not self._is_submenu_open():
            return -1
        if dist < self.sub_radius_inner or dist > self.sub_radius_outer:
            return -1
        children = self._active_children()
        start, end, step = self._submenu_geometry()
        if not angle_in_span(angle, start, end):
            return -1
        return index_in_span(angle, start, step, len(children))

    def _resolve_target(self, x: float, y: float) -> tuple[int, int]:
        """
        Resolves the (inner index, child index) pair under the pointer.

        While a submenu is open, the pointer may cross the dead band
        between the rings or drift slightly past the parent wedge
        borders (the corridor) without switching to another entry.
        """
        dx = x - self.total_radius
        dy = y - self.total_radius
        dist = math.hypot(dx, dy)
        if dist < self.radius_inner:
            return -1, -1
        angle = math.atan2(dy, dx)

        if self._is_submenu_open():
            child_index = self._get_child_index_at(angle, dist)
            if child_index >= 0:
                return self._active_index, child_index
            if dist <= self.sub_radius_outer:
                if self._angle_in_corridor(angle):
                    return self._active_index, -1
            else:
                return -1, -1

        return self._get_inner_index_at(x, y), -1

    def _set_hover(self, inner_index: int, child_index: int) -> bool:
        """
        Updates the hover state. Returns True if something changed.
        """
        if (
            inner_index == self._active_index
            and child_index == self._active_child_index
        ):
            return False
        self._active_index = inner_index
        self._active_child_index = child_index
        return True

    def _on_motion(self, controller, x, y):
        inner_index, child_index = self._resolve_target(x, y)
        if self._set_hover(inner_index, child_index):
            self.drawing_area.queue_draw()

    def _on_leave(self, controller):
        if self._set_hover(-1, -1):
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

        inner_index, child_index = self._resolve_target(x, y)
        self._activate_target(inner_index, child_index)

    def _activate_target(self, inner_index: int, child_index: int):
        if inner_index < 0:
            logger.debug("Release on background/nothing")
            self.popdown()
            return

        item = self._inner_items[inner_index]
        children = get_visible_children(item)

        if 0 <= child_index < len(children):
            child = children[child_index]
            logger.debug(
                f"Activating '{child.label}' with data '{child.data}'"
            )
            self.popdown()
            child.on_click.send(child)
            return

        if children:
            # Group entries have no action of their own; releasing on
            # one reveals its submenu and keeps the menu open.
            logger.debug(f"Release on group '{item.label}'")
            if self._set_hover(inner_index, -1):
                self.drawing_area.queue_draw()
            return

        logger.debug(f"Activating '{item.label}' with data '{item.data}'")
        self.popdown()
        item.on_click.send(item)

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
        items = self._inner_items
        if not items:
            return

        # Fetch theme colors from the style context
        style = drawing_area.get_style_context()
        colors = self._get_colors(style.get_color())

        # Use the actual center of the drawing area for robustness
        cx, cy = width / 2, height / 2
        step = (2 * math.pi) / len(items)

        # 1. Draw Background Ring
        bg = self._get_background_color(style, colors["fg"])
        ctx.new_path()
        ctx.arc(cx, cy, self.radius_outer, 0, 2 * math.pi)
        ctx.arc_negative(cx, cy, self.radius_inner, 2 * math.pi, 0)
        ctx.close_path()
        ctx.set_source_rgba(*bg, self.background_opacity)
        ctx.fill()

        # 2. Draw Slices and Icons
        for i, item in enumerate(items):
            self._draw_slice(ctx, cx, cy, i, step, item, colors)
            if self._collapsed and get_visible_children(item):
                self._draw_chevron(ctx, cx, cy, i, colors["fg"])

        # 3. Draw the open submenu ring
        if self._is_submenu_open():
            self._draw_submenu(ctx, cx, cy, style, colors)

        # 4. Draw Active Label (External)
        self._draw_active_label(ctx, cx, cy, items, step, colors)

    def _draw_slice(self, ctx, cx, cy, index, step, item, colors):
        start_angle = index * step
        end_angle = (index + 1) * step
        mid_angle = start_angle + (step / 2)

        is_active = index == self._active_index

        # Slice Shape
        ctx.new_path()
        ctx.arc(cx, cy, self.radius_outer, start_angle, end_angle)
        ctx.arc_negative(cx, cy, self.radius_inner, end_angle, start_angle)
        ctx.close_path()

        if is_active:
            ctx.set_source_rgba(*colors["slice_active"])
        else:
            ctx.set_source_rgba(*colors["slice_normal"])

        ctx.fill_preserve()

        # Border
        ctx.set_source_rgba(*colors["border"])
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
                self._paint_icon(
                    ctx, icon_pixbuf, icon_x, icon_y, colors["fg"]
                )

    def _draw_chevron(self, ctx, cx, cy, index, color):
        """
        Marks a group entry with a small outward-pointing triangle at
        the outer rim of its wedge.
        """
        mid_angle = (index + 0.5) * self._wedge_step()
        tip_dist = self.radius_outer - 3.0
        base_dist = tip_dist - 6.0
        spread = 5.0 / base_dist

        ctx.new_path()
        ctx.move_to(
            cx + math.cos(mid_angle) * tip_dist,
            cy + math.sin(mid_angle) * tip_dist,
        )
        ctx.line_to(
            cx + math.cos(mid_angle - spread) * base_dist,
            cy + math.sin(mid_angle - spread) * base_dist,
        )
        ctx.line_to(
            cx + math.cos(mid_angle + spread) * base_dist,
            cy + math.sin(mid_angle + spread) * base_dist,
        )
        ctx.close_path()
        ctx.set_source_rgba(*color)
        ctx.fill()

    def _draw_submenu(self, ctx, cx, cy, style, colors):
        children = self._active_children()
        if not children:
            return

        start, end, step = self._submenu_geometry()

        bg = self._get_background_color(style, colors["fg"])
        ctx.new_path()
        ctx.arc(cx, cy, self.sub_radius_outer, start, end)
        ctx.arc_negative(cx, cy, self.sub_radius_inner, end, start)
        ctx.close_path()
        ctx.set_source_rgba(*bg, self.background_opacity)
        ctx.fill()

        for i, child in enumerate(children):
            self._draw_sub_slice(ctx, cx, cy, start, step, i, child, colors)

    def _draw_sub_slice(self, ctx, cx, cy, start, step, index, child, colors):
        slice_start = start + index * step
        slice_end = slice_start + step

        ctx.new_path()
        ctx.arc(cx, cy, self.sub_radius_outer, slice_start, slice_end)
        ctx.arc_negative(cx, cy, self.sub_radius_inner, slice_end, slice_start)
        ctx.close_path()

        if index == self._active_child_index:
            ctx.set_source_rgba(*colors["slice_active"])
        else:
            ctx.set_source_rgba(*colors["slice_normal"])

        ctx.fill_preserve()

        ctx.set_source_rgba(*colors["border"])
        ctx.set_line_width(1)
        ctx.stroke()

        icon_dist = (self.sub_radius_inner + self.sub_radius_outer) / 2
        mid_angle = slice_start + (step / 2)
        ix = cx + math.cos(mid_angle) * icon_dist
        iy = cy + math.sin(mid_angle) * icon_dist

        if child.icon_name:
            icon_pixbuf = get_icon_pixbuf(child.icon_name, self.icon_size)
            if icon_pixbuf:
                icon_x = ix - (icon_pixbuf.get_width() / 2)
                icon_y = iy - (icon_pixbuf.get_height() / 2)
                self._paint_icon(
                    ctx, icon_pixbuf, icon_x, icon_y, colors["fg"]
                )

    def _draw_active_label(self, ctx, cx, cy, items, step, colors):
        if self._is_submenu_open():
            children = self._active_children()
            if 0 <= self._active_child_index < len(children):
                child = children[self._active_child_index]
                start, _end, child_step = self._submenu_geometry()
                angle = start + (self._active_child_index + 0.5) * child_step
                self._draw_label(
                    ctx,
                    cx,
                    cy,
                    angle,
                    child.label,
                    self.sub_radius_outer,
                    colors,
                )
            return

        if 0 <= self._active_index < len(items):
            item = items[self._active_index]
            mid_angle = (self._active_index + 0.5) * step
            self._draw_label(
                ctx,
                cx,
                cy,
                mid_angle,
                item.label,
                self.radius_outer,
                colors,
            )

    def _draw_label(self, ctx, cx, cy, angle, text, ring_radius, colors):
        """Draws the text outside ring_radius, aligned along the angle."""
        label_dist = ring_radius + self.label_gap
        lx = cx + math.cos(angle) * label_dist
        ly = cy + math.sin(angle) * label_dist

        ctx.save()
        self._apply_label_font(ctx)
        extents = ctx.text_extents(text)

        # Determine Alignment based on angle (cos)
        cos_a = math.cos(angle)

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
        ctx.text_path(text)
        ctx.set_source_rgba(*colors["outline"])
        ctx.set_line_width(self.label_outline_width)
        ctx.set_line_join(cairo.LINE_JOIN_ROUND)
        ctx.stroke_preserve()
        ctx.set_source_rgba(*colors["fg"])
        ctx.fill()
        ctx.restore()
