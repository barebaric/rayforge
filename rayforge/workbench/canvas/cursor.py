import math
from typing import Dict
import cairo
from gi.repository import Gdk, GLib  # type: ignore
from .region import ElementRegion, ROTATE_HANDLES

# Module-level caches for custom-rendered cursors to avoid recreating them.
_cursor_cache: Dict[int, Gdk.Cursor] = {}
_arc_cursor_cache: Dict[int, Gdk.Cursor] = {}

# This map defines the base angle for each resize handle, using a standard
# counter-clockwise (CCW) convention where 0 degrees is to the right.
_region_angles = {
    ElementRegion.MIDDLE_RIGHT: 0,
    ElementRegion.TOP_RIGHT: 45,
    ElementRegion.TOP_MIDDLE: 90,
    ElementRegion.TOP_LEFT: 135,
    ElementRegion.MIDDLE_LEFT: 180,
    ElementRegion.BOTTOM_LEFT: 225,
    ElementRegion.BOTTOM_MIDDLE: 270,
    ElementRegion.BOTTOM_RIGHT: 315,
    # Rotation handles are arcs with arrows.
    ElementRegion.ROTATE_TOP_RIGHT: 315,
    ElementRegion.ROTATE_TOP_LEFT: 45,
    ElementRegion.ROTATE_BOTTOM_LEFT: 135,
    ElementRegion.ROTATE_BOTTOM_RIGHT: 225,
    # Shear handles are bidirectional arrows.
    ElementRegion.SHEAR_TOP: 0,
    ElementRegion.SHEAR_BOTTOM: 0,
    ElementRegion.SHEAR_LEFT: 90,
    ElementRegion.SHEAR_RIGHT: 90,
}


def get_rotated_cursor(angle_deg: float) -> Gdk.Cursor:
    """
    Creates or retrieves from cache a custom two-headed arrow cursor
    rotated to the given angle.

    Args:
        angle_deg: The desired mathematical rotation (CCW) of the cursor.

    Returns:
        A Gdk.Cursor object.
    """
    # Round angle to nearest degree for effective caching
    angle_key = round(angle_deg)
    if angle_key in _cursor_cache:
        return _cursor_cache[angle_key]

    size = 32
    hotspot = size // 2

    # 1. Draw the cursor shape using Cairo
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    ctx.translate(hotspot, hotspot)
    ctx.rotate(math.radians(angle_deg))

    # Draw a white arrow with a black outline for visibility
    ctx.set_line_width(2)
    ctx.set_source_rgb(0, 0, 0)  # Black outline

    # Main line
    ctx.move_to(-10, 0)
    ctx.line_to(10, 0)

    # Arrowhead 1
    ctx.move_to(10, 0)
    ctx.line_to(6, -4)
    ctx.move_to(10, 0)
    ctx.line_to(6, 4)

    # Arrowhead 2
    ctx.move_to(-10, 0)
    ctx.line_to(-6, -4)
    ctx.move_to(-10, 0)
    ctx.line_to(-6, 4)
    ctx.stroke_preserve()  # Keep path for white fill

    # White inner fill
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(1)
    ctx.stroke()

    # 2. Convert Cairo surface to Gdk.Texture (GTK4 method)
    data = surface.get_data()
    bytes_data = GLib.Bytes.new(data)
    texture = Gdk.MemoryTexture.new(
        size,
        size,
        Gdk.MemoryFormat.B8G8R8A8_PREMULTIPLIED,
        bytes_data,
        surface.get_stride(),
    )

    # 3. Create Gdk.Cursor from the texture and cache it
    cursor = Gdk.Cursor.new_from_texture(texture, hotspot, hotspot)
    _cursor_cache[angle_key] = cursor
    return cursor


def get_rotated_arc_cursor(angle_deg: float) -> Gdk.Cursor:
    """
    Creates or retrieves from cache a custom rotation cursor (arc with arrows)
    rotated to the given angle.

    Args:
        angle_deg: The desired mathematical rotation (CCW) of the cursor.

    Returns:
        A Gdk.Cursor object.
    """
    angle_key = round(angle_deg)
    if angle_key in _arc_cursor_cache:
        return _arc_cursor_cache[angle_key]

    size = 33
    hotspot = size // 2

    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, size, size)
    ctx = cairo.Context(surface)
    ctx.translate(hotspot, hotspot)
    # Negate the mathematical angle to get the correct visual rotation (CW)
    ctx.rotate(math.radians(angle_deg))

    ctx.set_line_width(2)
    ctx.set_source_rgb(0, 0, 0)

    radius = 14
    start_angle = math.radians(225)
    end_angle = math.radians(315)

    # Draw the main arc path
    ctx.arc(0, 0, radius, start_angle, end_angle)

    def draw_arrowhead(point_angle: float, is_start_arrow: bool = False):
        """Draws a symmetric arrowhead at a given angle on the circle."""
        arrow_length = 5
        arrow_width = 5

        ctx.save()

        px = radius * math.cos(point_angle)
        py = radius * math.sin(point_angle)
        ctx.translate(px, py)

        # Rotate to match the arc's tangent (clockwise direction)
        tangent_angle = point_angle + math.pi / 2.0
        ctx.rotate(tangent_angle)

        if is_start_arrow:
            ctx.rotate(math.pi)

        # Draw a standard V-shape arrowhead pointing away from the tip.
        ctx.move_to(0, 0)
        ctx.line_to(-arrow_length, -arrow_width)
        ctx.move_to(0, 0)
        ctx.line_to(-arrow_length, arrow_width)

        ctx.restore()

    # Draw arrowheads at the start and end of the arc
    draw_arrowhead(start_angle, is_start_arrow=True)
    draw_arrowhead(end_angle, is_start_arrow=False)

    ctx.stroke_preserve()

    # White inner fill
    ctx.set_source_rgb(1, 1, 1)
    ctx.set_line_width(1)
    ctx.stroke()

    # Convert Cairo surface to Gdk.Texture
    data = surface.get_data()
    bytes_data = GLib.Bytes.new(data)
    texture = Gdk.MemoryTexture.new(
        size,
        size,
        Gdk.MemoryFormat.B8G8R8A8_PREMULTIPLIED,
        bytes_data,
        surface.get_stride(),
    )

    cursor = Gdk.Cursor.new_from_texture(texture, hotspot, hotspot)
    _arc_cursor_cache[angle_key] = cursor
    return cursor


def get_cursor_for_region(
    region: ElementRegion,
    angle: float,
    absolute: bool = False
) -> Gdk.Cursor:
    base_angle = _region_angles.get(region, 0) if not absolute else 0
    if region is None or region == ElementRegion.NONE:
        return Gdk.Cursor.new_from_name("default")
    elif region == ElementRegion.BODY:
        return Gdk.Cursor.new_from_name("move")
    elif region in ROTATE_HANDLES:
        cursor_angle = -base_angle + angle
        return get_rotated_arc_cursor(cursor_angle)
    else:  # must be a resize or shear region
        # The final visual angle of the cursor is the handle's base angle
        # plus the element's total world rotation angle.
        cursor_angle = -base_angle + angle
        return get_rotated_cursor(cursor_angle)
