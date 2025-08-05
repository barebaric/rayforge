import math
from typing import Dict, Optional
import cairo
from gi.repository import Gdk, GLib  # type: ignore
from .region import ElementRegion

# A module-level cache for custom-rendered cursors to avoid recreating them.
_cursor_cache: Dict[int, Gdk.Cursor] = {}
_region_angles = {
    ElementRegion.MIDDLE_RIGHT: 0,
    ElementRegion.TOP_RIGHT: 45,
    ElementRegion.TOP_MIDDLE: 90,
    ElementRegion.TOP_LEFT: 135,
    ElementRegion.MIDDLE_LEFT: 180,
    ElementRegion.BOTTOM_LEFT: 225,
    ElementRegion.BOTTOM_MIDDLE: 270,
    ElementRegion.BOTTOM_RIGHT: 315,
}


def get_rotated_cursor(angle_deg: float) -> Gdk.Cursor:
    """
    Creates or retrieves from cache a custom two-headed arrow cursor
    rotated to the given angle.

    Args:
        angle_deg: The desired rotation of the cursor in degrees.

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
    ctx.rotate(-math.radians(angle_deg))

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
    ctx.set_source_rgb(1, 1, 1)  # White
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


def get_cursor_for_region(
    region: Optional[ElementRegion], additional_angle: float
) -> Gdk.Cursor:
    if region is None or region == ElementRegion.NONE:
        return Gdk.Cursor.new_from_name("default")
    elif region == ElementRegion.BODY:
        return Gdk.Cursor.new_from_name("move")
    elif region == ElementRegion.ROTATION_HANDLE:
        return Gdk.Cursor.new_from_name("crosshair")
    else:  # must be a resize region
        # Create a custom rotated cursor
        base_angle = _region_angles.get(region, 0)
        cursor_angle = base_angle - additional_angle
        return get_rotated_cursor(cursor_angle)
