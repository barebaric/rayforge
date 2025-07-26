import math
from typing import Dict
import cairo
from gi.repository import Gdk, GLib  # type: ignore


# A module-level cache for custom-rendered cursors to avoid recreating them.
_cursor_cache: Dict[int, Gdk.Cursor] = {}


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
