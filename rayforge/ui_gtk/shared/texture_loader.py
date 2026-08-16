"""Helpers for displaying material textures in the UI."""

import logging
from functools import lru_cache
from pathlib import Path

import cairo
import numpy as np
import pyvips
from gi.repository import Gdk, GdkPixbuf, GLib, Gtk

from ...core.color import colorize_rgb
from ...core.material import Material

logger = logging.getLogger(__name__)

# Corner radius of swatch images, in pixels at the swatch's own scale
SWATCH_CORNER_RADIUS = 3

# Maximum number of decoded full-resolution textures to keep in memory
MAX_CACHED_TEXTURES = 16


def _rgba_bytes_to_texture(
    data: bytes, width: int, height: int
) -> Gdk.Texture:
    """Convert RGBA byte data into a Gdk.Texture."""
    pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
        GLib.Bytes.new(data),
        GdkPixbuf.Colorspace.RGB,
        True,
        8,
        width,
        height,
        width * 4,
    )
    return Gdk.Texture.new_for_pixbuf(pixbuf)


def _rounded_rect_mask(width: int, height: int, radius: int) -> np.ndarray:
    """
    Anti-aliased alpha mask of a rounded rectangle.

    Uses the signed distance to the rounded-rectangle boundary, so
    the mask is 255 in the interior and fades out over the corner
    arcs. This bakes the rounded corners into the texture itself,
    because GTK does not clip a Gtk.Image's painted content to the
    CSS border-radius.
    """
    y, x = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = np.clip(x, radius, width - 1 - radius)
    cy = np.clip(y, radius, height - 1 - radius)
    dist = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    alpha = np.clip(radius + 0.5 - dist, 0.0, 1.0)
    return (alpha * 255).astype(np.uint8)


def _apply_rounded_corners(
    data: bytes, width: int, height: int, radius: int
) -> bytes:
    """Intersect the alpha channel with a rounded-rectangle mask."""
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    mask = _rounded_rect_mask(width, height, radius)
    result = arr.copy()
    result[:, :, 3] = np.minimum(arr[:, :, 3], mask)
    return result.tobytes()


def _apply_tint(
    data: bytes,
    width: int,
    height: int,
    tint: tuple[float, float, float, float],
) -> bytes:
    """Recolor RGBA bytes to the tint color, preserving per-pixel shading."""
    if tint is None:
        return data
    arr = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 4)
    result = arr.astype(np.float32)
    result[:, :, :3] = colorize_rgb(result[:, :, :3], tint)
    result[:, :, 3] *= tint[3]
    return np.clip(result, 0, 255).astype(np.uint8).tobytes()


def load_texture_thumbnail(
    path: Path,
    size: int = 48,
    tint: tuple[float, float, float, float] | None = None,
) -> Gdk.Texture | None:
    """
    Load an image file as a square Gdk.Texture thumbnail.

    GDK only decodes PNG/JPEG/TIFF natively, so WebP (and anything
    else pyvips supports) is decoded with pyvips and converted to
    RGBA bytes first. The thumbnail gets rounded corners matching
    the swatch style. When *tint* is given, the pixels are multiplied
    by the tint color first.

    Args:
        path: Image file to load
        size: Maximum thumbnail dimension in pixels
        tint: Optional RGBA tint to multiply into the pixels

    Returns:
        A Gdk.Texture scaled to fit within size x size, or None if
        the image could not be loaded
    """
    try:
        image = pyvips.Image.thumbnail(str(path), size, height=size)
    except (pyvips.error.Error, OSError):
        logger.warning("Failed to load texture thumbnail: %s", path)
        return None

    image = image.colourspace("srgb")
    if not image.hasalpha():
        image = image.addalpha()
    image = image.cast("uchar")

    data = image.write_to_memory()
    if tint is not None:
        data = _apply_tint(data, image.width, image.height, tint)
    radius = min(SWATCH_CORNER_RADIUS, size // 3)
    data = _apply_rounded_corners(data, image.width, image.height, radius)
    return _rgba_bytes_to_texture(data, image.width, image.height)


def _solid_color_texture(
    rgba: tuple[int, int, int, int], size: int
) -> Gdk.Texture:
    """Create a solid-color square texture with rounded corners."""
    r, g, b, a = rgba
    data = bytes((r, g, b, a)) * (size * size)
    radius = min(SWATCH_CORNER_RADIUS, size // 3)
    data = _apply_rounded_corners(data, size, size, radius)
    return _rgba_bytes_to_texture(data, size, size)


def _make_swatch_image(texture: Gdk.Texture, size: int) -> Gtk.Image:
    """
    Wrap a texture in a fixed-size Gtk.Image.

    Gtk.Image with set_pixel_size() paints the paintable at exactly
    the requested size, so the widget always allocates `size` by
    `size`. Gtk.Picture is not used here: it follows the paintable's
    intrinsic size and can grow beyond the requested swatch size
    depending on the surrounding layout.
    """
    image = Gtk.Image.new_from_paintable(texture)
    image.set_pixel_size(size)
    image.set_valign(Gtk.Align.CENTER)
    return image


@lru_cache(maxsize=MAX_CACHED_TEXTURES)
def _load_texture_surface_cached(
    path_str: str, mtime_ns: int, size: int
) -> tuple[cairo.ImageSurface, bytearray] | None:
    """Decode a texture file at full resolution into a Cairo surface.

    The file identity (path, mtime, size) is part of the cache key so
    replaced files are picked up automatically. The returned tuple
    also carries the RGBA byte buffer that the surface references;
    the buffer must stay alive while the surface is used.
    """
    try:
        image = pyvips.Image.new_from_file(path_str).colourspace("srgb")
    except (pyvips.error.Error, OSError):
        logger.warning("Failed to load texture: %s", path_str)
        return None
    if not image.hasalpha():
        image = image.addalpha()
    image = image.cast("uchar")
    width = image.width
    height = image.height

    arr = np.frombuffer(image.write_to_memory(), dtype=np.uint8)
    arr = arr.reshape(height, width, 4)  # RGBA from pyvips
    if np.any(arr[:, :, 3] < 255):
        # Cairo ARGB32 expects premultiplied alpha
        premult = arr.astype(np.uint16)
        premult[:, :, 0] = premult[:, :, 0] * premult[:, :, 3] // 255
        premult[:, :, 1] = premult[:, :, 1] * premult[:, :, 3] // 255
        premult[:, :, 2] = premult[:, :, 2] * premult[:, :, 3] // 255
        out = premult.astype(np.uint8)
    else:
        out = arr.copy()

    # Cairo ARGB32 stores pixels in BGRA byte order
    out[:, :, [0, 2]] = out[:, :, [2, 0]]

    buffer = bytearray(out.tobytes())
    surface = cairo.ImageSurface.create_for_data(
        buffer, cairo.FORMAT_ARGB32, width, height, width * 4
    )
    return surface, buffer


def load_texture_cairo_surface(
    path: Path,
) -> tuple[cairo.ImageSurface, bytearray] | None:
    """
    Load a texture at full resolution as a Cairo image surface.

    The returned tuple also carries the RGBA byte buffer that the
    surface references; the buffer must stay alive while the surface
    is used. Surfaces are cached by (path, mtime, size), so replaced
    files are picked up automatically.
    """
    try:
        stat = path.stat()
    except OSError:
        return None
    return _load_texture_surface_cached(
        str(path), stat.st_mtime_ns, stat.st_size
    )


@lru_cache(maxsize=MAX_CACHED_TEXTURES)
def _tint_surface_cached(
    path_str: str,
    mtime_ns: int,
    size: int,
    tint: tuple[float, float, float, float],
) -> tuple[cairo.ImageSurface, bytearray] | None:
    """Decode a texture and colorize it with a tint, cached per path+tint.

    The cache key includes the file identity and the tint, so each
    (texture, tint) combination is computed once and reused across
    redraws instead of re-tinting a full-size copy on every draw.
    """
    loaded = _load_texture_surface_cached(path_str, mtime_ns, size)
    if loaded is None:
        return None
    surface, _buffer = loaded
    return tint_cairo_surface(surface, tint)


def tinted_texture_cairo_surface(
    path: Path, tint: tuple[float, float, float, float]
) -> tuple[cairo.ImageSurface, bytearray] | None:
    """Load a texture and return a tinted Cairo surface (cached)."""
    try:
        stat = path.stat()
    except OSError:
        return None
    return _tint_surface_cached(
        str(path), stat.st_mtime_ns, stat.st_size, tint
    )


def tint_cairo_surface(
    surface: cairo.ImageSurface, tint: tuple[float, float, float, float]
) -> tuple[cairo.ImageSurface, bytearray] | None:
    """
    Return a copy of a Cairo ARGB32 surface recolored to the tint color.

    The source surface (usually from the texture cache) is never
    modified; a new surface plus its backing buffer is returned so the
    tiled 2D-canvas rendering can show tintable materials tinted. The
    returned bytearray must be kept alive while the surface is used.

    Colorization shifts every pixel to the tint hue while preserving the
    texture's per-pixel shading (``luma * tint``), in premultiplied space.
    """
    if surface.get_format() != cairo.FORMAT_ARGB32:
        return None
    width = surface.get_width()
    height = surface.get_height()
    src = np.frombuffer(surface.get_data(), dtype=np.uint8).reshape(
        height, width, 4
    )  # BGRA byte order, premultiplied alpha
    out = src.astype(np.float32).copy()

    # Rec.709 luma over the premultiplied BGRA channels (R=idx2, G=idx1,
    # B=idx0). Alpha is already premultiplied into RGB, so this stays
    # consistent; textures are treated as fully opaque for tinting.
    luma = out[..., 2] * 0.2126 + out[..., 1] * 0.7152 + out[..., 0] * 0.0722
    out[..., 2] = luma * tint[0]  # red
    out[..., 1] = luma * tint[1]  # green
    out[..., 0] = luma * tint[2]  # blue

    out = np.clip(out, 0, 255).astype(np.uint8)
    buffer = bytearray(out.tobytes())
    tinted = cairo.ImageSurface.create_for_data(
        buffer, cairo.FORMAT_ARGB32, width, height, width * 4
    )
    return tinted, buffer


def create_material_swatch_texture(
    material: Material, size: int = 32
) -> Gdk.Texture:
    """
    Create the swatch texture for a material.

    Shows the material's texture thumbnail when one is available,
    otherwise a solid swatch in the material's display color. The
    swatch has rounded corners baked into the texture's alpha
    channel.
    """
    texture_path = material.get_texture_path()
    tint = material.appearance.get_tint_rgba()
    texture = (
        load_texture_thumbnail(texture_path, size=size, tint=tint)
        if texture_path is not None
        else None
    )
    if texture is None:
        r, g, b, _ = material.get_display_rgba()
        texture = _solid_color_texture(
            (int(r * 255), int(g * 255), int(b * 255), 255), size
        )
    return texture


def create_material_swatch(material: Material, size: int = 32) -> Gtk.Image:
    """
    Create a swatch widget for a material.

    Shows the material's texture thumbnail when one is available,
    otherwise a solid swatch in the material's display color. Both
    cases return the same Gtk.Image widget painted at exactly
    `size` pixels, so textured and plain materials always render
    at the same size. The swatch has rounded corners baked into
    the texture's alpha channel.

    Args:
        material: The material to show
        size: Swatch size in pixels

    Returns:
        A Gtk.Image suitable as an ActionRow prefix
    """
    return _make_swatch_image(
        create_material_swatch_texture(material, size), size
    )
