import importlib.resources
import logging
import pathlib
from functools import lru_cache

from gi.repository import Gdk, GdkPixbuf, Gio, GLib, Graphene, Gsk, Gtk

from ..resources import icons  # type: ignore

logger = logging.getLogger(__name__)

_icon_search_paths: list[pathlib.Path] = []

# Global cache for loaded icons to avoid repeated expensive operations
# We cache the Gio.Icon or icon name, not the Gtk.Image widget itself
_icon_cache: dict[str, Gio.Icon | str] = {}

# Shared renderer for rasterizing SVG icons with GTK instead of
# gdk-pixbuf; see _load_svg_pixbuf_with_gtk().
_svg_renderer: Gsk.Renderer | None = None


def register_icon_path(path):
    """
    Register an additional directory to search for icons.

    Addons should call this to make their icons available via
    ``get_icon()``.  Registered paths are searched before the
    built-in ``rayforge.resources.icons`` package.
    """
    p = pathlib.Path(path)
    if p.is_dir() and p not in _icon_search_paths:
        _icon_search_paths.append(p)


def get_icon_path(icon_name) -> pathlib.Path:
    """
    Retrieve the path of an icon, searching addon-registered paths
    first, then the built-in resource directory.
    """
    filename = f"{icon_name}.svg"
    for search_path in _icon_search_paths:
        candidate = search_path / filename
        if candidate.is_file():
            return candidate
    with importlib.resources.path(icons, filename) as path:
        return path


def get_icon(icon_name: str) -> Gtk.Image:
    """
    Retrieve a Gtk.Image, prioritizing a local file from the resource
    directory before falling back to the system theme.

    Icons are cached to avoid repeated expensive loading operations.
    """
    # Check cache first
    if icon_name in _icon_cache:
        cached_value = _icon_cache[icon_name]
        if isinstance(cached_value, str):  # icon name string
            return Gtk.Image.new_from_icon_name(cached_value)
        return Gtk.Image.new_from_gicon(cached_value)

    # First, attempt to load the icon from a local file path.
    path = get_icon_path(icon_name)
    if path and path.is_file():
        logger.debug(f"Using local icon for '{icon_name}' from: {path}")
        try:
            icon_file = Gio.File.new_for_path(str(path))
            icon = Gio.FileIcon.new(icon_file)
            _icon_cache[icon_name] = icon
            return Gtk.Image.new_from_gicon(icon)
        except GLib.Error as e:
            logger.error(f"Failed to load local icon '{icon_name}': {e}")
            # Continue to fallback...

    # If local file doesn't exist or failed to load, fall back to the theme.
    logger.debug(f"Icon for '{icon_name}' not found. Falling back to theme.")
    _icon_cache[icon_name] = icon_name
    return Gtk.Image.new_from_icon_name(icon_name)


def _get_svg_renderer() -> Gsk.Renderer:
    """Returns the shared renderer for GTK-based SVG rasterization."""
    global _svg_renderer
    if _svg_renderer is None:
        renderer = Gsk.CairoRenderer.new()
        renderer.realize(None)
        _svg_renderer = renderer
    return _svg_renderer


def _load_svg_pixbuf_with_gtk(path: pathlib.Path, size: int):
    """
    Rasterizes an SVG icon with GTK's SVG renderer.

    GTK renders the same files for Gtk.Image without involving
    gdk-pixbuf's SVG loader, which is not available in every build
    (the macOS app bundle is one such case). Returns None when GTK has
    no SVG renderer or rendering fails.
    """
    if not hasattr(Gtk, "Svg"):
        return None

    svg = Gtk.Svg.new_from_bytes(GLib.Bytes.new(path.read_bytes()))
    width = svg.get_intrinsic_width()
    height = svg.get_intrinsic_height()
    if width <= 0 or height <= 0:
        width = height = float(size)
    else:
        scale = size / max(width, height)
        width *= scale
        height *= scale

    snapshot = Gtk.Snapshot()
    svg.snapshot(snapshot, width, height)
    node = snapshot.to_node()
    if node is None:
        return None

    texture = _get_svg_renderer().render_texture(
        node, Graphene.Rect().init(0, 0, width, height)
    )
    if texture is None:
        return None
    return Gdk.pixbuf_get_from_texture(texture)


@lru_cache
def get_icon_pixbuf(icon_name: str, size: int = 24):
    """
    Retrieve a GdkPixbuf for Cairo rendering, prioritizing a local file
    from the resource directory.

    gdk-pixbuf loads SVG files through a separate loader module that is
    missing from some builds (notably the macOS app bundle). When that
    load fails, the icon is rendered with GTK's SVG renderer instead.

    Args:
        icon_name: Name of the icon (without .svg extension)
        size: Size of the icon in pixels

    Returns:
        GdkPixbuf.Pixbuf: The loaded icon as a pixbuf, or None if failed
    """
    # First, attempt to load the icon from a local file path.
    path = get_icon_path(icon_name)
    if path and path.is_file():
        logger.debug(f"Using local icon for '{icon_name}' from: {path}")
        try:
            pixbuf = GdkPixbuf.Pixbuf.new_from_file_at_scale(
                str(path), size, size, True
            )
            return pixbuf
        except GLib.Error as e:
            pixbuf = _load_svg_pixbuf_with_gtk(path, size)
            if pixbuf is not None:
                logger.debug(
                    f"Rendered icon '{icon_name}' with GTK because "
                    f"gdk-pixbuf failed to load it: {e}"
                )
                return pixbuf
            logger.error(f"Failed to load local icon '{icon_name}': {e}")

    # Return None if icon couldn't be loaded
    return None


def clear_icon_cache():
    """Clear the icon cache. Useful for testing or theme changes."""
    _icon_cache.clear()
