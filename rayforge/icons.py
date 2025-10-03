import importlib.resources
import logging
import pathlib
from typing import Dict, Union
from gi.repository import Gtk, Gio
from .resources import icons  # type: ignore


logger = logging.getLogger(__name__)

# Global cache for loaded icons to avoid repeated expensive operations
# We cache the Gio.Icon or icon name, not the Gtk.Image widget itself
_icon_cache: Dict[str, Union[Gio.Icon, str]] = {}


def get_icon_path(icon_name) -> pathlib.Path:
    """Retrieve the path of an icon inside the resource directory."""
    with importlib.resources.path(icons, f"{icon_name}.svg") as path:
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
        if isinstance(cached_value, Gio.Icon):
            return Gtk.Image.new_from_gicon(cached_value)
        else:  # icon name string
            return Gtk.Image.new_from_icon_name(cached_value)

    # First, attempt to load the icon from a local file path.
    path = get_icon_path(icon_name)
    if path and path.is_file():
        logger.debug(f"Using local icon for '{icon_name}' from: {path}")
        try:
            icon_file = Gio.File.new_for_path(str(path))
            icon = Gio.FileIcon.new(icon_file)
            _icon_cache[icon_name] = icon
            return Gtk.Image.new_from_gicon(icon)
        except Exception as e:
            logger.error(f"Failed to load local icon '{icon_name}': {e}")
            # Continue to fallback...

    # If local file doesn't exist or failed to load, fall back to the theme.
    logger.debug(f"Icon for '{icon_name}' not found. Falling back to theme.")
    _icon_cache[icon_name] = icon_name
    return Gtk.Image.new_from_icon_name(icon_name)


def clear_icon_cache():
    """Clear the icon cache. Useful for testing or theme changes."""
    _icon_cache.clear()
