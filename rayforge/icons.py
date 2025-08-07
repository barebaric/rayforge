import importlib.resources
import logging
import pathlib
from gi.repository import Gtk, Gio  # type: ignore
from .resources import icons  # type: ignore


logger = logging.getLogger(__name__)


def get_icon_path(icon_name) -> pathlib.Path:
    """Retrieve the path of an icon inside the resource directory."""
    with importlib.resources.path(icons, f"{icon_name}.svg") as path:
        return path


def get_icon(icon_name: str) -> Gtk.Image:
    """
    Retrieve a Gtk.Image, prioritizing a local file from the resource
    directory before falling back to the system theme.
    """
    # First, attempt to load the icon from a local file path.
    path = get_icon_path(icon_name)
    if path and path.is_file():
        logger.debug(f"Using local icon for '{icon_name}' from: {path}")
        try:
            icon_file = Gio.File.new_for_path(str(path))
            icon = Gio.FileIcon.new(icon_file)
            return Gtk.Image.new_from_gicon(icon)
        except Exception as e:
            logger.error(f"Failed to load local icon '{icon_name}': {e}")
            # Continue to fallback...

    # If local file doesn't exist or failed to load, fall back to the theme.
    logger.debug(f"Icon for '{icon_name}' not found. Falling back to theme.")
    return Gtk.Image.new_from_icon_name(icon_name)
