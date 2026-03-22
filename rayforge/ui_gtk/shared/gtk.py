import logging
from gettext import gettext as _
from typing import Optional, Tuple, cast
from gi.repository import Gtk, Gdk
from ...image.registry import FileFilter
from ...shared.util.once import once_per_object

logger = logging.getLogger(__name__)


def get_monitor_geometry() -> Optional[Gdk.Rectangle]:
    """
    Returns a rectangle for the current monitor dimensions. If not found,
    may return None.
    """
    display = Gdk.Display.get_default()
    if not display:
        return None

    monitors = display.get_monitors()
    if not monitors:
        return None
    monitor = cast(Gdk.Monitor, monitors[0])

    # Try to get the monitor under the cursor (heuristic for active
    # monitor). Note: Wayland has no concept of "primary monitor"
    # anymore, so Gdk.get_primary_monitor() is obsolete.
    # Fallback to the first monitor if no monitor is found under the cursor
    seat = display.get_default_seat()
    if not seat:
        return monitor.get_geometry()

    pointer = seat.get_pointer()
    if not pointer:
        return monitor.get_geometry()

    surface, x, y = pointer.get_surface_at_position()
    if not surface:
        return monitor.get_geometry()

    monitor_under_mouse = display.get_monitor_at_surface(surface)
    if not monitor_under_mouse:
        return monitor.get_geometry()

    return monitor_under_mouse.get_geometry()


def get_screen_size() -> Optional[Tuple[int, int]]:
    """Get the current monitor's screen size as (width, height)."""
    geometry = get_monitor_geometry()
    if not geometry:
        return None
    return (geometry.width, geometry.height)


@once_per_object
def apply_css(css: str):
    provider = Gtk.CssProvider()
    provider.load_from_string(css)
    display = Gdk.Display.get_default()
    if not display:
        logger.warning("No default Gdk display found. CSS may not apply.")
        return
    Gtk.StyleContext.add_provider_for_display(
        display,
        provider,
        Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
    )


def file_filter_to_gtk(
    filt: FileFilter, translate: bool = True
) -> Gtk.FileFilter:
    """
    Convert a FileFilter dataclass to a Gtk.FileFilter.

    Args:
        filt: The FileFilter to convert.
        translate: Whether to translate the label using gettext.

    Returns:
        A configured Gtk.FileFilter instance.
    """
    gtk_filter = Gtk.FileFilter()
    label = _(filt.label) if translate else filt.label
    gtk_filter.set_name(label)
    for ext in filt.extensions:
        gtk_filter.add_pattern(f"*{ext}")
    for mime_type in filt.mime_types:
        gtk_filter.add_mime_type(mime_type)
    return gtk_filter
