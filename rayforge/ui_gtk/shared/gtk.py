import logging
from ...shared.util.once import once_per_object
from gi.repository import Gtk, Gdk

logger = logging.getLogger(__name__)


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
