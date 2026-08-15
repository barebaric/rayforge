from gi.repository import Gdk, Gtk

# Workaround CSS for Adwaita row layout issues:
#
# 1. Spin buttons inside rows ignore ``set_width_chars()`` and size
#    themselves from their initial value, producing inconsistent entry
#    widths.  A global ``min-width`` on every ``Gtk.SpinButton`` inside
#    a row fixes this.
#
# 2. ``Adw.ComboRow`` renders the selected value in an inline
#    ``Gtk.ListView`` whose natural/min width is ~4 px.  When the row
#    has a subtitle (description), the title box expands (``hexpand``)
#    and squeezes the dropdown to near-zero width, showing only the
#    first letter of the selected value.  A ``min-width`` on the
#    inline list view prevents this collapse.
_ROW_MIN_WIDTH_CSS = (
    "row spinbutton { min-width: 100px; }"
    " row.combo listview.inline { min-width: 100px; }"
)

_css_loaded = False


def ensure_row_min_width(row: Gtk.Widget) -> None:
    """Load global CSS that enforces minimum widths on row children.

    Called once (idempotent) by :class:`SpinRow` and
    :class:`~rayforge.ui_gtk.varset.adapter.combo.ComboAdapter` to
    prevent Adwaita layout from collapsing spin buttons and combo
    dropdowns.
    """
    global _css_loaded
    if not _css_loaded:
        provider = Gtk.CssProvider()
        provider.load_from_string(_ROW_MIN_WIDTH_CSS)
        display = Gdk.Display.get_default()
        if display is not None:
            Gtk.StyleContext.add_provider_for_display(
                display,
                provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION,
            )
        _css_loaded = True


def get_spinrow_int(spinrow):
    # Workaround: Adw.SpinRow seems to have a bug that the value is not
    # always updated if it was edited using the keyboard in the edit
    # field. I.e. get_value() still returns the previous value.
    # So I convert it manually from text if possible.
    try:
        value = int(spinrow.get_text())
    except ValueError:
        value = int(spinrow.get_value())
    lower = spinrow.get_adjustment().get_lower()
    upper = spinrow.get_adjustment().get_upper()
    return int(max(lower, min(value, upper)))


def get_spinrow_float(spinrow):
    # Workaround: Adw.SpinRow seems to have a bug that the value is not
    # always updated if it was edited using the keyboard in the edit
    # field. I.e. get_value() still returns the previous value.
    # So I convert it manually from text if possible.
    try:
        value = float(spinrow.get_text())
    except ValueError:
        value = float(spinrow.get_value())
    lower = spinrow.get_adjustment().get_lower()
    upper = spinrow.get_adjustment().get_upper()
    return max(lower, min(value, upper))
