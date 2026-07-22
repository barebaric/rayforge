from gi.repository import Gdk, Gtk

_SPINROW_MIN_WIDTH_CSS = b"row spinbutton { min-width: 130px; }"

_css_loaded = False


def ensure_spinrow_min_width(row: Gtk.Widget) -> None:
    """Ensure a consistent minimum width on spin buttons inside rows.

    ``Adw.SpinRow.set_width_chars()`` delegates through the
    ``Gtk.Editable`` interface but the internal layout does not honour
    it, so rows with different adjustment ranges end up with
    differently-sized entry fields — and multi-digit values get
    clipped.  Loading a global CSS rule that sets ``min-width`` on
    every ``Gtk.SpinButton`` inside a row works reliably.
    """
    global _css_loaded
    if not _css_loaded:
        provider = Gtk.CssProvider()
        provider.load_from_data(_SPINROW_MIN_WIDTH_CSS)
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
