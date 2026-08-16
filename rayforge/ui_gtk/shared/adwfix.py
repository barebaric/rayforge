from gi.repository import Adw, Gdk, Gtk, Pango

# Workaround CSS for Adwaita row layout issues:
#
# 1. Spin buttons inside rows ignore ``set_width_chars()`` and size
#    themselves from their initial value, producing inconsistent entry
#    widths.  A global ``min-width`` on every ``Gtk.SpinButton`` inside
#    a row fixes this.
#
# 2. ``Adw.ComboRow`` renders the selected value in an inline
#    ``Gtk.ListView``.  Its default factory draws a left-aligned,
#    ellipsized label with ``width-chars=1``, which collapses the value
#    to a single letter + "…" whenever the row is squeezed (e.g. by an
#    expanded title box).  This module monkey-patches ``Adw.ComboRow``
#    (see :data:`_PATCHED_COMBO_ROW`) so the inline label is right-
#    aligned and never ellipsized, keeping its natural width and sitting
#    flush against the row's trailing edge -- no ``min-width`` hack
#    needed.  Rows that install a custom factory or ``use-subtitle``
#    keep their custom rendering untouched.
_ROW_MIN_WIDTH_CSS = "row spinbutton { min-width: 100px; }"

_css_loaded = False


def _on_combo_selected_changed(row, _pspec, list_item):
    """Show the checkmark in the popover on the selected item."""
    icon = list_item._factory_icon
    icon.set_opacity(
        1.0 if row.get_selected_item() == list_item.get_item() else 0.0
    )


def _on_combo_root_changed(box, _pspec, list_item):
    """Show the checkmark only inside the popover, not inline."""
    icon = list_item._factory_icon
    icon.set_visible(box.get_ancestor(Gtk.Popover) is not None)


def _make_combo_factory(
    row: Adw.ComboRow, xalign: float
) -> Gtk.SignalListItemFactory:
    """Factory replicating Adw.ComboRow's default item rendering.

    The default factory (see ``setup_item`` in adw-combo-row.c) draws a
    left-aligned, ellipsized label with ``width-chars=1``, which both
    misaligns the selected value and lets it collapse to a single letter
    when the row has a subtitle.  This factory keeps the trailing
    checkmark (popover only) but disables ellipsation so the label
    requests its natural width, and aligns the text as requested.
    """
    factory = Gtk.SignalListItemFactory()

    def on_setup(_factory, list_item):
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        label = Gtk.Label(xalign=xalign)
        label.set_ellipsize(Pango.EllipsizeMode.NONE)
        label.set_valign(Gtk.Align.CENTER)
        # Expand the label so it fills any extra width the row gives the
        # inline area; xalign then right-aligns the text within it.
        label.set_hexpand(True)
        label.set_halign(Gtk.Align.FILL)
        box.append(label)
        icon = Gtk.Image(icon_name="object-select-symbolic")
        box.append(icon)
        list_item.set_child(box)
        list_item._factory_box = box
        list_item._factory_icon = icon

    def on_bind(_factory, list_item):
        item = list_item.get_item()
        label = list_item._factory_box.get_first_child()
        text = ""
        if isinstance(item, Gtk.StringObject):
            text = item.get_string()
        elif item is not None:
            text = str(item)
        label.set_label(text)
        list_item._factory_selected_id = row.connect(
            "notify::selected-item", _on_combo_selected_changed, list_item
        )
        list_item._factory_root_id = list_item._factory_box.connect(
            "notify::root", _on_combo_root_changed, list_item
        )
        _on_combo_selected_changed(row, None, list_item)
        _on_combo_root_changed(list_item._factory_box, None, list_item)

    def on_unbind(_factory, list_item):
        row.disconnect(list_item._factory_selected_id)
        list_item._factory_box.disconnect(list_item._factory_root_id)
        list_item._factory_selected_id = None
        list_item._factory_root_id = None

    factory.connect("setup", on_setup)
    factory.connect("bind", on_bind)
    factory.connect("unbind", on_unbind)
    return factory


def _apply_combo_row_fix(row: Adw.ComboRow) -> None:
    """Right-align an Adw.ComboRow's inline value and disable ellipsation.

    The default factory draws the selected value left-aligned and
    ellipsized, which misaligns it within the row and collapses it to a
    single letter when the row has a subtitle.  This replaces the inline
    factory with a right-aligned, non-ellipsized one while keeping the
    popover list left-aligned with its selection checkmark.
    """
    row.set_factory(_make_combo_factory(row, 1.0))
    row.set_list_factory(_make_combo_factory(row, 0.0))


_ORIGINAL_COMBO_ROW = Adw.ComboRow
_PATCHED_COMBO_ROW = None


class _ComboRow(_ORIGINAL_COMBO_ROW):
    """An :class:`Adw.ComboRow` with the inline value right-aligned.

    Keeps the popover list left-aligned with its selection checkmark.
    Rows that install a custom factory or ``use-subtitle`` keep their
    custom rendering: setting either resets ``list-factory`` to ``None``
    so the custom factory governs both the popover and the inline
    display, matching upstream behavior.
    """

    __gtype_name__ = "RayforgeComboRow"

    def __init__(self, *args, **kwargs):
        use_custom = kwargs.get("factory") is not None
        use_custom = use_custom or bool(kwargs.get("use_subtitle"))
        super().__init__(*args, **kwargs)
        if not use_custom:
            _apply_combo_row_fix(self)

    def set_factory(self, factory=None):
        result = super().set_factory(factory)
        # A custom factory governs both the popover and the inline
        # display; drop our left-aligned list factory so the popover
        # uses the custom factory as upstream expects.
        if self.get_list_factory() is not None:
            super().set_list_factory(None)
        return result

    def set_use_subtitle(self, use_subtitle):
        result = super().set_use_subtitle(use_subtitle)
        if use_subtitle and self.get_list_factory() is not None:
            super().set_list_factory(None)
        return result


_PATCHED_COMBO_ROW = _ComboRow
Adw.ComboRow = _ComboRow


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
