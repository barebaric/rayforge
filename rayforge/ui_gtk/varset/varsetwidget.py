import logging
from typing import Any
from gi.repository import Gtk, Adw
from blinker import Signal
from ...core.varset import (
    ChoiceVar,
    HostnameVar,
    SliderFloatVar,
    TextAreaVar,
    Var,
    VarSet,
)
from ..icons import get_icon
from ...machine.transport.validators import is_valid_hostname_or_ip
from ..shared.adwfix import get_spinrow_int
from .var_row_factory import VarRowFactory, escape_title

logger = logging.getLogger(__name__)
NULL_CHOICE_LABEL = _("None Selected")


class VarSetWidget(Adw.PreferencesGroup):
    """
    A self-contained Adwaita Preferences Group that populates itself with
    rows based on a VarSet. It uses a VarRowFactory to generate the UI
    and supports both immediate updates and explicit "Apply" buttons.
    """

    def __init__(self, explicit_apply=False, **kwargs):
        super().__init__(**kwargs)
        self.explicit_apply = explicit_apply
        self.widget_map: dict[str, tuple[Adw.PreferencesRow, Var]] = {}
        self._created_rows = []
        self._factory = VarRowFactory()
        self.data_changed = Signal()

    def clear_dynamic_rows(self):
        """Removes only the rows dynamically created by populate()."""
        for row in self._created_rows:
            self.remove(row)
        self._created_rows.clear()
        self.widget_map.clear()

    def populate(self, var_set: VarSet):
        """
        Clears previous dynamic rows and builds new ones from a VarSet.
        Any static rows added manually are preserved.
        Reuse existing rows if possible to preserve state.
        """
        # Set the group's title and description from the VarSet
        if var_set.title:
            self.set_title(escape_title(var_set.title))
        if var_set.description:
            self.set_description(escape_title(var_set.description))

        # 1. Identify rows to remove (keys not present in new set)
        new_keys = {var.key for var in var_set}
        existing_keys = list(self.widget_map.keys())

        for key in existing_keys:
            if key not in new_keys:
                row, _ = self.widget_map.pop(key)
                self.remove(row)
                if row in self._created_rows:
                    self._created_rows.remove(row)

        # 2. Add or update rows
        for var in var_set:
            if var.key in self.widget_map:
                row, old_var = self.widget_map[var.key]

                # Determine if we need to rebuild the widget
                needs_rebuild = type(var) is not type(old_var)
                if not needs_rebuild and isinstance(var, ChoiceVar):
                    assert isinstance(old_var, ChoiceVar)
                    # Rebuild ChoiceVar if options changed
                    if var.choices != old_var.choices:
                        needs_rebuild = True

                if needs_rebuild:
                    self.remove(row)
                    if row in self._created_rows:
                        self._created_rows.remove(row)
                    del self.widget_map[var.key]
                else:
                    # Update reference and attributes (e.g. limits) without
                    # destroying
                    self.widget_map[var.key] = (row, var)
                    self._update_row_attributes(row, var)
                    continue

            # Create new row
            row = self._factory.create_row_for_var(var, "value")
            if row:
                self._wire_up_row(row, var)
                self.add(row)
                self._created_rows.append(row)
                self.widget_map[var.key] = (row, var)

    def _update_row_attributes(self, row: Adw.PreferencesRow, var: Var):
        """
        Updates an existing row's properties (title, subtitle, ranges)
        from a new Var definition.
        """
        if hasattr(row, "set_title") and var.label:
            row.set_title(escape_title(var.label))

        widget = getattr(row, "get_activatable_widget", lambda: None)()

        if isinstance(row, Adw.SpinRow):
            self._update_adjustment(row.get_adjustment(), var)
            if var.description:
                row.set_subtitle(var.description)

        elif isinstance(widget, Gtk.Scale):
            self._update_adjustment(widget.get_adjustment(), var)
            if var.description:
                row.set_tooltip_text(var.description)

        elif isinstance(row, Adw.EntryRow):
            if var.description:
                row.set_tooltip_text(var.description)

    def _update_adjustment(self, adj: Gtk.Adjustment, var: Var):
        """Updates the limits of an adjustment if the var defines them."""
        if not adj:
            return

        min_val = getattr(var, "min_val", None)
        max_val = getattr(var, "max_val", None)

        if min_val is not None:
            adj.set_lower(float(min_val))
        if max_val is not None:
            adj.set_upper(float(max_val))

    def get_values(self) -> dict[str, Any]:
        """Reads all current values from the UI widgets."""
        values = {}
        for key, (row, var) in self.widget_map.items():
            value = None
            widget = getattr(row, "get_activatable_widget", lambda: None)()
            if isinstance(var, TextAreaVar):
                text_view = getattr(row, "core_widget", None)
                if isinstance(text_view, Gtk.TextView):
                    buffer = text_view.get_buffer()
                    start, end = buffer.get_start_iter(), buffer.get_end_iter()
                    value = buffer.get_text(start, end, True)
            elif isinstance(var, SliderFloatVar) and isinstance(
                widget, Gtk.Scale
            ):
                min_val = var.min_val if var.min_val is not None else 0.0
                max_val = var.max_val if var.max_val is not None else 1.0
                percent = widget.get_value() / 100.0
                value = min_val + percent * (max_val - min_val)
            elif isinstance(widget, Gtk.Switch):
                value = widget.get_active()
            elif isinstance(row, Adw.EntryRow):
                value = row.get_text()
            elif isinstance(row, Adw.SpinRow):
                value = (
                    get_spinrow_int(row)
                    if var.var_type is int
                    else row.get_value()
                )
            elif isinstance(row, Adw.ComboRow):
                selected = row.get_selected_item()
                display_str = ""
                if selected:
                    display_str = selected.get_string()  # type: ignore

                if display_str == NULL_CHOICE_LABEL:
                    value = None
                elif isinstance(var, ChoiceVar):
                    value = var.get_value_for_display(display_str)
                else:  # For BaudrateVar, SerialPortVar
                    value = display_str

            values[key] = value
        return values

    def set_values(self, values: dict[str, Any]):
        """Sets the UI widgets from a dictionary of values."""
        for key, value in values.items():
            if key not in self.widget_map or value is None:
                continue

            row, var = self.widget_map[key]
            widget = getattr(row, "get_activatable_widget", lambda: None)()
            if isinstance(var, TextAreaVar):
                text_view = getattr(row, "core_widget", None)
                if isinstance(text_view, Gtk.TextView):
                    text_view.get_buffer().set_text(str(value))
            elif isinstance(var, SliderFloatVar) and isinstance(
                widget, Gtk.Scale
            ):
                min_val = var.min_val if var.min_val is not None else 0.0
                max_val = var.max_val if var.max_val is not None else 1.0
                range_size = max_val - min_val
                percent = 0.0
                if range_size > 1e-9:
                    percent = ((float(value) - min_val) / range_size) * 100.0
                widget.set_value(percent)
            elif isinstance(widget, Gtk.Switch):
                widget.set_active(bool(value))
            elif isinstance(row, Adw.EntryRow):
                row.set_text(str(value))
            elif isinstance(row, Adw.SpinRow):
                row.set_value(float(value))
            elif isinstance(row, Adw.ComboRow):
                model = row.get_model()
                if isinstance(model, Gtk.StringList):
                    display_str = NULL_CHOICE_LABEL
                    if value is not None:
                        display_str = (
                            var.get_display_for_value(str(value)) or str(value)
                            if isinstance(var, ChoiceVar)
                            else str(value)
                        )
                    for i in range(model.get_n_items()):
                        if model.get_string(i) == display_str:
                            row.set_selected(i)
                            break

    def _on_data_changed(self, key: str):
        self.data_changed.send(self, key=key)

    def _add_apply_button_if_needed(self, row, key):
        if not self.explicit_apply:
            return
        apply_button = Gtk.Button(
            child=get_icon("object-select-symbolic"),
            tooltip_text=_("Apply Change"),
        )
        apply_button.add_css_class("flat")
        apply_button.set_valign(Gtk.Align.CENTER)
        apply_button.connect("clicked", lambda b: self._on_data_changed(key))
        row.add_suffix(apply_button)

    def _wire_up_row(self, row: Adw.PreferencesRow, var: Var):
        """Connects signals for the row based on the explicit_apply setting."""
        widget = getattr(row, "get_activatable_widget", lambda: None)() or row

        if self.explicit_apply:
            self._add_apply_button_if_needed(row, var.key)
            if isinstance(row, Adw.EntryRow):
                row.connect("apply", lambda r: self._on_data_changed(var.key))
            return

        if isinstance(row, Adw.EntryRow):
            row.connect("apply", lambda r: self._on_data_changed(var.key))
        elif isinstance(row, Adw.SpinRow):
            row.connect(
                "notify::value", lambda r, p: self._on_data_changed(var.key)
            )
        elif isinstance(row, Adw.ComboRow):
            row.connect(
                "notify::selected-item",
                lambda r, p: self._on_data_changed(var.key),
            )
        elif isinstance(widget, Gtk.Switch):
            widget.connect(
                "state-set", lambda s, a: self._on_data_changed(var.key)
            )
        elif isinstance(widget, Gtk.Scale):
            widget.connect(
                "value-changed", lambda s: self._on_data_changed(var.key)
            )
        elif isinstance(var, HostnameVar) and isinstance(row, Adw.EntryRow):

            def on_validate(entry_row):
                if is_valid_hostname_or_ip(entry_row.get_text()):
                    entry_row.remove_css_class("error")
                else:
                    entry_row.add_css_class("error")

            row.connect("changed", on_validate)
            on_validate(row)
