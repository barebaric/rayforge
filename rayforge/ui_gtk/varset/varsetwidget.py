import logging
from typing import Any, Dict, Optional, Tuple
from gettext import gettext as _
from gi.repository import GLib, Gtk, Adw
from blinker import Signal
from ...core.varset import (
    ChoiceVar,
    HostnameVar,
    Var,
    VarSet,
)
from ..icons import get_icon
from ...machine.transport.validators import is_valid_hostname_or_ip
from .adapter import RowAdapter, create_row_for_var, escape_title

logger = logging.getLogger(__name__)

_DEBOUNCE_DELAY_MS = 300


class VarSetWidget(Adw.PreferencesGroup):
    """
    A self-contained Adwaita Preferences Group that populates itself with
    rows based on a VarSet. Supports both immediate updates and explicit
    "Apply" buttons, with built-in debouncing for rapid value changes.
    """

    def __init__(self, explicit_apply=False, debounce_ms=0, **kwargs):
        super().__init__(**kwargs)
        self.explicit_apply = explicit_apply
        self.debounce_ms = debounce_ms
        self.widget_map: Dict[str, Tuple[Adw.PreferencesRow, Var]] = {}
        self._adapters: Dict[str, RowAdapter] = {}
        self._created_rows = []
        self._apply_buttons = []
        self.data_changed = Signal()
        self._debounce_timer_id: Optional[int] = None
        self._pending_keys: set = set()

    def clear_dynamic_rows(self):
        """Removes only the rows dynamically created by populate()."""
        self._cancel_debounce()
        for row in self._created_rows:
            self.remove(row)
        self._created_rows.clear()
        self._apply_buttons.clear()
        self.widget_map.clear()
        self._adapters.clear()

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
            row, adapter = create_row_for_var(var, "value")
            if row:
                self._wire_up_row(row, var)
                self.add(row)
                self._created_rows.append(row)
                self.widget_map[var.key] = (row, var)
                if adapter is not None:
                    self._adapters[var.key] = adapter

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

    def get_values(self) -> Dict[str, Any]:
        """Reads all current values from the UI widgets."""
        values = {}
        for key in self.widget_map:
            adapter = self._adapters.get(key)
            if adapter is not None:
                values[key] = adapter.get_value()
            else:
                values[key] = None
        return values

    def set_values(self, values: Dict[str, Any]):
        """Sets the UI widgets from a dictionary of values."""
        for key, value in values.items():
            if key not in self.widget_map or value is None:
                continue
            adapter = self._adapters.get(key)
            if adapter is not None:
                adapter.set_value(value)

    def _on_data_changed(self, key: str):
        if self.debounce_ms > 0:
            self._pending_keys.add(key)
            self._schedule_debounce()
        else:
            self.data_changed.send(self, key=key)

    def _schedule_debounce(self):
        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)
        self._debounce_timer_id = GLib.timeout_add(
            self.debounce_ms, self._flush_debounce
        )

    def _cancel_debounce(self):
        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = None
        self._pending_keys.clear()

    def _flush_debounce(self):
        self._debounce_timer_id = None
        keys = set(self._pending_keys)
        self._pending_keys.clear()
        for key in keys:
            self.data_changed.send(self, key=key)

    def _add_apply_button_if_needed(self, row, key):
        if not self.explicit_apply:
            return
        apply_button = Gtk.Button(
            child=get_icon("check-symbolic"),
            tooltip_text=_("Apply Change"),
        )
        apply_button.add_css_class("flat")
        apply_button.set_valign(Gtk.Align.CENTER)
        apply_button.connect("clicked", lambda b: self._on_data_changed(key))
        row.add_suffix(apply_button)
        self._apply_buttons.append(apply_button)

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

    def set_apply_buttons_sensitive(self, sensitive: bool):
        """Set the sensitivity of all apply buttons."""
        for button in self._apply_buttons:
            button.set_sensitive(sensitive)
