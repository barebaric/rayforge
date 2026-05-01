import logging
from typing import Any, Dict, Optional, Tuple
from gettext import gettext as _
from gi.repository import GLib, Gtk, Adw
from blinker import Signal
from ...core.varset import Var, VarSet
from ..icons import get_icon
from .adapter import RowAdapter, create_row_for_var, escape_title

logger = logging.getLogger(__name__)

_DEBOUNCE_DELAY_MS = 300


class _VarSetRowManager:
    """
    Mixin providing all VarSet row management logic (populate, get/set
    values, debouncing, apply buttons). Subclasses must implement
    ``_add_row`` and ``_remove_row``, and may override
    ``_set_group_title`` and ``_set_group_description``.
    """

    def _init_varset(self, explicit_apply=False, debounce_ms=0):
        self.explicit_apply = explicit_apply
        self.debounce_ms = debounce_ms
        self.widget_map: Dict[str, Tuple[Adw.PreferencesRow, Var]] = {}
        self._adapters: Dict[str, RowAdapter] = {}
        self._created_rows = []
        self._apply_buttons = []
        self.data_changed = Signal()
        self._debounce_timer_id: Optional[int] = None
        self._pending_keys: set = set()

    def _add_row(self, row):
        raise NotImplementedError

    def _remove_row(self, row):
        raise NotImplementedError

    def _set_group_title(self, title):
        pass

    def _set_group_description(self, desc):
        pass

    def clear_dynamic_rows(self):
        """Removes only the rows dynamically created by populate()."""
        self._cancel_debounce()
        for row in self._created_rows:
            self._remove_row(row)
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
        if var_set.title:
            self._set_group_title(escape_title(var_set.title))
        if var_set.description:
            self._set_group_description(escape_title(var_set.description))

        new_keys = {var.key for var in var_set}
        existing_keys = list(self.widget_map.keys())

        for key in existing_keys:
            if key not in new_keys:
                row, _ = self.widget_map.pop(key)
                self._remove_row(row)
                if row in self._created_rows:
                    self._created_rows.remove(row)

        for var in var_set:
            if var.key in self.widget_map:
                row, old_var = self.widget_map[var.key]
                adapter = self._adapters.get(var.key)

                needs_rebuild = type(var) is not type(old_var)
                if not needs_rebuild and adapter is not None:
                    needs_rebuild = adapter.needs_rebuild(old_var, var)

                if needs_rebuild:
                    self._remove_row(row)
                    if row in self._created_rows:
                        self._created_rows.remove(row)
                    del self.widget_map[var.key]
                else:
                    self.widget_map[var.key] = (row, var)
                    adapter = self._adapters.get(var.key)
                    if adapter is not None:
                        adapter.update_from_var(var)
                    continue

            row, adapter = create_row_for_var(var, "value")
            if row:
                self._wire_up_row(row, var, adapter)
                self._add_row(row)
                self._created_rows.append(row)
                self.widget_map[var.key] = (row, var)
                if adapter is not None:
                    self._adapters[var.key] = adapter

    def get_values(self) -> Dict[str, Any]:
        values = {}
        for key in self.widget_map:
            adapter = self._adapters.get(key)
            if adapter is not None:
                values[key] = adapter.get_value()
            else:
                values[key] = None
        return values

    def set_values(self, values: Dict[str, Any]):
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

    def _wire_up_row(
        self,
        row: Adw.PreferencesRow,
        var: Var,
        adapter: Optional[RowAdapter],
    ):
        self._add_apply_button_if_needed(row, var.key)
        if adapter is not None:
            if not self.explicit_apply or adapter.has_natural_commit:
                adapter.changed.connect(
                    lambda sender: self._on_data_changed(var.key),
                    weak=False,
                )

    def set_apply_buttons_sensitive(self, sensitive: bool):
        for button in self._apply_buttons:
            button.set_sensitive(sensitive)


class VarSetWidget(Adw.PreferencesGroup, _VarSetRowManager):
    """
    A self-contained Adwaita Preferences Group that populates itself with
    rows based on a VarSet. Supports both immediate updates and explicit
    "Apply" buttons, with built-in debouncing for rapid value changes.
    """

    def __init__(self, explicit_apply=False, debounce_ms=0, **kwargs):
        Adw.PreferencesGroup.__init__(self, **kwargs)
        self._init_varset(explicit_apply, debounce_ms)

    def _add_row(self, row):
        self.add(row)

    def _remove_row(self, row):
        self.remove(row)

    def _set_group_title(self, title):
        self.set_title(title)

    def _set_group_description(self, desc):
        self.set_description(desc)


class VarSetRowList(Gtk.ListBox, _VarSetRowManager):
    """
    A Gtk.ListBox that populates itself with rows based on a VarSet.
    Intended for use inside Expander cards where Adw.PreferencesGroup
    styling would be visually inconsistent.
    """

    def __init__(self, explicit_apply=False, debounce_ms=0, **kwargs):
        Gtk.ListBox.__init__(self, **kwargs)
        self.set_selection_mode(Gtk.SelectionMode.NONE)
        self._init_varset(explicit_apply, debounce_ms)

    def _add_row(self, row):
        self.append(row)

    def _remove_row(self, row):
        self.remove(row)
