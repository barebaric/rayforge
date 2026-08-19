from gettext import gettext as _
from typing import Any

from blinker import Signal
from gi.repository import Adw, GLib, Gtk

from ...core.varset import Var, VarSet
from ..icons import get_icon
from .adapter import RowAdapter, create_row_for_var, escape_title

_DEBOUNCE_DELAY_MS = 300


class _VarSetRowManager:
    """
    Mixin providing all VarSet row management logic (populate, get/set
    values, debouncing, apply buttons). Subclasses must implement
    ``_add_row``, ``_remove_row``, ``_row_index``, and
    ``_insert_row_at``, and may override ``_set_group_title`` and
    ``_set_group_description``.
    """

    def _init_varset(
        self,
        explicit_apply=False,
        debounce_ms=0,
        show_reset=False,
        context_values=None,
    ):
        self.explicit_apply = explicit_apply
        self.debounce_ms = debounce_ms
        self.show_reset = show_reset
        self.widget_map: dict[str, tuple[Adw.PreferencesRow, Var]] = {}
        self._adapters: dict[str, RowAdapter] = {}
        self._created_rows = []
        self._apply_buttons = []
        self._reset_buttons = []
        self.data_changed = Signal()
        self._syncing = False
        self._debounce_timer_id: int | None = None
        self._pending_keys: set = set()
        #: Keys this widget has just committed and is currently
        #: delivering as ``data_changed``. The host's synchronous
        #: model update (``step.updated`` → ``sync_from_model``) is a
        #: round-trip of the widget's own edit; those keys must not be
        #: re-pushed into their rows or the in-progress edit is
        #: overwritten (e.g. clamping a typed value resets the cursor).
        self._committed_keys: set = set()
        #: Extra values visible to visible_when/sensitive_when that
        #: have no row in this widget (e.g. a key whose row lives in a
        #: sibling widget). Host pages push them via set_context_values.
        self._context_values: dict[str, Any] = dict(context_values or {})
        #: Keys managed by a composite adapter, not the primary var.
        #: Skipped during row creation so only one row is built per
        #: composite group, but included in get/set_values.
        self._related_keys: set[str] = set()
        #: Reverse map: related key -> primary key (for dispatch).
        self._related_to_primary: dict[str, str] = {}

    def _add_row(self, row):
        raise NotImplementedError

    def _remove_row(self, row):
        raise NotImplementedError

    def _row_index(self, row):
        raise NotImplementedError

    def _insert_row_at(self, row, index):
        raise NotImplementedError

    def _set_group_title(self, title):
        pass

    def _set_group_description(self, desc):
        pass

    def _should_emit_data_changed(self, key: str) -> bool:
        """Return False to suppress a ``data_changed`` emission.

        The base always emits. Subclasses override this to gate
        emissions on external state (e.g. an apply toggle).
        """
        return True

    def _on_row_created(
        self,
        row: Adw.PreferencesRow,
        var: Var,
        adapter: RowAdapter | None,
    ) -> None:
        """Called after a row is created and registered in widget_map.

        Subclasses override this to attach per-row decorations.
        """

    def clear_dynamic_rows(self):
        """Removes only the rows dynamically created by populate()."""
        self._cancel_debounce()
        self._committed_keys.clear()
        for row in self._created_rows:
            self._remove_row(row)
        self._created_rows.clear()
        self._apply_buttons.clear()
        self._reset_buttons.clear()
        self.widget_map.clear()
        self._adapters.clear()
        self._related_keys.clear()
        self._related_to_primary.clear()

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
                row, var = self.widget_map.pop(key)
                self._remove_row(row)
                if row in self._created_rows:
                    self._created_rows.remove(row)
                self._remove_extra_rows(key, var)
                self._adapters.pop(key, None)

        for var in var_set:
            if var.key in self._related_keys:
                continue

            if var.key in self.widget_map:
                row, old_var = self.widget_map[var.key]
                adapter = self._adapters.get(var.key)

                needs_rebuild = type(var) is not type(old_var)
                if not needs_rebuild and adapter is not None:
                    needs_rebuild = adapter.needs_rebuild(old_var, var)

                if needs_rebuild:
                    # Remember the row's position so the rebuilt row
                    # can be inserted at the same spot instead of being
                    # appended at the end.
                    insert_index = self._row_index(row)
                    self._remove_row(row)
                    if row in self._created_rows:
                        self._created_rows.remove(row)
                    del self.widget_map[var.key]
                    self._remove_extra_rows(var.key, old_var)
                    self._adapters.pop(var.key, None)
                else:
                    self.widget_map[var.key] = (row, var)
                    adapter = self._adapters.get(var.key)
                    if adapter is not None:
                        adapter.update_from_var(var)
                    continue
            else:
                insert_index = None

            row, adapter = create_row_for_var(var, "value")
            if row:
                self.widget_map[var.key] = (row, var)
                self._wire_up_row(row, var, adapter)
                if insert_index is not None:
                    self._insert_row_at(row, insert_index)
                else:
                    self._add_row(row)
                self._created_rows.append(row)
                self._add_extra_rows(var, adapter)
                if adapter is not None:
                    self._adapters[var.key] = adapter
                    if adapter.related_keys:
                        for rk in adapter.related_keys:
                            self._related_keys.add(rk)
                            self._related_to_primary[rk] = var.key

        self._update_visibility()

    def _add_extra_rows(self, var, adapter):
        if adapter is None:
            return
        for extra in adapter.extra_rows():
            self._add_row(extra)
            self._created_rows.append(extra)

    def _remove_extra_rows(self, key, var):
        """Remove rows that a composite adapter appended after its
        primary row (e.g. the second half of a min/max pair)."""
        adapter = self._adapters.get(key)
        if adapter is None:
            return
        for extra in adapter.extra_rows():
            self._remove_row(extra)
            if extra in self._created_rows:
                self._created_rows.remove(extra)

    def get_values(self) -> dict[str, Any]:
        values = {}
        for key in self.widget_map:
            adapter = self._adapters.get(key)
            if adapter is not None:
                values[key] = adapter.get_value_for_key(key)
            else:
                values[key] = None
        for key in self._related_keys:
            primary = self._related_to_primary.get(key)
            if primary is None:
                continue
            adapter = self._adapters.get(primary)
            if adapter is not None:
                values[key] = adapter.get_value_for_key(key)
        return values

    def set_values(self, values: dict[str, Any]):
        """Push values into the rows without emitting ``data_changed``.

        Programmatic value updates must not look like user edits, so
        the whole batch is applied under a syncing guard. Visibility is
        re-evaluated afterwards because predicates may depend on the
        pushed values.
        """
        self._syncing = True
        try:
            for key, value in values.items():
                if value is None:
                    continue
                if key in self._related_keys:
                    primary = self._related_to_primary.get(key)
                    if primary is None:
                        continue
                    adapter = self._adapters.get(primary)
                    if adapter is not None:
                        adapter.set_value_for_key(key, value)
                elif key in self.widget_map:
                    adapter = self._adapters.get(key)
                    if adapter is not None:
                        adapter.set_value(value)
        finally:
            self._syncing = False
        self._update_visibility()

    def sync_from_model(self, values: dict[str, Any]):
        """Update rows from the model, skipping keys with pending
        (debounced) user edits, then re-evaluate visibility.

        Used by host pages to resync the widget when the underlying
        object changes externally (e.g. undo). For an authoritative
        resync that overrides pending edits (e.g. recipe apply), call
        :meth:`cancel_pending` first and then :meth:`set_values`.
        """
        pending = set(self._pending_keys)
        pending |= set(self._committed_keys)
        filtered = {k: v for k, v in values.items() if k not in pending}
        self.set_values(filtered)

    def refresh(self):
        """Re-evaluate visibility and adapter value dependencies."""
        self._update_visibility()

    def row_for(self, key: str) -> Adw.PreferencesRow | None:
        """The row widget managing the given key, if any."""
        primary = self._related_to_primary.get(key, key)
        entry = self.widget_map.get(primary)
        if entry is None:
            return None
        return entry[0]

    def adapter_for(self, key: str) -> RowAdapter | None:
        """The adapter managing the given key, if any."""
        primary = self._related_to_primary.get(key, key)
        return self._adapters.get(primary)

    def keys(self) -> list[str]:
        """All keys managed by this widget, in insertion order."""
        keys = list(self.widget_map.keys())
        for key in self._related_keys:
            if key not in keys:
                keys.append(key)
        return keys

    def cancel_pending(self):
        """Cancel any scheduled debounced emissions without committing."""
        self._cancel_debounce()

    def flush_pending(self):
        """Immediately emit any pending (debounced) changes."""
        if self._debounce_timer_id is not None:
            GLib.source_remove(self._debounce_timer_id)
            self._debounce_timer_id = None
        self._flush_debounce()

    def _on_data_changed(self, key: str):
        if self._syncing:
            return
        if not self._should_emit_data_changed(key):
            self._update_visibility()
            return
        self._emit_data_changed(key)
        adapter = self._adapters.get(key)
        if adapter is not None:
            for rk in adapter.related_keys:
                if self._should_emit_data_changed(rk):
                    self._emit_data_changed(rk)
        self._update_visibility()

    def _emit_data_changed(self, key: str):
        if self.debounce_ms > 0:
            self._pending_keys.add(key)
            self._schedule_debounce()
        else:
            self._deliver_data_changed(key)

    def _deliver_data_changed(self, key: str):
        """Emit ``data_changed`` for ``key`` while marking it committed.

        The host answers synchronously by pushing the new value into the
        model, which round-trips back through :meth:`sync_from_model`.
        Marking the key as committed during delivery keeps that
        round-trip from rewriting the row the user is still editing.
        """
        self._committed_keys.add(key)
        try:
            self.data_changed.send(self, key=key)
        finally:
            self._committed_keys.discard(key)

    def _update_visibility(self):
        """Re-evaluate ``visible_when``/``sensitive_when`` callbacks
        and adapter value dependencies.

        Called after populate and after each immediate (non-debounced)
        data_changed emission. Debounced emissions trigger it via
        ``_flush_debounce``. Context values (keys without a row here)
        are merged into the values dict.
        """
        values = self.get_values()
        if self._context_values:
            values = {**values, **self._context_values}
        for row, var in self.widget_map.values():
            if var.visible_when is not None:
                row.set_visible(var.visible_when(values))
            if var.sensitive_when is not None:
                row.set_sensitive(var.sensitive_when(values))
        for adapter in set(self._adapters.values()):
            adapter.update_from_values(values)
        # Composite adapters may render extra rows that must follow
        # the primary row's predicates.
        for key, adapter in self._adapters.items():
            if not adapter.extra_rows():
                continue
            var = self.widget_map[key][1]
            if var.visible_when is None and var.sensitive_when is None:
                continue
            for extra in adapter.extra_rows():
                if var.visible_when is not None:
                    extra.set_visible(var.visible_when(values))
                if var.sensitive_when is not None:
                    extra.set_sensitive(var.sensitive_when(values))

    def set_context_values(self, values: dict[str, Any]):
        """Provide values for keys that have no row in this widget.

        Host pages use this when a ``visible_when``/``sensitive_when``
        predicate depends on a key whose row lives in a sibling widget
        (e.g. the raster mode row controlling the power section).
        Visibility is re-evaluated immediately.
        """
        self._context_values.update(values)
        self._update_visibility()

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
            self._deliver_data_changed(key)
        self._update_visibility()

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

    def _add_reset_button_if_needed(self, row, var, adapter):
        if not self.show_reset:
            return
        reset_button = Gtk.Button(
            child=get_icon("undo-symbolic"),
            tooltip_text=_("Reset to Default"),
        )
        reset_button.add_css_class("flat")
        reset_button.set_valign(Gtk.Align.CENTER)
        reset_button.connect(
            "clicked",
            lambda b: (
                adapter.set_value(var.default)
                if var.default is not None
                else None
            ),
        )
        row.add_suffix(reset_button)
        self._reset_buttons.append(reset_button)

    def _wire_up_row(
        self,
        row: Adw.PreferencesRow,
        var: Var,
        adapter: RowAdapter | None,
    ):
        self._add_apply_button_if_needed(row, var.key)
        self._add_reset_button_if_needed(row, var, adapter)
        if adapter is not None and (
            not self.explicit_apply or adapter.has_natural_commit
        ):
            adapter.changed.connect(
                lambda sender: self._on_data_changed(var.key),
                weak=False,
            )
        self._on_row_created(row, var, adapter)

    def set_apply_buttons_sensitive(self, sensitive: bool):
        for button in self._apply_buttons:
            button.set_sensitive(sensitive)


class VarSetWidget(Adw.PreferencesGroup, _VarSetRowManager):
    """
    A self-contained Adwaita Preferences Group that populates itself with
    rows based on a VarSet. Supports both immediate updates and explicit
    "Apply" buttons, with built-in debouncing for rapid value changes.
    """

    def __init__(
        self,
        explicit_apply=False,
        debounce_ms=0,
        show_reset=False,
        context_values=None,
        **kwargs,
    ):
        Adw.PreferencesGroup.__init__(self, **kwargs)
        self._init_varset(
            explicit_apply, debounce_ms, show_reset, context_values
        )

    def _add_row(self, row):
        self.add(row)

    def _remove_row(self, row):
        self.remove(row)

    def _row_index(self, row):
        return row.get_index()

    def _insert_row_at(self, row, index):
        # Adw.PreferencesGroup nests rows inside an internal Gtk.ListBox
        # that is not exposed as a direct child. Find it by searching
        # descendants and insert there.
        list_box = self._find_list_box(self)
        if list_box is not None:
            list_box.insert(row, index)
        else:
            self.add(row)

    @staticmethod
    def _find_list_box(widget):
        if isinstance(widget, Gtk.ListBox):
            return widget
        child = widget.get_first_child()
        while child is not None:
            result = VarSetWidget._find_list_box(child)
            if result is not None:
                return result
            child = child.get_next_sibling()
        return None

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

    def __init__(
        self,
        explicit_apply=False,
        debounce_ms=0,
        show_reset=False,
        context_values=None,
        **kwargs,
    ):
        Gtk.ListBox.__init__(self, **kwargs)
        self.set_selection_mode(Gtk.SelectionMode.NONE)
        self._init_varset(
            explicit_apply, debounce_ms, show_reset, context_values
        )

    def _add_row(self, row):
        self.append(row)

    def _remove_row(self, row):
        self.remove(row)

    def _row_index(self, row):
        return row.get_index()

    def _insert_row_at(self, row, index):
        self.insert(row, index)
