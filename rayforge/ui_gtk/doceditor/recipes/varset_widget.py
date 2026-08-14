"""Recipe-mode varset widget.

The generic varset machinery (``VarSetWidget``) has no awareness of
recipes. :class:`RecipeVarSetWidget` extends it via the hooks
``_should_emit_data_changed`` and ``_on_row_created`` to add per-row
apply toggles, dim-but-active visuals, ``apply_changed``, and
``setting_dicts`` I/O.
"""

from gettext import gettext as _
from typing import Any, Protocol, cast

from blinker import Signal
from gi.repository import Adw, Gtk

from ....core.varset import Var
from ...icons import get_icon
from ...varset.adapter import RowAdapter
from ...varset.varsetwidget import VarSetWidget


class _Prefixable(Protocol):
    """A preferences row that accepts native prefix widgets.

    PyGObject does not expose ``add_prefix`` on the bare
    :class:`Adw.PreferencesRow` base, only on its concrete row
    subclasses, so the recipe-mode toggle is attached through this
    protocol.
    """

    def add_prefix(self, widget: Gtk.Widget) -> None: ...


class RecipeVarSetWidget(VarSetWidget):
    """A :class:`VarSetWidget` with recipe-mode apply toggles.

    Each row gets a toggle button as a native prefix: toggle on = the
    recipe applies the setting, off = leave unchanged. While off,
    ``data_changed`` is suppressed and the row is dimmed (but stays
    interactive). Toggle state is announced via ``apply_changed``
    and can be read/written through :meth:`set_apply_state` /
    :meth:`get_apply_state`. Recipe entries are round-tripped via
    :meth:`set_setting_dicts` / :meth:`get_setting_dicts`.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._apply_toggles: dict[str, Gtk.ToggleButton] = {}
        self.apply_changed = Signal()

    def clear_dynamic_rows(self):
        super().clear_dynamic_rows()
        self._apply_toggles.clear()

    # --- Hook overrides --------------------------------------------

    def _should_emit_data_changed(self, key: str) -> bool:
        primary = self._related_to_primary.get(key, key)
        return self.get_apply_state(primary)

    def _on_row_created(
        self,
        row: Adw.PreferencesRow,
        var: Var,
        adapter: RowAdapter | None,
    ) -> None:
        self._build_apply_toggle(row, var.key, initial_apply=False)

    # --- Apply toggle ----------------------------------------------

    def set_apply_state(self, key: str, applied: bool):
        """Set the apply toggle state for a key.

        For composite adapters the toggle lives on the primary key's
        row; related keys share the same toggle.
        """
        primary = self._related_to_primary.get(key, key)
        toggle = self._apply_toggles.get(primary)
        if toggle is not None:
            toggle.set_active(applied)
        self._update_apply_visual(primary)

    def get_apply_state(self, key: str) -> bool:
        """Whether the recipe applies the setting for the given key."""
        primary = self._related_to_primary.get(key, key)
        toggle = self._apply_toggles.get(primary)
        return toggle.get_active() if toggle is not None else False

    def _build_apply_toggle(
        self, row: Adw.PreferencesRow, key: str, initial_apply: bool
    ) -> None:
        """Attach the recipe-mode apply toggle as a native prefix."""
        toggle = Gtk.ToggleButton()
        toggle.add_css_class("flat")
        toggle.set_valign(Gtk.Align.CENTER)
        toggle.set_active(initial_apply)
        toggle.set_tooltip_text(_("Apply this setting to the step"))
        toggle.connect("toggled", self._on_apply_toggled, key)
        cast(_Prefixable, row).add_prefix(toggle)
        self._apply_toggles[key] = toggle
        self._update_apply_visual(key)

    def _on_apply_toggled(self, toggle: Gtk.ToggleButton, key: str):
        applied = toggle.get_active()
        self.apply_changed.send(self, key=key, state=applied)
        self._update_apply_visual(key)
        if applied:
            primary = self._related_to_primary.get(key, key)
            adapter = self._adapters.get(primary)
            if adapter is not None:
                self._on_data_changed(primary)
            for rk in self._related_keys:
                if self._related_to_primary.get(rk) == primary:
                    self._on_data_changed(rk)

    def _update_apply_visual(self, key: str):
        """Dim the row while the apply toggle is off and swap the icon."""
        toggle = self._apply_toggles.get(key)
        if toggle is None:
            return
        applied = toggle.get_active()
        icon = get_icon("check-symbolic" if applied else "disabled-symbolic")
        icon.set_valign(Gtk.Align.CENTER)
        toggle.set_child(icon)
        row, _ = self.widget_map.get(key, (None, None))
        if row is not None:
            row.set_opacity(1.0 if applied else 0.5)

    # --- setting_dicts I/O -----------------------------------------

    def _all_keys(self) -> list[str]:
        """All keys managed by this widget, in insertion order."""
        keys = list(self.widget_map.keys())
        for key in self._related_keys:
            if key not in keys:
                keys.append(key)
        return keys

    def set_setting_dicts(self, setting_dicts: list[dict[str, Any]]):
        """Prefill row values and apply-toggle states from recipe entries.

        Each entry carries ``name``, ``value`` and ``recipe_apply``.
        Entries whose key has no row are ignored. ``None`` values are
        not pushed (matching :meth:`set_values`). Toggle states default
        to off for keys without an explicit entry.
        """
        apply_states: dict[str, bool] = {}
        values: dict[str, Any] = {}
        for d in setting_dicts:
            name = d.get("name")
            if name is None:
                continue
            apply_states[name] = bool(d.get("recipe_apply", False))
            if d.get("value") is not None:
                values[name] = d["value"]
        self.set_values(values)
        for key in list(self._apply_toggles.keys()):
            self.set_apply_state(key, apply_states.get(key, False))

    def get_setting_dicts(self) -> list[dict[str, Any]]:
        """Collect recipe entries from the current row state.

        Returns one dict per rendered key (including related keys):
        ``{"name": key, "value": value, "recipe_apply": bool}``.
        Keys with ``None`` values are included so the caller knows the
        row exists; the caller may drop them.
        """
        result: list[dict[str, Any]] = []
        values = self.get_values()
        for key in self._all_keys():
            result.append(
                {
                    "name": key,
                    "value": values.get(key),
                    "recipe_apply": self.get_apply_state(key),
                }
            )
        return result
