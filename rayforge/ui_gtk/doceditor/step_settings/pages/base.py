"""Base class for a step's settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, ClassVar, cast

from gi.repository import Adw, GLib, Gtk

from .....core.undo.property_cmd import ChangePropertyCommand
from .....core.varset import VarSet
from .....shared.util.glib import DebounceMixin
from ....icons import get_icon
from ....shared.pref_rows import SpeedSpinRow
from ....shared.preferences_page import TrackedPreferencesPage
from ....varset.adapter import escape_title
from ....varset.varsetwidget import VarSetWidget
from ..recipe_control_widget import RecipeControlWidget

if TYPE_CHECKING:
    from .....doceditor.editor import DocEditor
    from .....machine.models.machine import Machine


class StepSettingsPage(DebounceMixin, TrackedPreferencesPage):
    """Base class for a step type's settings page.

    Subclasses compose settings into titled sections. Sections are
    normally rendered from the step's ``recipe_varset_groups()`` via
    :meth:`add_varset_section`; the varset machinery builds the rows
    and the page persists changes through :meth:`set_step_property`.
    The page starts with an identity section holding the step name and
    the recipe control; set ``show_identity`` to False to omit it (for
    auxiliary pages).
    """

    show_identity = True

    #: Declares extra settings pages as ``(method_name, title,
    #: icon_name)`` tuples. Each method returns a :class:`StepSettingsPage`
    #: that the step settings dialog adds as an additional tab.
    extra_pages: ClassVar[tuple[tuple[str, str, str], ...]] = ()

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__()
        self.editor = editor
        self.step = step
        self.doc = editor.doc
        self.history_manager = editor.doc.history_manager
        producer_type = step.ASSEMBLER_NAME or "unknown"
        self.key = f"{producer_type.lower()}/step-settings"
        self.path_prefix = "/step-settings/"
        self._sections: list[Adw.PreferencesGroup] = []
        self._rows: list[Any] = []
        self._varset_widgets: list[tuple[VarSetWidget, VarSet]] = []
        self._speed_warning_icons: dict[Gtk.Widget, Gtk.Image] = {}
        if self.show_identity:
            self._add_identity_section()
        # Keep varset rows in sync with the model (undo, recipe apply,
        # external edits).
        self.step.updated.connect(self._sync_widgets_to_model)
        # Rebuild machine-dependent rows (head dropdowns, speed
        # bounds) when the active machine changes.
        config = editor.context.config
        config.changed.connect(self._on_config_changed)

    def _add_identity_section(self):
        name_row = Adw.EntryRow(title=_("Name"))
        name_row.set_text(self.step.name)
        name_row.connect("changed", self._on_name_changed)
        self.recipe_control = RecipeControlWidget(self.editor, self.step)
        self.recipe_control.recipe_applied.connect(self._on_recipe_applied)
        self.add_section(
            _("General"),
            name_row,
            self.recipe_control,
            description=_("Step name and recipe settings."),
        )

    def _on_name_changed(self, row):
        new_name = row.get_text().strip()
        if not new_name or new_name == self.step.name:
            return
        self.editor.step.rename_step(self.step, new_name)

    def _on_recipe_applied(self, *args):
        # A recipe apply is authoritative: cancel pending (debounced)
        # edits and push the applied values into the rows.
        for widget, var_set in self._varset_widgets:
            widget.cancel_pending()
            values = {
                var.key: getattr(self.step, var.key, None) for var in var_set
            }
            widget.set_values(values)
        self._update_machine_bounds()

    def get_machine(self) -> "Machine | None":
        return self.editor.context.machine

    def get_selected_head(self):
        machine = self.get_machine()
        if machine is None:
            return None
        return self.step.get_selected_head(machine)

    def set_step_property(
        self,
        key: str,
        new_value: Any,
        name: str | None = None,
    ):
        current = getattr(self.step, key, None)
        if current == new_value:
            return

        def _notify():
            self.step.updated.send(self.step)

        setter_name = f"set_{key}"
        setter = getattr(self.step, setter_name, None)
        command = ChangePropertyCommand(
            target=self.step,
            property_name=key,
            new_value=new_value,
            setter_method_name=setter_name if setter else None,
            name=name or _("Change {key}").format(key=key.replace("_", " ")),
            on_change_callback=None if setter else _notify,
        )
        self.history_manager.execute(command)

    def add_section(
        self,
        title: str | None,
        *widgets: Gtk.Widget,
        description: str | None = None,
    ) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup()
        if title:
            group.set_title(title)
        if description:
            group.set_description(description)
        for widget in widgets:
            self._rows.append(widget)
            group.add(widget)
        self.add(group)
        self._sections.append(group)
        return group

    def add_group(self, group: Adw.PreferencesGroup):
        self.add(group)
        self._sections.append(group)

    def add_varset_section(
        self,
        title: str | None,
        var_set: VarSet,
        description: str | None = None,
        widget_cls: type[VarSetWidget] = VarSetWidget,
    ) -> VarSetWidget:
        """Render a varset group as a preferences section wired to the
        step.

        Populates a :class:`VarSetWidget` from the given ``var_set``
        and pushes the step's current values in. User changes flow
        back through ``data_changed`` → :meth:`_on_varset_data_changed`
        → :meth:`set_step_property`. Pass ``widget_cls`` to use a
        custom varset widget for the section.
        """
        widget = widget_cls(debounce_ms=300)
        if title:
            widget.set_title(escape_title(title))
        if description:
            widget.set_description(escape_title(description))
        widget.populate(var_set)
        widget.set_values(
            {var.key: getattr(self.step, var.key, None) for var in var_set}
        )
        widget.data_changed.connect(self._on_varset_data_changed)
        self.add(widget)
        self._sections.append(widget)
        self._varset_widgets.append((widget, var_set))
        self._update_speed_warnings()
        return widget

    def _varset_for_keys(
        self, var_set: VarSet, keys: set[str] | list[str]
    ) -> VarSet:
        """Subset of a varset holding only the given keys.

        When ``keys`` is a list, the returned vars preserve that order;
        when it is a set, the original varset order is kept.
        """
        if isinstance(keys, list):
            key_to_var = {var.key: var for var in var_set}
            return VarSet(
                vars=[key_to_var[k] for k in keys if k in key_to_var]
            )
        return VarSet(vars=[var for var in var_set if var.key in keys])

    def _on_varset_data_changed(self, widget: VarSetWidget, key: str):
        value = widget.get_values().get(key)
        self.set_step_property(key, value)

    def _sync_widgets_to_model(self, *args):
        """Resync rows from the model, preserving in-progress edits.

        Called on ``step.updated`` (undo, external edits). Recipe
        application overrides pending edits instead; see
        :meth:`_on_recipe_applied`.
        """
        for widget, var_set in self._varset_widgets:
            values = {
                var.key: getattr(self.step, var.key, None) for var in var_set
            }
            widget.sync_from_model(values)
        self._update_machine_bounds()

    def _on_config_changed(self, *args):
        """React to config changes (including active machine switches).

        Rebuilds every varset section so machine-dependent vars (e.g.
        head-selection dropdowns) pick up the new machine's heads, then
        calls :meth:`_on_machine_changed` for subclass-specific
        updates.
        """
        for i, (widget, old_var_set) in enumerate(self._varset_widgets):
            keys = {var.key for var in old_var_set}
            new_var_set = self._rebuild_varset(keys)
            if new_var_set is None:
                continue
            widget.populate(new_var_set)
            widget.set_values(
                {
                    var.key: getattr(self.step, var.key, None)
                    for var in new_var_set
                }
            )
            self._varset_widgets[i] = (widget, new_var_set)
        self._on_machine_changed()

    def _rebuild_varset(self, keys: set[str]) -> VarSet | None:
        """Re-derive a varset subset from the step's current recipe.

        Returns a :class:`VarSet` holding only the given ``keys``,
        re-instantiated from ``step.recipe_varset()`` so
        machine-dependent vars (head selection) reflect the active
        machine. Returns ``None`` when the step has no recipe varset.
        """
        full = self.step.recipe_varset()
        return VarSet(vars=[v for v in full if v.key in keys])

    def _on_machine_changed(self):
        """Hook called after the active machine changes.

        Subclasses override this to update machine-dependent visibility
        and derived state. The base re-syncs speed-row bounds.
        """
        self._update_machine_bounds()

    def _update_machine_bounds(self):
        """Sync speed-row bounds with the active machine's limits.

        The upper bound is the machine's ceiling, so user edits clamp to
        it (GTK clamps the entry to the adjustment's upper bound). A
        speed loaded from a project may still be shown above it: the
        spin row displays such a value via its text and flags it with a
        warning (see :meth:`_update_speed_warnings`) instead of clamping
        the loaded value. Called on model sync and machine change.
        """
        machine = self.get_machine()
        if machine is None:
            return
        supports_travel = machine.supports_travel_speed()
        for widget, _var_set in self._varset_widgets:
            row = widget.row_for("cut_speed")
            if row is not None:
                cast(SpeedSpinRow, row).set_range(
                    1.0, float(machine.max_cut_speed)
                )
            row = widget.row_for("travel_speed")
            if row is not None:
                cast(SpeedSpinRow, row).set_range(
                    1.0, float(machine.max_travel_speed)
                )
                row.set_visible(supports_travel)
        self._update_speed_warnings()

    def _update_speed_warnings(self):
        """Flag speed rows whose value exceeds the machine's ceiling.

        A step loaded from a project (or edited on a slower machine) may
        hold a speed above the active machine's maximum. Such rows show a
        warning icon that explains the problem on hover. The warning text
        is produced by the step via :meth:`~rayforge.core.step.Step.check`.
        """
        machine = self.get_machine()
        if machine is None:
            return
        warnings = self.step.check(machine)
        message = "\n".join(warnings) if warnings else ""
        supports_travel = machine.supports_travel_speed()
        for widget, _var_set in self._varset_widgets:
            row = widget.row_for("cut_speed")
            if row is not None:
                self._set_row_warning(
                    cast(SpeedSpinRow, row),
                    self.step.cut_speed > machine.max_cut_speed,
                    message,
                )
            row = widget.row_for("travel_speed")
            if row is not None:
                self._set_row_warning(
                    cast(SpeedSpinRow, row),
                    supports_travel
                    and self.step.travel_speed > machine.max_travel_speed,
                    message,
                )

    def _set_row_warning(
        self, row: Gtk.Widget, active: bool, message: str
    ) -> None:
        """Attach or remove a warning icon on any settings row.

        The icon explains the problem on hover via ``message``.
        """
        icon = self._speed_warning_icons.get(row)
        if icon is None:
            icon = get_icon("warning-symbolic")
            icon.add_css_class("warning")
            icon.set_valign(Gtk.Align.CENTER)
            if isinstance(row, SpeedSpinRow):
                # Place the icon to the LEFT of the input box (the
                # first child of the suffix box), so it is visually
                # tied to the value it flags rather than sitting at
                # the far right.
                suffix_box = row.get_spin_button().get_parent()
                cast(Gtk.Box, suffix_box).insert_child_after(icon, None)
            else:
                cast(Adw.ActionRow, row).add_suffix(icon)
            self._speed_warning_icons[row] = icon
        icon.set_visible(active)
        icon.set_tooltip_text(message if active else "")

    def _cleanup(self):
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        for widget, _var_set in self._varset_widgets:
            widget.cancel_pending()
        config = self.editor.context.config
        config.changed.disconnect(self._on_config_changed)
