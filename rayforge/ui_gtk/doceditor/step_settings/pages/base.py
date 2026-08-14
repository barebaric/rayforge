"""Base class for a step's settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, ClassVar, cast

from gi.repository import Adw, GLib, Gtk

from .....core.undo.property_cmd import ChangePropertyCommand
from .....core.varset import VarSet
from .....shared.util.glib import DebounceMixin
from ....shared.pref_rows import SpeedSpinRow
from ....shared.preferences_page import TrackedPreferencesPage
from ....varset.adapter import escape_title
from ....varset.varsetwidget import VarSetWidget
from ..recipe_control_widget import RecipeControlWidget
from ..rows import StepRow

if TYPE_CHECKING:
    from .....doceditor.editor import DocEditor


def _to_widget(item: Any, editor: "DocEditor", step: Any) -> Gtk.Widget:
    if isinstance(item, type):
        item = item(editor, step)
    if isinstance(item, StepRow):
        return item.widget
    return item


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
        if self.show_identity:
            self._add_identity_section()
        # Keep varset rows in sync with the model (undo, recipe apply,
        # external edits).
        self.step.updated.connect(self._sync_widgets_to_model)

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
        self._sync_widgets_to_model()

    def get_machine(self):
        return getattr(self.editor.context, "machine", None)

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
        *rows: Any,
        description: str | None = None,
    ) -> Adw.PreferencesGroup:
        group = Adw.PreferencesGroup()
        if title:
            group.set_title(title)
        if description:
            group.set_description(description)
        for item in rows:
            if isinstance(item, type):
                item = item(self.editor, self.step)
            self._rows.append(item)
            group.add(_to_widget(item, self.editor, self.step))
        self.add(group)
        self._sections.append(group)
        return group

    def add_row(self, row: Any):
        if not self._sections:
            self.add_section(None)
        self._rows.append(row)
        self._sections[-1].add(_to_widget(row, self.editor, self.step))

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
        return widget

    def _varset_for_keys(self, var_set: VarSet, keys: set[str]) -> VarSet:
        """Subset of a varset holding only the given keys."""
        return VarSet(vars=[var for var in var_set if var.key in keys])

    def _on_varset_data_changed(self, widget: VarSetWidget, key: str):
        value = widget.get_values().get(key)
        self.set_step_property(key, value)

    def _sync_widgets_to_model(self, *args):
        for row in self._rows:
            resync = getattr(row, "resync", None)
            if callable(resync):
                resync()
        for widget, var_set in self._varset_widgets:
            values = {
                var.key: getattr(self.step, var.key, None) for var in var_set
            }
            widget.sync_from_model(values)
        self._update_machine_bounds()

    def _update_machine_bounds(self):
        """Sync speed-row bounds with the step's machine limits.

        The base speed vars resolve their upper bound from the active
        machine; the per-step ``max_cut_speed``/``max_travel_speed``
        may differ (e.g. after a head change), so push them into the
        rows on every model sync.
        """
        for widget, _var_set in self._varset_widgets:
            for key, attr in (
                ("cut_speed", "max_cut_speed"),
                ("travel_speed", "max_travel_speed"),
            ):
                row = widget.row_for(key)
                if row is None:
                    continue
                max_speed = getattr(self.step, attr, None)
                if max_speed:
                    cast(SpeedSpinRow, row).set_range(1.0, float(max_speed))

    def _cleanup(self):
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        for row in self._rows:
            cleanup = getattr(row, "cleanup", None)
            if callable(cleanup):
                cleanup()
        for widget, _var_set in self._varset_widgets:
            widget.cancel_pending()
