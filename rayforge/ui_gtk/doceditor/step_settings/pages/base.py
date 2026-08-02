"""Base class for a step's settings page."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Any, List, Optional

from gi.repository import Adw, GLib, Gtk

from rayforge.core.undo.property_cmd import ChangePropertyCommand
from rayforge.shared.util.glib import DebounceMixin
from rayforge.ui_gtk.doceditor.recipe_control_widget import RecipeControlWidget
from rayforge.ui_gtk.doceditor.step_settings.rows import StepRow
from rayforge.ui_gtk.shared.preferences_page import TrackedPreferencesPage

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


def _to_widget(item: Any, editor: "DocEditor", step: Any) -> Gtk.Widget:
    if isinstance(item, type):
        item = item(editor, step)
    if isinstance(item, StepRow):
        return item.widget
    return item


class StepSettingsPage(DebounceMixin, TrackedPreferencesPage):
    """Base class for a step type's settings page.

    Subclasses compose row widgets into titled sections via
    ``add_section``. The page normally starts with a section holding
    the step name and the recipe control; set ``show_identity`` to
    False to omit it (for auxiliary pages).
    """

    show_identity = True

    def __init__(self, editor: "DocEditor", step: Any):
        super().__init__()
        self.editor = editor
        self.step = step
        self.doc = editor.doc
        self.history_manager = editor.doc.history_manager
        producer_type = step.ASSEMBLER_NAME or "unknown"
        self.key = f"{producer_type.lower()}/step-settings"
        self.path_prefix = "/step-settings/"
        self._sections: List[Adw.PreferencesGroup] = []
        self._rows: List[Any] = []
        if self.show_identity:
            self._add_identity_section()

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
        pass

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
        name: Optional[str] = None,
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
        title: Optional[str],
        *rows: Any,
        description: Optional[str] = None,
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

    def _sync_widgets_to_model(self, *args):
        pass

    def _cleanup(self):
        if self._debounce_timer > 0:
            GLib.source_remove(self._debounce_timer)
            self._debounce_timer = 0
        for row in self._rows:
            cleanup = getattr(row, "cleanup", None)
            if callable(cleanup):
                cleanup()
