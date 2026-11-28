"""Dialog for configuring workflow-level settings."""

from gettext import gettext as _
from typing import TYPE_CHECKING, Optional

from gi.repository import Adw, Gdk, Gtk

from ...core.workflow import Workflow
from ..shared.patched_dialog_window import PatchedDialogWindow
from .workflow_post_processors import WorkflowPostProcessorGroup

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


class WorkflowSettingsDialog(PatchedDialogWindow):
    """Configures a workflow's name, color and post processors."""

    def __init__(
        self,
        workflow: Workflow,
        transient_for: Gtk.Window | None,
        editor: Optional["DocEditor"] = None,
        **kwargs,
    ):
        super().__init__(transient_for=transient_for, **kwargs)
        self.workflow = workflow
        self.editor = editor
        self._is_initializing = True

        self.set_title(_("{name} - Settings").format(name=workflow.name))
        self.set_default_size(600, -1)
        self.set_modal(False)

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        main_box.append(header)
        close_button = Gtk.Button(label=_("Close"))
        close_button.add_css_class("suggested-action")
        close_button.connect("clicked", lambda w: self.close())
        header.pack_end(close_button)

        content = Adw.PreferencesPage()
        main_box.append(content)

        general_group = Adw.PreferencesGroup(
            title=_("General"),
            description=_(
                "Basic workflow settings such as its name and color."
            ),
        )
        content.add(general_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        self.name_row.set_text(workflow.name)
        self.name_row.connect("changed", self._on_name_changed)
        general_group.add(self.name_row)

        color_dialog = Gtk.ColorDialog()
        color_dialog.set_with_alpha(False)
        self.color_button = Gtk.ColorDialogButton(dialog=color_dialog)
        self.color_button.set_size_request(45, 45)
        self.color_button.set_valign(Gtk.Align.CENTER)
        rgba = Gdk.RGBA()
        rgba.parse(workflow.color)
        self.color_button.set_rgba(rgba)
        self.color_button.connect("notify::rgba", self._on_color_changed)

        color_row = Adw.ActionRow(
            title=_("Workflow Color"),
            subtitle=_("Color used for this workflow's operations"),
        )
        color_row.add_suffix(self.color_button)
        general_group.add(color_row)

        self.enabled_row = Adw.SwitchRow(
            title=_("Enabled"),
            subtitle=_("Include this workflow when running the job"),
        )
        self.enabled_row.set_active(workflow.enabled)
        self.enabled_row.connect("notify::active", self._on_enabled_changed)
        general_group.add(self.enabled_row)

        self._is_initializing = False

        self.post_processor_group = WorkflowPostProcessorGroup(
            workflow, editor
        )
        content.add(self.post_processor_group)

    def _on_name_changed(self, row: Adw.EntryRow):
        if self._is_initializing:
            return
        new_name = row.get_text().strip()
        if not new_name or new_name == self.workflow.name:
            return
        if self.editor is not None:
            self.editor.workflow.rename_workflow(self.workflow, new_name)
        else:
            self.workflow.name = new_name
        self.set_title(_("{name} - Settings").format(name=new_name))

    def _on_color_changed(self, button: Gtk.ColorDialogButton, _param):
        if self._is_initializing:
            return
        rgba = button.get_rgba()
        r = round(rgba.red * 255)
        g = round(rgba.green * 255)
        b = round(rgba.blue * 255)
        hex_color = f"#{r:02x}{g:02x}{b:02x}"
        if hex_color == self.workflow.color:
            return
        if self.editor is not None:
            self.editor.workflow.set_workflow_color(self.workflow, hex_color)
        else:
            self.workflow.set_color(hex_color)

    def _on_enabled_changed(self, row: Adw.SwitchRow, _param):
        if self._is_initializing:
            return
        enabled = row.get_active()
        if enabled == self.workflow.enabled:
            return
        if self.editor is not None:
            self.editor.workflow.set_workflow_enabled(self.workflow, enabled)
        else:
            self.workflow.set_enabled(enabled)
