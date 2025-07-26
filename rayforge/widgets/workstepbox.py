from gi.repository import Gtk  # type: ignore
from blinker import Signal
from ..models.doc import Doc
from ..models.workplan import WorkStep
from ..util.resources import get_icon_path
from ..undo.property_cmd import ChangePropertyCommand
from .groupbox import GroupBox
from .workstepsettings import WorkStepSettingsDialog


class WorkStepBox(GroupBox):
    def __init__(self, doc: Doc, workstep: WorkStep, prefix=""):
        super().__init__(workstep.name, workstep.get_summary())
        self.doc = doc
        self.workstep = workstep
        self.prefix = prefix
        self.delete_clicked = Signal()

        self.visibility_on_icon = Gtk.Image.new_from_file(
            get_icon_path("visibility-on")
        )
        self.visibility_off_icon = Gtk.Image.new_from_file(
            get_icon_path("visibility-off")
        )

        # Store the button as an instance attribute to update it on undo/redo
        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_active(workstep.visible)
        self.add_button(self.visibility_button)
        self.visibility_button.connect("clicked", self.on_button_view_click)

        icon = Gtk.Image.new_from_file(get_icon_path("settings"))
        button = Gtk.Button()
        button.set_child(icon)
        self.add_button(button)
        button.connect("clicked", self.on_button_properties_clicked)

        icon = Gtk.Image.new_from_file(get_icon_path("delete"))
        button = Gtk.Button()
        button.set_child(icon)
        self.add_button(button)
        button.connect("clicked", self.on_button_delete_clicked)

        # Connect to the model's changed signal to keep the UI in sync
        self.workstep.changed.connect(self.on_workstep_changed)
        self.on_workstep_changed(self.workstep)  # trigger initial UI update

    def set_prefix(self, prefix):
        self.prefix = prefix

    def on_workstep_changed(self, sender, **kwargs):
        # Update title and subtitle
        self.title_label.set_label(f"{self.prefix}{self.workstep.name}")
        self.subtitle_label.set_label(self.workstep.get_summary())

        # Sync the visibility button's state and icon with the model.
        # This is crucial for undo/redo to update the UI correctly.
        is_visible = self.workstep.visible
        self.visibility_button.set_active(is_visible)
        if is_visible:
            self.visibility_button.set_child(self.visibility_on_icon)
        else:
            self.visibility_button.set_child(self.visibility_off_icon)

    def on_button_view_click(self, button):
        new_visibility = button.get_active()

        # Only create a command if the user's action changes the state.
        # This prevents command creation when the UI is updated
        # programmatically.
        if new_visibility == self.workstep.visible:
            return

        command = ChangePropertyCommand(
            target=self.workstep,
            property_name="visible",
            new_value=new_visibility,
            setter_method_name="set_visible",
            name=_("Toggle workstep visibility"),
        )
        self.doc.history_manager.execute(command)

    def on_button_properties_clicked(self, button):
        # The dialog now needs the doc object to access the history manager
        dialog = WorkStepSettingsDialog(self.doc, self.workstep)
        dialog.present(self)
        dialog.changed.connect(self.on_workstep_changed)

    def on_button_delete_clicked(self, button):
        self.delete_clicked.send(self, workstep=self.workstep)
