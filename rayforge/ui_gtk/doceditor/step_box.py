from typing import TYPE_CHECKING
from gi.repository import Gtk, Adw
from blinker import Signal
from ...core.step import Step
from ..icons import get_icon
from ...core.undo.property_cmd import ChangePropertyCommand
from ...shared.units.formatter import format_value
from ...context import get_context
from .step_settings_dialog import StepSettingsDialog

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor


class StepBox(Adw.ActionRow):
    def __init__(
        self,
        editor: "DocEditor",
        step: Step,
        prefix: str = "",
    ):
        super().__init__()
        self.set_margin_start(0)
        self.set_margin_end(0)
        self.editor = editor
        self.doc = editor.doc
        self.step = step
        self.prefix = prefix
        self.delete_clicked = Signal()

        # Store the switch as an instance attribute to update it on undo/redo
        self.visibility_switch = Gtk.Switch()
        self.visibility_switch.set_active(step.visible)
        self.visibility_switch.set_valign(Gtk.Align.CENTER)
        self.add_suffix(self.visibility_switch)
        self.visibility_switch.connect("state-set", self.on_switch_state_set)

        button = Gtk.Button()
        button.set_child(get_icon("settings-symbolic"))
        button.set_valign(Gtk.Align.CENTER)
        self.add_suffix(button)
        button.connect("clicked", self.on_button_properties_clicked)

        button = Gtk.Button()
        button.set_child(get_icon("delete-symbolic"))
        button.set_valign(Gtk.Align.CENTER)
        self.add_suffix(button)
        button.connect("clicked", self.on_button_delete_clicked)

        # Connect to the model's signals to keep the UI in sync
        self.step.updated.connect(self.on_step_changed)
        self.step.visibility_changed.connect(self.on_step_changed)
        get_context().config.changed.connect(self.on_step_changed)
        self.on_step_changed(self.step)  # trigger initial UI update

    def do_destroy(self):
        """Overrides GObject.Object.do_destroy to disconnect signals."""
        self.step.updated.disconnect(self.on_step_changed)
        self.step.visibility_changed.disconnect(self.on_step_changed)
        get_context().config.changed.disconnect(self.on_step_changed)

    def set_prefix(self, prefix):
        self.prefix = prefix

    def on_step_changed(self, sender, **kwargs):
        # Update title and subtitle
        self.set_title(f"{self.prefix}{self.step.name}")

        power_percent = int(self.step.power * 100)
        formatted_speed = format_value(self.step.cut_speed, "speed")
        self.set_subtitle(f"{power_percent}% power, {formatted_speed}")

        # Sync the visibility switch's state with the model.
        is_visible = self.step.visible
        self.visibility_switch.set_active(is_visible)

    def on_switch_state_set(self, switch, state):
        command = ChangePropertyCommand(
            target=self.step,
            property_name="visible",
            new_value=state,
            setter_method_name="set_visible",
            name=_("Toggle step visibility"),
        )
        self.doc.history_manager.execute(command)

    def on_button_properties_clicked(self, button):
        parent_window = self.get_root()
        dialog = StepSettingsDialog(
            self.editor,
            self.step,
            transient_for=parent_window,
        )
        dialog.present()

    def on_button_delete_clicked(self, button):
        self.delete_clicked.send(self, step=self.step)
