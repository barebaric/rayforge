from typing import cast, Dict, Tuple
from gi.repository import Gtk, Adw
from ..icons import get_icon
from ...machine.models.machine import Machine
from ...machine.models.macro import Macro, MacroTrigger
from .gcode_editor import GcodeEditorDialog


class HookList(Adw.PreferencesGroup):
    """
    An Adwaita widget for displaying and managing a static list of G-code
    hooks.
    """

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(title=_("G-code Hooks"), **kwargs)
        self.machine = machine
        # Store references to the rows and their widgets to update them later
        self.trigger_widgets: Dict[
            MacroTrigger, Tuple[Adw.ActionRow, Gtk.Button, Gtk.Switch]
        ] = {}
        self._setup_ui()
        # Connect to machine changes to update row state if a macro is
        # added/removed, or if the dialect changes.
        self.machine.changed.connect(self._on_machine_changed)

    def _setup_ui(self):
        """Builds the user interface for the editor."""
        self.set_description(
            _(
                "Add custom G-code to be executed "
                "at specific points in the job."
            )
        )

        for trigger in MacroTrigger:
            row = Adw.ActionRow()
            row.set_title(trigger.name.replace("_", " ").title())

            # Switch to enable/disable
            switch = Gtk.Switch(valign=Gtk.Align.CENTER)
            switch.connect("notify::active", self._on_enable_toggled, trigger)
            row.add_suffix(switch)

            # "Reset to Default" button
            reset_button = Gtk.Button(valign=Gtk.Align.CENTER)
            reset_button.set_child(get_icon("undo-symbolic"))
            reset_button.add_css_class("flat")
            reset_button.set_tooltip_text(_("Reset to Default"))
            reset_button.connect("clicked", self._on_reset_clicked, trigger)
            row.add_suffix(reset_button)

            # "Edit" button
            edit_button = Gtk.Button(valign=Gtk.Align.CENTER)
            edit_button.set_child(get_icon("edit-symbolic"))
            edit_button.add_css_class("flat")
            edit_button.connect("clicked", self._on_edit_clicked, trigger)
            row.add_suffix(edit_button)
            row.set_activatable_widget(edit_button)

            self.add(row)

            # Store references to the created widgets, keyed by trigger
            self.trigger_widgets[trigger] = (row, reset_button, switch)
            self._update_row_state(trigger)

    def _update_row_state(self, trigger: MacroTrigger):
        """Sets the row's subtitle and widget visibility."""
        row, reset_button, switch = self.trigger_widgets[trigger]

        row.set_subtitle(trigger.value)

        # A hook is considered "customized" if its key exists in the
        # dictionary,
        # regardless of whether the code is empty or not.
        is_customized = trigger in self.machine.hookmacros

        reset_button.set_visible(is_customized)
        switch.set_visible(is_customized)

        macro = self.machine.hookmacros.get(trigger)
        if macro:
            switch.set_active(macro.enabled)

    def _on_machine_changed(self, sender: Machine, **kwargs):
        """Update all row styles when the machine model changes."""
        for trigger in self.trigger_widgets.keys():
            self._update_row_state(trigger)

    def _on_enable_toggled(self, switch: Gtk.Switch, _, trigger: MacroTrigger):
        """Handles the state change of the enable/disable switch for a hook."""
        macro = self.machine.hookmacros.get(trigger)
        if macro:
            is_active = switch.get_active()
            if macro.enabled != is_active:
                macro.enabled = is_active
                self.machine.changed.send(self.machine)

    def _on_reset_clicked(self, button: Gtk.Button, trigger: MacroTrigger):
        """Shows a confirmation dialog before resetting a hook macro."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        hook_name = trigger.name.replace("_", " ").title()

        dialog = Adw.MessageDialog(
            transient_for=parent,
            heading=_("Reset '{hook_name}' to Default?").format(
                hook_name=hook_name
            ),
            body=_(
                "This will remove your custom G-code for this hook. "
                "The machine will revert to using its built-in default "
                "macro. This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("reset", _("Reset"))
        dialog.set_response_appearance(
            "reset", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")
        dialog.connect("response", self._on_reset_response, trigger)
        dialog.present()

    def _on_reset_response(
        self,
        dialog: Adw.MessageDialog,
        response_id: str,
        trigger: MacroTrigger,
    ):
        """Handles the response from the reset confirmation dialog."""
        if response_id != "reset":
            return

        if trigger in self.machine.hookmacros:
            del self.machine.hookmacros[trigger]
            self.machine.changed.send(self.machine)

    def _on_edit_clicked(self, button: Gtk.Button, trigger: MacroTrigger):
        """Handles the 'Edit' button click for a specific trigger."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))

        # If a macro already exists (even if empty), edit it directly.
        # Otherwise, create a new one.
        existing_macro = self.machine.hookmacros.get(trigger)
        if existing_macro is not None:
            macro_to_edit = existing_macro
        else:
            macro_to_edit = Macro(
                name=trigger.name.replace("_", " ").title(),
                code=[_("# Your G-code here")],
            )

        # Pass the list of available macros for the include popover
        existing_macros = list(self.machine.macros.values())

        editor_dialog = GcodeEditorDialog(
            parent,
            macro_to_edit,
            allow_name_edit=False,
            existing_macros=existing_macros,
        )
        editor_dialog.connect(
            "close-request",
            self._on_edit_dialog_closed,
            trigger,
            macro_to_edit,
        )
        editor_dialog.present()

    def _on_edit_dialog_closed(
        self, dialog: GcodeEditorDialog, trigger: MacroTrigger, macro: Macro
    ):
        """
        Handles closing the editor. If saved, updates the macro in the
        machine model's hook dictionary.
        """
        if dialog.saved:
            # Always save the macro, even if its code is empty.
            # This correctly represents the user's intent.
            self.machine.hookmacros[trigger] = macro
            self.machine.changed.send(self.machine)
