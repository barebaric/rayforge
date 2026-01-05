from typing import cast, Optional
from gi.repository import Adw, Gtk
from ...context import get_context
from ...machine.models.machine import Machine
from ...machine.models.profile import MachineProfile
from .settings_dialog import MachineSettingsDialog
from .profile_selector import MachineProfileSelectorDialog
from ..shared.round_button import RoundButton
from ..icons import get_icon


class MachineSettingsPage(Adw.PreferencesPage):
    """A settings page for adding, removing, and managing machines."""

    def __init__(self, **kwargs):
        """Initializes the Machine Settings page."""
        super().__init__(**kwargs)
        self.set_title(_("Machines"))
        self.set_icon_name("drive-harddisk-symbolic")

        self.machines_group = Adw.PreferencesGroup()
        self.machines_group.set_title(_("Configured Machines"))
        self.machines_group.set_description(_("Add or remove machines."))
        self.add(self.machines_group)

        # This listbox will contain the machine rows.
        self.machine_list_box = Gtk.ListBox()
        self.machine_list_box.set_selection_mode(Gtk.SelectionMode.NONE)
        self.machine_list_box.get_style_context().add_class("boxed-list")
        self.machines_group.add(self.machine_list_box)

        self._populate_machines_list()

        # Add button
        add_button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        add_button_box.set_halign(Gtk.Align.CENTER)
        add_button = RoundButton(label=_("+"))
        add_button_box.append(add_button)
        self.machines_group.add(add_button_box)

        # Signals
        context = get_context()
        add_button.connect("clicked", self._on_add_machine_clicked)
        context.machine_mgr.machine_added.connect(
            self._on_machine_list_changed
        )
        context.machine_mgr.machine_removed.connect(
            self._on_machine_list_changed
        )
        context.machine_mgr.machine_updated.connect(
            self._on_machine_list_changed
        )
        context.config.changed.connect(self._on_machine_list_changed)

    def _populate_machines_list(self):
        """Clears and rebuilds the rows within the ListBox."""
        context = get_context()
        machine_mgr = context.machine_mgr
        config = context.config

        while child := self.machine_list_box.get_row_at_index(0):
            self.machine_list_box.remove(child)

        sorted_machines = sorted(
            machine_mgr.machines.values(), key=lambda m: m.name.lower()
        )
        active_machine_id = config.machine.id if config.machine else None

        for machine in sorted_machines:
            row = Adw.ActionRow(title=machine.name)

            # Use a box to hold an icon, ensuring consistent row alignment.
            icon_placeholder = Gtk.Box()
            icon_placeholder.set_valign(Gtk.Align.CENTER)
            icon_placeholder.set_size_request(24, -1)
            row.add_prefix(icon_placeholder)

            is_valid, error_msg = machine.validate_driver_setup()

            if not is_valid:
                icon = get_icon("warning-symbolic")
                icon.get_style_context().add_class("warning")
                tooltip = error_msg or _(
                    "This machine has an invalid configuration."
                )
                icon.set_tooltip_text(tooltip)
                icon_placeholder.append(icon)
                row.set_subtitle(tooltip)
            elif machine.id == active_machine_id:
                icon = get_icon("check-circle-symbolic")
                icon.set_tooltip_text(_("This is the active machine."))
                icon_placeholder.append(icon)
                row.set_subtitle(machine.id)
            else:
                row.set_subtitle(machine.id)

            buttons_box = Gtk.Box(spacing=6)
            row.add_suffix(buttons_box)

            edit_button = Gtk.Button(
                child=get_icon("edit-symbolic"),
                valign=Gtk.Align.CENTER,
            )
            edit_button.connect(
                "clicked", self._on_edit_machine_clicked, machine
            )
            buttons_box.append(edit_button)

            delete_button = Gtk.Button(
                child=get_icon("delete-symbolic"),
                valign=Gtk.Align.CENTER,
            )
            delete_button.get_style_context().add_class("destructive-action")
            delete_button.connect(
                "clicked", self._on_delete_machine_clicked, machine
            )
            buttons_box.append(delete_button)

            self.machine_list_box.append(row)

    def _on_machine_list_changed(self, sender, **kwargs):
        """Handler to rebuild the list when machines change."""
        self._populate_machines_list()

    def _on_edit_machine_clicked(self, button, machine: Machine):
        """Opens the detailed settings dialog for a specific machine."""
        dialog = MachineSettingsDialog(
            machine=machine,
            transient_for=self.get_ancestor(Gtk.Window),
        )
        dialog.present()

    def _on_delete_machine_clicked(self, button, machine: Machine):
        """Shows a confirmation dialog before deleting a machine."""
        dialog = Adw.MessageDialog(
            transient_for=cast(
                Optional[Gtk.Window], self.get_ancestor(Gtk.Window)
            ),
            modal=True,
            heading=_("Delete ‘{name}’?").format(name=machine.name),
            body=_(
                "This machine profile and all its settings will be "
                "permanently removed. This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("cancel")

        dialog.connect("response", self._on_delete_confirm_response, machine)
        dialog.present()

    def _on_delete_confirm_response(
        self, dialog, response_id: str, machine: Machine
    ):
        """Handles the response from the delete confirmation dialog."""
        if response_id == "delete":
            get_context().machine_mgr.remove_machine(machine.id)
        dialog.close()

    def _on_add_machine_clicked(self, button):
        """Shows a dialog to select a machine profile to add."""
        dialog = MachineProfileSelectorDialog(
            transient_for=cast(
                Optional[Gtk.Window], self.get_ancestor(Gtk.Window)
            )
        )
        dialog.profile_selected.connect(self._on_profile_selected_for_add)
        dialog.present()

    def _on_profile_selected_for_add(self, sender, *, profile: MachineProfile):
        """Creates a machine and opens its settings editor."""
        machine_mgr = get_context().machine_mgr
        new_machine = profile.create_machine(get_context())
        machine_mgr.add_machine(new_machine)

        editor_dialog = MachineSettingsDialog(
            machine=new_machine,
            transient_for=self.get_ancestor(Gtk.Window),
        )
        editor_dialog.present()
