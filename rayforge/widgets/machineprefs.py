from gi.repository import Adw, Gtk  # type: ignore

from ..config import machine_mgr
from ..models.machine import Machine
from ..models.machineprofile import PROFILES
from .machinesettings import MachineSettingsDialog
from .roundbutton import RoundButton


class MachinePreferencesPage(Adw.PreferencesPage):
    """A preferences page for adding, removing, and managing machines."""

    def __init__(self, **kwargs):
        """Initializes the Machine Preferences page."""
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
        add_button.connect("clicked", self._on_add_machine_clicked)
        machine_mgr.machine_added.connect(self._on_machine_list_changed)
        machine_mgr.machine_removed.connect(self._on_machine_list_changed)
        machine_mgr.machine_updated.connect(self._on_machine_list_changed)

    def _populate_machines_list(self):
        """Clears and rebuilds the rows within the ListBox."""
        while child := self.machine_list_box.get_row_at_index(0):
            self.machine_list_box.remove(child)

        sorted_machines = sorted(
            machine_mgr.machines.values(), key=lambda m: m.name.lower()
        )

        for machine in sorted_machines:
            row = Adw.ActionRow(title=machine.name, subtitle=machine.id)

            buttons_box = Gtk.Box(spacing=6)

            edit_button = Gtk.Button(
                icon_name="document-edit-symbolic",
                valign=Gtk.Align.CENTER,
            )
            edit_button.connect(
                "clicked", self._on_edit_machine_clicked, machine
            )
            buttons_box.append(edit_button)

            delete_button = Gtk.Button(
                icon_name="edit-delete-symbolic",
                valign=Gtk.Align.CENTER,
            )
            delete_button.get_style_context().add_class("destructive-action")
            delete_button.connect(
                "clicked", self._on_delete_machine_clicked, machine
            )
            buttons_box.append(delete_button)

            row.add_suffix(buttons_box)
            self.machine_list_box.append(row)

    def _on_machine_list_changed(self, sender, machine_id, **kwargs):
        """Handler to rebuild the list when machines are added or removed."""
        self._populate_machines_list()

    def _on_edit_machine_clicked(self, button, machine: Machine):
        """Opens the detailed settings dialog for a specific machine."""
        dialog = MachineSettingsDialog(machine=machine)
        dialog.present(self)

    def _on_delete_machine_clicked(self, button, machine: Machine):
        """Shows a confirmation dialog before deleting a machine."""
        dialog = Adw.MessageDialog(
            transient_for=self.get_root(),
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
            machine_mgr.remove_machine(machine.id)
        dialog.close()

    def _on_add_machine_clicked(self, button):
        """Shows a dialog to select a machine profile to add."""
        dialog = Adw.MessageDialog(
            transient_for=self.get_root(),
            modal=True,
            heading=_("Add a New Machine"),
            body=_("Select a machine profile to use as a template."),
        )

        combo = Gtk.ComboBoxText()
        for i, profile in enumerate(PROFILES):
            combo.insert(i, str(i), profile.name)
        combo.set_active(0)

        dialog.set_extra_child(combo)
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("add", _("Add"))
        dialog.set_response_appearance("add", Adw.ResponseAppearance.SUGGESTED)
        dialog.set_default_response("add")

        dialog.connect("response", self._on_add_dialog_response)
        dialog.present()

    def _on_add_dialog_response(self, dialog, response_id: str):
        """Handles the response from the add machine dialog."""
        if response_id == "add":
            combo = dialog.get_extra_child()
            active_id_str = combo.get_active_id()
            if active_id_str:
                profile_index = int(active_id_str)
                selected_profile = PROFILES[profile_index]

                new_machine = selected_profile.create_machine()
                machine_mgr.add_machine(new_machine)
        dialog.close()
