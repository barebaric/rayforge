from gi.repository import Gtk, Adw  # type: ignore
from ..models.machine import Camera
from .cameraproperties import CameraProperties
from .cameraselectiondialog import CameraSelectionDialog


class CameraPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Camera"), icon_name="camera-photo-symbolic", **kwargs
        )
        self.machine = machine

        # List of Cameras
        camera_list_group = Adw.PreferencesGroup(title=_("Cameras"))
        self.add(camera_list_group)
        self.camera_list = Gtk.ListBox()
        self.camera_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.camera_list.set_show_separators(True)
        camera_list_group.add(self.camera_list)

        # Add and Remove buttons for cameras
        camera_button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=5,
            halign=Gtk.Align.END,
        )
        add_camera_button = Gtk.Button(icon_name="list-add-symbolic")
        add_camera_button.connect("clicked", self.on_add_camera)
        remove_camera_button = Gtk.Button(icon_name="list-remove-symbolic")
        remove_camera_button.connect("clicked", self.on_remove_camera)
        camera_button_box.append(add_camera_button)
        camera_button_box.append(remove_camera_button)
        camera_list_group.add(camera_button_box)

        # Configuration panel for the selected Camera
        self.camera_properties_widget = CameraProperties(None)
        self.add(self.camera_properties_widget)

        # Connect signals for cameras
        self.camera_list.connect("row-selected", self.on_camera_selected)

        # Populate the list with existing Cameras
        self.populate_camera_list()

    def populate_camera_list(self):
        """Populate the list of Cameras."""
        for camera in self.machine.cameras:
            row = Adw.ActionRow(
                title=_("Camera: {name}").format(name=camera.name)
            )
            row.set_margin_top(5)
            row.set_margin_bottom(5)
            self.camera_list.append(row)
        row = self.camera_list.get_row_at_index(0)
        if row:
            self.camera_list.select_row(row)

    def on_add_camera(self, button):
        """Add a new Camera to the machine."""
        dialog = CameraSelectionDialog(self.get_ancestor(Gtk.Window))
        dialog.present()
        dialog.connect("response", self.on_camera_selection_dialog_response)

    def on_camera_selection_dialog_response(self, dialog, response_id):
        if response_id == "select":
            device_id = dialog.selected_device_id
            if device_id:
                # Check if a camera with this device_id already exists
                if any(c.device_id == device_id for c in self.machine.cameras):
                    # Optionally, show a message to the user that camera
                    # exists
                    return

                new_camera = Camera(
                    _("Camera {device_id}").format(device_id=device_id),
                    device_id,
                )
                new_camera.enabled = True
                self.machine.add_camera(new_camera)
                row = Adw.ActionRow(
                    title=_("Camera: {camera_name}").format(
                        camera_name=new_camera.name
                    )
                )
                row.set_margin_top(5)
                row.set_margin_bottom(5)
                self.camera_list.append(row)
                self.camera_list.select_row(row)
        dialog.destroy()

    def on_remove_camera(self, button):
        """Remove the selected Camera from the machine."""
        selected_row = self.camera_list.get_selected_row()
        if selected_row:
            index = selected_row.get_index()
            camera = self.machine.cameras[index]
            camera.enabled = False
            self.machine.remove_camera(camera)
            self.camera_list.remove(selected_row)
            self.camera_properties_widget.set_camera(None)

    def on_camera_selected(self, listbox, row):
        """Update the configuration panel when a Camera is selected."""
        if row is not None:
            index = row.get_index()
            selected_camera = self.machine.cameras[index]
            self.camera_properties_widget.set_camera(selected_camera)
        else:
            self.camera_properties_widget.set_camera(None)
