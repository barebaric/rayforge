from gi.repository import Gtk, Adw  # type: ignore
from blinker import Signal
from typing import List
from ..models.camera import Camera
from .properties_widget import CameraProperties
from .selection_dialog import CameraSelectionDialog


class CameraPreferencesPage(Adw.PreferencesPage):

    camera_add_requested = Signal()
    """Signal emitted when a user requests to add a camera.
    Sends: sender, device_id (str)
    """
    camera_remove_requested = Signal()
    """Signal emitted when a user requests to remove a camera.
    Sends: sender, camera (Camera)
    """

    def __init__(self, **kwargs):
        super().__init__(
            title=_("Camera"), icon_name="camera-photo-symbolic", **kwargs
        )
        self._cameras: List[Camera] = []

        # List of Cameras
        camera_list_group = Adw.PreferencesGroup(
            title=_("Cameras"),
            description=_(
                "Stream a camera image directly onto the work surface."
            ),
        )
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

    def set_cameras(self, cameras: List[Camera]):
        """Sets the list of cameras to be displayed and refreshes the UI."""
        self._cameras = cameras
        self._populate_camera_list()

    def _populate_camera_list(self):
        """Populate the list of Cameras from the internal list."""
        selected_row = self.camera_list.get_selected_row()
        selected_index = selected_row.get_index() if selected_row else -1

        # Clear the listbox
        while child := self.camera_list.get_row_at_index(0):
            self.camera_list.remove(child)

        # Repopulate from the internal list
        for camera in self._cameras:
            row = Adw.ActionRow(
                title=_("Camera: {name}").format(name=camera.name)
            )
            row.set_margin_top(5)
            row.set_margin_bottom(5)
            self.camera_list.append(row)

        # Restore selection
        if 0 <= selected_index < len(self._cameras):
            row = self.camera_list.get_row_at_index(selected_index)
        elif self._cameras:
            # If previous selection is invalid, select the first item
            row = self.camera_list.get_row_at_index(0)
        else:
            row = None

        if row:
            self.camera_list.select_row(row)
        else:
            # Explicitly set properties to None if list is empty
            self.camera_properties_widget.set_camera(None)

    def on_add_camera(self, button):
        """Show a dialog to select a new camera device."""
        dialog = CameraSelectionDialog(self.get_ancestor(Gtk.Window))
        dialog.present()
        dialog.connect("response", self.on_camera_selection_dialog_response)

    def on_camera_selection_dialog_response(self, dialog, response_id):
        if response_id == "select":
            device_id = dialog.selected_device_id
            if device_id:
                # Check for duplicates in the current list
                if any(c.device_id == device_id for c in self._cameras):
                    return
                # Emit a signal to request the addition
                self.camera_add_requested.send(self, device_id=device_id)
        dialog.destroy()

    def on_remove_camera(self, button):
        """Emit a signal to request removal of the selected Camera."""
        selected_row = self.camera_list.get_selected_row()
        if selected_row:
            index = selected_row.get_index()
            camera_to_remove = self._cameras[index]
            self.camera_remove_requested.send(self, camera=camera_to_remove)

    def on_camera_selected(self, listbox, row):
        """Update the configuration panel when a Camera is selected."""
        if row is not None:
            index = row.get_index()
            selected_camera = self._cameras[index]
            self.camera_properties_widget.set_camera(selected_camera)
        else:
            self.camera_properties_widget.set_camera(None)
