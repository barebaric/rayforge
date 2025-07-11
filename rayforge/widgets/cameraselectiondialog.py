from gi.repository import Gtk, Adw
from typing import Optional
from ..models.camera import Camera


class CameraSelectionDialog(Adw.MessageDialog):
    def __init__(self, parent, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading="Select Camera",
            body="Please select an available camera device:",
            close_response="cancel",
            **kwargs
        )
        self.set_size_request(300, 200)
        self.selected_device_id: Optional[str] = None

        self.list_box = Gtk.ListBox()
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_show_separators(True)
        self.list_box.connect("row-activated", self.on_row_activated)

        scroll_window = Gtk.ScrolledWindow()
        scroll_window.set_child(self.list_box)
        scroll_window.set_vexpand(True)
        scroll_window.set_hexpand(True)

        self.set_extra_child(scroll_window)

        self.add_response("select", "Select")
        self.set_response_enabled(
            "select", False
        )  # Disable until a selection is made
        self.set_default_response("cancel")

        self.list_available_cameras()

        self.list_box.connect("row-selected", self.on_row_selected)

    def list_available_cameras(self):
        available_devices = Camera.list_available_devices()
        if not available_devices:
            row = Adw.ActionRow(title="No cameras found.")
            self.list_box.append(row)
            self.set_response_enabled("select", False)
            return

        for device_id in available_devices:
            row = Adw.ActionRow(title=f"Device ID: {device_id}")
            row.device_id = device_id  # Store device_id in the row
            self.list_box.append(row)

    def on_row_activated(self, list_box, row):
        self.selected_device_id = row.device_id
        self.response("select")

    def on_row_selected(self, list_box, row):
        if row:
            self.set_response_enabled("select", True)
            self.selected_device_id = row.device_id
        else:
            self.set_response_enabled("select", False)
            self.selected_device_id = None
