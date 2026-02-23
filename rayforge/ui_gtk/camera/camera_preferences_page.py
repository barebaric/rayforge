from gi.repository import Gtk, Adw
from blinker import Signal
from typing import List, cast, Optional
from ...camera.models.camera import Camera
from ...camera.controller import CameraController
from ..icons import get_icon
from ..shared.preferences_group import PreferencesGroupWithButton
from .properties_widget import CameraProperties
from .selection_dialog import CameraSelectionDialog


class CameraRow(Gtk.Box):
    """A widget representing a single Camera in a ListBox."""

    def __init__(self, camera: Camera):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.camera = camera
        self.delete_button: Gtk.Button
        self.title_label: Gtk.Label
        self.subtitle_label: Gtk.Label
        self._setup_ui()

        # Signals
        self.remove_clicked = Signal()
        """Signal emitted when the remove button is clicked.
        Sends: sender, camera (Camera)
        """

    def _setup_ui(self):
        """Builds the user interface for the row."""
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=0, hexpand=True
        )
        self.append(labels_box)

        self.title_label = Gtk.Label(
            label=self.camera.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        labels_box.append(self.title_label)

        self.subtitle_label = Gtk.Label(
            label=self._get_subtitle_text(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        self.subtitle_label.add_css_class("dim-label")
        labels_box.append(self.subtitle_label)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.connect("clicked", self._on_remove_clicked)
        self.append(self.delete_button)

    def _get_subtitle_text(self) -> str:
        """Generates the subtitle text from camera properties."""
        return _("Device ID: {device_id}").format(
            device_id=self.camera.device_id
        )

    def _on_remove_clicked(self, button: Gtk.Button):
        """Emits a signal requesting the removal of this camera."""
        self.remove_clicked.send(self, camera=self.camera)


class CameraListEditor(PreferencesGroupWithButton):
    """An Adwaita widget for displaying and managing a list of cameras."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Add New Camera"), **kwargs)
        self._setup_ui()

        # Signals
        self.add_requested = Signal()
        """Signal emitted when the 'Add New Camera' button is clicked."""
        self.remove_requested = Signal()
        """Signal emitted when a camera's remove button is clicked.
        Sends: sender, camera (Camera)
        """

    def _setup_ui(self):
        """Configures the widget's list box and placeholder."""
        placeholder = Gtk.Label(
            label=_("No cameras configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.list_box.set_show_separators(True)

    def set_cameras(self, cameras: List[Camera]):
        """Rebuilds the list to match the provided list of cameras."""
        selected_camera = None
        selected_row = self.list_box.get_selected_row()
        if selected_row:
            camera_row_widget = cast(CameraRow, selected_row.get_child())
            selected_camera = camera_row_widget.camera

        row_count = 0
        while self.list_box.get_row_at_index(row_count):
            row_count += 1

        new_selection_index = -1
        for i, camera in enumerate(cameras):
            if camera == selected_camera:
                new_selection_index = i

            if i < row_count:
                row = self.list_box.get_row_at_index(i)
                if not row:
                    continue
                camera_row = cast(CameraRow, row.get_child())
                camera_row.camera = camera
                camera_row.title_label.set_label(camera.name)
                camera_row.subtitle_label.set_label(
                    camera_row._get_subtitle_text()
                )
            else:
                list_box_row = Gtk.ListBoxRow()
                list_box_row.set_child(self.create_row_widget(camera))
                self.list_box.append(list_box_row)

        while row_count > len(cameras):
            last_row = self.list_box.get_row_at_index(row_count - 1)
            if last_row:
                self.list_box.remove(last_row)
            row_count -= 1

        if new_selection_index >= 0:
            row = self.list_box.get_row_at_index(new_selection_index)
            self.list_box.select_row(row)
        elif len(cameras) > 0:
            row = self.list_box.get_row_at_index(0)
            self.list_box.select_row(row)
        else:
            if self.list_box.get_selected_row():
                self.list_box.unselect_all()
            else:
                self.list_box.emit("row-selected", None)

    def create_row_widget(self, item: Camera) -> Gtk.Widget:
        """Creates a CameraRow for the given camera item."""
        row = CameraRow(item)
        row.remove_clicked.connect(self._on_row_remove_clicked)
        return row

    def _on_add_clicked(self, button: Gtk.Button):
        """Emits the add_requested signal."""
        self.add_requested.send(self)

    def _on_row_remove_clicked(self, row_widget: CameraRow, camera: Camera):
        """Bubbles up the remove_requested signal from a CameraRow."""
        self.remove_requested.send(self, camera=camera)


class CameraEnhancementGroup(Adw.PreferencesGroup):
    """A widget for noise reduction and image enhancement."""

    def __init__(self, **kwargs):
        super().__init__(
            title=_("Image Enhancement"),
            description=_("Reduce noise and improve image stability."),
            **kwargs
        )
        self._controller: Optional[CameraController] = None
        self._updating = False

        # Scale 0-100 mapped to denoise factor 0.0 - 0.95
        self.denoise_scale = Gtk.Scale.new_with_range(Gtk.Orientation.HORIZONTAL, 0, 100, 1)
        self.denoise_scale.set_hexpand(True)
        self.denoise_scale.set_value_pos(Gtk.PositionType.LEFT)
        self.denoise_scale.connect("value-changed", self._on_value_changed)
        
        row = Adw.ActionRow(title=_("Noise Reduction"))
        row.set_subtitle(_("Temporal averaging. Higher values remove more noise but cause trailing."))
        row.add_suffix(self.denoise_scale)
        self.add(row)

    def set_controller(self, controller: Optional[CameraController]):
        self._controller = controller
        self.set_sensitive(controller is not None)

        if not controller:
            return
        
        self._updating = True
        # Read the current config. Defaults to 0.0
        denoise_val = getattr(controller.config, 'denoise', 0.0)
        # Convert 0.0-0.95 range to 0-100 slider
        self.denoise_scale.set_value(denoise_val * 100.0)
        self._updating = False

    def _on_value_changed(self, scale):
        if self._updating or not self._controller:
            return
        
        # Convert 0-100 slider to 0.0-0.95 range
        val = scale.get_value() / 100.0
        # Hard clamp to 0.95 to avoid accidental infinite freeze
        if val > 0.95:
            val = 0.95
            
        self._controller.config.denoise = val


class CameraDistortionGroup(Adw.PreferencesGroup):
    """A widget for correcting fisheye/wide-angle lens distortion."""

    def __init__(self, **kwargs):
        super().__init__(
            title=_("Lens Distortion Correction (Fisheye)"),
            description=_("Straighten bowed lines using Radial (k1, k2) and Tangential (p1, p2) parameters. Note: Values are usually very small (e.g., 0.005)."),
            **kwargs
        )
        self._controller: Optional[CameraController] = None
        self._updating = False

        self.k1_spin = self._create_spin_row(_("Radial 1 (k1)"))
        self.k2_spin = self._create_spin_row(_("Radial 2 (k2)"))
        self.p1_spin = self._create_spin_row(_("Tangential 1 (p1)"))
        self.p2_spin = self._create_spin_row(_("Tangential 2 (p2)"))

    def _create_spin_row(self, title: str) -> Gtk.SpinButton:
        row = Adw.ActionRow(title=title)
        spin = Gtk.SpinButton(
            adjustment=Gtk.Adjustment(
                value=0.0,
                lower=-10.0,
                upper=10.0,
                step_increment=0.001,
                page_increment=0.01
            ),
            digits=4,
            numeric=True
        )
        spin.set_valign(Gtk.Align.CENTER)
        spin.connect("value-changed", self._on_value_changed)
        row.add_suffix(spin)
        self.add(row)
        return spin

    def set_controller(self, controller: Optional[CameraController]):
        self._controller = controller
        self.set_sensitive(controller is not None)

        if not controller:
            return

        self._updating = True
        self.k1_spin.set_value(getattr(controller.config, 'distortion_k1', 0.0))
        self.k2_spin.set_value(getattr(controller.config, 'distortion_k2', 0.0))
        self.p1_spin.set_value(getattr(controller.config, 'distortion_p1', 0.0))
        self.p2_spin.set_value(getattr(controller.config, 'distortion_p2', 0.0))
        self._updating = False

    def _on_value_changed(self, spin: Gtk.SpinButton):
        if self._updating or not self._controller:
            return

        self._controller.config.distortion_k1 = self.k1_spin.get_value()
        self._controller.config.distortion_k2 = self.k2_spin.get_value()
        self._controller.config.distortion_p1 = self.p1_spin.get_value()
        self._controller.config.distortion_p2 = self.p2_spin.get_value()


class CameraPositioningGroup(Adw.PreferencesGroup):
    """A widget for fine-tuning the bounding box and stretching of the camera image."""

    def __init__(self, **kwargs):
        super().__init__(
            title=_("Image Stretching & Positioning"),
            description=_("Adjust the top, bottom, left, and right bounds. Use positive or negative values to stretch or offset the image representation."),
            **kwargs
        )
        self._controller: Optional[CameraController] = None
        self._updating = False

        self.top_spin = self._create_spin_row(_("Top"))
        self.bottom_spin = self._create_spin_row(_("Bottom"))
        self.left_spin = self._create_spin_row(_("Left"))
        self.right_spin = self._create_spin_row(_("Right"))

    def _create_spin_row(self, title: str) -> Gtk.SpinButton:
        row = Adw.ActionRow(title=title)
        spin = Gtk.SpinButton(
            adjustment=Gtk.Adjustment(
                value=0.0,
                lower=-99999.0,
                upper=99999.0,
                step_increment=1.0,
                page_increment=10.0
            ),
            digits=2,
            numeric=True
        )
        spin.set_valign(Gtk.Align.CENTER)
        spin.connect("value-changed", self._on_value_changed)
        row.add_suffix(spin)
        self.add(row)
        return spin

    def set_controller(self, controller: Optional[CameraController]):
        self._controller = controller
        self.set_sensitive(controller is not None)

        if not controller:
            return

        self._updating = True
        self.top_spin.set_value(getattr(controller.config, 'offset_top', 0.0))
        self.bottom_spin.set_value(getattr(controller.config, 'offset_bottom', 0.0))
        self.left_spin.set_value(getattr(controller.config, 'offset_left', 0.0))
        self.right_spin.set_value(getattr(controller.config, 'offset_right', 0.0))
        self._updating = False

    def _on_value_changed(self, spin: Gtk.SpinButton):
        if self._updating or not self._controller:
            return

        self._controller.config.offset_top = self.top_spin.get_value()
        self._controller.config.offset_bottom = self.bottom_spin.get_value()
        self._controller.config.offset_left = self.left_spin.get_value()
        self._controller.config.offset_right = self.right_spin.get_value()


class CameraPreferencesPage(Adw.PreferencesPage):
    def __init__(self, **kwargs):
        super().__init__(
            title=_("Camera"), icon_name="camera-photo-symbolic", **kwargs
        )
        self._controllers: List[CameraController] = []
        self._cameras: List[Camera] = []

        # Signals
        self.camera_add_requested = Signal()
        """Signal emitted when a user requests to add a camera.
        Sends: sender, device_id (str)
        """
        self.camera_remove_requested = Signal()
        """Signal emitted when a user requests to remove a camera.
        Sends: sender, camera (Camera)
        """

        # List of Cameras, using the new reusable widget
        self.camera_list_editor = CameraListEditor(
            title=_("Cameras"),
            description=_(
                "Stream a camera image directly onto the work surface."
            ),
        )
        self.add(self.camera_list_editor)

        # Configuration panel for the selected Camera
        self.camera_properties_widget = CameraProperties(None)
        self.add(self.camera_properties_widget)
        
        # New Image Enhancement Group (Denoise)
        self.camera_enhancement_widget = CameraEnhancementGroup()
        self.add(self.camera_enhancement_widget)

        # Lens Distortion Group (Fisheye Correction)
        self.camera_distortion_widget = CameraDistortionGroup()
        self.add(self.camera_distortion_widget)

        # Positioning widget for the selected Camera
        self.camera_positioning_widget = CameraPositioningGroup()
        self.add(self.camera_positioning_widget)

        # Connect signals
        self.camera_list_editor.add_requested.connect(self.on_add_camera)
        self.camera_list_editor.remove_requested.connect(self.on_remove_camera)
        self.camera_list_editor.list_box.connect(
            "row-selected", self.on_camera_selected
        )

    def set_controllers(self, controllers: List[CameraController]):
        """Sets the list of camera controllers and refreshes the UI."""
        self._controllers = controllers
        self._cameras = [c.config for c in controllers]
        self.camera_list_editor.set_cameras(self._cameras)

    def on_add_camera(self, sender):
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

    def on_remove_camera(self, sender, camera: Camera):
        """Emit a signal to request removal of the selected Camera."""
        self.camera_remove_requested.send(self, camera=camera)

    def on_camera_selected(self, listbox, row):
        """Update the configuration panel when a Camera is selected."""
        if row is not None:
            camera_row = cast(CameraRow, row.get_child())
            selected_camera = camera_row.camera
            # Find the controller that matches this camera model
            selected_controller = next(
                (c for c in self._controllers if c.config == selected_camera),
                None,
            )
            self.camera_properties_widget.set_controller(selected_controller)
            self.camera_enhancement_widget.set_controller(selected_controller)
            self.camera_distortion_widget.set_controller(selected_controller)
            self.camera_positioning_widget.set_controller(selected_controller)
        else:
            self.camera_properties_widget.set_controller(None)
            self.camera_enhancement_widget.set_controller(None)
            self.camera_distortion_widget.set_controller(None)
            self.camera_positioning_widget.set_controller(None)
