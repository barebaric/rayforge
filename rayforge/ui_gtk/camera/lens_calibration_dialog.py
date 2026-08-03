import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..shared.patched_dialog_window import PatchedDialogWindow
from .display_widget import CameraDisplay
from .lens_calibration_widget import LensCalibrationWidget

logger = logging.getLogger(__name__)


class LensCalibrationDialog(PatchedDialogWindow):
    def __init__(self, parent, controller: CameraController, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1150,
            default_height=750,
            title=_("{camera_name} - Lens Calibration").format(
                camera_name=controller.config.name
            ),
            **kwargs,
        )
        self.controller = controller
        self.camera = controller.config

        self._setup_ui()

    def _setup_ui(self):
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        main_box.set_margin_start(32)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        content.append(main_box)

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)

        self.camera_display = CameraDisplay(self.controller)
        self.camera_display.set_hexpand(True)
        self.camera_display.set_vexpand(True)
        self.camera_display.set_halign(Gtk.Align.FILL)

        left_box.append(self.camera_display)
        main_box.append(left_box)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        main_box.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(32)
        settings_box.set_margin_top(4)
        settings_box.set_margin_bottom(12)
        right_scroll.set_child(settings_box)

        self.calibration_widget = LensCalibrationWidget(self.camera)
        settings_box.append(self.calibration_widget)

    def do_close_request(self, *args) -> bool:
        logger.debug(
            f"LensCalibrationDialog closing for camera {self.camera.name}"
        )
        self.calibration_widget.stop()
        self.camera_display.stop()
        return False
