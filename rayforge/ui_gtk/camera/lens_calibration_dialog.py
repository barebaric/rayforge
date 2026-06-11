import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..shared.patched_dialog_window import PatchedDialogWindow
from .display_widget import CameraDisplay

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

        self._updating_ui = False

        self._setup_ui()

    def _setup_ui(self):
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        main_box.set_margin_start(12)
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

        calibration_group = Adw.PreferencesGroup(
            title=_("Lens Calibration"),
            description=_(
                "Correct lens distortion for straighter lines. "
                "Use the wizard for automatic calibration or adjust "
                "manually."
            ),
        )

        self.wizard_button = Gtk.Button(
            label=_("Wizard"), valign=Gtk.Align.CENTER, margin_start=8
        )
        self.wizard_button.connect("clicked", self._on_wizard_clicked)
        calibration_group.set_header_suffix(self.wizard_button)

        settings_box.append(calibration_group)

        self._distortion_rows = {}
        for key, title, subtitle in [
            (
                "distortion_k1",
                _("Radial 1 (k1)"),
                _("First order radial distortion"),
            ),
            (
                "distortion_k2",
                _("Radial 2 (k2)"),
                _("Second order radial distortion"),
            ),
            (
                "distortion_k3",
                _("Radial 3 (k3)"),
                _("Third order radial distortion"),
            ),
            (
                "distortion_p1",
                _("Tangential 1 (p1)"),
                _("First order tangential distortion"),
            ),
            (
                "distortion_p2",
                _("Tangential 2 (p2)"),
                _("Second order tangential distortion"),
            ),
        ]:
            row = self._create_spin_row(
                title, subtitle, getattr(self.camera, key), key
            )
            self._distortion_rows[key] = row
            calibration_group.add(row)

        self.camera.settings_changed.connect(self._on_camera_settings_changed)

    def _create_spin_row(
        self, title: str, subtitle: str, value: float, config_key: str
    ) -> Adw.SpinRow:
        row = Adw.SpinRow(
            title=title,
            subtitle=subtitle,
            adjustment=Gtk.Adjustment(
                value=value,
                lower=-10.0,
                upper=10.0,
                step_increment=0.001,
                page_increment=0.01,
            ),
            digits=4,
            numeric=True,
        )
        row.connect(
            "notify::value",
            self._on_distortion_value_changed,
            config_key,
        )
        return row

    def _on_camera_settings_changed(self, camera):
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            for key, row in self._distortion_rows.items():
                row.set_value(getattr(camera, key))
        finally:
            self._updating_ui = False

    def _on_distortion_value_changed(
        self, spin_row: Adw.SpinRow, pspec, config_key: str
    ):
        if self._updating_ui:
            return
        setattr(self.camera, config_key, spin_row.get_value())

    def _on_wizard_clicked(self, button):
        from .calibration_wizard import CalibrationWizard

        window = self.get_ancestor(Gtk.Window)
        if not isinstance(window, Gtk.Window):
            return

        wizard = CalibrationWizard(window, self.controller)
        wizard.present()

    def do_close_request(self, *args) -> bool:
        logger.debug(
            f"LensCalibrationDialog closing for camera {self.camera.name}"
        )
        self.camera.settings_changed.disconnect(
            self._on_camera_settings_changed
        )
        self.camera_display.stop()
        return False
