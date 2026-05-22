import logging
from gettext import gettext as _
from typing import List, Optional, Tuple

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..shared.adwfix import get_spinrow_int
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.slider import create_slider_row
from .display_widget import CameraDisplay

logger = logging.getLogger(__name__)


class CameraImageSettingsDialog(PatchedDialogWindow):
    def __init__(self, parent, controller: CameraController, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1150,
            default_height=750,
            title=_("{camera_name} - Camera Image Settings").format(
                camera_name=controller.config.name
            ),
            **kwargs,
        )
        self.controller = controller
        self.camera = controller.config

        # Guard to prevent signal loops when updating UI programmatically
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

        image_group = Adw.PreferencesGroup(
            title=_("Camera Image Settings"),
            description=_("Adjust image quality and appearance parameters."),
        )
        settings_box.append(image_group)

        self._resolution_values: List[Optional[Tuple[int, int]]] = [None]
        resolution_labels = [_("Default")]
        for w, h in self.controller.available_resolutions:
            self._resolution_values.append((w, h))
            resolution_labels.append(f"{w} × {h}")

        # Add custom resolution option
        self._resolution_values.append((-1, -1))
        resolution_labels.append(_("Custom..."))

        self.resolution_store = Gtk.StringList.new(resolution_labels)
        self.resolution_row = Adw.ComboRow(
            title=_("Resolution"),
            subtitle=_(
                "Camera capture resolution. "
                "Default uses the camera's native setting."
            ),
            model=self.resolution_store,
        )
        self.resolution_row.connect(
            "notify::selected", self._on_resolution_changed
        )
        image_group.add(self.resolution_row)

        # Custom Resolution Inputs (Hidden by default)
        self.custom_width_row = Adw.SpinRow(
            title=_("Custom Width"),
            adjustment=Gtk.Adjustment(
                value=1920, lower=16, upper=16384, step_increment=1
            ),
            numeric=True,
            visible=False,
        )
        self.custom_width_row.connect(
            "notify::value", self._on_custom_res_changed
        )
        image_group.add(self.custom_width_row)

        self.custom_height_row = Adw.SpinRow(
            title=_("Custom Height"),
            adjustment=Gtk.Adjustment(
                value=1080, lower=16, upper=16384, step_increment=1
            ),
            numeric=True,
            visible=False,
        )
        self.custom_height_row.connect(
            "notify::value", self._on_custom_res_changed
        )
        image_group.add(self.custom_height_row)

        # Apply current resolution selection after widgets exist
        self._sync_resolution_selection()

        self.yuyv_row = Adw.ActionRow(
            title=_("Prefer YUYV Format"),
            subtitle=_(
                "Use uncompressed YUYV instead of MJPEG. Fixes green "
                "artifacts on some USB cameras but may reduce resolution "
                "or frame rate on USB 2.0."
            ),
        )
        self.yuyv_switch = Gtk.Switch()
        self.yuyv_switch.set_valign(Gtk.Align.CENTER)
        self.yuyv_switch.set_active(self.camera.prefer_yuyv)
        self.yuyv_switch.connect("notify::active", self.on_yuyv_toggled)
        self.yuyv_row.add_suffix(self.yuyv_switch)
        self.yuyv_row.set_activatable_widget(self.yuyv_switch)
        image_group.add(self.yuyv_row)

        self.auto_white_balance_row = Adw.ActionRow(
            title=_("Auto White Balance"),
            subtitle=_("Automatically adjust white balance"),
        )
        self.auto_white_balance_switch = Gtk.Switch()
        self.auto_white_balance_switch.set_valign(Gtk.Align.CENTER)
        self.auto_white_balance_switch.set_active(
            self.camera.white_balance is None
        )
        self.auto_white_balance_switch.connect(
            "notify::active", self.on_auto_white_balance_toggled
        )
        self.auto_white_balance_row.add_suffix(self.auto_white_balance_switch)
        self.auto_white_balance_row.set_activatable_widget(
            self.auto_white_balance_switch
        )
        image_group.add(self.auto_white_balance_row)

        initial_wb = (
            self.camera.white_balance
            if self.camera.white_balance is not None
            else 4000
        )
        self.wb_adjustment = Gtk.Adjustment(
            value=initial_wb,
            lower=2500,
            upper=10000,
            step_increment=10,
            page_increment=100,
        )
        wb_row, self.white_balance_scale = create_slider_row(
            title=_("White Balance (Kelvin)"),
            subtitle=_("Color temperature for accurate color representation"),
            adjustment=self.wb_adjustment,
            digits=0,
            on_value_changed=lambda s: self.on_white_balance_changed(s),
        )
        image_group.add(wb_row)

        self.white_balance_scale.set_sensitive(
            self.camera.white_balance is not None
        )

        row, self.contrast_scale = self._create_slider_row(
            title=_("Contrast"),
            subtitle=_("Difference between light and dark areas"),
            initial_val=self.camera.contrast,
            callback=self.on_contrast_changed,
            lower=0.0,
            upper=100.0,
            step=0.01,
            page=10.0,
            digits=2,
        )
        image_group.add(row)

        row, self.brightness_scale = self._create_slider_row(
            title=_("Brightness"),
            subtitle=_("Overall lightness or darkness of the image"),
            initial_val=self.camera.brightness,
            callback=self.on_brightness_changed,
            lower=-100.0,
            upper=100.0,
            step=0.01,
            page=10.0,
            digits=2,
        )
        image_group.add(row)

        row, self.denoise_scale = self._create_slider_row(
            title=_("Noise Reduction"),
            subtitle=_("Temporal averaging, higher values cause trailing"),
            initial_val=self.camera.denoise * 100.0,
            callback=self.on_denoise_changed,
            lower=0.0,
            upper=100.0,
            step=1.0,
            page=10.0,
            digits=0,
        )
        image_group.add(row)

        row, self.transparency_scale = self._create_slider_row(
            title=_("Transparency"),
            subtitle=_("Transparency on the worksurface"),
            initial_val=self.camera.transparency,
            callback=self.on_transparency_changed,
            lower=0.0,
            upper=1.0,
            step=0.01,
            page=0.1,
            digits=2,
        )
        image_group.add(row)

        calibration_group = Adw.PreferencesGroup(
            title=_("Lens Calibration"),
            description=_(
                "Correct lens distortion for straighter lines. "
                "Use the wizard for automatic calibration or adjust manually."
            ),
            margin_top=12,
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
        self.controller.resolutions_probed.connect(self._on_resolutions_probed)

    def _on_camera_settings_changed(self, camera):
        for key, row in self._distortion_rows.items():
            row.set_value(getattr(camera, key))

    def _on_resolution_changed(self, combo_row, pspec):
        if self._updating_ui:
            return

        idx = combo_row.get_selected()
        if 0 <= idx < len(self._resolution_values):
            val = self._resolution_values[idx]

            if val == (-1, -1):
                # Custom Selected
                self.custom_width_row.set_visible(True)
                self.custom_height_row.set_visible(True)

                w = get_spinrow_int(self.custom_width_row)
                h = get_spinrow_int(self.custom_height_row)
                self.camera.resolution = (w, h)
            else:
                # Standard Selected
                self.custom_width_row.set_visible(False)
                self.custom_height_row.set_visible(False)
                self.camera.resolution = val

    def _on_custom_res_changed(self, spin_row, pspec):
        if self._updating_ui:
            return

        idx = self.resolution_row.get_selected()
        if 0 <= idx < len(self._resolution_values):
            if self._resolution_values[idx] == (-1, -1):
                w = get_spinrow_int(self.custom_width_row)
                h = get_spinrow_int(self.custom_height_row)
                self.camera.resolution = (w, h)

    def _on_resolutions_probed(self, controller):
        self._resolution_values = [None]
        labels = [_("Default")]

        for w, h in controller.available_resolutions:
            self._resolution_values.append((w, h))
            labels.append(f"{w} × {h}")

        self._resolution_values.append((-1, -1))
        labels.append(_("Custom..."))

        self.resolution_store = Gtk.StringList.new(labels)
        self.resolution_row.set_model(self.resolution_store)
        self._sync_resolution_selection()

    def _sync_resolution_selection(self):
        self._updating_ui = True
        try:
            res = self.camera.resolution
            if res is None:
                self.resolution_row.set_selected(0)
                self.custom_width_row.set_visible(False)
                self.custom_height_row.set_visible(False)
            elif res in self._resolution_values:
                idx = self._resolution_values.index(res)
                self.resolution_row.set_selected(idx)
                self.custom_width_row.set_visible(False)
                self.custom_height_row.set_visible(False)
            else:
                # It's a custom resolution not found in the probed list
                idx = self._resolution_values.index((-1, -1))
                self.resolution_row.set_selected(idx)
                self.custom_width_row.set_value(res[0])
                self.custom_height_row.set_value(res[1])
                self.custom_width_row.set_visible(True)
                self.custom_height_row.set_visible(True)
        finally:
            self._updating_ui = False

    def _create_slider_row(
        self,
        title,
        subtitle,
        initial_val,
        callback,
        lower,
        upper,
        step,
        page,
        digits,
    ):
        adj = Gtk.Adjustment(
            value=initial_val,
            lower=lower,
            upper=upper,
            step_increment=step,
            page_increment=page,
        )
        return create_slider_row(
            title=title,
            subtitle=subtitle,
            adjustment=adj,
            digits=digits,
            on_value_changed=callback,
        )

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
            "notify::value", self._on_distortion_value_changed, config_key
        )
        return row

    def on_white_balance_changed(self, scale):
        if not self.auto_white_balance_switch.get_active():
            self.camera.white_balance = scale.get_value()

    def on_auto_white_balance_toggled(self, switch_row, pspec):
        is_auto = switch_row.get_active()
        self.white_balance_scale.set_sensitive(not is_auto)
        if is_auto:
            self.camera.white_balance = None
        else:
            self.camera.white_balance = self.wb_adjustment.get_value()

    def on_yuyv_toggled(self, switch_row, pspec):
        self.camera.prefer_yuyv = switch_row.get_active()

    def on_contrast_changed(self, scale):
        self.camera.contrast = scale.get_value()

    def on_brightness_changed(self, scale):
        self.camera.brightness = scale.get_value()

    def on_denoise_changed(self, scale):
        val = scale.get_value() / 100.0
        if val > 0.95:
            val = 0.95
        self.camera.denoise = val

    def on_transparency_changed(self, scale):
        self.camera.transparency = scale.get_value()

    def _on_distortion_value_changed(
        self, spin_row: Adw.SpinRow, pspec, config_key: str
    ):
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
            f"CameraImageSettingsDialog closing for camera {self.camera.name}"
        )
        self.camera.settings_changed.disconnect(
            self._on_camera_settings_changed
        )
        self.controller.resolutions_probed.disconnect(
            self._on_resolutions_probed
        )
        self.camera_display.stop()
        return False
