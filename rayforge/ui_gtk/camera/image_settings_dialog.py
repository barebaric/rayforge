import logging
from gettext import gettext as _
from gi.repository import Gtk, Adw
from ...camera.controller import CameraController
from ..shared.patched_dialog_window import PatchedMessageDialog
from .display_widget import CameraDisplay

logger = logging.getLogger(__name__)


class CameraImageSettingsDialog(PatchedMessageDialog):
    def __init__(self, parent, controller: CameraController, **kwargs):
        super().__init__(
            transient_for=parent,
            modal=True,
            heading=_("{camera_name} - Camera Image Settings").format(
                camera_name=controller.config.name
            ),
            close_response="cancel",
            **kwargs,
        )
        self.controller = controller
        self.camera = controller.config

        # Main Horizontal Box for Side-by-Side layout
        main_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        main_box.set_margin_start(16)
        main_box.set_margin_end(16)
        main_box.set_margin_top(16)
        main_box.set_margin_bottom(16)

        # Force the overall dialog to be significantly larger
        # so everything fits without needing to scroll
        main_box.set_size_request(1150, 750)
        self.set_extra_child(main_box)

        # ==========================================
        # LEFT SIDE: Camera Display
        # ==========================================
        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)

        self.camera_display = CameraDisplay(self.controller)
        self.camera_display.set_hexpand(True)
        self.camera_display.set_vexpand(True)

        # PREVENT STRETCHING: Center the widget instead of forcing it to fill.
        # This preserves the camera's correct aspect ratio.
        self.camera_display.set_halign(Gtk.Align.CENTER)
        self.camera_display.set_valign(Gtk.Align.CENTER)

        left_box.append(self.camera_display)
        main_box.append(left_box)

        # ==========================================
        # RIGHT SIDE: Settings Panel
        # ==========================================
        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        right_scroll.set_propagate_natural_width(True)
        right_scroll.set_propagate_natural_height(True)
        right_scroll.set_min_content_width(480)
        main_box.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, spacing=16
        )
        right_scroll.set_child(settings_box)

        # --- Group: Camera Image Settings ---
        image_group = Adw.PreferencesGroup(title=_("Camera Image Settings"))
        settings_box.append(image_group)

        # Auto White Balance Switch
        self.auto_white_balance_switch = Adw.SwitchRow(
            title=_("Auto White Balance"),
            subtitle=_("Automatically adjust white balance"),
        )
        self.auto_white_balance_switch.set_active(
            self.camera.white_balance is None
        )
        self.auto_white_balance_switch.connect(
            "notify::active", self.on_auto_white_balance_toggled
        )
        image_group.add(self.auto_white_balance_switch)

        # White Balance Manual Slider
        self.wb_adjustment = Gtk.Adjustment(
            lower=2500, upper=10000, step_increment=10, page_increment=100
        )
        self.white_balance_scale = Gtk.Scale.new(
            Gtk.Orientation.HORIZONTAL, self.wb_adjustment
        )
        self.white_balance_scale.set_size_request(
            200, -1
        )  # Uniform slider width
        self.white_balance_scale.set_valign(Gtk.Align.CENTER)
        self.white_balance_scale.set_digits(0)
        self.white_balance_scale.set_value_pos(Gtk.PositionType.RIGHT)

        self.wb_adjustment.set_value(
            self.camera.white_balance
            if self.camera.white_balance is not None
            else 4000
        )
        self.white_balance_scale.connect(
            "value-changed", self.on_white_balance_changed
        )

        wb_row = Adw.ActionRow(title=_("White Balance (Kelvin)"))
        wb_row.add_suffix(self.white_balance_scale)
        image_group.add(wb_row)

        self.white_balance_scale.set_sensitive(
            self.camera.white_balance is not None
        )

        # Other Sliders
        row, self.contrast_scale = self._create_slider_row(
            title=_("Contrast"),
            subtitle=None,
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
            subtitle=None,
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
            subtitle=_("Temporal averaging. Higher values cause trailing."),
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

        # --- Group: Lens Distortion ---
        distortion_group = Adw.PreferencesGroup(
            title=_("Lens Distortion Correction"),
            description=_(
                "Straighten bowed lines using Radial (k1, k2) and Tangential "
                "(p1, p2) parameters."
            ),
        )
        settings_box.append(distortion_group)

        distortion_group.add(
            self._create_spin_row(
                _("Radial 1 (k1)"),
                _("First order radial distortion"),
                self.camera.distortion_k1,
                "distortion_k1",
            )
        )
        distortion_group.add(
            self._create_spin_row(
                _("Radial 2 (k2)"),
                _("Second order radial distortion"),
                self.camera.distortion_k2,
                "distortion_k2",
            )
        )
        distortion_group.add(
            self._create_spin_row(
                _("Tangential 1 (p1)"),
                _("First order tangential distortion"),
                self.camera.distortion_p1,
                "distortion_p1",
            )
        )
        distortion_group.add(
            self._create_spin_row(
                _("Tangential 2 (p2)"),
                _("Second order tangential distortion"),
                self.camera.distortion_p2,
                "distortion_p2",
            )
        )

        # Buttons
        self.add_response("close", _("Close"))
        self.set_default_response("cancel")
        self.connect("response", self.on_dialog_response)

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
        scale = Gtk.Scale.new(Gtk.Orientation.HORIZONTAL, adj)
        # Setting a 200px width for all scales makes them nicely aligned
        # while leaving enough room so the title text won't vertically wrap
        scale.set_size_request(200, -1)
        scale.set_valign(Gtk.Align.CENTER)
        scale.set_digits(digits)
        scale.set_value_pos(Gtk.PositionType.RIGHT)
        scale.connect("value-changed", callback)

        row = Adw.ActionRow(title=title)
        if subtitle:
            row.set_subtitle(subtitle)
        row.add_suffix(scale)

        return row, scale

    def _create_spin_row(
        self, title: str, subtitle: str, value: float, config_key: str
    ) -> Adw.ActionRow:
        row = Adw.ActionRow(title=title, subtitle=subtitle)
        spin = Gtk.SpinButton(
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
        spin.set_valign(Gtk.Align.CENTER)
        spin.connect(
            "value-changed", self._on_distortion_value_changed, config_key
        )
        row.add_suffix(spin)
        return row

    # --- Callbacks ---
    def on_white_balance_changed(self, adjustment):
        if not self.auto_white_balance_switch.get_active():
            self.camera.white_balance = adjustment.get_value()

    def on_auto_white_balance_toggled(self, switch_row, pspec):
        is_auto = switch_row.get_active()
        self.white_balance_scale.set_sensitive(not is_auto)
        if is_auto:
            self.camera.white_balance = None
        else:
            self.camera.white_balance = self.wb_adjustment.get_value()

    def on_contrast_changed(self, adjustment):
        self.camera.contrast = adjustment.get_value()

    def on_brightness_changed(self, adjustment):
        self.camera.brightness = adjustment.get_value()

    def on_denoise_changed(self, adjustment):
        val = adjustment.get_value() / 100.0
        if val > 0.95:
            val = 0.95
        self.camera.denoise = val

    def on_transparency_changed(self, adjustment):
        self.camera.transparency = adjustment.get_value()

    def _on_distortion_value_changed(
        self, spin: Gtk.SpinButton, config_key: str
    ):
        setattr(self.camera, config_key, spin.get_value())

    def on_dialog_response(self, dialog, response_id):
        if response_id == "close" or response_id == "cancel":
            logger.debug(
                "CameraImageSettingsDialog closing, calling "
                f"CameraDisplay.stop() for camera {self.camera.name}"
            )
            self.camera_display.stop()
            self.close()
