"""Reusable image-settings controls for a camera.

Composed by both :class:`CameraImageSettingsDialog` and the camera
wizard's image-settings page so the two stay in sync.
"""

import logging
from gettext import gettext as _
from typing import List, Optional, Tuple

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ..shared.pref_rows.base import SpinRow
from ..shared.slider import create_slider_row
from .display_widget import CameraDisplay

logger = logging.getLogger(__name__)


class CameraImageSettings(Gtk.Box):
    """Live preview + image-quality controls for one camera."""

    def __init__(self, controller: CameraController, **kwargs):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, **kwargs)
        self.controller = controller
        self.camera = controller.config
        self._updating_ui = False
        self._build_ui()
        self.camera.settings_changed.connect(self._on_camera_settings_changed)
        self.controller.resolutions_probed.connect(self._on_resolutions_probed)

    def _build_ui(self) -> None:
        self.set_spacing(16)

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)

        self.camera_display = CameraDisplay(self.controller)
        self.camera_display.set_hexpand(True)
        self.camera_display.set_vexpand(True)
        self.camera_display.set_halign(Gtk.Align.FILL)

        left_box.append(self.camera_display)
        self.append(left_box)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(0)
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

        self.custom_width_row = SpinRow(
            _("Custom Width"),
            lower=16,
            upper=16384,
            numeric=True,
            value=1920,
        )
        self.custom_width_row.value_changed.connect(
            self._on_custom_res_changed
        )
        self.custom_width_row.set_visible(False)
        image_group.add(self.custom_width_row)

        self.custom_height_row = SpinRow(
            _("Custom Height"),
            lower=16,
            upper=16384,
            numeric=True,
            value=1080,
        )
        self.custom_height_row.value_changed.connect(
            self._on_custom_res_changed
        )
        self.custom_height_row.set_visible(False)
        image_group.add(self.custom_height_row)

        self._sync_resolution_selection()

        self.yuyv_row = Adw.ActionRow(
            title=_("Prefer YUYV Format"),
            subtitle=_(
                "Use uncompressed YUYV instead of MJPEG. Fixes green "
                "artifacts on some USB cameras but may reduce "
                "resolution or frame rate on USB 2.0."
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

    def stop(self) -> None:
        self.camera_display.stop()

    # ----- signal handlers ---------------------------------------------

    def _on_camera_settings_changed(self, camera) -> None:
        pass

    def _on_resolution_changed(self, combo_row, pspec) -> None:
        if self._updating_ui:
            return
        idx = combo_row.get_selected()
        if 0 <= idx < len(self._resolution_values):
            val = self._resolution_values[idx]
            if val == (-1, -1):
                self.custom_width_row.set_visible(True)
                self.custom_height_row.set_visible(True)
                w = self.custom_width_row.get_int_value()
                h = self.custom_height_row.get_int_value()
                self.camera.resolution = (w, h)
            else:
                self.custom_width_row.set_visible(False)
                self.custom_height_row.set_visible(False)
                self.camera.resolution = val

    def _on_custom_res_changed(self, spin_row) -> None:
        if self._updating_ui:
            return
        idx = self.resolution_row.get_selected()
        if 0 <= idx < len(self._resolution_values):
            if self._resolution_values[idx] == (-1, -1):
                w = self.custom_width_row.get_int_value()
                h = self.custom_height_row.get_int_value()
                self.camera.resolution = (w, h)

    def _on_resolutions_probed(self, controller) -> None:
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

    def _sync_resolution_selection(self) -> None:
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

    def on_white_balance_changed(self, scale) -> None:
        if not self.auto_white_balance_switch.get_active():
            self.camera.white_balance = scale.get_value()

    def on_auto_white_balance_toggled(self, switch_row, pspec) -> None:
        is_auto = switch_row.get_active()
        self.white_balance_scale.set_sensitive(not is_auto)
        if is_auto:
            self.camera.white_balance = None
        else:
            self.camera.white_balance = self.wb_adjustment.get_value()

    def on_yuyv_toggled(self, switch_row, pspec) -> None:
        self.camera.prefer_yuyv = switch_row.get_active()

    def on_contrast_changed(self, scale) -> None:
        self.camera.contrast = scale.get_value()

    def on_brightness_changed(self, scale) -> None:
        self.camera.brightness = scale.get_value()

    def on_denoise_changed(self, scale) -> None:
        val = scale.get_value() / 100.0
        if val > 0.95:
            val = 0.95
        self.camera.denoise = val

    def on_transparency_changed(self, scale) -> None:
        self.camera.transparency = scale.get_value()


__all__ = ["CameraImageSettings"]
