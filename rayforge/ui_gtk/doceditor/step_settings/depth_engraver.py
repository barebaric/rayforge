from typing import Dict, Any, TYPE_CHECKING, cast, Optional
from gi.repository import Gtk, Adw, GObject
import numpy as np
from ....image.dither import DitherAlgorithm
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.depth import DepthEngraver, DepthMode
from ....shared.util.glib import DebounceMixin
from ...shared.adwfix import get_spinrow_int, get_spinrow_float
from .base import StepComponentSettingsWidget
from .direction_preview import DirectionPreview
from .histogram_preview import HistogramPreview

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class DepthEngraverSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the DepthEngraver producer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        producer = cast(DepthEngraver, OpsProducer.from_dict(target_dict))

        super().__init__(
            editor,
            title,
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        # Mode selection dropdown
        mode_choices = [m.name.replace("_", " ").title() for m in DepthMode]
        mode_row = Adw.ComboRow(
            title=_("Mode"), model=Gtk.StringList.new(mode_choices)
        )
        mode_row.set_selected(list(DepthMode).index(producer.depth_mode))
        self.add(mode_row)

        # --- Threshold (for Constant Power mode) ---
        threshold_adj = Gtk.Adjustment(
            lower=0,
            upper=255,
            step_increment=1,
            page_increment=10,
            value=producer.threshold,
        )
        self.threshold_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=threshold_adj,
            digits=0,
            draw_value=True,
        )
        self.threshold_scale.set_size_request(200, -1)
        self.threshold_scale.connect(
            "value-changed", self._on_threshold_changed
        )

        self.threshold_row = Adw.ActionRow(
            title=_("Threshold"),
            subtitle=_("Brightness cutoff for black/white (0-255)"),
        )
        self.threshold_row.add_suffix(self.threshold_scale)
        self.add(self.threshold_row)

        # --- Dither Algorithm (for Dither mode) ---
        dither_choices = [
            m.name.replace("_", " ").title() for m in DitherAlgorithm
        ]
        self.dither_algorithm_row = Adw.ComboRow(
            title=_("Dither Algorithm"),
            subtitle=_("Algorithm for converting grayscale to binary"),
            model=Gtk.StringList.new(dither_choices),
        )
        current_algo = (
            producer.dither_algorithm or DitherAlgorithm.FLOYD_STEINBERG
        )
        self.dither_algorithm_row.set_selected(
            list(DitherAlgorithm).index(current_algo)
        )
        self.dither_algorithm_row.connect(
            "notify::selected", self._on_dither_algorithm_changed
        )
        self.add(self.dither_algorithm_row)

        # --- Cross-Hatch & Scan Angle with Preview ---
        angle_adj = Gtk.Adjustment(
            lower=0,
            upper=360,
            step_increment=1,
            page_increment=15,
            value=producer.scan_angle,
        )
        self.angle_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=angle_adj,
            digits=0,
            draw_value=True,
        )
        self.angle_scale.set_size_request(200, -1)
        self.angle_scale.connect("value-changed", self._on_angle_changed)

        self.direction_preview = DirectionPreview(
            producer.scan_angle, producer.cross_hatch
        )

        preview_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        preview_box.append(self.direction_preview)
        preview_box.append(self.angle_scale)

        self.scan_angle_row = Adw.ActionRow(
            title=_("Scan Angle"),
            subtitle=_("Angle of scan lines in degrees"),
        )
        self.scan_angle_row.add_suffix(preview_box)
        self.add(self.scan_angle_row)

        self.cross_hatch_row = Adw.SwitchRow(
            title=_("Cross-Hatch"),
            subtitle=_("Add a second pass at 90 degrees"),
        )
        self.cross_hatch_row.set_active(producer.cross_hatch)
        self.cross_hatch_row.connect(
            "notify::active", self._on_cross_hatch_changed
        )
        self.add(self.cross_hatch_row)

        # --- Histogram ---
        self.histogram_preview = HistogramPreview()
        self.histogram_row = Adw.ActionRow(title=_("Histogram"))
        self.histogram_row.add_suffix(self.histogram_preview)
        self.add(self.histogram_row)

        # --- Power Modulation Settings ---
        self.min_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=0.1,
            value=producer.min_power * 100,
        )
        self.min_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.min_power_adj,
            digits=1,
            draw_value=True,
        )
        self.min_power_scale.set_size_request(200, -1)
        self.min_power_row = Adw.ActionRow(
            title=_("Min Power"),
            subtitle=_(
                "Power for lightest areas, as a % of the step's main power"
            ),
        )
        self.min_power_row.add_suffix(self.min_power_scale)
        self.add(self.min_power_row)

        self.max_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=0.1,
            value=producer.max_power * 100,
        )
        self.max_power_scale = Gtk.Scale(
            orientation=Gtk.Orientation.HORIZONTAL,
            adjustment=self.max_power_adj,
            digits=1,
            draw_value=True,
        )
        self.max_power_scale.set_size_request(200, -1)
        self.max_power_row = Adw.ActionRow(
            title=_("Max Power"),
            subtitle=_(
                "Power for darkest areas, as a % of the step's main power"
            ),
        )
        self.max_power_row.add_suffix(self.max_power_scale)
        self.add(self.max_power_row)

        self._update_power_labels(producer.invert)

        # --- Multi-Pass Settings ---
        levels_adj = Gtk.Adjustment(
            lower=1,
            upper=255,
            step_increment=1,
            value=producer.num_depth_levels,
        )
        self.levels_row = Adw.SpinRow(
            title=_("Number of Depth Levels"), adjustment=levels_adj
        )
        self.add(self.levels_row)

        z_step_adj = Gtk.Adjustment(
            lower=0, upper=50, step_increment=0.1, value=producer.z_step_down
        )
        self.z_step_row = Adw.SpinRow(
            title=_("Z Step-Down per Level (mm)"),
            adjustment=z_step_adj,
            digits=2,
        )
        self.add(self.z_step_row)

        # Connect signals
        mode_row.connect("notify::selected", self._on_mode_changed)

        self.min_power_handler_id = self.min_power_scale.connect(
            "value-changed", self._on_min_power_scale_changed
        )
        self.max_power_handler_id = self.max_power_scale.connect(
            "value-changed", self._on_max_power_scale_changed
        )

        self.levels_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed,
                "num_depth_levels",
                get_spinrow_int(r),
            ),
        )
        self.z_step_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed, "z_step_down", get_spinrow_float(r)
            ),
        )

        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Engrave white areas instead of black areas"),
        )
        self.invert_row.set_active(producer.invert)
        self.invert_row.connect("notify::active", self._on_invert_changed)
        self.add(self.invert_row)

        line_interval_adj = Gtk.Adjustment(
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            value=producer.line_interval_mm or 0.1,
        )
        self.line_interval_row = Adw.SpinRow(
            title=_("Line Interval"),
            subtitle=_(
                "Distance between scan lines in mm. Leave at 0 to use laser "
                "spot size."
            ),
            adjustment=line_interval_adj,
            digits=2,
        )
        self.line_interval_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_line_interval_changed, get_spinrow_float(r)
            ),
        )
        self.add(self.line_interval_row)

        self.mode_row = mode_row
        self._compute_and_update_histogram(producer.invert)
        self._on_mode_changed(mode_row, None)

    def _compute_and_update_histogram(self, invert: bool):
        layer = self.step.layer
        if not layer:
            self.histogram_preview.update_histogram(None, 0.0, 1.0)
            return

        workpieces = layer.all_workpieces
        if not workpieces:
            self.histogram_preview.update_histogram(None, 0.0, 1.0)
            return

        workpiece = workpieces[0]
        size = workpiece.size
        if not size or size[0] <= 0 or size[1] <= 0:
            self.histogram_preview.update_histogram(None, 0.0, 1.0)
            return

        pixels_per_mm = self.step.pixels_per_mm
        width_px = int(size[0] * pixels_per_mm[0])
        height_px = int(size[1] * pixels_per_mm[1])

        if width_px <= 0 or height_px <= 0:
            self.histogram_preview.update_histogram(None, 0.0, 1.0)
            return

        surface = workpiece.render_to_pixels(width_px, height_px)
        if not surface:
            self.histogram_preview.update_histogram(None, 0.0, 1.0)
            return

        width_px = surface.get_width()
        height_px = surface.get_height()
        stride = surface.get_stride()
        buf = surface.get_data()
        data_with_padding = np.ndarray(
            shape=(height_px, stride // 4, 4), dtype=np.uint8, buffer=buf
        )
        data = data_with_padding[:, :width_px, :]

        alpha = data[:, :, 3].astype(np.float32) / 255.0
        r = data[:, :, 2].astype(np.float32)
        g = data[:, :, 1].astype(np.float32)
        b = data[:, :, 0].astype(np.float32)

        alpha_safe = np.maximum(alpha, 1e-6)
        r_unpremult = np.clip(r / alpha_safe, 0, 255)
        g_unpremult = np.clip(g / alpha_safe, 0, 255)
        b_unpremult = np.clip(b / alpha_safe, 0, 255)

        r_blended = 255.0 - (255.0 - r_unpremult) * alpha
        g_blended = 255.0 - (255.0 - g_unpremult) * alpha
        b_blended = 255.0 - (255.0 - b_unpremult) * alpha

        gray_image = (
            0.2989 * r_blended + 0.5870 * g_blended + 0.114 * b_blended
        ).astype(np.uint8)

        if invert:
            alpha_mask = alpha > 0
            gray_image[alpha_mask] = 255 - gray_image[alpha_mask]

        histogram, _ = np.histogram(
            gray_image[alpha > 0], bins=64, range=(0, 255)
        )

        min_threshold = self.min_power_adj.get_value() / 100.0
        max_threshold = self.max_power_adj.get_value() / 100.0
        self.histogram_preview.update_histogram(
            histogram, min_threshold, max_threshold
        )

    def _commit_power_range_change(self):
        """Commits the min/max power to the model via command(s)."""
        min_p = self.min_power_adj.get_value() / 100.0
        max_p = self.max_power_adj.get_value() / 100.0

        params = self.target_dict.setdefault("params", {})
        min_changed = abs(params.get("min_power", 0.0) - min_p) > 1e-6
        max_changed = abs(params.get("max_power", 0.0) - max_p) > 1e-6

        if not min_changed and not max_changed:
            return

        with self.history_manager.transaction(_("Change Power Range")):
            if min_changed:
                self.editor.step.set_step_param(
                    params, "min_power", min_p, _("Change Min Power")
                )
            if max_changed:
                self.editor.step.set_step_param(
                    params, "max_power", max_p, _("Change Max Power")
                )
        self.step.updated.send(self.step)

    def _on_min_power_scale_changed(self, scale: Gtk.Scale):
        new_min_value = self.min_power_adj.get_value()

        GObject.signal_handler_block(
            self.max_power_scale, self.max_power_handler_id
        )

        if self.max_power_adj.get_value() < new_min_value:
            self.max_power_adj.set_value(new_min_value)

        GObject.signal_handler_unblock(
            self.max_power_scale, self.max_power_handler_id
        )

        self._update_histogram_thresholds()
        self._debounce(self._commit_power_range_change)

    def _on_max_power_scale_changed(self, scale: Gtk.Scale):
        new_max_value = self.max_power_adj.get_value()

        GObject.signal_handler_block(
            self.min_power_scale, self.min_power_handler_id
        )

        if self.min_power_adj.get_value() > new_max_value:
            self.min_power_adj.set_value(new_max_value)

        GObject.signal_handler_unblock(
            self.min_power_scale, self.min_power_handler_id
        )

        self._update_histogram_thresholds()
        self._debounce(self._commit_power_range_change)

    def _update_histogram_thresholds(self):
        if self.histogram_preview.histogram is not None:
            min_threshold = self.min_power_adj.get_value() / 100.0
            max_threshold = self.max_power_adj.get_value() / 100.0
            self.histogram_preview.update_histogram(
                self.histogram_preview.histogram, min_threshold, max_threshold
            )

    def _on_mode_changed(self, row, _):
        selected_idx = row.get_selected()
        selected_mode = list(DepthMode)[selected_idx]
        is_power_mode = selected_mode == DepthMode.POWER_MODULATION
        is_constant_power = selected_mode == DepthMode.CONSTANT_POWER
        is_dither = selected_mode == DepthMode.DITHER
        is_multi_pass = selected_mode == DepthMode.MULTI_PASS

        self.min_power_row.set_visible(is_power_mode)
        self.max_power_row.set_visible(is_power_mode)
        self.histogram_row.set_visible(is_power_mode)

        self.threshold_row.set_visible(is_constant_power)
        self.dither_algorithm_row.set_visible(is_dither)

        self.levels_row.set_visible(is_multi_pass)
        self.z_step_row.set_visible(is_multi_pass)

        self._on_param_changed("depth_mode", selected_mode.name)

    def _on_dither_algorithm_changed(self, row, _):
        selected_idx = row.get_selected()
        selected_algo = list(DitherAlgorithm)[selected_idx]
        self._on_param_changed("dither_algorithm", selected_algo.value)

    def _on_threshold_changed(self, scale):
        value = int(scale.get_value())
        self._debounce(self._on_param_changed, "threshold", value)

    def _on_angle_changed(self, scale):
        value = float(scale.get_value())
        self.direction_preview.update(value, self.cross_hatch_row.get_active())
        self._debounce(self._on_param_changed, "scan_angle", value)

    def _on_cross_hatch_changed(self, w, _):
        cross_hatch = w.get_active()
        self.direction_preview.update(
            self.angle_scale.get_value(), cross_hatch
        )
        self._on_param_changed("cross_hatch", cross_hatch)

    def _update_power_labels(self, invert: bool):
        """Update min/max power labels based on invert setting."""
        lightest_subtitle = _(
            "Power for lightest areas, as a % of the step's main power"
        )
        darkest_subtitle = _(
            "Power for darkest areas, as a % of the step's main power"
        )

        if invert:
            self.min_power_row.set_title(_("Min Power (Black)"))
            self.min_power_row.set_subtitle(darkest_subtitle)
            self.max_power_row.set_title(_("Max Power (White)"))
            self.max_power_row.set_subtitle(lightest_subtitle)
        else:
            self.min_power_row.set_title(_("Min Power (White)"))
            self.min_power_row.set_subtitle(lightest_subtitle)
            self.max_power_row.set_title(_("Max Power (Black)"))
            self.max_power_row.set_subtitle(darkest_subtitle)

    def _on_invert_changed(self, w, pspec):
        invert = w.get_active()
        self._update_power_labels(invert)
        self._compute_and_update_histogram(invert)
        self._on_param_changed("invert", invert)

    def _on_line_interval_changed(self, value: Optional[float]):
        if value is not None and value <= 0:
            value = None
        self._on_param_changed("line_interval_mm", value)

    def _on_param_changed(self, key: str, value: Any):
        target_dict = self.target_dict.setdefault("params", {})
        self.editor.step.set_step_param(
            target_dict=target_dict,
            key=key,
            new_value=value,
            name=_("Change Depth Engraving setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
