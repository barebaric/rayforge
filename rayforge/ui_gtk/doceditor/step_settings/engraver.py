from typing import Dict, Any, TYPE_CHECKING, cast, Optional
from gi.repository import Gtk, Adw, GObject
import numpy as np
from ....image.dither import DitherAlgorithm
from ....image.util import compute_auto_levels
from ....pipeline.producer.base import OpsProducer
from ....pipeline.producer.raster import (
    Rasterizer,
    DepthMode,
)
from ....shared.util.glib import DebounceMixin
from ...shared.adwfix import get_spinrow_int, get_spinrow_float
from .base import StepComponentSettingsWidget
from .direction_preview import DirectionPreview
from .histogram_preview import HistogramPreview

if TYPE_CHECKING:
    from ....core.step import Step
    from ....doceditor.editor import DocEditor


class EngraverSettingsWidget(DebounceMixin, StepComponentSettingsWidget):
    """UI for configuring the Rasterizer producer."""

    def __init__(
        self,
        editor: "DocEditor",
        title: str,
        target_dict: Dict[str, Any],
        page: Adw.PreferencesPage,
        step: "Step",
        **kwargs,
    ):
        producer = cast(Rasterizer, OpsProducer.from_dict(target_dict))

        super().__init__(
            editor,
            title,
            description=_("Configure how the laser engraves your image."),
            target_dict=target_dict,
            page=page,
            step=step,
            **kwargs,
        )

        # Mode selection dropdown
        mode_choices = [m.display_name for m in DepthMode]
        self.mode_row = Adw.ComboRow(
            title=_("Mode"), model=Gtk.StringList.new(mode_choices)
        )
        self.mode_row.set_selected(list(DepthMode).index(producer.depth_mode))
        self.add(self.mode_row)

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
            title=_("Engraving Method"),
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

        # --- Raster Geometry Group ---
        self._build_raster_geometry_group(producer)

        # --- Histogram (Black/White Point) ---
        self.histogram_preview = HistogramPreview()
        self.histogram_preview.set_points(
            producer.black_point, producer.white_point
        )
        self.histogram_preview.auto_mode = producer.auto_levels
        self.histogram_preview.black_point_changed.connect(
            self._on_black_point_changed
        )
        self.histogram_preview.white_point_changed.connect(
            self._on_white_point_changed
        )

        self.auto_levels_row = Adw.SwitchRow(
            title=_("Auto Levels"),
            subtitle=_("Automatically adjust black/white points"),
        )
        self.auto_levels_row.set_active(producer.auto_levels)
        self.auto_levels_row.connect(
            "notify::active", self._on_auto_levels_changed
        )
        self.add(self.auto_levels_row)

        self.histogram_row = Adw.ActionRow(
            title=_("Brightness Range"),
            subtitle=(
                _("Auto-adjusted based on image content")
                if producer.auto_levels
                else _("Drag markers to set black/white points")
            ),
        )
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

        angle_incr_adj = Gtk.Adjustment(
            lower=0,
            upper=180,
            step_increment=1,
            value=producer.angle_increment,
        )
        self.angle_incr_row = Adw.SpinRow(
            title=_("Rotate Angle Per Pass"),
            subtitle=_("Degrees to rotate each successive pass"),
            adjustment=angle_incr_adj,
            digits=0,
        )
        self.add(self.angle_incr_row)

        # Connect signals
        self.mode_row.connect("notify::selected", self._on_mode_changed)

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
        self.angle_incr_row.connect(
            "changed",
            lambda r: self._debounce(
                self._on_param_changed,
                "angle_increment",
                get_spinrow_float(r),
            ),
        )

        self._compute_and_update_histogram(producer.invert)
        self._on_mode_changed(self.mode_row, None)

    def _build_raster_geometry_group(self, producer: Rasterizer):
        """Builds the Engraving Pattern preferences group."""
        group = Adw.PreferencesGroup(
            title=_("Engraving Pattern"),
            description=_(
                "Settings that control the scan line pattern and orientation."
            ),
        )
        self.page.add(group)

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
        self.angle_scale.set_margin_top(30)
        self.angle_scale.set_margin_bottom(30)
        self.angle_scale.set_size_request(200, -1)
        self.angle_scale.connect("value-changed", self._on_angle_changed)

        self.direction_preview = DirectionPreview(
            producer.scan_angle, producer.cross_hatch
        )

        preview_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        preview_box.append(self.direction_preview)
        preview_box.append(self.angle_scale)

        self.scan_angle_row = Adw.ActionRow(
            title=_("Angle"),
            subtitle=_("Angle of scan lines in degrees"),
        )
        self.scan_angle_row.add_suffix(preview_box)
        group.add(self.scan_angle_row)

        self.cross_hatch_row = Adw.SwitchRow(
            title=_("Cross-Hatch"),
            subtitle=_("Add a second pass at 90 degrees"),
        )
        self.cross_hatch_row.set_active(producer.cross_hatch)
        self.cross_hatch_row.connect(
            "notify::active", self._on_cross_hatch_changed
        )
        group.add(self.cross_hatch_row)

        line_interval_adj = Gtk.Adjustment(
            lower=0.01,
            upper=10.0,
            step_increment=0.01,
            value=producer.line_interval_mm or 0.1,
        )
        self.line_interval_row = Adw.SpinRow(
            title=_("Line Spacing"),
            subtitle=_(
                "Distance between scan lines in machine units. "
                "Leave at 0 to use laser spot size"
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
        group.add(self.line_interval_row)

        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Engrave white areas instead of black areas"),
        )
        self.invert_row.set_active(producer.invert)
        self.invert_row.connect("notify::active", self._on_invert_changed)
        group.add(self.invert_row)

    def _compute_and_update_histogram(self, invert: bool):
        layer = self.step.layer
        if not layer:
            self.histogram_preview.update_histogram(None)
            return

        workpieces = layer.all_workpieces
        if not workpieces:
            self.histogram_preview.update_histogram(None)
            return

        workpiece = workpieces[0]
        size = workpiece.size
        if not size or size[0] <= 0 or size[1] <= 0:
            self.histogram_preview.update_histogram(None)
            return

        pixels_per_mm = self.step.pixels_per_mm
        width_px = int(size[0] * pixels_per_mm[0])
        height_px = int(size[1] * pixels_per_mm[1])

        if width_px <= 0 or height_px <= 0:
            self.histogram_preview.update_histogram(None)
            return

        surface = workpiece.render_to_pixels(width_px, height_px)
        if not surface:
            self.histogram_preview.update_histogram(None)
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

        self.histogram_preview.update_histogram(histogram)

        auto_black, auto_white = compute_auto_levels(gray_image[alpha > 0])
        self.histogram_preview.set_auto_points(auto_black, auto_white)

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

        self._debounce(self._commit_power_range_change)

    def _on_mode_changed(self, row, _):
        selected_idx = row.get_selected()
        selected_mode = list(DepthMode)[selected_idx]
        is_power_mode = selected_mode == DepthMode.POWER_MODULATION
        is_constant_power = selected_mode == DepthMode.CONSTANT_POWER
        is_dither = selected_mode == DepthMode.DITHER
        is_multi_pass = selected_mode == DepthMode.MULTI_PASS

        self.min_power_row.set_visible(is_power_mode)
        self.max_power_row.set_visible(is_power_mode)

        uses_grayscale = is_power_mode or is_multi_pass
        self.histogram_row.set_visible(uses_grayscale)
        self.auto_levels_row.set_visible(uses_grayscale)

        self.threshold_row.set_visible(is_constant_power)
        self.dither_algorithm_row.set_visible(is_dither)

        self.levels_row.set_visible(is_multi_pass)
        self.z_step_row.set_visible(is_multi_pass)
        self.angle_incr_row.set_visible(is_multi_pass)

        self._on_param_changed("depth_mode", selected_mode.name)

    def _on_black_point_changed(self, sender, black_point: int):
        self._on_param_changed("black_point", black_point)

    def _on_white_point_changed(self, sender, white_point: int):
        self._on_param_changed("white_point", white_point)

    def _on_auto_levels_changed(self, w, _):
        auto_levels = w.get_active()
        self.histogram_preview.auto_mode = auto_levels
        if auto_levels:
            self.histogram_row.set_subtitle(
                _("Auto-adjusted based on image content")
            )
        else:
            self.histogram_row.set_subtitle(
                _("Drag markers to set black/white points")
            )
        self._on_param_changed("auto_levels", auto_levels)

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
            name=_("Change engraving setting"),
            on_change_callback=lambda: self.step.updated.send(self.step),
        )
