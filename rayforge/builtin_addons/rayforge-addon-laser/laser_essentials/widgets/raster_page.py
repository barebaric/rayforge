from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import numpy as np
from gi.repository import Adw, GLib, GObject, Gtk
from raygeo.image.grayscale import compute_auto_levels
from raygeo.image.scan import ScanMode

from rayforge.image.dither import DitherAlgorithm
from rayforge.image.util import (
    get_visible_grayscale_values,
)
from rayforge.machine.models.laser import LaserHead
from rayforge.pipeline.stage.assembler_helpers import DepthMode
from rayforge.ui_gtk.shared.direction_preview import DirectionPreview
from rayforge.ui_gtk.shared.histogram_preview import HistogramPreview
from rayforge.ui_gtk.shared.pref_rows import (
    AngleSpinRow,
    LengthSpinRow,
    SpinRow,
)
from rayforge.ui_gtk.shared.slider import create_slider, create_slider_row

from .rows import LaserStepSettingsPage

_SCAN_MODES = [ScanMode.SEGMENTED, ScanMode.FULL_SWEEP]

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor


class RasterSettingsPage(LaserStepSettingsPage):
    """UI for configuring the EngraveStep."""

    cut_speed_title = _("Engrave Speed")

    def __init__(
        self,
        editor: "DocEditor",
        step: Any,
    ):
        super().__init__(editor, step)
        engrave_group = self.add_section(
            _("Engrave"),
            description=_("Raster the image onto the material."),
        )
        histogram_group = self.add_section(
            _("Power"),
            description=_("Power modulation and brightness range."),
        )

        mode_choices = [m.display_name for m in DepthMode]
        self.mode_row = Adw.ComboRow(
            title=_("Mode"), model=Gtk.StringList.new(mode_choices)
        )
        self.mode_row.set_selected(
            list(DepthMode).index(DepthMode[step.depth_mode])
        )
        self._add(engrave_group, self.mode_row)

        # --- Threshold (for Constant Power mode) ---
        threshold_adj = Gtk.Adjustment(
            lower=0,
            upper=255,
            step_increment=1,
            page_increment=10,
            value=step.threshold,
        )
        self.threshold_row, self.threshold_scale = create_slider_row(
            title=_("Threshold"),
            adjustment=threshold_adj,
            subtitle=_("Brightness cutoff for black/white (0-255)"),
            digits=0,
            on_value_changed=lambda s: self._on_threshold_changed(s),
        )
        self._add(engrave_group, self.threshold_row)

        # --- Dither Algorithm (for Dither mode) ---
        dither_choices = [m.display_name for m in DitherAlgorithm]
        self.dither_algorithm_row = Adw.ComboRow(
            title=_("Engraving Method"),
            subtitle=_("Algorithm for converting grayscale to binary"),
            model=Gtk.StringList.new(dither_choices),
        )
        current_algo = step.dither_algorithm or DitherAlgorithm.FLOYD_STEINBERG
        self.dither_algorithm_row.set_selected(
            list(DitherAlgorithm).index(current_algo)
        )
        self.dither_algorithm_row.connect(
            "notify::selected", self._on_dither_algorithm_changed
        )
        self._add(engrave_group, self.dither_algorithm_row)

        # --- Raster Geometry ---
        self._build_raster_geometry_group(engrave_group)

        # --- Histogram (Black/White Point) ---
        self.histogram_preview = HistogramPreview()
        self.histogram_preview.set_points(step.black_point, step.white_point)
        self.histogram_preview.auto_mode = step.auto_levels
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
        self.auto_levels_row.set_active(step.auto_levels)
        self.auto_levels_row.connect(
            "notify::active", self._on_auto_levels_changed
        )
        self._add(histogram_group, self.auto_levels_row)

        self.histogram_row = Adw.ActionRow(
            title=_("Brightness Range"),
            subtitle=(
                _("Auto-adjusted based on image content")
                if step.auto_levels
                else _("Drag markers to set black/white points")
            ),
        )
        self.histogram_row.add_suffix(self.histogram_preview)
        self._add(histogram_group, self.histogram_row)

        # --- Power Modulation Settings ---
        self.min_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=0.1,
            value=step.min_power_level * 100,
        )
        self.min_power_row, self.min_power_scale = create_slider_row(
            title=_("Min Power"),
            adjustment=self.min_power_adj,
            subtitle=_(
                "Power for lightest areas, as a % of the step's main power"
            ),
            digits=1,
        )
        self._add(histogram_group, self.min_power_row)

        self.max_power_adj = Gtk.Adjustment(
            lower=0,
            upper=100,
            step_increment=0.1,
            value=step.max_power_level * 100,
        )
        self.max_power_row, self.max_power_scale = create_slider_row(
            title=_("Max Power"),
            adjustment=self.max_power_adj,
            subtitle=_(
                "Power for darkest areas, as a % of the step's main power"
            ),
            digits=1,
        )
        self._add(histogram_group, self.max_power_row)

        self.power_levels_row = SpinRow(
            _("Power Levels"),
            _("Number of discrete power steps (lower = fewer moves)"),
            lower=2,
            upper=256,
            digits=0,
            value=step.num_power_levels,
        )
        self.power_levels_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_param_changed,
                "num_power_levels",
                r.get_int_value(),
            ),
        )
        self._add(histogram_group, self.power_levels_row)

        self._update_power_labels(step.invert)

        # --- Multi-Pass Settings ---
        self.levels_row = SpinRow(
            _("Number of Depth Levels"),
            lower=1,
            upper=255,
            value=step.num_depth_levels,
        )
        self._add(engrave_group, self.levels_row)

        self.z_step_row = LengthSpinRow(
            _("Z Step-Down per Level"),
            upper=50,
            value_in_base=step.z_step_down,
        )
        self._add(engrave_group, self.z_step_row)
        self.z_step_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_param_changed,
                "z_step_down",
                r.get_value_in_base_units(),
            )
        )

        self.angle_incr_row = AngleSpinRow(
            _("Rotate Angle Per Pass"),
            _("Degrees to rotate each successive pass"),
            lower=0,
            upper=180,
            digits=0,
            value=step.angle_increment,
        )
        self._add(engrave_group, self.angle_incr_row)

        # Connect signals
        self.mode_row.connect("notify::selected", self._on_mode_changed)

        self.min_power_handler_id = self.min_power_scale.connect(
            "value-changed", self._on_min_power_scale_changed
        )
        self.max_power_handler_id = self.max_power_scale.connect(
            "value-changed", self._on_max_power_scale_changed
        )

        self.levels_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_param_changed,
                "num_depth_levels",
                r.get_int_value(),
            ),
        )
        self.angle_incr_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_param_changed,
                "angle_increment",
                r.get_value(),
            ),
        )

        GLib.idle_add(self._compute_and_update_histogram, step.invert)
        self._on_mode_changed(self.mode_row, None)

    def _add(self, group, widget):
        self._rows.append(widget)
        group.add(widget)

    def _build_raster_geometry_group(self, group):
        """Builds the Engraving Pattern preferences group."""

        # --- Cross-Hatch & Scan Angle with Preview ---
        angle_adj = Gtk.Adjustment(
            lower=0,
            upper=360,
            step_increment=0.1,
            page_increment=15,
            value=self.step.scan_angle,
        )
        self.angle_scale = create_slider(
            adjustment=angle_adj,
            digits=1,
            draw_value=True,
            on_value_changed=lambda s: self._on_angle_changed(s),
        )

        self.direction_preview = DirectionPreview(
            self.step.scan_angle, self.step.cross_hatch
        )

        preview_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        preview_box.append(self.direction_preview)
        preview_box.append(self.angle_scale)

        self.scan_angle_row = Adw.ActionRow(
            title=_("Angle"),
            subtitle=_("Angle of scan lines in degrees"),
        )
        self.scan_angle_row.add_suffix(preview_box)
        self._add(group, self.scan_angle_row)

        self.cross_hatch_row = Adw.SwitchRow(
            title=_("Cross-Hatch"),
            subtitle=_("Add a second pass at 90 degrees"),
        )
        self.cross_hatch_row.set_active(self.step.cross_hatch)
        self.cross_hatch_row.connect(
            "notify::active", self._on_cross_hatch_changed
        )
        self._add(group, self.cross_hatch_row)

        scan_mode_choices = [
            _("Segmented"),
            _("Full Sweep"),
        ]
        self.scan_mode_row = Adw.ComboRow(
            title=_("Scan Mode"),
            subtitle=_(
                "Segmented: moves between content regions. "
                "Full Sweep: scans full width with laser toggling."
            ),
            model=Gtk.StringList.new(scan_mode_choices),
        )
        self.scan_mode_row.set_selected(
            _SCAN_MODES.index(getattr(ScanMode, self.step.scan_mode))
        )
        self.scan_mode_row.connect(
            "notify::selected", self._on_scan_mode_changed
        )
        self._add(group, self.scan_mode_row)

        head = self.get_selected_head()
        laser = head if isinstance(head, LaserHead) else None
        default_line_interval_mm = laser.spot_size_mm[1] if laser else 0.1
        default_sample_interval_mm = (
            laser.spot_size_mm[0] / 2.0 if laser else 0.05
        )

        self.line_interval_row = LengthSpinRow(
            _("Line Spacing"),
            _("Distance between scan lines"),
            lower=0.001,
            upper=20.0,
            step_increment=0.01,
            digits=3,
            value_in_base=(
                self.step.line_interval_mm
                if self.step.line_interval_mm is not None
                else default_line_interval_mm
            ),
        )
        self._add(group, self.line_interval_row)
        self.line_interval_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_line_interval_changed,
                r.get_value_in_base_units(),
            )
        )

        self.sample_interval_row = LengthSpinRow(
            _("Sample Interval"),
            _(
                "Distance between power samples along scan line. "
                "Lower values improve accuracy, but increase output size. "
            ),
            lower=0.001,
            upper=20.0,
            step_increment=0.01,
            digits=3,
            value_in_base=(
                self.step.sample_interval_mm
                if self.step.sample_interval_mm is not None
                else default_sample_interval_mm
            ),
        )
        self._add(group, self.sample_interval_row)
        self.sample_interval_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_sample_interval_changed,
                r.get_value_in_base_units(),
            )
        )

        default_dot_width_correction_mm = (
            laser.spot_size_mm[0] / 2.0 if laser else 0.05
        )
        self.dot_width_correction_row = LengthSpinRow(
            _("Dot Width Correction"),
            _(
                "Reduces engrave length at both ends to compensate "
                "for physical dot width"
            ),
            upper=5.0,
            step_increment=0.01,
            digits=3,
            value_in_base=(
                self.step.dot_width_correction_mm
                if self.step.dot_width_correction_mm is not None
                else default_dot_width_correction_mm
            ),
        )
        self._add(group, self.dot_width_correction_row)
        self.dot_width_correction_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_dot_width_correction_changed,
                r.get_value_in_base_units(),
            )
        )

        self.bidir_x_offset_row = LengthSpinRow(
            _("Bidirectional Scan Offset"),
            _(
                "Corrects X misalignment between left-to-right and "
                "right-to-left raster passes"
            ),
            lower=-5.0,
            upper=5.0,
            step_increment=0.01,
            digits=3,
            value_in_base=self.step.bidir_x_offset_mm,
        )
        self._add(group, self.bidir_x_offset_row)
        self.bidir_x_offset_row.value_changed.connect(
            lambda r: self._debounce(
                self._on_bidir_x_offset_changed,
                r.get_value_in_base_units(),
            )
        )

        self.invert_row = Adw.SwitchRow(
            title=_("Invert"),
            subtitle=_("Engrave white areas instead of black areas"),
        )
        self.invert_row.set_active(self.step.invert)
        self.invert_row.connect("notify::active", self._on_invert_changed)
        self._add(group, self.invert_row)

    def _compute_and_update_histogram(self, invert: bool):
        layer = self.step.layer
        if not layer:
            self.histogram_preview.update_histogram(None)
            return

        workpieces = layer.all_workpieces
        if not workpieces:
            self.histogram_preview.update_histogram(None)
            return

        pixels_per_mm = self.step.pixels_per_mm
        all_gray_values = []

        for workpiece in workpieces:
            size = workpiece.size
            if not size or size[0] <= 0 or size[1] <= 0:
                continue

            width_px = int(size[0] * pixels_per_mm[0])
            height_px = int(size[1] * pixels_per_mm[1])

            if width_px <= 0 or height_px <= 0:
                continue

            max_px = 256
            if width_px > max_px or height_px > max_px:
                scale = min(max_px / width_px, max_px / height_px)
                width_px = max(int(width_px * scale), 1)
                height_px = max(int(height_px * scale), 1)

            surface = workpiece.render_to_pixels(width_px, height_px)
            if not surface:
                continue

            gray_values = get_visible_grayscale_values(surface, invert)
            if gray_values.size > 0:
                all_gray_values.append(gray_values)

        if not all_gray_values:
            self.histogram_preview.update_histogram(None)
            return

        combined_gray = np.concatenate(all_gray_values)

        histogram, _ = np.histogram(combined_gray, bins=64, range=(0, 255))

        self.histogram_preview.update_histogram(histogram)

        auto_black, auto_white = compute_auto_levels(combined_gray)
        self.histogram_preview.set_auto_points(auto_black, auto_white)

    def _commit_power_range_change(self):
        """Commits the min/max power to the step via commands."""
        min_p = self.min_power_adj.get_value() / 100.0
        max_p = self.max_power_adj.get_value() / 100.0

        min_changed = abs(self.step.min_power_level - min_p) > 1e-6
        max_changed = abs(self.step.max_power_level - max_p) > 1e-6

        if not min_changed and not max_changed:
            return

        with self.history_manager.transaction(_("Change Power Range")):
            if min_changed:
                self.set_step_property("min_power_level", min_p)
            if max_changed:
                self.set_step_property("max_power_level", max_p)

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

    def _on_mode_changed(self, row, pspec):
        selected_idx = row.get_selected()
        selected_mode = list(DepthMode)[selected_idx]
        is_power_mode = selected_mode == DepthMode.POWER_MODULATION
        is_constant_power = selected_mode == DepthMode.CONSTANT_POWER
        is_dither = selected_mode == DepthMode.DITHER
        is_multi_pass = selected_mode == DepthMode.MULTI_PASS

        self.min_power_row.set_visible(is_power_mode)
        self.max_power_row.set_visible(is_power_mode)
        self.sample_interval_row.set_visible(is_power_mode)
        self.power_levels_row.set_visible(is_power_mode)

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

    def _on_auto_levels_changed(self, w, pspec):
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

    def _on_dither_algorithm_changed(self, row, pspec):
        selected_idx = row.get_selected()
        selected_algo = list(DitherAlgorithm)[selected_idx]
        self._on_param_changed("dither_algorithm", selected_algo)

    def _on_threshold_changed(self, scale):
        value = int(scale.get_value())
        self._debounce(self._on_param_changed, "threshold", value)

    def _on_angle_changed(self, scale):
        value = float(scale.get_value())
        self.direction_preview.update(value, self.cross_hatch_row.get_active())
        self._debounce(self._on_param_changed, "scan_angle", value)

    def _on_cross_hatch_changed(self, w, pspec):
        cross_hatch = w.get_active()
        self.direction_preview.update(
            self.angle_scale.get_value(), cross_hatch
        )
        self._on_param_changed("cross_hatch", cross_hatch)

    def _on_scan_mode_changed(self, row, pspec):
        selected_idx = row.get_selected()
        selected_mode = _SCAN_MODES[selected_idx]
        self._on_param_changed("scan_mode", selected_mode.name)

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

    def _on_line_interval_changed(self, value: float | None):
        if value is not None and value <= 0:
            value = None
        self._on_param_changed("line_interval_mm", value)

    def _on_sample_interval_changed(self, value: float | None):
        if value is not None and value <= 0:
            value = None
        self._on_param_changed("sample_interval_mm", value)

    def _on_dot_width_correction_changed(self, value: float | None):
        # Unlike line/sample interval, 0.0 is a meaningful explicit value
        # here (no correction), not a sentinel for "reset to auto".
        self._on_param_changed("dot_width_correction_mm", value or 0.0)

    def _on_bidir_x_offset_changed(self, value: float | None):
        self._on_param_changed("bidir_x_offset_mm", value or 0.0)

    def _on_param_changed(self, key: str, value: Any):
        self.set_step_property(key, value)
