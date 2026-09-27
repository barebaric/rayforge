"""Camera wizard page: capture target frames and solve calibration."""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ....camera.calibration.calibrator import CameraCalibrator
from ....camera.calibration.result import CalibrationResult
from ....camera.calibration.target import CalibrationTarget
from ..capture_surface import CalibrationCaptureSurface
from .base_page import CameraWizardPage

logger = logging.getLogger(__name__)


class CapturePage(CameraWizardPage):
    step_name = "capture"
    title = _("Capture Frames")
    MIN_FRAMES = 5
    RECOMMENDED_FRAMES = 8

    def __init__(self, wizard, controller):
        super().__init__(wizard, controller)
        self._target: CalibrationTarget | None = None
        self.calibrator: CameraCalibrator | None = None
        self._calibration_result: CalibrationResult | None = None
        self._calibration_applied = False
        self._capture_surface: CalibrationCaptureSurface | None = None

    @property
    def capture_button(self) -> Gtk.Button | None:
        return self._capture_btn

    @property
    def clear_button(self) -> Gtk.Button | None:
        return self._clear_btn

    @property
    def calibrate_button(self) -> Gtk.Button | None:
        return self._calibrate_btn

    def set_target(self, target: CalibrationTarget | None) -> None:
        self._target = target
        if self._capture_surface is not None:
            self._capture_surface.target = target

    def build(self) -> Gtk.Box:
        self.root = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)
        self.root.append(left_box)

        preview_frame = Gtk.Frame(
            halign=Gtk.Align.FILL,
            valign=Gtk.Align.FILL,
            hexpand=True,
            vexpand=True,
        )
        preview_frame.add_css_class("card")
        left_box.append(preview_frame)

        self._capture_surface = CalibrationCaptureSurface(
            self.controller, self._target
        )
        preview_frame.set_child(self._capture_surface)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        self.root.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(12)
        settings_box.set_margin_top(4)
        settings_box.set_margin_bottom(12)
        right_scroll.set_child(settings_box)

        info_group = Adw.PreferencesGroup(
            title=_("Instructions"),
            description=_(
                "Capture the card at different positions. Important: "
                "include the image corners and edges for accurate "
                "distortion correction."
            ),
        )
        settings_box.append(info_group)

        status_group = Adw.PreferencesGroup(
            title=_("Status"),
            description=_("Progress of the calibration capture process."),
        )
        settings_box.append(status_group)

        self.frames_row = Adw.ActionRow(title=_("Captured Frames"))
        self.frames_row.set_subtitle("0")
        status_group.add(self.frames_row)

        self.corners_row = Adw.ActionRow(title=_("Corners Detected"))
        self.corners_row.set_subtitle("0")
        status_group.add(self.corners_row)

        self.coverage_row = Adw.ActionRow(title=_("Coverage"))
        self.coverage_row.set_subtitle(_("Not started"))
        status_group.add(self.coverage_row)

        self.status_row = Adw.ActionRow(title=_("Status"))
        self.status_row.set_subtitle(_("Move card to capture more positions"))
        status_group.add(self.status_row)

        self.progress_bar = Gtk.ProgressBar(
            show_text=True,
            text=_("Capture Progress"),
            margin_top=6,
        )
        status_group.add(self.progress_bar)

        self._capture_btn = Gtk.Button(label=_("Capture Frame"))
        self._capture_btn.add_css_class("suggested-action")
        self._capture_btn.connect("clicked", self._on_capture_clicked)

        self._clear_btn = Gtk.Button(label=_("Clear"))
        self._clear_btn.add_css_class("flat")
        self._clear_btn.connect("clicked", self._on_clear_clicked)

        self._calibrate_btn = Gtk.Button(label=_("Calibrate"))
        self._calibrate_btn.set_sensitive(False)
        self._calibrate_btn.connect("clicked", self._on_calibrate_clicked)

        return self.root

    def enter(self) -> None:
        if self._capture_surface is not None:
            self._capture_surface.start()
        self._calibration_applied = False
        self._init_calibrator()

    def leave(self) -> None:
        if self._capture_surface is not None:
            self._capture_surface.stop()

    def footer_buttons(self) -> list:
        return [self._clear_btn, self._capture_btn, self._calibrate_btn]

    def _init_calibrator(self) -> None:
        if self._target is None:
            return
        if self.calibrator is not None:
            self.calibrator.clear()
        self.calibrator = CameraCalibrator(self._target)
        self.calibrator.frame_added.connect(self._on_frame_added)
        self.calibrator.frame_rejected.connect(self._on_frame_rejected)
        if self._capture_surface:
            self._capture_surface.target = self._target
        self._update_capture_status()

    def _on_capture_clicked(self, button) -> None:
        if self._capture_surface is None or self.calibrator is None:
            return
        raw_image = self.controller.raw_image_data
        if raw_image is None:
            logger.warning("No image data available")
            return
        success, _count, _ = self.calibrator.detect_and_add_frame(raw_image)
        if success:
            self._update_capture_status()

    def _on_frame_added(self, sender, count: int, total: int) -> None:
        logger.debug(f"Frame added with {count} corners (total: {total})")

    def _on_frame_rejected(self, sender, reason: str, **kwargs) -> None:
        logger.debug(f"Frame rejected: {reason}")

    def _on_clear_clicked(self, button) -> None:
        if self.calibrator:
            self.calibrator.clear()
            self._update_capture_status()

    def _update_capture_status(self) -> None:
        if self.calibrator is None:
            return
        frame_count = self.calibrator.frame_count
        total_corners = self.calibrator.total_corners

        self.frames_row.set_subtitle(f"{frame_count}")
        avg = total_corners / frame_count if frame_count > 0 else 0
        self.corners_row.set_subtitle(
            f"{total_corners} total ({avg:.0f} per frame avg)"
        )

        if frame_count > 0:
            coverage_level, _msg = self.calibrator.get_coverage_quality()
            if coverage_level == "good":
                self.coverage_row.set_subtitle(_("Good"))
            elif coverage_level == "warning":
                self.coverage_row.set_subtitle(_("Limited — reach edges"))
            else:
                self.coverage_row.set_subtitle(_("Poor — reach all corners"))
        else:
            self.coverage_row.set_subtitle(_("Not started"))

        can_calibrate, status_msg = self.calibrator.calibration_status()
        self.status_row.set_subtitle(status_msg)
        self._calibrate_btn.set_sensitive(can_calibrate)

        progress = min(1.0, frame_count / self.RECOMMENDED_FRAMES)
        self.progress_bar.set_fraction(progress)

    def _on_calibrate_clicked(self, button) -> None:
        if self.calibrator is None:
            return
        resolution = self.controller.resolution
        result = self.calibrator.calibrate(resolution)
        if result is None:
            _ready, reason = self.calibrator.calibration_status()
            self.wizard.show_error(_("Calibration Failed"), reason)
            return
        self._calibration_result = result
        self._show_result_dialog()

    def _show_result_dialog(self) -> None:
        if self._calibration_result is None:
            return
        result = self._calibration_result
        dialog = Adw.MessageDialog(
            transient_for=self.wizard,
            modal=True,
            heading=_("Calibration Complete"),
            body=_(
                "RMS Error: {rms:.4f} pixels\n"
                "Quality: {quality}\n"
                "Frames used: {frames}"
            ).format(
                rms=result.rms_error,
                quality=result.quality_rating.title(),
                frames=result.num_frames_used,
            ),
        )
        dialog.add_response("discard", _("Discard"))
        dialog.add_response("save", _("Save Calibration"))
        dialog.set_response_appearance(
            "save", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.connect("response", self._on_result_dialog_response)
        dialog.present()

    def _on_result_dialog_response(self, dialog, response_id) -> None:
        dialog.destroy()
        if response_id != "save":
            return
        self._apply_calibration()
        self.show_toast(_("Lens calibration saved."))
        self.wizard.advance()

    def _apply_calibration(self) -> None:
        if self._calibration_result is None:
            return
        self.controller.config.set_calibration_result(self._calibration_result)
        self._calibration_applied = True
        logger.info("Calibration applied to camera configuration")

    def on_next_requested(self) -> bool:
        """Warn before leaving captured frames uncalibrated.

        Capturing frames only fills the calibrator's buffer; the lens
        model is solved by the Calibrate button. Advancing without it
        silently discards the captures, so confirm the user means it.
        """
        if self._calibration_applied or self.calibrator is None:
            return True
        if self.calibrator.frame_count == 0:
            return True
        dialog = Adw.MessageDialog(
            transient_for=self.wizard,
            modal=True,
            heading=_("Skip Lens Calibration?"),
            body=_(
                "You captured {frames} frame(s) but have not run "
                "Calibrate yet. Continuing discards them and leaves "
                "the lens uncorrected."
            ).format(frames=self.calibrator.frame_count),
        )
        dialog.add_response("skip", _("Skip Anyway"))
        dialog.add_response("stay", _("Stay Here"))
        dialog.set_response_appearance(
            "skip", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.set_default_response("stay")
        dialog.connect("response", self._on_skip_dialog_response)
        dialog.present()
        return False

    def _on_skip_dialog_response(self, dialog, response_id) -> None:
        dialog.destroy()
        if response_id == "skip":
            self.wizard.advance()


__all__ = ["CapturePage"]
