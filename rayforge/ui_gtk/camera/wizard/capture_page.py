"""Camera wizard page: capture Charuco frames and solve calibration."""

import logging
from gettext import gettext as _
from typing import Optional

from gi.repository import Adw, Gtk

from ....camera.calibration.calibrator import CameraCalibrator
from ....camera.calibration.charuco import CharucoBoard
from ....camera.calibration.result import CalibrationResult
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
        self._board: Optional[CharucoBoard] = None
        self.calibrator: Optional[CameraCalibrator] = None
        self._calibration_result: Optional[CalibrationResult] = None
        self._capture_surface: Optional[CalibrationCaptureSurface] = None

    @property
    def capture_button(self) -> Optional[Gtk.Button]:
        return self._capture_btn

    @property
    def clear_button(self) -> Optional[Gtk.Button]:
        return self._clear_btn

    @property
    def calibrate_button(self) -> Optional[Gtk.Button]:
        return self._calibrate_btn

    def set_board(self, board: Optional[CharucoBoard]) -> None:
        self._board = board
        if self._capture_surface is not None:
            self._capture_surface.board = board

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
            self.controller, self._board
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
        self._init_calibrator()

    def leave(self) -> None:
        pass

    def footer_buttons(self) -> list:
        return [self._clear_btn, self._capture_btn, self._calibrate_btn]

    def _init_calibrator(self) -> None:
        if self._board is None:
            return
        if self.calibrator is not None:
            self.calibrator.clear()
        self.calibrator = CameraCalibrator(self._board)
        self.calibrator.frame_added.connect(self._on_frame_added)
        self.calibrator.frame_rejected.connect(self._on_frame_rejected)
        if self._capture_surface:
            self._capture_surface.board = self._board
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
        if response_id == "save":
            self._apply_calibration()
        self.wizard.close()

    def _apply_calibration(self) -> None:
        if self._calibration_result is None:
            return
        self.controller.config.set_calibration_result(self._calibration_result)
        logger.info("Calibration applied to camera configuration")

    def stop(self) -> None:
        if self._capture_surface is not None:
            self._capture_surface.stop()


__all__ = ["CapturePage"]
