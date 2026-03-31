import logging
import os
from gettext import gettext as _
from typing import Optional, Tuple, List

import cv2
import numpy as np

try:
    import pymupdf
except ImportError:
    import fitz as pymupdf
from gi.repository import Adw, Gdk, GdkPixbuf, GLib, Gtk, Graphene

from ...camera.calibration.calibrator import CameraCalibrator
from ...camera.calibration.charuco import CharucoBoard
from ...camera.calibration.result import CalibrationResult
from ...camera.controller import CameraController
from ...context import get_context
from ...shared.units.formatter import format_value
from ..shared.patched_dialog_window import PatchedDialogWindow
from ..shared.unit_spin_row import UnitSpinRowHelper

logger = logging.getLogger(__name__)


def _numpy_to_pixbuf(image: np.ndarray) -> Optional[GdkPixbuf.Pixbuf]:
    if image is None:
        return None
    if len(image.shape) == 2:
        rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    elif image.shape[2] == 3:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        return None

    height, width = rgb.shape[:2]
    rgb_bytes = GLib.Bytes.new(rgb.tobytes())
    return GdkPixbuf.Pixbuf.new_from_bytes(
        rgb_bytes,
        GdkPixbuf.Colorspace.RGB,
        False,
        8,
        width,
        height,
        width * 3,
    )


class CalibrationCaptureSurface(Gtk.Widget):
    def __init__(
        self,
        controller: CameraController,
        board: Optional[CharucoBoard] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.controller = controller
        self.board = board
        self._last_corners: Optional[List[Tuple[float, float]]] = None
        self._last_ids: Optional[List[int]] = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(750, 500)

        self.controller.subscribe()
        self.controller.image_captured.connect(self._on_image_captured)

    def stop(self):
        self.controller.unsubscribe()

    def _on_image_captured(self, _):
        self.queue_draw()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        width = self.get_width()
        height = self.get_height()
        if width <= 0 or height <= 0:
            return

        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        raw_image = self.controller.raw_image_data
        if raw_image is not None:
            pixbuf = _numpy_to_pixbuf(raw_image)
            if pixbuf:
                img_w = pixbuf.get_width()
                img_h = pixbuf.get_height()

                scale = min(width / img_w, height / img_h)
                scaled_w = img_w * scale
                scaled_h = img_h * scale
                offset_x = (width - scaled_w) / 2
                offset_y = (height - scaled_h) / 2

                ctx.save()
                ctx.translate(offset_x, offset_y)
                ctx.scale(scale, scale)
                Gdk.cairo_set_source_pixbuf(ctx, pixbuf, 0, 0)
                ctx.paint()
                ctx.restore()

                if self.board is not None:
                    detection = self.board.detect(raw_image)
                    if detection is not None:
                        corners, ids = detection
                        self._last_corners = corners
                        self._last_ids = ids

                        ctx.save()
                        ctx.translate(offset_x, offset_y)
                        ctx.scale(scale, scale)

                        for i, pt in enumerate(corners):
                            ctx.arc(pt[0], pt[1], 4, 0, 2 * 3.14159)
                            ctx.set_source_rgba(0, 1, 0, 0.8)
                            ctx.fill()
                            ctx.set_source_rgba(1, 1, 1, 1)
                            ctx.set_line_width(1.0)
                            ctx.stroke()

                        ctx.restore()
                    else:
                        self._last_corners = None
                        self._last_ids = None
        else:
            ctx.set_source_rgb(0.1, 0.1, 0.1)
            ctx.rectangle(0, 0, width, height)
            ctx.fill()

            ctx.set_source_rgb(0.5, 0.5, 0.5)
            ctx.set_font_size(14)
            text = _("Waiting for camera...")
            extents = ctx.text_extents(text)
            ctx.move_to(
                (width - extents.width) / 2,
                (height + extents.height) / 2,
            )
            ctx.show_text(text)

    @property
    def last_detection(
        self,
    ) -> Optional[Tuple[List[Tuple[float, float]], List[int]]]:
        if self._last_corners and self._last_ids:
            return self._last_corners, self._last_ids
        return None


class CalibrationWizard(PatchedDialogWindow):
    MIN_FRAMES = 5
    RECOMMENDED_FRAMES = 8
    DEFAULT_CARD_RATIO = 0.7

    def __init__(
        self,
        parent: Gtk.Window,
        controller: CameraController,
        **kwargs,
    ):
        super().__init__(
            transient_for=parent,
            modal=True,
            default_width=1150,
            default_height=750,
            title=_("Lens Calibration Wizard"),
            **kwargs,
        )

        self.controller = controller
        self._board: Optional[CharucoBoard] = None
        self._preview_pixbuf: Optional[GdkPixbuf.Pixbuf] = None
        self.calibrator: Optional[CameraCalibrator] = None
        self._calibration_result: Optional[CalibrationResult] = None
        self._capture_surface: Optional[CalibrationCaptureSurface] = None

        machine = get_context().machine
        if machine:
            unused_x, unused_y, wa_w, wa_h = machine.work_area
            self._card_width = min(100.0, wa_w * self.DEFAULT_CARD_RATIO)
            self._card_height = min(140.0, wa_h * self.DEFAULT_CARD_RATIO)
        else:
            self._card_width = 80.0
            self._card_height = 100.0

        self._setup_ui()
        self._update_card_preview()

    def _setup_ui(self):
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(content)

        header = Adw.HeaderBar()
        content.append(header)

        self._main_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            margin_top=12,
            margin_bottom=12,
            margin_start=12,
            margin_end=12,
        )
        content.append(self._main_box)

        self._stack = Gtk.Stack()
        self._stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self._main_box.append(self._stack)

        self._setup_card_page()
        self._setup_capture_page()

        self._button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12,
            halign=Gtk.Align.END,
            margin_top=12,
        )
        self._main_box.append(self._button_box)

        self._back_btn = Gtk.Button(label=_("Back"))
        self._back_btn.add_css_class("flat")
        self._back_btn.connect("clicked", self._on_back_clicked)
        self._back_btn.set_visible(False)
        self._button_box.append(self._back_btn)

        self._cancel_btn = Gtk.Button(label=_("Cancel"))
        self._cancel_btn.add_css_class("flat")
        self._cancel_btn.connect("clicked", lambda _: self.close())
        self._button_box.append(self._cancel_btn)

        self._next_btn = Gtk.Button(label=_("Next"))
        self._next_btn.add_css_class("suggested-action")
        self._next_btn.connect("clicked", self._on_next_clicked)
        self._button_box.append(self._next_btn)

        self._capture_btn = Gtk.Button(label=_("Capture Frame"))
        self._capture_btn.add_css_class("suggested-action")
        self._capture_btn.connect("clicked", self._on_capture_clicked)
        self._capture_btn.set_visible(False)
        self._button_box.append(self._capture_btn)

        self._clear_btn = Gtk.Button(label=_("Clear"))
        self._clear_btn.add_css_class("flat")
        self._clear_btn.connect("clicked", self._on_clear_clicked)
        self._clear_btn.set_visible(False)
        self._button_box.append(self._clear_btn)

        self._calibrate_btn = Gtk.Button(label=_("Calibrate"))
        self._calibrate_btn.set_sensitive(False)
        self._calibrate_btn.connect("clicked", self._on_calibrate_clicked)
        self._calibrate_btn.set_visible(False)
        self._button_box.append(self._calibrate_btn)

        self._stack.connect("notify::visible-child", self._on_page_changed)

    def _setup_card_page(self):
        page = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        self._stack.add_named(page, "card")

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)
        page.append(left_box)

        preview_frame = Gtk.Frame(
            halign=Gtk.Align.FILL,
            valign=Gtk.Align.FILL,
            hexpand=True,
            vexpand=True,
        )
        preview_frame.add_css_class("card")
        left_box.append(preview_frame)

        self.preview_image = Gtk.Picture(
            halign=Gtk.Align.CENTER,
            valign=Gtk.Align.CENTER,
        )
        self.preview_image.set_content_fit(Gtk.ContentFit.CONTAIN)
        self.preview_image.set_size_request(400, 400)
        preview_frame.set_child(self.preview_image)

        right_scroll = Gtk.ScrolledWindow()
        right_scroll.set_policy(Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC)
        page.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(24)
        settings_box.set_margin_top(4)
        settings_box.set_margin_bottom(12)
        right_scroll.set_child(settings_box)

        intro_group = Adw.PreferencesGroup(
            title=_("Instructions"),
            description=_(
                "Print a calibration card to correct lens distortion. "
                "The card size should fit within your camera view."
            ),
        )
        settings_box.append(intro_group)

        size_group = Adw.PreferencesGroup(
            title=_("Card Size"),
            description=_("Adjust to fit your work surface."),
        )
        settings_box.append(size_group)

        width_spin = Adw.SpinRow(
            title=_("Width"),
            subtitle=_("Card width"),
            adjustment=Gtk.Adjustment(
                lower=20.0,
                upper=300.0,
                step_increment=5.0,
                page_increment=20.0,
            ),
        )
        self._width_helper = UnitSpinRowHelper(
            spin_row=width_spin,
            quantity="length",
            max_value_in_base=300.0,
        )
        self._width_helper.set_value_in_base_units(self._card_width)
        self._width_helper.changed.connect(self._on_size_changed)
        size_group.add(width_spin)

        height_spin = Adw.SpinRow(
            title=_("Height"),
            subtitle=_("Card height"),
            adjustment=Gtk.Adjustment(
                lower=20.0,
                upper=300.0,
                step_increment=5.0,
                page_increment=20.0,
            ),
        )
        self._height_helper = UnitSpinRowHelper(
            spin_row=height_spin,
            quantity="length",
            max_value_in_base=300.0,
        )
        self._height_helper.set_value_in_base_units(self._card_height)
        self._height_helper.changed.connect(self._on_size_changed)
        size_group.add(height_spin)

        info_group = Adw.PreferencesGroup(
            title=_("Generated Pattern"),
            description=_("Details about the calibration pattern."),
            margin_top=12,
        )
        settings_box.append(info_group)

        self.squares_row = Adw.ActionRow(title=_("Grid Size"))
        info_group.add(self.squares_row)

        self.square_size_row = Adw.ActionRow(title=_("Square Size"))
        info_group.add(self.square_size_row)

        self._card_size_row = Adw.ActionRow(title=_("Physical Size"))
        info_group.add(self._card_size_row)

        save_pdf_row = Adw.ActionRow(
            title=_("Save to PDF"),
            subtitle=_("Export the calibration card for printing"),
        )
        save_pdf_btn = Gtk.Button(label=_("Save"), valign=Gtk.Align.CENTER)
        save_pdf_btn.connect("clicked", self._on_save_pdf)
        save_pdf_row.add_suffix(save_pdf_btn)
        save_pdf_row.set_activatable_widget(save_pdf_btn)
        info_group.add(save_pdf_row)

    def _setup_capture_page(self):
        page = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=16)
        self._stack.add_named(page, "capture")

        left_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        left_box.set_hexpand(True)
        left_box.set_vexpand(True)
        page.append(left_box)

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
        page.append(right_scroll)

        settings_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=12,
            width_request=500,
            hexpand=False,
        )
        settings_box.set_margin_start(12)
        settings_box.set_margin_end(24)
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

    def _on_size_changed(self, helper):
        self._card_width = self._width_helper.get_value_in_base_units()
        self._card_height = self._height_helper.get_value_in_base_units()
        self._update_card_preview()

    def _update_card_preview(self):
        config = CharucoBoard.recommend_config(
            card_width_mm=self._card_width,
            card_height_mm=self._card_height,
        )
        self._board = CharucoBoard(config)

        self.squares_row.set_subtitle(
            f"{config.squares_x} x {config.squares_y} squares"
        )
        self.square_size_row.set_subtitle(
            format_value(config.square_length_mm, "length")
        )

        card_w, card_h = self._board.card_size_mm
        self._card_size_row.set_subtitle(
            f"{format_value(card_w, 'length')} x "
            f"{format_value(card_h, 'length')}"
        )

        px_per_mm = 8
        img_w = int(card_w * px_per_mm)
        img_h = int(card_h * px_per_mm)
        image = self._board.generate_image(output_size=(img_w, img_h))

        if image is not None:
            self._preview_pixbuf = _numpy_to_pixbuf(image)
            self.preview_image.set_pixbuf(self._preview_pixbuf)

    def _on_page_changed(self, stack, pspec):
        visible_child = stack.get_visible_child_name()
        if visible_child == "card":
            self._cancel_btn.set_visible(True)
            self._next_btn.set_visible(True)
            self._back_btn.set_visible(False)
            self._capture_btn.set_visible(False)
            self._clear_btn.set_visible(False)
            self._calibrate_btn.set_visible(False)
        elif visible_child == "capture":
            self._cancel_btn.set_visible(True)
            self._cancel_btn.set_label(_("Cancel"))
            self._next_btn.set_visible(False)
            self._back_btn.set_visible(True)
            self._capture_btn.set_visible(True)
            self._clear_btn.set_visible(True)
            self._calibrate_btn.set_visible(True)
            self._init_calibrator()

    def _init_calibrator(self):
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

    def _on_back_clicked(self, button):
        self._stack.set_visible_child_name("card")

    def _on_next_clicked(self, button):
        if self._board is None:
            return
        self._stack.set_visible_child_name("capture")

    def _on_capture_clicked(self, button):
        if self._capture_surface is None or self.calibrator is None:
            return

        raw_image = self.controller.raw_image_data
        if raw_image is None:
            logger.warning("No image data available")
            return

        success, count, _ = self.calibrator.detect_and_add_frame(raw_image)
        if success:
            self._update_capture_status()

    def _on_frame_added(self, sender, count: int, total: int):
        logger.debug(f"Frame added with {count} corners (total: {total})")

    def _on_frame_rejected(self, sender, reason: str, **kwargs):
        logger.debug(f"Frame rejected: {reason}")

    def _on_clear_clicked(self, button):
        if self.calibrator:
            self.calibrator.clear()
            self._update_capture_status()

    def _update_capture_status(self):
        if self.calibrator is None:
            return

        frame_count = self.calibrator.frame_count
        total_corners = self.calibrator.total_corners

        self.frames_row.set_subtitle(f"{frame_count}")

        avg_per_frame = total_corners / frame_count if frame_count > 0 else 0
        self.corners_row.set_subtitle(
            f"{total_corners} total ({avg_per_frame:.0f} per frame avg)"
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

    def _on_calibrate_clicked(self, button):
        if self.calibrator is None:
            return

        resolution = self.controller.resolution
        result = self.calibrator.calibrate(resolution)

        if result is None:
            _ready, reason = self.calibrator.calibration_status()
            self._show_error(
                _("Calibration Failed"),
                reason,
            )
            return

        self._calibration_result = result
        self._show_result_dialog()

    def _show_error(self, title: str, message: str):
        dialog = Adw.MessageDialog(
            transient_for=self,
            modal=True,
            heading=title,
            body=message,
        )
        dialog.add_response("ok", _("OK"))
        dialog.present()

    def _show_result_dialog(self):
        if self._calibration_result is None:
            return

        result = self._calibration_result

        dialog = Adw.MessageDialog(
            transient_for=self,
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

    def _on_result_dialog_response(self, dialog, response_id):
        dialog.destroy()
        if response_id == "save":
            self._apply_calibration()
        self.close()

    def _apply_calibration(self):
        if self._calibration_result is None:
            return

        self.controller.config.set_calibration_result(self._calibration_result)
        logger.info("Calibration applied to camera configuration")

    def _on_save_pdf(self, button):
        dialog = Gtk.FileDialog()
        dialog.set_title(_("Save Calibration Card"))
        dialog.set_initial_name("calibration_card.pdf")

        dialog.save(self, None, self._on_save_dialog_response)

    def _on_save_dialog_response(self, dialog, result):
        try:
            file = dialog.save_finish(result)
            if file:
                self._save_pdf(file.get_path())
        except GLib.Error:
            pass

    def _save_pdf(self, filepath: str):
        if self._board is None:
            return

        card_w_mm, card_h_mm = self._board.card_size_mm

        dpi = 300
        px_per_mm = dpi / 25.4
        img_w = int(card_w_mm * px_per_mm)
        img_h = int(card_h_mm * px_per_mm)

        image = self._board.generate_image(output_size=(img_w, img_h))
        if image is None:
            return

        page_w = card_w_mm / 25.4 * 72
        page_h = card_h_mm / 25.4 * 72

        doc = pymupdf.open()
        page = doc.new_page(width=page_w, height=page_h)

        temp_path = filepath.replace(".pdf", "_temp.png")
        cv2.imwrite(temp_path, image)

        rect = pymupdf.Rect(0, 0, page_w, page_h)
        page.insert_image(rect, filename=temp_path)

        doc.save(filepath)
        doc.close()

        os.remove(temp_path)

        logger.info(f"Calibration card saved to {filepath}")

        toast = Adw.Toast(title=_("Calibration card saved"))
        self.toast_overlay.add_toast(toast)

    def close(self):
        if self._capture_surface:
            self._capture_surface.stop()
        super().close()
