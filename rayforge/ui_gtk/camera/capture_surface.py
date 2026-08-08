"""Live capture surface that overlays Charuco detections."""

import logging
from gettext import gettext as _
from typing import Optional

import cv2
import numpy as np
from gi.repository import Gdk, GdkPixbuf, GLib, Graphene, Gtk

from ...camera.calibration.charuco import CharucoBoard
from ...camera.controller import CameraController

logger = logging.getLogger(__name__)


def numpy_to_pixbuf(image: np.ndarray) -> Optional[GdkPixbuf.Pixbuf]:
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
        self._last_corners: Optional[list[tuple[float, float]]] = None
        self._last_ids: Optional[list[int]] = None

        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(750, 500)

        self.controller.subscribe()
        self.controller.image_captured.connect(self._on_image_captured)

    def stop(self) -> None:
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
            pixbuf = numpy_to_pixbuf(raw_image)
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

                        for pt in corners:
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
    ) -> Optional[tuple[list[tuple[float, float]], list[int]]]:
        if self._last_corners and self._last_ids:
            return self._last_corners, self._last_ids
        return None


__all__ = ["CalibrationCaptureSurface", "numpy_to_pixbuf"]
