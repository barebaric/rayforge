import logging
import numpy as np
from gi.repository import Gtk, GdkPixbuf, GLib
from ..models.camera import Camera


logger = logging.getLogger(__name__)


class CameraDisplay(Gtk.Box):
    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.set_orientation(Gtk.Orientation.VERTICAL)
        self.set_spacing(6)

        self.image = Gtk.Image()
        self.image.set_hexpand(True)
        self.image.set_vexpand(True)
        self.set_size_request(320, 240)  # Set a reasonable default size
        self.append(self.image)

        # For now, we'll just call update_display manually.
        # TODO: More frequent auto-updates
        self.update_display()

    def update_display(self):
        image_data = self.camera.image_data
        if image_data is None or not isinstance(image_data, np.ndarray):
            self.image.set_from_icon_name("image-missing")
            logger.info(
                "No image data available for camera %s, "
                "displaying placeholder.",
                self.camera.name
            )
            return

        try:
            height, width, channels = image_data.shape
            rowstride = width * channels

            if image_data.dtype != np.uint8:
                image_data = image_data.astype(np.uint8)

            # Assuming RGB or RGBA. If BGR, conversion is needed.
            has_alpha = True if channels == 4 else False
            pixbuf = GdkPixbuf.Pixbuf.new_from_bytes(
                GLib.Bytes.new(image_data.tobytes()),
                GdkPixbuf.Colorspace.RGB,
                has_alpha,
                8,  # bits per sample
                width,
                height,
                rowstride
            )
            self.image.set_from_pixbuf(pixbuf)
            logger.info(
                "Image for camera %s displayed successfully.", self.camera.name
            )

        except Exception as e:
            logger.error(
                "Error processing image data for camera %s: %s",
                self.camera.name, e
            )
            self.image.set_from_icon_name("image-missing")

    def apply_transformation(self):
        # This method will apply self.camera.transform_matrix to the displayed
        # image and will be implemented in Phase 4.
        logger.info(
            "Applying transformation (placeholder) for camera %s: %s",
            self.camera.name, self.camera.transform_matrix
        )
