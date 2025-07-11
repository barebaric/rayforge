import logging
from gi.repository import Gtk, Gdk, GdkPixbuf, cairo
from ..models.camera import Camera


logger = logging.getLogger(__name__)


class CameraDisplay(Gtk.DrawingArea):
    def __init__(self, camera: Camera):
        super().__init__()
        self.camera = camera
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(640, 480)
        self.set_draw_func(self.on_draw)
        self.start()
        self.connect("destroy", self.on_destroy)

    def start(self):
        """
        Starts the camera display by connecting to the image_captured signal
        and enabling the camera.
        """
        logger.debug(
            "CameraDisplay.start called for camera %s (instance: %s)",
            self.camera.name, id(self)
        )
        self.queue_draw()
        self.camera.image_captured.connect(self.on_image_captured)
        self.camera.settings_changed.connect(self.on_settings_changed)

    def stop(self):
        """
        Stops the camera display by disconnecting the image_captured signal.
        This method should be called when the display is no longer needed.
        """
        logger.debug(
            "CameraDisplay.stop called for camera %s (instance: %s)",
            self.camera.name, id(self)
        )
        self.camera.image_captured.disconnect(self.on_image_captured)
        self.camera.settings_changed.disconnect(self.on_settings_changed)

    def on_draw(self, widget, cr, width, height):
        """
        Draw handler for the Gtk.DrawingArea. Scales and draws the camera's
        pixbuf.
        """
        if not self.camera.enabled:
            self._draw_disabled_message(cr, width, height)
            return False

        pixbuf = self.camera.pixbuf
        if pixbuf is None:
            logger.debug("No pixbuf available for camera %s",
                         self.camera.name)
            self._draw_no_image_message(cr, width, height)
            return False

        if width <= 0 or height <= 0:
            return False

        scaled_pixbuf = pixbuf.scale_simple(
            width,
            height,
            GdkPixbuf.InterpType.BILINEAR
        )
        logger.debug(f"Scaled pixbuf to {width}x{height} for camera "
                     f"{self.camera.name}")

        Gdk.cairo_set_source_pixbuf(cr, scaled_pixbuf, 0, 0)
        cr.paint()
        return False

    def _draw_message(self, cr, width, height, message):
        """Helper to draw a message in the center of the widget."""
        cr.set_source_rgb(0.5, 0.5, 0.5)  # Grey color for text
        cr.select_font_face("Sans", cairo.FontSlant.NORMAL,
                            cairo.FontWeight.BOLD)
        cr.set_font_size(24)

        # Get text extents
        _, _, text_width, text_height, _, _ = cr.text_extents(message)

        # Calculate position to center the text
        x = (width - text_width) / 2
        y = (height + text_height) / 2

        cr.move_to(x, y)
        cr.show_text(message)

    def _draw_disabled_message(self, cr, width, height):
        """Draws a 'Camera Disabled' message."""
        self._draw_message(cr, width, height, "Camera Disabled")

    def _draw_no_image_message(self, cr, width, height):
        """Draws a 'No Image' message."""
        self._draw_message(cr, width, height, "No Image")

    def on_image_captured(self, camera):
        """Callback for the camera's image_captured signal."""
        logger.debug(
            f"CameraDisplay.on_image_captured called for camera "
            f"{self.camera.name} (instance: {id(self)})"
        )
        self.queue_draw()

    def on_settings_changed(self, camera):
        """Callback for the camera's settings_changed signal."""
        logger.debug(
            f"Settings changed, redrawing for camera {self.camera.name}"
        )
        self.queue_draw()

    def on_destroy(self, widget):
        """Callback for when the CameraDisplay widget is destroyed."""
        logger.debug(
            f"CameraDisplay.on_destroy called for camera "
            f"{self.camera.name} (instance: {id(self)})"
        )
        self.stop()
