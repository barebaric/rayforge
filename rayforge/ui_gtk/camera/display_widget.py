import logging
from gi.repository import Gtk, Gdk, GdkPixbuf, Graphene, Pango, PangoCairo
from ...camera.controller import CameraController


logger = logging.getLogger(__name__)


class CameraDisplay(Gtk.DrawingArea):
    def __init__(self, controller: CameraController, **kwargs):
        super().__init__(**kwargs)
        self.controller = controller
        self.camera = controller.config
        self.set_hexpand(True)
        self.set_vexpand(True)
        self.set_size_request(640, 480)
        self.marked_points = []
        self.active_point_index = -1
        self.start()
        self.connect("destroy", self.on_destroy)

    def start(self):
        """
        Starts the camera display by connecting to the image_captured signal
        and subscribing to the controller.
        """
        logger.debug(
            "CameraDisplay.start called for camera %s (instance: %s)",
            self.camera.name,
            id(self),
        )
        self.queue_draw()
        self.controller.image_captured.connect(self.on_image_captured)
        self.camera.settings_changed.connect(self.on_settings_changed)
        self.controller.subscribe()

    def stop(self):
        """
        Stops the camera display by disconnecting the image_captured signal
        and unsubscribing from the controller.
        """
        logger.debug(
            "CameraDisplay.stop called for camera %s (instance: %s)",
            self.camera.name,
            id(self),
        )
        self.controller.image_captured.disconnect(self.on_image_captured)
        self.camera.settings_changed.disconnect(self.on_settings_changed)
        self.controller.unsubscribe()

    def set_marked_points(self, points, active_point_index=-1):
        self.marked_points = points or []
        self.active_point_index = active_point_index
        self.queue_draw()

    def do_snapshot(self, snapshot):
        """
        Draw handler for the Gtk.DrawingArea. Scales and draws the camera's
        pixbuf while maintaining aspect ratio.
        """
        width, height = self.get_width(), self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        if not self.camera.enabled:
            self._draw_disabled_message(ctx, width, height)
            return

        pixbuf = self.controller.pixbuf
        if pixbuf is None:
            logger.debug("No pixbuf available for camera %s", self.camera.name)
            self._draw_no_image_message(ctx, width, height)
            return

        if width <= 0 or height <= 0:
            return

        img_width = pixbuf.get_width()
        img_height = pixbuf.get_height()

        scale = min(width / img_width, height / img_height)
        scaled_w = img_width * scale
        scaled_h = img_height * scale
        offset_x = (width - scaled_w) / 2
        offset_y = (height - scaled_h) / 2

        scaled_pixbuf = pixbuf.scale_simple(
            int(scaled_w), int(scaled_h), GdkPixbuf.InterpType.BILINEAR
        )

        if scaled_pixbuf:
            Gdk.cairo_set_source_pixbuf(ctx, scaled_pixbuf, offset_x, offset_y)
            ctx.paint()

        if self.marked_points:
            for i, point_data in enumerate(self.marked_points):
                if point_data is None:
                    continue
                x, y = point_data
                display_x = offset_x + x * scale
                display_y = offset_y + scaled_h - (y * scale)

                if i == self.active_point_index:
                    ctx.set_source_rgb(1.0, 0.5, 0.0)
                    ctx.arc(display_x, display_y, 6, 0, 2 * 3.1416)
                    ctx.fill_preserve()
                    ctx.set_source_rgb(0.8, 0.4, 0.0)
                else:
                    ctx.set_source_rgb(0.53, 0.81, 0.98)
                    ctx.arc(display_x, display_y, 6, 0, 2 * 3.1416)
                    ctx.fill_preserve()
                    ctx.set_source_rgb(0.0, 0.0, 0.5)
                ctx.set_line_width(1.5)
                ctx.stroke()

    def _draw_message(self, ctx, width, height, message):
        """Helper to draw a message in the center of the widget."""
        ctx.set_source_rgb(0.5, 0.5, 0.5)  # Grey color for text

        # Use Pango to set font options
        font_desc = Pango.FontDescription()
        font_desc.set_family("Sans")
        font_desc.set_style(Pango.Style.NORMAL)
        font_desc.set_weight(Pango.Weight.BOLD)
        font_desc.set_size(24 * Pango.SCALE)  # Pango units

        layout = PangoCairo.create_layout(ctx)
        layout.set_font_description(font_desc)

        # Get text extents
        _, _, text_width, text_height, _, _ = ctx.text_extents(message)

        # Calculate position to center the text
        x = (width - text_width) / 2
        y = (height + text_height) / 2

        ctx.move_to(x, y)
        ctx.show_text(message)

    def _draw_disabled_message(self, ctx, width, height):
        """Draws a 'Camera Disabled' message."""
        self._draw_message(ctx, width, height, "Camera Disabled")

    def _draw_no_image_message(self, ctx, width, height):
        """Draws a 'No Image' message."""
        self._draw_message(ctx, width, height, "No Image")

    def on_image_captured(self, controller):
        """Callback for the camera's image_captured signal."""
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
