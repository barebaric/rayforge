from __future__ import annotations
import logging
import cairo
import cv2
from rayforge.widgets.canvas import CanvasElement
from rayforge.models.camera import Camera


logger = logging.getLogger(__name__)


class CameraImageElement(CanvasElement):
    def __init__(self, camera: Camera, **kwargs):
        super().__init__(x=0, y=0, width=0, height=0, **kwargs)
        self.selectable = False
        self.camera = camera
        self.camera.image_captured.connect(self._on_image_captured)
        self.camera.settings_changed.connect(self._on_settings_changed)
        self.set_visible(self.camera.enabled)

    def _on_image_captured(self, sender):
        # logger.debug("CameraImageElement: Image captured, marking dirty.")
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def _on_settings_changed(self, sender):
        logger.debug("CameraImageElement: Settings changed, marking dirty.")
        self.set_visible(self.camera.enabled)
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def allocate(self, force: bool = False):
        if self.parent:
            self.width = self.parent.width
            self.height = self.parent.height
        return super().allocate(force)

    def render(
        self,
        clip: tuple[int, int, int, int] | None = None,
        force: bool = False,
    ):
        # Call super's render to handle dirty flags and children
        super().render(clip=clip, force=force)

        if not self.visible or self.surface is None or self.canvas is None:
            logger.debug(
                "Render skipped: not visible, no surface, or no canvas"
            )
            return

        image_data = self.camera.image_data
        if image_data is None:
            logger.warning("No image data available")
            return

        # Use canvas dimensions for output size
        output_width = self.canvas.root.width if self.canvas else self.width
        output_height = self.canvas.root.height if self.canvas else self.height

        if output_width <= 0 or output_height <= 0:
            logger.warning(
                "Invalid output dimensions: width=%d, height=%d",
                output_width,
                output_height,
            )
            return

        # Apply perspective transformation if corresponding points are set
        if self.camera.image_to_world:
            try:
                physical_area = (
                    (0, 0),
                    (self.canvas.width_mm, self.canvas.height_mm),
                )
                image_data = self.camera.get_work_surface_image(
                    output_size=(output_width, output_height),
                    physical_area=physical_area,
                )
                if image_data is None:
                    logger.error("get_work_surface_image returned None")
                    return
            except ValueError as e:
                logger.warning(
                    f"Failed to apply perspective transformation: {e}"
                )
                # Fallback to raw image

        # Convert transformed NumPy array (BGR) to BGRA for Cairo's ARGB32
        # format
        try:
            cairo_data = cv2.cvtColor(image_data, cv2.COLOR_BGR2BGRA)
        except cv2.error as e:
            logger.error(f"Failed to convert image to BGRA: {e}")
            return

        height, width, _ = cairo_data.shape

        # Create cairo.ImageSurface from NumPy array
        try:
            image_surface = cairo.ImageSurface.create_for_data(
                cairo_data, cairo.FORMAT_ARGB32, width, height
            )
        except cairo.Error as e:
            logger.error(f"Failed to create Cairo surface: {e}")
            return

        # Draw the transformed image onto this element's Cairo context
        ctx = cairo.Context(self.surface)
        ctx.save()
        ctx.set_source_surface(image_surface, 0, 0)
        ctx.paint_with_alpha(self.camera.transparency)
        ctx.restore()
