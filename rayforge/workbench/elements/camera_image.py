from __future__ import annotations
import logging
import cairo
import cv2
import numpy as np
from ...camera.models.camera import Camera
from ..canvas import CanvasElement


logger = logging.getLogger(__name__)

# Cap the maximum dimension for the expensive warp operation.
# This gives a good balance between quality on high-zoom and performance.
MAX_PROCESSING_DIMENSION = 2048


class CameraImageElement(CanvasElement):
    def __init__(self, camera: Camera, **kwargs):
        super().__init__(x=0, y=0, width=0, height=0, **kwargs)
        self.selectable = False
        self.camera = camera
        self.camera.image_captured.connect(self._on_state_changed)
        self.camera.changed.connect(self._on_camera_model_changed)
        self.camera.settings_changed.connect(self._on_state_changed)
        self.camera.subscribe()
        self.set_visible(self.camera.enabled)

        # Cache for the processed cairo surface and its underlying data buffer.
        self._cached_surface: cairo.ImageSurface | None = None
        self._cached_surface_data: np.ndarray | None = None
        # A key representing the state that generated the cached surface.
        self._cached_key: tuple | None = None
        # A flag to prevent scheduling multiple recomputations.
        self._recomputing: bool = False

    def _on_camera_model_changed(self, sender):
        """
        Handles changes in the camera model, such as being enabled or disabled.

        The element's visibility depends on both its model's `enabled` state
        and the global visibility toggle on the `WorkSurface`. This handler
        ensures the element's visibility is correctly re-evaluated when the
        model changes at runtime.
        """
        if not self.canvas:
            return  # Cannot update visibility without canvas context
        is_globally_visible = self.canvas._cam_visible
        should_be_visible = is_globally_visible and self.camera.enabled
        if self.visible != should_be_visible:
            self.set_visible(should_be_visible)

    def remove(self):
        """
        Extends the base remove to unsubscribe from the camera stream before
        being removed from the canvas.
        """
        self.camera.unsubscribe()
        self.camera.image_captured.disconnect(self._on_state_changed)
        self.camera.changed.disconnect(self._on_camera_model_changed)
        self.camera.settings_changed.disconnect(self._on_state_changed)
        super().remove()

    def _on_state_changed(self, sender):
        """
        Handles any change that makes the current cache stale.
        Invalidates the key to trigger a recompute on the next draw, but
        keeps the old surface and data to prevent flickering.
        """
        self._cached_key = None
        self.mark_dirty()
        if self.canvas:
            self.canvas.queue_draw()

    def allocate(self, force: bool = False):
        if self.parent:
            self.width = self.parent.width
            self.height = self.parent.height
        return super().allocate(force)

    def draw(self, ctx: cairo.Context):
        assert self.canvas, "Canvas must be set before drawing"
        super().draw(ctx)

        # 1. Always draw the last valid computed surface to prevent flicker.
        if self._cached_surface:
            ctx.save()
            cached_width = self._cached_surface.get_width()
            cached_height = self._cached_surface.get_height()
            if cached_width > 0 and cached_height > 0:
                scale_x = self.width / cached_width
                scale_y = self.height / cached_height
                ctx.scale(scale_x, scale_y)
                ctx.set_source_surface(self._cached_surface, 0, 0)
                ctx.paint()
            ctx.restore()

        # 2. Determine if a new, updated surface needs to be computed.
        output_width = round(
            self.canvas.root.width if self.canvas else self.width
        )
        output_height = round(
            self.canvas.root.height if self.canvas else self.height
        )

        if (
            self.camera.image_data is None
            or output_width <= 0
            or output_height <= 0
        ):
            return

        physical_area = None
        if self.camera.image_to_world:
            physical_area = (
                (0, 0),
                (self.canvas.width_mm, self.canvas.height_mm),
            )

        current_key = (
            id(self.camera.image_data),
            output_width,
            output_height,
            physical_area,
            self.camera.transparency,
        )

        # 3. Recompute if needed.
        if self._cached_key != current_key:
            self._recompute_surface(current_key)

    def _recompute_surface(self, key_for_this_job: tuple):
        """
        Performs the expensive image processing in an idle callback.
        This function's broad exception handling is crucial to prevent
        crashing the main UI loop.
        """
        if self._recomputing:
            return
        self._recomputing = True
        try:
            image_data = self.camera.image_data
            img_data_id, width, height, p_area, transp = key_for_this_job

            if image_data is None or id(image_data) != img_data_id:
                # A newer frame has already arrived; this job is stale.
                return

            # Generate both the surface and its data buffer.
            result = self._generate_surface(
                image_data, (width, height), p_area, transp
            )

            if result:
                new_surface, new_surface_data = result
                # Store both to keep the data buffer alive.
                self._cached_surface = new_surface
                self._cached_surface_data = new_surface_data
                self._cached_key = key_for_this_job
                if self.canvas:
                    self.canvas.queue_draw()

        except Exception as e:
            logger.error(f"Failed to recompute camera surface: {e}")
        finally:
            self._recomputing = False

    def _generate_surface(
        self,
        image_data: np.ndarray,
        output_size: tuple[int, int],
        physical_area: tuple | None,
        transparency: float,
    ) -> tuple[cairo.ImageSurface, np.ndarray] | None:
        """
        Contains the core image processing logic, creating a Cairo surface
        and returning it along with its data buffer.
        """
        processed_image = image_data

        if self.camera.image_to_world and physical_area:
            processing_width, processing_height = output_size
            if (
                max(processing_width, processing_height)
                > MAX_PROCESSING_DIMENSION
            ):
                scale = MAX_PROCESSING_DIMENSION / max(
                    processing_width, processing_height
                )
                processing_width = round(processing_width * scale)
                processing_height = round(processing_height * scale)

            # Let ValueError propagate to the caller's handler.
            transformed_image = self.camera.get_work_surface_image(
                output_size=(processing_width, processing_height),
                physical_area=physical_area,
            )

            if transformed_image is None:
                logger.warning(
                    "Perspective transformation failed, skipping frame."
                )
                return None
            processed_image = transformed_image

        if processed_image.shape[2] == 3:
            bgra_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2BGRA)
        else:
            bgra_image = processed_image.copy()

        if transparency < 1.0:
            if not bgra_image.flags["WRITEABLE"]:
                bgra_image = bgra_image.copy()
            bgra_image[:, :, 3] = bgra_image[:, :, 3] * transparency

        height, width, _ = bgra_image.shape

        # Create a new data buffer that Cairo will use.
        surface_data = np.copy(bgra_image)
        new_surface = cairo.ImageSurface.create_for_data(
            surface_data,  # type: ignore
            cairo.FORMAT_ARGB32,
            width,
            height,  # type: ignore
        )

        # Return both the surface and its data to ensure the buffer is not
        # garbage collected while the C-level surface is in use.
        return new_surface, surface_data
