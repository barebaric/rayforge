from __future__ import annotations
import logging
import cairo
import cv2
import numpy as np
from gi.repository import GLib
from ....camera.controller import CameraController
from ...canvas import CanvasElement
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from ..surface import WorkSurface


logger = logging.getLogger(__name__)

# Cap the maximum dimension for the expensive warp operation.
# This gives a good balance between quality on high-zoom and performance.
MAX_PROCESSING_DIMENSION = 2048


class CameraImageElement(CanvasElement):
    def __init__(self, controller: CameraController, **kwargs):
        # We are not a standard buffered element because we manage our own
        # surface cache to prevent flicker. We will use a custom draw() method.
        super().__init__(
            x=0, y=0, width=1.0, height=1.0, buffered=False, **kwargs
        )
        self.selectable = False
        self.controller = controller
        self.camera = controller.config  # Convenience alias for the data model
        self.controller.image_captured.connect(self._on_state_changed)
        self.camera.changed.connect(self._on_camera_model_changed)
        self.camera.settings_changed.connect(self._on_state_changed)
        self.set_visible(self.camera.enabled)

        # Cache for the processed cairo surface and its underlying data buffer.
        self._cached_surface: cairo.ImageSurface | None = None
        self._cached_surface_data: np.ndarray | None = None
        # A key representing the state that generated the cached surface.
        self._cached_key: tuple | None = None

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
        worksurface = cast("WorkSurface", self.canvas)
        is_globally_visible = worksurface._cam_visible
        should_be_visible = is_globally_visible and self.camera.enabled
        if self.visible != should_be_visible:
            self.set_visible(should_be_visible)

    def remove(self):
        """
        Extends the base remove to disconnect signals before being removed
        from the canvas. Subscription is managed by the WorkSurface.
        """
        self.controller.image_captured.disconnect(self._on_state_changed)
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
        """
        Ensures our element's dimensions always match the canvas'.
        """
        worksurface = cast("WorkSurface", self.canvas)
        self.set_size(worksurface.width_mm, worksurface.height_mm)
        return super().allocate(force)

    def draw(self, ctx: cairo.Context):
        """
        Draws the cached camera surface, scaled correctly to fit the element's
        bounds, and triggers a recomputation if the camera state has changed.
        """
        assert self.canvas, "Canvas must be set before drawing"
        worksurface = cast("WorkSurface", self.canvas)

        # 1. Draw the last valid computed surface to prevent flicker.
        if self._cached_surface:
            ctx.save()
            source_w = self._cached_surface.get_width()
            source_h = self._cached_surface.get_height()

            if (
                source_w > 0
                and source_h > 0
                and self.width > 0
                and self.height > 0
            ):
                # This logic is equivalent to the standard way of drawing a
                # surface onto a rectangle in the base CanvasElement, but is
                # reimplemented here as this element manages its own cache.

                # Scale the context so that drawing a (source_w x source_h)
                # area will fill the element's (width x height) rectangle.
                scale_x = self.width / source_w
                scale_y = self.height / source_h
                ctx.scale(scale_x, scale_y)

                # The world is Y-up, but the cairo surface is Y-down.
                # Flip the Y axis to match.
                ctx.translate(0, source_h)
                ctx.scale(1, -1)

                # Set the cached surface as the source and paint.
                ctx.set_source_surface(self._cached_surface, 0, 0)
                ctx.get_source().set_filter(cairo.FILTER_GOOD)
                ctx.paint()

            ctx.restore()

        # 2. Check if a new surface needs to be computed.
        # The output size for the recomputation should be the pixel dimensions
        # of the canvas widget itself, not the mm dimensions of the work area.
        output_width = self.canvas.get_width()
        output_height = self.canvas.get_height()

        if (
            self.controller.image_data is None
            or output_width <= 0
            or output_height <= 0
        ):
            return

        physical_area = None
        if self.camera.image_to_world:
            physical_area = (
                (0, 0),
                (worksurface.width_mm, worksurface.height_mm),
            )

        current_key = (
            id(self.controller.image_data),
            output_width,
            output_height,
            physical_area,
            self.camera.transparency,
        )

        # 3. Recompute if needed, but in a non-blocking way.
        if self._cached_key != current_key:
            GLib.idle_add(self._process_and_update_cache, current_key)

    def _process_and_update_cache(self, key_for_this_job: tuple) -> bool:
        """The actual work, to be run by GLib.idle_add."""
        image_data = self.controller.image_data
        img_data_id, width, height, p_area, transp = key_for_this_job

        if image_data is None or id(image_data) != img_data_id:
            # A newer frame has already arrived; this job is stale.
            return False  # Stop the idle add

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

        # This function should only run once per schedule.
        return False

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
            transformed_image = self.controller.get_work_surface_image(
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
