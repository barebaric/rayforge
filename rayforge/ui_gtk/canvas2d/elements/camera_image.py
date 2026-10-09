from __future__ import annotations

import logging
from typing import TYPE_CHECKING, cast

import cairo
import cv2
import numpy as np
from gi.repository import GLib

from ....camera.controller import CameraController
from ....camera.outside_view import OutsideView
from ...canvas import CanvasElement

if TYPE_CHECKING:
    from ..surface import WorkSurface


logger = logging.getLogger(__name__)

# Cap the maximum dimension for the expensive warp operation.
# This gives a good balance between quality on high-zoom and performance.
MAX_PROCESSING_DIMENSION = 2048


def _processing_size(output_size: tuple[int, int]) -> tuple[int, int]:
    """Caps an output size to MAX_PROCESSING_DIMENSION."""
    processing_width, processing_height = output_size
    if max(processing_width, processing_height) > MAX_PROCESSING_DIMENSION:
        scale = MAX_PROCESSING_DIMENSION / max(
            processing_width, processing_height
        )
        processing_width = round(processing_width * scale)
        processing_height = round(processing_height * scale)
    return processing_width, processing_height


def _surface_for_data(
    data: np.ndarray,
) -> tuple[cairo.ImageSurface, np.ndarray]:
    """Wraps a BGRA buffer in a Cairo surface, returning both."""
    height, width, _ = data.shape
    surface_data = np.ascontiguousarray(data)
    surface = cairo.ImageSurface.create_for_data(
        surface_data,  # type: ignore
        cairo.FORMAT_ARGB32,
        width,
        height,  # type: ignore
    )
    return surface, surface_data


class CameraImageElement(CanvasElement):
    def __init__(
        self,
        controller: CameraController,
        outside_view_supported: bool = False,
        **kwargs,
    ):
        # We are not a standard buffered element because we manage our own
        # surface cache to prevent flicker. We will use a custom draw() method.
        # With outside view support the element draws beyond its bounds, so
        # it clips the workspace image itself instead.
        super().__init__(
            x=0,
            y=0,
            width=1.0,
            height=1.0,
            buffered=False,
            clip=not outside_view_supported,
            **kwargs,
        )
        self.selectable = False
        self.outside_view_supported = outside_view_supported
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
        # The key of the processing job that is scheduled but not yet run.
        self._pending_key: tuple | None = None
        # Cache for the optional view outside the workspace: its surface,
        # data buffer, view, and the outside signature it was made for.
        self._outside_cache: (
            tuple[cairo.ImageSurface, np.ndarray, OutsideView, tuple] | None
        ) = None

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
        outside = self._valid_outside_cache()

        # 1. Draw the last valid computed surface to prevent flicker.
        if self._cached_surface:
            ctx.save()
            if not self.clip:
                if outside:
                    # Pixel-exact edges keep the two passes seamless.
                    ctx.set_antialias(cairo.ANTIALIAS_NONE)
                ctx.rectangle(0, 0, self.width, self.height)
                ctx.clip()
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

        if outside:
            self._draw_outside(ctx, outside[0], outside[2])

        # 2. Check if a new surface needs to be computed.
        current_key = self._current_key()
        if current_key is None:
            return

        # 3. Recompute if needed, but in a non-blocking way.
        if (
            self._cached_key != current_key
            and self._pending_key != current_key
        ):
            self._pending_key = current_key
            GLib.idle_add(self._process_and_update_cache, current_key)

    def _physical_area(self) -> tuple | None:
        """The workspace area to align to, or None without alignment."""
        if not self.camera.image_to_world:
            return None
        worksurface = cast("WorkSurface", self.canvas)
        return ((0, 0), (worksurface.width_mm, worksurface.height_mm))

    def _outside_margin_request(self) -> float | None:
        """The outside view margin to render, or None to skip the pass."""
        camera = self.camera
        if (
            not self.outside_view_supported
            or not camera.outside_view_enabled
            or not camera.image_to_world
            or camera.outside_view_margin_mm <= 0
            or camera.outside_view_transparency <= 0
        ):
            return None
        return camera.outside_view_margin_mm

    def _outside_signature(self) -> tuple | None:
        """Identifies the geometry an outside view is valid for."""
        margin = self._outside_margin_request()
        if margin is None:
            return None
        return (self.camera.image_to_world, self._physical_area(), margin)

    def _valid_outside_cache(self) -> tuple | None:
        """The cached outside view, if it matches the current settings."""
        if self._outside_cache is None:
            return None
        signature = self._outside_signature()
        if signature is None or self._outside_cache[3] != signature:
            return None
        return self._outside_cache

    def _current_key(self) -> tuple | None:
        """A key for the state the cached surfaces should reflect."""
        if not self.canvas:
            return None
        # The output size for the recomputation should be the pixel
        # dimensions of the canvas widget itself, not the mm dimensions of
        # the work area.
        output_width = self.canvas.get_width()
        output_height = self.canvas.get_height()
        image_data = self.controller.image_data

        if image_data is None or output_width <= 0 or output_height <= 0:
            return None

        return (
            id(image_data),
            output_width,
            output_height,
            self._physical_area(),
            self.camera.transparency,
            self._outside_margin_request(),
        )

    def _draw_outside(
        self,
        ctx: cairo.Context,
        surface: cairo.ImageSurface,
        view: OutsideView,
    ):
        """
        Paints the outside view around, but never inside, the workspace,
        at the outside view transparency.
        """
        (x_min, y_min), (x_max, y_max) = view.area
        area_w = x_max - x_min
        area_h = y_max - y_min
        source_w = surface.get_width()
        source_h = surface.get_height()
        if area_w <= 0 or area_h <= 0 or source_w <= 0 or source_h <= 0:
            return

        ctx.save()
        ctx.set_antialias(cairo.ANTIALIAS_NONE)
        ctx.set_fill_rule(cairo.FILL_RULE_EVEN_ODD)
        ctx.rectangle(x_min, y_min, area_w, area_h)
        ctx.rectangle(0, 0, self.width, self.height)
        ctx.clip()

        ctx.translate(x_min, y_min)
        ctx.scale(area_w / source_w, area_h / source_h)
        # The world is Y-up, but the cairo surface is Y-down.
        ctx.translate(0, source_h)
        ctx.scale(1, -1)

        ctx.set_source_surface(surface, 0, 0)
        ctx.get_source().set_filter(cairo.FILTER_GOOD)
        ctx.paint_with_alpha(self.camera.outside_view_transparency)
        ctx.restore()

    def _process_and_update_cache(self, key_for_this_job: tuple) -> bool:
        """The actual work, to be run by GLib.idle_add."""
        if self._pending_key == key_for_this_job:
            self._pending_key = None

        if key_for_this_job != self._current_key():
            # A newer frame or a settings change has arrived, or the
            # element was removed; this job is stale.
            return False  # Stop the idle add

        image_data = self.controller.image_data
        assert image_data is not None
        _, width, height, p_area, transp, margin = key_for_this_job

        outside = None
        if p_area is not None and margin is not None:
            result, outside = self._generate_surfaces(
                (width, height), p_area, transp, margin
            )
        else:
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
            self._outside_cache = outside
            if self.canvas:
                self.canvas.queue_draw()

        # This function should only run once per schedule.
        return False

    def _generate_surfaces(
        self,
        output_size: tuple[int, int],
        physical_area: tuple,
        transparency: float,
        margin_mm: float,
    ) -> tuple[tuple | None, tuple | None]:
        """
        Creates the workspace surface and the outside view surface from a
        single frame. The outside result also carries the view and the
        signature it is valid for.
        """
        signature = self._outside_signature()
        workspace, view = self.controller.get_work_surface_images(
            output_size=_processing_size(output_size),
            physical_area=physical_area,
            outside_margin_mm=margin_mm,
            max_dimension=MAX_PROCESSING_DIMENSION,
        )
        if workspace is None:
            logger.warning("Image transformation failed, skipping frame.")
            return None, None

        outside = None
        if view is not None and signature is not None:
            surface, data = _surface_for_data(view.image)
            outside = (surface, data, view, signature)
        return self._surface_from_image(workspace, transparency), outside

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

        if physical_area:
            transformed_image = self.controller.get_work_surface_image(
                output_size=_processing_size(output_size),
                physical_area=physical_area,
            )

            if transformed_image is None:
                logger.warning("Image transformation failed, skipping frame.")
                return None
            processed_image = transformed_image

        return self._surface_from_image(processed_image, transparency)

    def _surface_from_image(
        self, processed_image: np.ndarray, transparency: float
    ) -> tuple[cairo.ImageSurface, np.ndarray]:
        """
        Applies the camera opacity to an image and creates a Cairo surface,
        returning it along with its data buffer.
        """
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
