from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, TYPE_CHECKING, Tuple, Dict, Any
import logging
import warnings
from dataclasses import dataclass, field

from ..core.vectorization_spec import TraceSpec

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

if TYPE_CHECKING:
    from ..core.source_asset_segment import SourceAssetSegment
    from ..core.workpiece import RenderContext
    from ..image.structures import ImportResult


logger = logging.getLogger(__name__)


@dataclass
class RenderSpecification:
    """Instructions from a Renderer on how to execute a render job."""

    width: int
    height: int
    data: bytes
    kwargs: Dict[str, Any] = field(default_factory=dict)
    crop_rect: Optional[Tuple[int, int, int, int]] = None
    apply_mask: bool = True


class Renderer(ABC):
    """
    An abstract base class for any object that can render raw data to a
    pixel image. Renderers are stateless singletons.
    """

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        """
        Calculates the strategy for rendering. Subclasses will override this.
        The default implementation is a simple pass-through.
        """
        return RenderSpecification(
            width=target_size[0],
            height=target_size[1],
            data=source_context.data,
            apply_mask=True,
        )

    @abstractmethod
    def render_base_image(
        self,
        data: bytes,
        width: int,
        height: int,
        **kwargs,
    ) -> Optional[pyvips.Image]:
        """
        Renders raw data into a pyvips Image of the specified dimensions.
        This method performs the raw format conversion (e.g. SVG->Bitmap,
        PDF->Bitmap) but does NOT handle cropping, masking, or high-level
        caching, which are handled by the WorkPiece.

        Args:
            data: The raw bytes to render.
            width: The target pixel width.
            height: The target pixel height.
            **kwargs: Optional format-specific arguments (e.g. 'boundaries'
                      for vector renderers).

        Returns:
            A pyvips.Image, or None if rendering fails.
        """
        raise NotImplementedError

    def render_preview_image(
        self,
        import_result: "ImportResult",
        target_width: int,
        target_height: int,
    ) -> Optional[pyvips.Image]:
        """
        Generates a high-resolution preview image from a full ImportResult.
        This allows a renderer to use context from parsing and vectorization
        to create the most accurate background image for the import dialog.

        The base implementation is a fallback for simple raster renderers.

        Args:
            import_result: The complete result of the import operation.
            target_width: The target pixel width for the preview.
            target_height: The target pixel height for the preview.

        Returns:
            A pyvips.Image, or None if rendering fails.
        """
        source = import_result.payload.source
        # For previews, always prefer the pre-processed (e.g., trimmed) data.
        data_to_render = source.base_render_data or source.original_data
        if not data_to_render:
            return None

        # Delegate directly to render_base_image, which is expected to handle
        # rendering to the target dimensions. This is more efficient than
        # loading a full-res image and then thumbnailing.
        return self.render_base_image(
            data=data_to_render, width=target_width, height=target_height
        )


class RasterRenderer(Renderer):
    """
    A base renderer for raster formats that handles the complex logic for
    high-resolution rendering of cropped segments.
    """

    def compute_render_spec(
        self,
        segment: Optional["SourceAssetSegment"],
        target_size: Tuple[int, int],
        source_context: "RenderContext",
    ) -> "RenderSpecification":
        """
        Calculates the render specification for a raster source. If the
        source is cropped, it computes the upscaled render dimensions and
        crop rectangle necessary to produce a sharp final image.
        """
        target_width, target_height = target_size
        source_px_dims = source_context.source_pixel_dims
        original_data = source_context.original_data

        # A traced item is treated as a raster for cropping purposes.
        is_vector = segment is not None and not isinstance(
            segment.vectorization_spec, TraceSpec
        )

        # This logic applies only to non-vector sources that are cropped and
        # for which we have the original, uncropped data and dimensions.
        if (
            segment
            and segment.crop_window_px is not None
            and not is_vector
            and original_data
            and source_px_dims
        ):
            source_w, source_h = source_px_dims
            crop_x_f, crop_y_f, crop_w_f, crop_h_f = segment.crop_window_px
            crop_w, crop_h = float(crop_w_f), float(crop_h_f)

            if crop_w > 0 and crop_h > 0:
                # Upscale the full original image so the crop area matches the
                # target pixel dimensions.
                scale_x = target_width / crop_w
                scale_y = target_height / crop_h
                render_width = max(1, int(source_w * scale_x))
                render_height = max(1, int(source_h * scale_y))

                # Calculate the crop rectangle in the upscaled image's coords.
                scaled_x = int(crop_x_f * scale_x)
                scaled_y = int(crop_y_f * scale_y)
                scaled_w = int(crop_w * scale_x)
                scaled_h = int(crop_h * scale_y)
                crop_rect = (scaled_x, scaled_y, scaled_w, scaled_h)

                return RenderSpecification(
                    width=render_width,
                    height=render_height,
                    data=original_data,  # Use original for full render
                    crop_rect=crop_rect,
                    apply_mask=True,
                )

        # Fallback for non-cropped images or if required data is missing.
        # This is a standard, direct render.
        return RenderSpecification(
            width=target_width,
            height=target_height,
            data=source_context.data,
            apply_mask=True,
        )
