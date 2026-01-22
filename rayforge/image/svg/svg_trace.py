from __future__ import annotations
import math
import logging
from typing import Optional, Dict, Any, Tuple

from ...core.geo import Geometry
from ...core.matrix import Matrix
from ...core.vectorization_spec import (
    TraceSpec,
    VectorizationSpec,
)
from ..base_importer import (
    ImporterFeature,
)
from .. import image_util
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
from ..tracing import trace_surface, VTRACER_PIXEL_LIMIT
from .renderer import SVG_RENDERER
from .svg_base import SvgImporterBase
from ..engine import NormalizationEngine

logger = logging.getLogger(__name__)


class SvgTraceImporter(SvgImporterBase):
    """
    Imports SVG files by rendering them to a high-resolution bitmap and then
    tracing the result.
    """

    label = "SVG (Trace Strategy)"
    mime_types = ()
    extensions = ()
    features = {ImporterFeature.BITMAP_TRACING}

    def __init__(self, data: bytes, source_file: Optional[Any] = None):
        super().__init__(data, source_file)
        self.traced_artefacts: Dict[str, Any] = {}

    def parse(self) -> Optional[ParsingResult]:
        # 1. Use base class to get dimensions and units
        basics = self._calculate_parsing_basics()
        if not basics:
            # Errors already added by basics
            return None

        # Unpack
        (
            _,
            document_bounds,
            unit_to_mm,
            untrimmed_document_bounds,
            world_frame,
        ) = basics

        # Create a temporary result to generate the background transform
        temp_result = ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=[],
            untrimmed_document_bounds=untrimmed_document_bounds,
            world_frame_of_reference=world_frame,
            background_world_transform=None,  # type: ignore
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            document_bounds, temp_result
        )

        # 2. Define single layer (Trace-specific logic)
        # For tracing, we treat the whole content as one "layer"
        layer_geometries = [
            LayerGeometry(
                layer_id="__default__",
                name="Traced Content",
                content_bounds=document_bounds,
            )
        ]

        return ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_document_bounds=untrimmed_document_bounds,
            geometry_is_relative_to_bounds=True,
            is_cropped_to_content=True,
            world_frame_of_reference=world_frame,
            background_world_transform=bg_item.world_matrix,
        )

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        if not isinstance(spec, TraceSpec):
            raise TypeError("SvgTraceImporter requires a TraceSpec")

        self.traced_artefacts = {}

        # Use the TRIMMED data bounds to determine render size.
        document_bounds = parse_result.document_bounds
        native_unit_to_mm = parse_result.native_unit_to_mm

        w_native = document_bounds[2]
        h_native = document_bounds[3]
        w_mm = w_native * native_unit_to_mm
        h_mm = h_native * native_unit_to_mm

        if w_mm <= 0 or h_mm <= 0:
            logger.warning("Cannot trace SVG: failed to determine size.")
            self.add_warning(
                _("Cannot determine valid dimensions for tracing.")
            )
            return VectorizationResult({}, parse_result)

        aspect = w_mm / h_mm if h_mm > 0 else 1.0
        TARGET_DIM = math.sqrt(VTRACER_PIXEL_LIMIT)

        if aspect >= 1.0:
            w_px = int(TARGET_DIM)
            h_px = int(TARGET_DIM / aspect)
        else:
            h_px = int(TARGET_DIM)
            w_px = int(TARGET_DIM * aspect)
        w_px, h_px = max(1, w_px), max(1, h_px)

        # Render the TRIMMED data to capture the correct content area.
        data_to_render = self.trimmed_data or self.raw_data
        vips_image = SVG_RENDERER.render_base_image(
            data_to_render, width=w_px, height=h_px
        )

        if not vips_image:
            logger.error("Failed to render SVG to vips image for tracing.")
            self.add_error(_("Failed to rasterize SVG for tracing."))
            return VectorizationResult({}, parse_result)

        if w_mm > 0 and h_mm > 0:
            xres = w_px / w_mm
            yres = h_px / h_mm
            vips_image = vips_image.copy(xres=xres, yres=yres)

        # Store the PNG data only for the duration of this import process.
        # DO NOT attach it to the final SourceAsset.
        png_data = vips_image.pngsave_buffer()
        self.traced_artefacts["png_data"] = png_data
        self.traced_artefacts["width_px"] = vips_image.width
        self.traced_artefacts["height_px"] = vips_image.height

        normalized_vips = image_util.normalize_to_rgba(vips_image)
        if not normalized_vips:
            self.add_error(_("Failed to normalize image data."))
            return VectorizationResult({}, parse_result)

        surface = image_util.vips_rgba_to_cairo_surface(normalized_vips)

        geometries = trace_surface(surface, spec)

        combined_geo = Geometry()
        if geometries:
            for geo in geometries:
                geo.close_gaps()
                combined_geo.extend(geo)

        rendered_width = vips_image.width
        rendered_height = vips_image.height
        mm_per_px_x, mm_per_px_y = image_util.get_mm_per_pixel(vips_image)

        # document_bounds contains the offset (vb_x, vb_y) in native units.
        offset_x_native = document_bounds[0]
        offset_y_native = document_bounds[1]

        offset_x_mm = offset_x_native * native_unit_to_mm
        offset_y_mm = offset_y_native * native_unit_to_mm
        offset_x_px = offset_x_mm / mm_per_px_x if mm_per_px_x > 0 else 0
        offset_y_px = offset_y_mm / mm_per_px_y if mm_per_px_y > 0 else 0

        # Shift the geometry to match the original world position
        if not combined_geo.is_empty():
            shift_matrix = Matrix.translation(offset_x_px, offset_y_px)
            combined_geo.transform(shift_matrix.to_4x4_numpy())

        trace_document_bounds = (
            offset_x_px,
            offset_y_px,
            float(rendered_width),
            float(rendered_height),
        )

        trace_untrimmed_bounds: Optional[Tuple[float, float, float, float]] = (
            None
        )
        if parse_result.untrimmed_document_bounds:
            u_native = parse_result.untrimmed_document_bounds
            # Convert untrimmed native size to trace pixels
            u_w_mm = u_native[2] * native_unit_to_mm
            u_h_mm = u_native[3] * native_unit_to_mm
            u_w_px = u_w_mm / mm_per_px_x if mm_per_px_x > 0 else 0
            u_h_px = u_h_mm / mm_per_px_y if mm_per_px_y > 0 else 0
            trace_untrimmed_bounds = (0.0, 0.0, u_w_px, u_h_px)

        # Calculate authoritative world frame for the traced image
        t_ref_bounds = trace_untrimmed_bounds or trace_document_bounds
        t_x, t_y, t_w, t_h = t_ref_bounds
        t_w_mm = t_w * mm_per_px_x
        t_h_mm = t_h * mm_per_px_x
        t_x_mm = t_x * mm_per_px_x
        t_y_mm = (t_h - (t_y + t_h)) * mm_per_px_x  # Invert Y for Y-Up
        trace_world_frame = (t_x_mm, t_y_mm, t_w_mm, t_h_mm)

        # Create temporary result to calculate background transform
        temp_trace_result = ParsingResult(
            document_bounds=trace_document_bounds,
            native_unit_to_mm=mm_per_px_x,
            is_y_down=True,
            layers=[],
            untrimmed_document_bounds=trace_untrimmed_bounds,
            world_frame_of_reference=trace_world_frame,
            background_world_transform=None,  # type: ignore
        )

        bg_item_trace = NormalizationEngine.calculate_layout_item(
            trace_document_bounds, temp_trace_result
        )

        trace_parse_result = ParsingResult(
            document_bounds=trace_document_bounds,
            native_unit_to_mm=mm_per_px_x,
            is_y_down=True,
            layers=[],
            geometry_is_relative_to_bounds=False,
            untrimmed_document_bounds=trace_untrimmed_bounds,
            world_frame_of_reference=trace_world_frame,
            background_world_transform=bg_item_trace.world_matrix,
        )

        return VectorizationResult(
            geometries_by_layer={None: combined_geo},
            source_parse_result=trace_parse_result,
        )
