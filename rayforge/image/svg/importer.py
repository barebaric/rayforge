import io
import math
import logging
from typing import List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

from svgelements import (
    SVG,
    Arc,
    Close,
    CubicBezier,
    Line,
    Move,
    Path,
    QuadraticBezier,
)

from ...core.geo import Geometry
from ...core.source_asset_segment import SourceAssetSegment
from ...core.item import DocItem
from ...core.matrix import Matrix
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ...core.workpiece import WorkPiece
from ..base_importer import Importer, ImportPayload
from .. import image_util
from ..tracing import trace_surface, VTRACER_PIXEL_LIMIT
from .renderer import SVG_RENDERER
from .svgutil import PPI, get_natural_size, trim_svg

logger = logging.getLogger(__name__)


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)

    def get_doc_items(
        self, vectorization_spec: Optional["VectorizationSpec"] = None
    ) -> Optional[ImportPayload]:
        """
        Generates DocItems from SVG data.

        If a TraceSpec is provided, it renders the SVG to a bitmap and
        traces it. This is robust but may lose fidelity.

        Otherwise, it attempts to parse the SVG path and shape data
        directly for a high-fidelity vector import.
        """
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
        )

        if isinstance(vectorization_spec, TraceSpec):
            # Path 1: Render to bitmap and trace
            items = self._get_doc_items_from_trace(source, vectorization_spec)
        else:
            # Path 2: Direct vector parsing with pre-trimming
            trimmed_data = trim_svg(self.raw_data)
            source.base_render_data = trimmed_data
            self._populate_metadata(source)
            items = self._get_doc_items_direct(source)

        if not items:
            return None

        return ImportPayload(source=source, items=items)

    def _populate_metadata(self, source: SourceAsset):
        """Calculates and stores metadata for direct SVG import."""
        metadata = {}
        try:
            # Get size of original, untrimmed SVG
            untrimmed_size = get_natural_size(source.original_data)
            if untrimmed_size:
                source.width_mm = untrimmed_size[0]
                source.height_mm = untrimmed_size[1]
                metadata["untrimmed_width_mm"] = untrimmed_size[0]
                metadata["untrimmed_height_mm"] = untrimmed_size[1]

            # Get size of the new, trimmed SVG
            if source.base_render_data:
                trimmed_size = get_natural_size(source.base_render_data)
                if trimmed_size:
                    metadata["trimmed_width_mm"] = trimmed_size[0]
                    metadata["trimmed_height_mm"] = trimmed_size[1]

                # Get viewBox from trimmed SVG for direct import
                root = ET.fromstring(source.base_render_data)
                vb_str = root.get("viewBox")
                if vb_str:
                    metadata["viewbox"] = tuple(map(float, vb_str.split()))

            source.metadata.update(metadata)
        except Exception as e:
            logger.warning(f"Could not calculate SVG metadata: {e}")

    def _get_doc_items_from_trace(
        self, source: SourceAsset, vectorization_spec: TraceSpec
    ) -> Optional[List[DocItem]]:
        """
        Renders the original SVG data to a bitmap, traces it, and creates a
        single masked WorkPiece.
        """
        size_mm = get_natural_size(source.original_data)
        if not size_mm or not size_mm[0] or not size_mm[1]:
            logger.warning("Cannot trace SVG: failed to determine size.")
            return None

        # Populate intrinsic dimensions
        source.width_mm, source.height_mm = size_mm

        # Calculate render dimensions that preserve the original aspect ratio,
        # maximizing the render resolution for better tracing quality.
        w_mm, h_mm = size_mm
        aspect = w_mm / h_mm if h_mm > 0 else 1.0
        TARGET_DIM = math.sqrt(VTRACER_PIXEL_LIMIT)

        if aspect >= 1.0:  # Landscape or square
            w_px = int(TARGET_DIM)
            h_px = int(TARGET_DIM / aspect)
        else:  # Portrait
            h_px = int(TARGET_DIM)
            w_px = int(TARGET_DIM * aspect)
        w_px, h_px = max(1, w_px), max(1, h_px)

        vips_image = SVG_RENDERER.render_base_image(
            source.original_data, width=w_px, height=h_px
        )
        if not vips_image:
            logger.error("Failed to render SVG to vips image for tracing.")
            return None

        # Manually set the resolution metadata on the rendered image. This is
        # crucial for create_single_workpiece_from_trace to calculate the
        # correct physical size of the cropped area.
        if w_mm > 0 and h_mm > 0:
            xres = w_px / w_mm  # pixels per mm
            yres = h_px / h_mm  # pixels per mm
            vips_image = vips_image.copy(xres=xres, yres=yres)

        normalized_vips = image_util.normalize_to_rgba(vips_image)
        if not normalized_vips:
            return None
        surface = image_util.vips_rgba_to_cairo_surface(normalized_vips)

        geometries = trace_surface(surface, vectorization_spec)

        # Use the standard helper for creating a single, masked workpiece
        return image_util.create_single_workpiece_from_trace(
            geometries,
            source,
            vips_image,
            vectorization_spec,
            self.source_file.stem,
        )

    def _get_doc_items_direct(
        self, source: SourceAsset
    ) -> Optional[List[DocItem]]:
        """
        Orchestrates the direct parsing of SVG data into DocItems.
        """
        if not source.base_render_data:
            logger.error("source has no data to process for direct import")
            return None

        # 1. Establish authoritative dimensions in millimeters.
        final_dims_mm = self._get_final_dimensions(source)
        if not final_dims_mm:
            msg = (
                "SVG is missing width or height attributes; "
                "falling back to trace method for direct import."
            )
            logger.warning(msg)
            return self._get_doc_items_from_trace(source, TraceSpec())
        final_width_mm, final_height_mm = final_dims_mm

        # 2. Parse SVG data into an object model.
        svg = self._parse_svg_data(source)
        if svg is None:
            return None

        # 3. Convert SVG shapes to internal geometry (in pixel coordinates).
        geo = self._convert_svg_to_geometry(svg, final_dims_mm)

        # 4. Get pixel dimensions for normalization.
        pixel_dims = self._get_pixel_dimensions(svg)
        if not pixel_dims:
            msg = (
                "Could not determine valid pixel dimensions from SVG; "
                "falling back to trace method."
            )
            logger.warning(msg)
            return self._get_doc_items_from_trace(source, TraceSpec())
        width_px, height_px = pixel_dims

        # 5. Normalize geometry to a 0-1 unit square (Y-down).
        self._normalize_geometry(geo, width_px, height_px)

        # 6. Create the final workpiece.
        wp = self._create_workpiece(
            geo, source, final_width_mm, final_height_mm
        )
        return [wp]

    def _get_final_dimensions(
        self, source: SourceAsset
    ) -> Optional[Tuple[float, float]]:
        """
        Extracts the final width and height in millimeters from source
        metadata.
        """
        width = source.metadata.get("trimmed_width_mm")
        height = source.metadata.get("trimmed_height_mm")
        if width and height:
            return width, height
        return None

    def _parse_svg_data(self, source: SourceAsset) -> Optional[SVG]:
        """Parses SVG byte data into an svgelements.SVG object."""
        if not source.base_render_data:
            logger.error("Source has no working_data to parse.")
            return None
        try:
            svg_stream = io.BytesIO(source.base_render_data)
            return SVG.parse(svg_stream, ppi=PPI)
        except Exception as e:
            logger.error(f"Failed to parse SVG for direct import: {e}")
            return None

    def _get_pixel_dimensions(self, svg: SVG) -> Optional[Tuple[float, float]]:
        """
        Extracts the pixel width and height from a parsed SVG object.
        """
        if svg.width is None or svg.height is None:
            return None

        width_px = (
            svg.width.px if hasattr(svg.width, "px") else float(svg.width)
        )
        height_px = (
            svg.height.px if hasattr(svg.height, "px") else float(svg.height)
        )

        if width_px <= 1e-9 or height_px <= 1e-9:
            return None

        msg = (
            "Normalizing vectors using final pixel dimensions from "
            "svgelements: {width_px:.3f}px x {height_px:.3f}px"
        )
        logger.debug(msg)
        return width_px, height_px

    def _convert_svg_to_geometry(
        self, svg: SVG, final_dims_mm: Tuple[float, float]
    ) -> Geometry:
        """
        Converts an SVG object into a Geometry object in pixel coordinates.
        """
        geo = Geometry()
        final_width_mm, final_height_mm = final_dims_mm

        # Calculate tolerance for curve flattening.
        avg_dim = max(final_width_mm, final_height_mm, 1.0)
        avg_scale = avg_dim / 960  # Assuming typical viewBox size
        tolerance = 0.1 / avg_scale if avg_scale > 1e-9 else 0.1

        for shape in svg.elements():
            try:
                path = Path(shape)
                path.reify()  # Apply transforms
                self._add_path_to_geometry(path, geo, tolerance)
            except (AttributeError, TypeError):
                continue  # Skip non-shape elements like <defs>
        return geo

    def _add_path_to_geometry(
        self, path: Path, geo: Geometry, tolerance: float
    ) -> None:
        """Converts a single Path object's segments to Geometry commands."""
        for seg in path:
            # Use a local variable to help strict type checkers.
            end = seg.end
            if end is None or end.x is None or end.y is None:
                continue

            if isinstance(seg, Move):
                geo.move_to(float(end.x), float(end.y))
            elif isinstance(seg, Line):
                geo.line_to(float(end.x), float(end.y))
            elif isinstance(seg, Close):
                geo.close_path()
            elif isinstance(seg, Arc):
                self._add_arc_to_geometry(seg, geo)
            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                self._flatten_bezier_to_geometry(seg, geo, tolerance)

    def _add_arc_to_geometry(self, seg: Arc, geo: Geometry) -> None:
        """Adds an Arc segment to the Geometry."""
        # Local variables help type checkers confirm non-None status.
        start = seg.start
        center = seg.center
        end = seg.end

        if (
            start is None
            or start.x is None
            or start.y is None
            or center is None
            or center.x is None
            or center.y is None
            or end is None
            or end.x is None
            or end.y is None
        ):
            return

        start_x, start_y = float(start.x), float(start.y)
        center_x, center_y = float(center.x), float(center.y)

        center_offset_x = center_x - start_x
        center_offset_y = center_y - start_y
        # Per SVG spec, sweep-flag=1 is positive-angle (clockwise).
        # svgelements preserves this as sweep=1 and correctly flips it on
        # transforms with negative determinants.
        is_clockwise = bool(seg.sweep)
        geo.arc_to(
            float(end.x),
            float(end.y),
            center_offset_x,
            center_offset_y,
            clockwise=is_clockwise,
        )

    def _flatten_bezier_to_geometry(
        self,
        seg: Union[CubicBezier, QuadraticBezier],
        geo: Geometry,
        tolerance: float,
    ) -> None:
        """Flattens a Bezier curve into a series of lines in the Geometry."""
        # Use a local variable to help Pylance avoid 'Unbound' issues.
        end = seg.end

        if end is None:
            return
        if end.x is None or end.y is None:
            return

        length = seg.length()
        end_x, end_y = float(end.x), float(end.y)

        # If the curve is very short, treat it as a straight line.
        if length is None or length <= 1e-9:
            geo.line_to(end_x, end_y)
            return

        num_steps = max(2, int(length / tolerance))

        for i in range(1, num_steps + 1):
            t = i / num_steps
            p = seg.point(t)
            if p is not None and p.x is not None and p.y is not None:
                geo.line_to(float(p.x), float(p.y))

    def _normalize_geometry(
        self, geo: Geometry, width_px: float, height_px: float
    ) -> None:
        """
        Normalizes geometry to a 0-1 unit box in a Y-down coordinate system.
        """
        # Normalize from pixel space to a (0,0)-(1,1) unit box.
        # Since SVG coordinates are already Y-down, we don't need to flip.
        if width_px > 0 and height_px > 0:
            norm_matrix = Matrix.scale(1.0 / width_px, 1.0 / height_px)
            geo.transform(norm_matrix.to_4x4_numpy())

    def _create_workpiece(
        self,
        geo: Geometry,
        source: SourceAsset,
        width_mm: float,
        height_mm: float,
    ) -> WorkPiece:
        """Creates and configures the final WorkPiece."""
        gen_config = SourceAssetSegment(
            source_asset_uid=source.uid,
            segment_mask_geometry=geo,
            vectorization_spec=PassthroughSpec(),
            width_mm=width_mm,
            height_mm=height_mm,
        )
        wp = WorkPiece(
            name=self.source_file.stem,
            source_segment=gen_config,
        )
        wp.set_size(width_mm, height_mm)
        wp.pos = (0, 0)
        logger.info(
            f"Workpiece set size: {width_mm:.3f}mm x {height_mm:.3f}mm"
        )
        return wp
