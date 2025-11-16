import io
from typing import Optional, List, Tuple, Union
import logging
from xml.etree import ElementTree as ET
from svgelements import (
    SVG,
    Path,
    Move,
    Line,
    Close,
    Arc,
    CubicBezier,
    QuadraticBezier,
)

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.item import DocItem
from ...core.matrix import Matrix
from ...core.vectorization_spec import VectorizationSpec, TraceSpec
from ...core.workpiece import WorkPiece
from ...core.generation_config import GenerationConfig
from ...core.vectorization_spec import PassthroughSpec
from ..base_importer import Importer, ImportPayload
from ..tracing import trace_surface
from .renderer import SVG_RENDERER
from .svgutil import get_natural_size, trim_svg, PPI

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
        # Process the SVG: trim it
        trimmed_data = trim_svg(self.raw_data)

        # Create import source.
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
        )
        # Store the trimmed data as the base render data for tracing.
        source.base_render_data = trimmed_data

        # Read metadata.
        metadata = {}
        try:
            # Get size of original, untrimmed SVG
            untrimmed_size = get_natural_size(self.raw_data)
            if untrimmed_size:
                metadata["untrimmed_width_mm"] = untrimmed_size[0]
                metadata["untrimmed_height_mm"] = untrimmed_size[1]

            # Get size of the new, trimmed SVG
            trimmed_size = get_natural_size(trimmed_data)
            if trimmed_size:
                metadata["trimmed_width_mm"] = trimmed_size[0]
                metadata["trimmed_height_mm"] = trimmed_size[1]

            # Get viewBox from trimmed SVG for direct import
            root = ET.fromstring(trimmed_data)
            vb_str = root.get("viewBox")
            if vb_str:
                metadata["viewbox"] = tuple(map(float, vb_str.split()))

            source.metadata.update(metadata)
        except Exception as e:
            logger.warning(f"Could not calculate SVG metadata: {e}")

        if isinstance(vectorization_spec, TraceSpec):
            # Path 1: Render to bitmap and trace
            items = self._get_doc_items_from_trace(source, vectorization_spec)
        else:
            # Path 2: Direct vector parsing
            items = self._get_doc_items_direct(source)

        if not items:
            return None

        return ImportPayload(source=source, items=items)

    def _get_doc_items_from_trace(
        self, source: SourceAsset, vectorization_spec: TraceSpec
    ) -> Optional[List[DocItem]]:
        """
        Renders trimmed SVG data to a bitmap, traces it, and creates a
        WorkPiece.
        """
        size_mm = None
        if source.metadata:
            w = source.metadata.get("trimmed_width_mm")
            h = source.metadata.get("trimmed_height_mm")
            if w is not None and h is not None:
                size_mm = (w, h)

        # If we can't determine a size, we can't trace. Return None.
        if not size_mm or not size_mm[0] or not size_mm[1]:
            logger.warning("failed to find a size")
            return None

        if not source.base_render_data:
            logger.error("source has no data to trace")
            return None

        w_mm, h_mm = size_mm
        w_px, h_px = 2048, 2048

        surface = SVG_RENDERER.render_to_pixels_from_data(
            source.base_render_data, w_px, h_px
        )

        wp = WorkPiece(name=self.source_file.stem)

        if surface:
            geometries = trace_surface(surface)
            if geometries:
                combined_geo = Geometry()
                for geo in geometries:
                    geo.close_gaps()
                    combined_geo.commands.extend(geo.commands)

                min_x, min_y, max_x, max_y = combined_geo.rect()
                mask_geo = Geometry()
                mask_geo.move_to(min_x, min_y)
                mask_geo.line_to(max_x, min_y)
                mask_geo.line_to(max_x, max_y)
                mask_geo.line_to(min_x, max_y)
                mask_geo.close_path()

                # Normalize the pixel-based geometry to a 1x1 unit square
                if surface.get_width() > 0 and surface.get_height() > 0:
                    norm_x = 1.0 / surface.get_width()
                    norm_y = 1.0 / surface.get_height()
                    norm_matrix = Matrix.scale(norm_x, norm_y)
                    combined_geo.transform(norm_matrix.to_4x4_numpy())

                wp.vectors = combined_geo
                gen_config = GenerationConfig(
                    source_asset_uid=source.uid,
                    segment_mask_geometry=mask_geo,
                    vectorization_spec=vectorization_spec,
                )
                wp.generation_config = gen_config

        # Always set the size. If tracing failed, the workpiece will be empty
        # but correctly sized.
        wp.set_size(size_mm[0], size_mm[1])

        return [wp]

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

        # 5. Normalize geometry to a 0-1 unit square and flip Y-axis.
        self._normalize_and_flip_geometry(geo, width_px, height_px)

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
        is_clockwise = seg.sweep == 0
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
                # Assertions help type checkers confirm state in complex loops.
                assert p and p.x is not None and p.y is not None
                geo.line_to(float(p.x), float(p.y))

    def _normalize_and_flip_geometry(
        self, geo: Geometry, width_px: float, height_px: float
    ) -> None:
        """
        Normalizes geometry to a 0-1 unit box and flips the Y-axis.
        """
        # Normalize from pixel space to a (0,0)-(1,1) unit box.
        norm_matrix = Matrix.scale(1.0 / width_px, 1.0 / height_px)
        geo.transform(norm_matrix.to_4x4_numpy())

        # Flip the Y-down SVG coordinate system to be Y-up.
        # This is a scale by -1 on Y, then a translation by +1 on Y.
        scale_flip_matrix = Matrix.scale(1.0, -1.0)
        translate_flip_matrix = Matrix.translation(0, 1.0)
        geo.transform(scale_flip_matrix.to_4x4_numpy())
        geo.transform(translate_flip_matrix.to_4x4_numpy())

    def _create_workpiece(
        self,
        geo: Geometry,
        source: SourceAsset,
        width_mm: float,
        height_mm: float,
    ) -> WorkPiece:
        """Creates and configures the final WorkPiece."""
        passthrough_spec = PassthroughSpec()
        gen_config = GenerationConfig(
            source_asset_uid=source.uid,
            segment_mask_geometry=Geometry(),
            vectorization_spec=passthrough_spec,
        )
        wp = WorkPiece(
            name=self.source_file.stem,
            vectors=geo,
            generation_config=gen_config,
        )
        wp.set_size(width_mm, height_mm)
        wp.pos = (0, 0)
        logger.info(
            f"Workpiece set size: {width_mm:.3f}mm x {height_mm:.3f}mm"
        )
        return wp
