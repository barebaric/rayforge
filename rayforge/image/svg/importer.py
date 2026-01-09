import io
import math
import logging
from typing import List, Optional, Tuple
import cairo
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
    Group,
)
import numpy as np

from ...core.geo import (
    Geometry,
    GEO_ARRAY_COLS,
    CMD_TYPE_LINE,
    COL_TYPE,
    COL_X,
    COL_Y,
)
from ...core.geo.simplify import simplify_points_to_array
from ...core.item import DocItem
from ...core.layer import Layer
from ...core.matrix import Matrix
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
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
from .svgutil import (
    PPI,
    MM_PER_PX,
    get_natural_size,
    trim_svg,
    extract_layer_manifest,
)

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
        # Determine if we have active layers (for splitting logic later)
        active_layer_ids = None
        if isinstance(vectorization_spec, PassthroughSpec):
            active_layer_ids = vectorization_spec.active_layer_ids

        # Use raw data for source to avoid corruption issues with
        # pre-filtering.
        # Layer filtering is handled during geometry extraction.
        render_data = self.raw_data

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
        )

        if isinstance(vectorization_spec, TraceSpec):
            # Path 1: Render to bitmap and trace
            # Note: TraceSpec doesn't currently support layer filtering in UI
            items = self._get_doc_items_from_trace(source, vectorization_spec)
        else:
            # Path 2: Direct vector parsing with pre-trimming
            trimmed_data = trim_svg(render_data)
            source.base_render_data = trimmed_data
            source.metadata["is_vector"] = True
            self._populate_metadata(source)
            items = self._get_doc_items_direct(source, active_layer_ids)

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
                try:
                    root = ET.fromstring(source.base_render_data)
                    vb_str = root.get("viewBox")
                    if vb_str:
                        metadata["viewbox"] = tuple(map(float, vb_str.split()))
                except ET.ParseError:
                    pass

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

        # This makes the high-res raster available to the preview dialog.
        source.base_render_data = vips_image.pngsave_buffer()

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
        self, source: SourceAsset, active_layer_ids: Optional[List[str]] = None
    ) -> Optional[List[DocItem]]:
        """
        Orchestrates the direct parsing of SVG data into DocItems.
        """
        if not source.base_render_data:
            logger.error("source has no data to process for direct import")
            return None

        # 1. Parse SVG data into an object model first.
        #    This allows us to get robust dimensions from svgelements if the
        #    simple metadata extraction failed (e.g. missing attributes).
        svg = self._parse_svg_data(source)
        if svg is None:
            return None

        # 2. Get pixel dimensions for normalization.
        pixel_dims = self._get_pixel_dimensions(svg)
        if not pixel_dims:
            msg = (
                "Could not determine valid pixel dimensions from SVG; "
                "falling back to trace method."
            )
            logger.warning(msg)
            return self._get_doc_items_from_trace(source, TraceSpec())
        width_px, height_px = pixel_dims

        # 3. Establish authoritative dimensions in millimeters.
        final_dims_mm = self._get_final_dimensions(source)
        if not final_dims_mm:
            # Fallback: Use dimensions derived from svgelements
            # svgelements normalizes units to 96 DPI (usually)
            final_dims_mm = (width_px * MM_PER_PX, height_px * MM_PER_PX)
            logger.info(
                "Using svgelements dimensions as fallback: "
                f"{final_dims_mm[0]:.2f}mm x {final_dims_mm[1]:.2f}mm"
            )

        final_width_mm, final_height_mm = final_dims_mm

        # 4. Handle Split Layers if requested
        if active_layer_ids:
            return self._create_split_items(
                svg,
                active_layer_ids,
                source,
                final_dims_mm,
                width_px,
                height_px,
            )

        # 5. Standard path (merged import)
        # Convert SVG shapes to internal geometry (in pixel coordinates).
        # This now preserves circular arcs.
        pristine_geo = self._convert_svg_to_geometry(svg, final_dims_mm)
        logger.info(
            f"SVG Import: Pristine geometry created with "
            f"{len(pristine_geo)} commands."
        )

        # If the SVG contained no parsable vector paths, abort the import.
        if pristine_geo.is_empty():
            logger.info(
                "Direct SVG import resulted in empty geometry. "
                "No items created."
            )
            return None

        # Calculate normalization matrix for the new path
        normalization_matrix = Matrix.scale(1.0 / width_px, 1.0 / height_px)

        # Create the segment with the pristine geometry and its normalization
        # matrix
        segment = SourceAssetSegment(
            source_asset_uid=source.uid,
            vectorization_spec=PassthroughSpec(),
            pristine_geometry=pristine_geo,
            normalization_matrix=normalization_matrix,
        )

        wp = WorkPiece(name=self.source_file.stem, source_segment=segment)
        wp.natural_width_mm = final_width_mm
        wp.natural_height_mm = final_height_mm
        wp.set_size(final_width_mm, final_height_mm)
        wp.pos = (0, 0)
        logger.info(
            f"Workpiece set size: "
            f"{final_width_mm:.3f}mm x {final_height_mm:.3f}mm"
        )
        return [wp]

    def _create_split_items(
        self,
        svg: SVG,
        layer_ids: List[str],
        source: SourceAsset,
        final_dims_mm: Tuple[float, float],
        width_px: float,
        height_px: float,
    ) -> List[DocItem]:
        """
        Creates separate Layer items containing WorkPieces for each selected
        layer ID. The WorkPieces share the same size and transform but use
        different geometry masks.
        """
        # Create a master WorkPiece to use as the matrix template.
        master_pristine_geo = self._convert_svg_to_geometry(svg, final_dims_mm)

        # If the master geometry is empty, there's nothing to split.
        if master_pristine_geo.is_empty():
            return []

        normalization_matrix = Matrix.scale(1.0 / width_px, 1.0 / height_px)
        master_segment = SourceAssetSegment(
            source_asset_uid=source.uid,
            vectorization_spec=PassthroughSpec(),
            pristine_geometry=master_pristine_geo,
            normalization_matrix=normalization_matrix,
        )
        master_wp = WorkPiece(
            name=self.source_file.stem, source_segment=master_segment
        )
        master_wp.natural_width_mm = final_dims_mm[0]
        master_wp.natural_height_mm = final_dims_mm[1]
        master_wp.set_size(final_dims_mm[0], final_dims_mm[1])
        master_wp.pos = (0, 0)

        final_items: List[DocItem] = []
        manifest = extract_layer_manifest(self.raw_data)
        layer_names = {m["id"]: m["name"] for m in manifest}

        # Prepare geometry containers for each requested layer
        layer_geoms = {lid: Geometry() for lid in layer_ids}

        # Calculate tolerance
        # Use a fixed, high-precision tolerance for consistency
        tolerance = 0.05

        def _get_all_shapes(group: Group):
            """Recursively yields all shapes from a group."""
            for item in group:
                if isinstance(item, Group):
                    yield from _get_all_shapes(item)
                else:
                    yield item

        # Iterate over each direct child of the SVG root. If it's a Group
        # that matches a requested layer ID, process all shapes within it.
        for element in svg:
            if not isinstance(element, Group):
                continue

            lid = element.id
            if lid and lid in layer_geoms:
                # This group is a layer we want to import.
                # Process all shapes within this group (and its subgroups).
                for shape in _get_all_shapes(element):
                    try:
                        path = Path(shape)
                        self._add_path_to_geometry(
                            path, layer_geoms[lid], tolerance
                        )
                    except (AttributeError, TypeError):
                        pass  # Ignore non-shape elements like <defs>

        # Create DocItems from populated geometries
        for lid in layer_ids:
            pristine_layer_geo = layer_geoms[lid]

            if not pristine_layer_geo.is_empty():
                segment = SourceAssetSegment(
                    source_asset_uid=source.uid,
                    vectorization_spec=PassthroughSpec(),
                    layer_id=lid,
                    pristine_geometry=pristine_layer_geo,
                    normalization_matrix=normalization_matrix,
                )

                # Create WorkPiece
                wp_name = layer_names.get(lid, f"Layer {lid}")
                wp = WorkPiece(name=wp_name, source_segment=segment)
                wp.matrix = master_wp.matrix
                wp.natural_width_mm = master_wp.natural_width_mm
                wp.natural_height_mm = master_wp.natural_height_mm

                # Create Container Layer
                new_layer = Layer(name=wp_name)
                new_layer.add_child(wp)
                final_items.append(new_layer)

        if not final_items:
            # Fallback if no specific layers found (shouldn't happen
            # with filter)
            return [master_wp]

        return final_items

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
        Preserves circular arcs.
        """
        geo = Geometry()

        # Use a fixed, high-precision tolerance for linearization and
        # buffering. SVG coordinates are usually 96 DPI.
        # 0.05 px ~= 0.013 mm, which is sufficient for high fidelity import.
        tolerance = 0.05

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
        """
        Converts a single Path object's segments to Geometry commands using a
        hybrid approach: circular arcs are added directly, while all other
        curves are flattened using Cairo. This avoids slow Python-based
        recursion for curve linearization by:
        1. Constructing the path in a temporary Cairo context.
        2. Utilizing `ctx.copy_path_flat()` to get pre-flattened line data.
        3. Buffering points for efficient ingestion into the Geometry object.
        """
        logger.debug(f"Processing path with {len(path)} segments.")
        # Create a throwaway 1x1 surface just to get a Context
        surface = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
        ctx = cairo.Context(surface)

        # Set tolerance for flattening (conversion of curves to lines)
        ctx.set_tolerance(tolerance)

        # Feed svgelements path data into the Cairo context or Geometry
        for i, seg in enumerate(path):
            is_circular_arc = False
            if (
                isinstance(seg, Arc)
                and seg.rx is not None
                and seg.ry is not None
            ):
                # Check if it's a circular arc (or very close to one)
                if abs(seg.rx - seg.ry) < 1e-3:
                    is_circular_arc = True

            if is_circular_arc:
                logger.debug(
                    f"Segment {i} is a circular arc, adding directly."
                )
                # 1. Flush any pending non-arc segments from the cairo context
                self._process_cairo_flat_path(ctx, geo, tolerance)

                # 2. Add the arc directly to our Geometry
                start = seg.start
                end = seg.end
                center = seg.center
                if (
                    start is not None
                    and end is not None
                    and center is not None
                    and start.x is not None
                    and start.y is not None
                    and end.x is not None
                    and end.y is not None
                    and center.real is not None
                    and center.imag is not None
                ):
                    i_offset = center.real - start.x
                    j_offset = center.imag - start.y

                    # SVG sweep_flag: 1 is "positive-angle" direction.
                    # In SVG's Y-down, this is CW. Our geometry expects
                    # CW=True.
                    clockwise = bool(seg.sweep)
                    geo.arc_to_as_bezier(
                        float(end.x),
                        float(end.y),
                        i_offset,
                        j_offset,
                        clockwise=clockwise,
                        z=0.0,
                    )
                # 3. Update Cairo's current point to keep it in sync
                if end is not None and end.x is not None and end.y is not None:
                    ctx.move_to(float(end.x), float(end.y))
            else:
                logger.debug(
                    f"Segment {i} is {type(seg).__name__}, adding to cairo."
                )
                # For all other segment types, feed them into Cairo
                start = seg.start
                end = seg.end
                if end is None:
                    continue
                # Robust checking before accessing coordinates
                if end.x is None or end.y is None:
                    continue

                end_pt = (float(end.x), float(end.y))

                if isinstance(seg, Move):
                    ctx.move_to(end_pt[0], end_pt[1])

                elif isinstance(seg, Line):
                    ctx.line_to(end_pt[0], end_pt[1])

                elif isinstance(seg, Close):
                    ctx.close_path()

                elif isinstance(seg, CubicBezier):
                    c1 = seg.control1
                    c2 = seg.control2
                    # Ensure control points exist before accessing properties
                    if (
                        c1 is not None
                        and c2 is not None
                        and c1.x is not None
                        and c1.y is not None
                        and c2.x is not None
                        and c2.y is not None
                    ):
                        ctx.curve_to(
                            float(c1.x),
                            float(c1.y),
                            float(c2.x),
                            float(c2.y),
                            end_pt[0],
                            end_pt[1],
                        )

                elif isinstance(seg, QuadraticBezier):
                    # Promote Quadratic to Cubic for Cairo
                    # CP1 = Start + (2/3)*(Control - Start)
                    # CP2 = End + (2/3)*(Control - End)
                    c = seg.control
                    if (
                        start is not None
                        and c is not None
                        and start.x is not None
                        and start.y is not None
                        and c.x is not None
                        and c.y is not None
                    ):
                        sx, sy = float(start.x), float(start.y)
                        cx, cy = float(c.x), float(c.y)
                        ex, ey = end_pt

                        c1x = sx + (2.0 / 3.0) * (cx - sx)
                        c1y = sy + (2.0 / 3.0) * (cy - sy)
                        c2x = ex + (2.0 / 3.0) * (cx - ex)
                        c2y = ey + (2.0 / 3.0) * (cy - ey)

                        ctx.curve_to(c1x, c1y, c2x, c2y, ex, ey)

                elif isinstance(seg, Arc):  # Elliptical arc
                    for cubic in seg.as_cubic_curves():
                        c1, c2, e = cubic.control1, cubic.control2, cubic.end
                        if (
                            c1 is not None
                            and c2 is not None
                            and e is not None
                            and c1.x is not None
                            and c1.y is not None
                            and c2.x is not None
                            and c2.y is not None
                            and e.x is not None
                            and e.y is not None
                        ):
                            ctx.curve_to(
                                float(c1.x),
                                float(c1.y),
                                float(c2.x),
                                float(c2.y),
                                float(e.x),
                                float(e.y),
                            )

        # Flush any remaining segments from the cairo context
        self._process_cairo_flat_path(ctx, geo, tolerance)

    def _process_cairo_flat_path(
        self, ctx: cairo.Context, geo: Geometry, tolerance: float
    ):
        """Helper to flush flattened bezier/line data from Cairo to Geometry"""
        flat_path_iter = ctx.copy_path_flat()
        point_buffer: List[Tuple[float, float]] = []

        def flush_buffer():
            if len(point_buffer) > 1:
                # Convert to numpy array
                pts_arr = np.array(point_buffer, dtype=np.float64)

                # Simplify using the numpy-based RDP function
                simplified_arr = simplify_points_to_array(pts_arr, tolerance)

                # We need to add LineTo commands for points[1:]
                # points[0] is the start (already MovedTo or LinedTo)
                count = len(simplified_arr)
                if count > 1:
                    # Allocate block
                    # Shape: (count - 1, 7)
                    block = np.zeros(
                        (count - 1, GEO_ARRAY_COLS), dtype=np.float64
                    )

                    # Set Command Type
                    block[:, COL_TYPE] = CMD_TYPE_LINE

                    # Set X, Y
                    block[:, COL_X] = simplified_arr[1:, 0]
                    block[:, COL_Y] = simplified_arr[1:, 1]

                    # Bulk append to geometry
                    geo.append_numpy_data(block)

        # Process the flattened Cairo commands
        for type, points in flat_path_iter:
            if type == cairo.PATH_MOVE_TO:
                flush_buffer()
                p = (points[0], points[1])
                point_buffer = [p]
                geo.move_to(p[0], p[1], 0.0)

            elif type == cairo.PATH_LINE_TO:
                p = (points[0], points[1])
                point_buffer.append(p)

            elif type == cairo.PATH_CLOSE_PATH:
                flush_buffer()
                geo.close_path()
                point_buffer = []

        # Flush any remaining segments
        flush_buffer()
        # After flushing, clear the Cairo path
        ctx.new_path()

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
