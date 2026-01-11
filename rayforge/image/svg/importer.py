import io
import math
import logging
from typing import List, Optional, Tuple
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

from ...core.geo import Geometry
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

            # Use analytical trimming to avoid raster clipping of control
            # points
            trimmed_data = self._analytical_trim(render_data)

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

    def _analytical_trim(self, data: bytes) -> bytes:
        """
        Trims the SVG using vector geometry bounds instead of rasterization.
        This handles elements that extend beyond the document bounds (overflow)
        without clipping.
        """
        try:
            stream = io.BytesIO(data)
            svg = SVG.parse(stream, ppi=PPI)

            # Determine scaling baked into svgelements
            # svgelements.SVG.width/height are in pixels (96 DPI usually)
            width_px = (
                svg.width.px if hasattr(svg.width, "px") else float(svg.width)
            )
            height_px = (
                svg.height.px
                if hasattr(svg.height, "px")
                else float(svg.height)
            )

            vb_x, vb_y, vb_w, vb_h = 0, 0, width_px, height_px
            if svg.viewbox:
                vb_x = svg.viewbox.x
                vb_y = svg.viewbox.y
                vb_w = svg.viewbox.width
                vb_h = svg.viewbox.height

            scale_x = width_px / vb_w if vb_w > 0 else 1.0
            scale_y = height_px / vb_h if vb_h > 0 else 1.0

            # Convert to geometry to find bounds
            geo = self._convert_svg_to_geometry(svg)
            if geo.is_empty():
                return data

            min_x, min_y, max_x, max_y = geo.rect()

            # Unscale to get User Unit bounds
            user_min_x = (min_x / scale_x) + vb_x
            user_max_x = (max_x / scale_x) + vb_x
            user_min_y = (min_y / scale_y) + vb_y
            user_max_y = (max_y / scale_y) + vb_y

            user_w = user_max_x - user_min_x
            user_h = user_max_y - user_min_y

            # Avoid tiny boxes or invalid results
            if user_w <= 1e-6 or user_h <= 1e-6:
                return data

            # Noise filter: if the new trim box is effectively identical to
            # the original viewbox, return the original data to avoid float
            # noise.
            if (
                abs(user_min_x - vb_x) < 1e-4
                and abs(user_min_y - vb_y) < 1e-4
                and abs(user_w - vb_w) < 1e-4
                and abs(user_h - vb_h) < 1e-4
            ):
                return data

            # Manipulate XML
            root = ET.fromstring(data)

            # Update ViewBox to exactly enclose the geometry
            new_vb = f"{user_min_x:g} {user_min_y:g} {user_w:g} {user_h:g}"
            root.set("viewBox", new_vb)

            # Update width/height to reflect the physical size of the new
            # viewbox. We preserve the original unit scale ratio.
            new_width_px = user_w * scale_x
            new_height_px = user_h * scale_y

            root.set("width", f"{new_width_px:.4f}px")
            root.set("height", f"{new_height_px:.4f}px")

            # Remove preserveAspectRatio to avoid confusion, forcing 1:1 map
            if "preserveAspectRatio" in root.attrib:
                del root.attrib["preserveAspectRatio"]

            return ET.tostring(root)

        except Exception as e:
            logger.warning(f"Analytical trim failed: {e}")
            return data

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

        # Ensure the source asset has pixel dimensions.
        # These are required for correct viewBox calculation in split/crop
        # operations.
        source.width_px = int(width_px)
        source.height_px = int(height_px)

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
        pristine_geo = self._convert_svg_to_geometry(svg)
        if pristine_geo.is_empty():
            logger.info(
                "Direct SVG import resulted in empty geometry. "
                "No items created."
            )
            return None

        # Get the geometry bounding box.
        geo_min_x, geo_min_y, geo_max_x, geo_max_y = pristine_geo.rect()
        geo_width = geo_max_x - geo_min_x
        geo_height = geo_max_y - geo_min_y

        if geo_width <= 0 or geo_height <= 0:
            logger.warning("SVG import resulted in zero-dimension geometry.")
            return None

        # --- Coordinate System Normalization ---
        # svgelements usually applies the viewport transform (User->Pixels).
        # We need to detect this and ensure we are working in User Units.

        vb = source.metadata.get("viewbox")
        user_unit_to_mm_x = 1.0
        user_unit_to_mm_y = 1.0

        if vb:
            vb_x, vb_y, vb_w, vb_h = vb

            # Calculate implied scale factor (Native / User)
            svg_scale_x = width_px / vb_w if vb_w > 0 else 1.0
            svg_scale_y = height_px / vb_h if vb_h > 0 else 1.0

            # If significant scale detected, assume svgelements baked it in.
            # We must revert it to get User Units.
            if abs(svg_scale_x - 1.0) > 0.1:
                inv_scale = Matrix.scale(1.0 / svg_scale_x, 1.0 / svg_scale_y)
                pristine_geo.transform(inv_scale.to_4x4_numpy())

                # Update local bounds variables
                geo_min_x /= svg_scale_x
                geo_min_y /= svg_scale_y
                geo_max_x /= svg_scale_x
                geo_max_y /= svg_scale_y
                geo_width /= svg_scale_x
                geo_height /= svg_scale_y

            # Now geometry is in User Units.
            # Calculate conversion factor to MM based on ViewBox and
            # Final Dimensions.
            if vb_w > 0:
                user_unit_to_mm_x = final_width_mm / vb_w
            if vb_h > 0:
                user_unit_to_mm_y = final_height_mm / vb_h
        else:
            # No ViewBox: Assume 1:1 mapping (Pixels=UserUnits)
            if width_px > 0:
                user_unit_to_mm_x = final_width_mm / width_px
            if height_px > 0:
                user_unit_to_mm_y = final_height_mm / height_px

        # The normalization matrix maps the geometry (User Units) to 0-1 box.
        norm_scale = Matrix.scale(1.0 / geo_width, 1.0 / geo_height)
        norm_translate = Matrix.translation(-geo_min_x, -geo_min_y)
        normalization_matrix = norm_scale @ norm_translate

        # Create the segment
        segment = SourceAssetSegment(
            source_asset_uid=source.uid,
            vectorization_spec=PassthroughSpec(),
            pristine_geometry=pristine_geo,
            normalization_matrix=normalization_matrix,
        )

        # Physical size
        wp_width_mm = geo_width * user_unit_to_mm_x
        wp_height_mm = geo_height * user_unit_to_mm_y

        # Position (X)
        wp_pos_x_mm = geo_min_x * user_unit_to_mm_x

        # Position (Y)
        # Flip from Y-Down User Space to Y-Up World Space.
        # We align relative to the bottom of the "Page" (ViewBox).
        if vb:
            _, vb_y, _, vb_h = vb
            visual_bottom_y = vb_y + vb_h
            dist_from_bottom_units = visual_bottom_y - geo_max_y
            wp_pos_y_mm = dist_from_bottom_units * user_unit_to_mm_y
        else:
            # Fallback: assume User Units = Pixels
            wp_pos_y_mm = (height_px - geo_max_y) * user_unit_to_mm_y

        # Correction for Trim offset to restore absolute position
        if (
            "untrimmed_height_mm" in source.metadata
            and "untrimmed_width_mm" in source.metadata
        ):
            if vb:
                # X: Add the trim offset (vb_x)
                wp_pos_x_mm += vb[0] * user_unit_to_mm_x

                # Y: Restore absolute Y-Up position
                # geo_max_y is relative to ViewBox top.
                # Absolute User Y from top = geo_max_y + vb_y
                untrimmed_h_mm = source.metadata["untrimmed_height_mm"]
                abs_geo_bottom_y_units = geo_max_y + vb[1]
                wp_pos_y_mm = untrimmed_h_mm - (
                    abs_geo_bottom_y_units * user_unit_to_mm_y
                )

        wp = WorkPiece(name=self.source_file.stem, source_segment=segment)
        wp.natural_width_mm = wp_width_mm
        wp.natural_height_mm = wp_height_mm
        wp.set_size(wp_width_mm, wp_height_mm)
        wp.pos = (wp_pos_x_mm, wp_pos_y_mm)

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
        master_pristine_geo = self._convert_svg_to_geometry(svg)
        if master_pristine_geo.is_empty():
            return []

        # Logic for Full-Page Normalization
        vb = source.metadata.get("viewbox")
        norm_width = width_px
        norm_height = height_px
        norm_off_x = 0
        norm_off_y = 0

        # Define scale variables early to avoid NameError/Unbound
        svg_scale_x = 1.0
        svg_scale_y = 1.0
        vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, 0.0, 0.0

        if vb:
            vb_x, vb_y, vb_w, vb_h = vb
            # Detect Scale (same as direct)
            svg_scale_x = width_px / vb_w if vb_w > 0 else 1.0
            svg_scale_y = height_px / vb_h if vb_h > 0 else 1.0

            if abs(svg_scale_x - 1.0) > 0.1:
                inv_scale = Matrix.scale(1.0 / svg_scale_x, 1.0 / svg_scale_y)
                master_pristine_geo.transform(inv_scale.to_4x4_numpy())

            # For Split, we normalize to the ViewBox.
            # svgelements shifts coordinates to the ViewBox origin (0,0),
            # so we do not need to subtract vb_x/vb_y here (offset is 0).
            norm_width = vb_w
            norm_height = vb_h
            norm_off_x = 0
            norm_off_y = 0

        # Normalization Matrix maps ViewBox to 0-1
        norm_matrix = Matrix.scale(
            1.0 / norm_width, 1.0 / norm_height
        ) @ Matrix.translation(-norm_off_x, -norm_off_y)

        # Master WP Size = Full Page Size
        # Master WP Pos = 0,0 (relative to page)

        master_segment = SourceAssetSegment(
            source_asset_uid=source.uid,
            vectorization_spec=PassthroughSpec(),
            pristine_geometry=master_pristine_geo,
            normalization_matrix=norm_matrix,
        )
        master_wp = WorkPiece(
            name=self.source_file.stem, source_segment=master_segment
        )
        master_wp.natural_width_mm = final_dims_mm[0]
        master_wp.natural_height_mm = final_dims_mm[1]
        master_wp.set_size(final_dims_mm[0], final_dims_mm[1])

        # Calculate absolute position to restore original layout
        pos_x, pos_y = 0.0, 0.0
        if vb:
            scale_x = final_dims_mm[0] / vb_w if vb_w > 0 else 1.0
            scale_y = final_dims_mm[1] / vb_h if vb_h > 0 else 1.0

            pos_x = vb_x * scale_x

            if "untrimmed_height_mm" in source.metadata:
                untrimmed_h = source.metadata["untrimmed_height_mm"]
                # Bottom of ViewBox in Y-down user units is vb_y + vb_h
                pos_y = untrimmed_h - ((vb_y + vb_h) * scale_y)

        master_wp.pos = (pos_x, pos_y)

        final_items: List[DocItem] = []
        manifest = extract_layer_manifest(self.raw_data)
        layer_names = {m["id"]: m["name"] for m in manifest}

        layer_geoms = {lid: Geometry() for lid in layer_ids}

        def _get_all_shapes(group: Group):
            for item in group:
                if isinstance(item, Group):
                    yield from _get_all_shapes(item)
                else:
                    yield item

        for element in svg:
            if not isinstance(element, Group):
                continue
            lid = element.id
            if lid and lid in layer_geoms:
                for shape in _get_all_shapes(element):
                    try:
                        path = Path(shape)
                        self._add_path_to_geometry(path, layer_geoms[lid])
                    except (AttributeError, TypeError):
                        pass

        for lid in layer_ids:
            pristine_layer_geo = layer_geoms[lid]

            # Apply same inverse scale if needed
            if vb and abs(svg_scale_x - 1.0) > 0.1:
                inv_scale = Matrix.scale(1.0 / svg_scale_x, 1.0 / svg_scale_y)
                pristine_layer_geo.transform(inv_scale.to_4x4_numpy())

            if not pristine_layer_geo.is_empty():
                segment = SourceAssetSegment(
                    source_asset_uid=source.uid,
                    vectorization_spec=PassthroughSpec(),
                    layer_id=lid,
                    pristine_geometry=pristine_layer_geo,
                    normalization_matrix=norm_matrix,
                )

                wp_name = layer_names.get(lid, f"Layer {lid}")
                wp = WorkPiece(name=wp_name, source_segment=segment)
                wp.matrix = master_wp.matrix
                wp.natural_width_mm = master_wp.natural_width_mm
                wp.natural_height_mm = master_wp.natural_height_mm

                new_layer = Layer(name=wp_name)
                new_layer.add_child(wp)
                final_items.append(new_layer)

        if not final_items:
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

    def _convert_svg_to_geometry(self, svg: SVG) -> Geometry:
        """
        Converts an SVG object into a Geometry object in pixel coordinates.
        Preserves curves as Béziers.
        """
        geo = Geometry()

        for shape in svg.elements():
            try:
                path = Path(shape)
                path.reify()  # Apply transforms
                self._add_path_to_geometry(path, geo)
            except (AttributeError, TypeError):
                continue  # Skip non-shape elements like <defs>
        return geo

    def _add_path_to_geometry(self, path: Path, geo: Geometry) -> None:
        """
        Converts a single Path object's segments to Geometry commands.
        Curves are added as Béziers instead of being linearized.
        """
        for seg in path:
            # Check for a valid end point, which is required for most segments.
            end_pt = (0.0, 0.0)
            if not isinstance(seg, Close):
                if seg.end is None or seg.end.x is None or seg.end.y is None:
                    continue
                end_pt = (float(seg.end.x), float(seg.end.y))

            if isinstance(seg, Move):
                geo.move_to(end_pt[0], end_pt[1])

            elif isinstance(seg, Line):
                geo.line_to(end_pt[0], end_pt[1])

            elif isinstance(seg, Close):
                geo.close_path()

            elif isinstance(seg, CubicBezier):
                if (
                    seg.control1 is not None
                    and seg.control1.x is not None
                    and seg.control1.y is not None
                    and seg.control2 is not None
                    and seg.control2.x is not None
                    and seg.control2.y is not None
                ):
                    c1 = (float(seg.control1.x), float(seg.control1.y))
                    c2 = (float(seg.control2.x), float(seg.control2.y))
                    geo.bezier_to(
                        end_pt[0], end_pt[1], c1[0], c1[1], c2[0], c2[1]
                    )
                else:
                    geo.line_to(end_pt[0], end_pt[1])

            elif isinstance(seg, QuadraticBezier):
                if (
                    seg.start is not None
                    and seg.start.x is not None
                    and seg.start.y is not None
                    and seg.control is not None
                    and seg.control.x is not None
                    and seg.control.y is not None
                ):
                    sx, sy = float(seg.start.x), float(seg.start.y)
                    cx, cy = float(seg.control.x), float(seg.control.y)
                    ex, ey = end_pt

                    c1x = sx + (2.0 / 3.0) * (cx - sx)
                    c1y = sy + (2.0 / 3.0) * (cy - sy)
                    c2x = ex + (2.0 / 3.0) * (cx - ex)
                    c2y = ey + (2.0 / 3.0) * (cy - ey)

                    geo.bezier_to(ex, ey, c1x, c1y, c2x, c2y)
                else:
                    geo.line_to(end_pt[0], end_pt[1])

            elif isinstance(seg, Arc):
                # svgelements handles Arc -> Cubic conversion
                for cubic in seg.as_cubic_curves():
                    if (
                        cubic.end is not None
                        and cubic.end.x is not None
                        and cubic.end.y is not None
                        and cubic.control1 is not None
                        and cubic.control1.x is not None
                        and cubic.control1.y is not None
                        and cubic.control2 is not None
                        and cubic.control2.x is not None
                        and cubic.control2.y is not None
                    ):
                        e = (float(cubic.end.x), float(cubic.end.y))
                        c1 = (float(cubic.control1.x), float(cubic.control1.y))
                        c2 = (float(cubic.control2.x), float(cubic.control2.y))
                        geo.bezier_to(e[0], e[1], c1[0], c1[1], c2[0], c2[1])
                    elif (
                        cubic.end is not None
                        and cubic.end.x is not None
                        and cubic.end.y is not None
                    ):
                        geo.line_to(float(cubic.end.x), float(cubic.end.y))

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
