from __future__ import annotations
import io
import math
import logging
from typing import List, Optional, Tuple, Dict
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
from ...core.matrix import Matrix
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import (
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
    LayerInfo,
)
from .. import image_util
from ..assembler import ItemAssembler
from ..engine import NormalizationEngine
from ..structures import ParsingResult, LayerGeometry
from ..tracing import trace_surface, VTRACER_PIXEL_LIMIT
from .renderer import SVG_RENDERER
from .svgutil import (
    PPI,
    MM_PER_PX,
    get_natural_size,
    extract_layer_manifest,
)

logger = logging.getLogger(__name__)

# Type Aliases for cleaner signatures
ViewBoxType = Optional[Tuple[float, float, float, float]]
ParsingFactsType = Optional[Tuple[float, float, ViewBoxType]]


class SvgImporter(Importer):
    label = "SVG files"
    mime_types = ("image/svg+xml",)
    extensions = (".svg",)
    features = {
        ImporterFeature.BITMAP_TRACING,
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
    }

    def scan(self) -> ImportManifest:
        """
        Scans the SVG file for layer names and overall dimensions without a
        full render.
        """
        warnings = []
        layers = []
        size_mm = None
        try:
            size_mm = get_natural_size(self.raw_data)
            layer_data = extract_layer_manifest(self.raw_data)
            layers = [
                LayerInfo(id=layer["id"], name=layer["name"])
                for layer in layer_data
            ]
        except ET.ParseError as e:
            logger.warning(f"SVG scan failed for {self.source_file.name}: {e}")
            warnings.append(
                _("Could not parse SVG. File may be corrupt or invalid.")
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during SVG scan for "
                f"{self.source_file.name}: {e}",
                exc_info=True,
            )
            warnings.append(
                _("An unexpected error occurred while scanning the SVG.")
            )

        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=size_mm,
            warnings=warnings,
        )

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
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
            # Note: TraceSpec doesn't currently support layer filtering in UI
            items = self._get_doc_items_from_trace(source, vectorization_spec)
        else:
            # Path 2: Direct vector parsing with pre-trimming

            # Use analytical trimming to avoid raster clipping of control
            # points
            trimmed_data = self._analytical_trim(self.raw_data)

            source.base_render_data = trimmed_data
            source.metadata["is_vector"] = True
            self._populate_metadata(source)
            items = self._get_doc_items_direct(source, vectorization_spec)

        if items is None:
            return None

        # If items is an empty list, it's a valid file with no content.
        # Return a payload with the source but no items.
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
        self,
        source: SourceAsset,
        vectorization_spec: Optional[VectorizationSpec] = None,
    ) -> Optional[List[DocItem]]:
        """
        Orchestrates the direct parsing of SVG data into DocItems using the
        Parse -> Plan -> Assemble pipeline.
        """
        if not source.base_render_data:
            logger.error("source has no data to process for direct import")
            return None

        # 1. Parse SVG data into an object model first.
        svg = self._parse_svg_data(source)
        if svg is None:
            return None

        # Handle the edge case of a file with no explicit dimensions and no
        # content.
        # svgelements will default the size, but we can detect the absence of
        # original width/height attributes.
        has_explicit_dims = svg.values is not None and (
            "width" in svg.values or "height" in svg.values
        )
        if not has_explicit_dims:
            geo = self._convert_svg_to_geometry(svg)
            if geo.is_empty():
                return None  # Fails the import as expected by test_edge_cases.

        facts = self._get_svg_parsing_facts(svg)
        if not facts:
            return None
        width_px, height_px, vb = facts

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

        spec = vectorization_spec or PassthroughSpec()
        parse_result = self._parse_to_result(svg, source, final_dims_mm)

        if not parse_result:
            logger.error("Failed to parse SVG into a structured result.")
            return None

        engine = NormalizationEngine()
        plan = engine.calculate_layout(parse_result, spec)

        if not plan:
            logger.info(
                "Direct SVG import resulted in empty layout plan. "
                "No items created."
            )
            return []

        geometries: Dict[Optional[str], Geometry]
        if len(plan) == 1 and plan[0].layer_id is None:
            geometries = {None: self._convert_svg_to_geometry(svg)}
        else:
            layer_ids_in_plan = [
                item.layer_id for item in plan if item.layer_id
            ]
            geometries = self._parse_geometry_by_layer(svg, layer_ids_in_plan)

        assembler = ItemAssembler()
        return assembler.create_items(
            source,
            plan,
            spec,
            self.source_file.stem,
            geometries,
            layer_manifest=extract_layer_manifest(self.raw_data),
        )

    def _parse_geometry_by_layer(
        self, svg: SVG, layer_ids: List[str]
    ) -> Dict[Optional[str], Geometry]:
        """
        Parses an SVG object and returns a dictionary mapping layer IDs to
        their corresponding Geometry.
        """
        layer_geoms: Dict[Optional[str], Geometry] = {
            lid: Geometry() for lid in layer_ids
        }

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
        return layer_geoms

    def _parse_to_result(
        self, svg: SVG, source: SourceAsset, final_dims_mm: Tuple[float, float]
    ) -> Optional[ParsingResult]:
        """
        Performs Phase 1 of the import pipeline: Parsing.
        It extracts all geometric facts from the SVG into a ParsingResult DTO,
        using the file's native coordinate system (pixels).
        """
        # Step 1: Extract SVG parsing facts
        facts = self._get_svg_parsing_facts(svg)
        if not facts:
            return None
        width_px, height_px, viewbox = facts

        # Calculate the correct scale factor based on final dimensions.
        # This aligns the parsing result units (pixels) with physical reality.
        unit_to_mm = final_dims_mm[0] / width_px if width_px > 0 else 1.0

        # Step 3: Get untrimmed page bounds for positioning reference
        untrimmed_page_bounds_px: Optional[
            Tuple[float, float, float, float]
        ] = None
        untrimmed_w_mm = source.metadata.get("untrimmed_width_mm")
        untrimmed_h_mm = source.metadata.get("untrimmed_height_mm")
        if untrimmed_w_mm and untrimmed_h_mm and unit_to_mm > 0:
            untrimmed_w_px = untrimmed_w_mm / unit_to_mm
            untrimmed_h_px = untrimmed_h_mm / unit_to_mm
            untrimmed_page_bounds_px = (0, 0, untrimmed_w_px, untrimmed_h_px)

        # Step 4: Parse geometry for all layers
        all_layer_ids = [
            elem.id for elem in svg if isinstance(elem, Group) and elem.id
        ]
        if not all_layer_ids:
            # Handle SVGs with no layers by treating the whole file as one
            all_layer_ids = ["__default__"]
            geometries_by_layer = {
                "__default__": self._convert_svg_to_geometry(svg)
            }
        else:
            geometries_by_layer = self._parse_geometry_by_layer(
                svg, all_layer_ids
            )

        # Step 5: Calculate content bounds and assemble LayerGeometry objects
        layer_geometries: List[LayerGeometry] = []
        for layer_id, geo in geometries_by_layer.items():
            if layer_id is None:
                continue
            if not geo.is_empty():
                min_x, min_y, max_x, max_y = geo.rect()
                layer_geometries.append(
                    LayerGeometry(
                        layer_id=layer_id,
                        content_bounds=(
                            min_x,
                            min_y,
                            max_x - min_x,
                            max_y - min_y,
                        ),
                    )
                )

        # Step 6: Assemble the final ParsingResult, ensuring all bounds are
        # consistently in PIXELS.
        page_bounds_px: Tuple[float, float, float, float]
        if viewbox:
            # Convert the viewbox (in user units) to pixel bounds.
            vb_x, vb_y, vb_w, vb_h = viewbox
            scale_x = width_px / vb_w if vb_w > 0 else 1.0
            scale_y = height_px / vb_h if vb_h > 0 else 1.0
            page_bounds_px = (
                vb_x * scale_x,
                vb_y * scale_y,
                vb_w * scale_x,
                vb_h * scale_y,
            )
        else:
            # No viewbox, the page bounds are simply the pixel dimensions.
            page_bounds_px = (0.0, 0.0, width_px, height_px)

        return ParsingResult(
            page_bounds=page_bounds_px,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_page_bounds=untrimmed_page_bounds_px,
        )

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

    def _get_svg_parsing_facts(self, svg: SVG) -> ParsingFactsType:
        """
        Extracts top-level facts (dimensions, viewbox) from a parsed SVG
        object.
        """
        if svg.width is None or svg.height is None:
            return None

        width_px: float = (
            float(svg.width.px)
            if hasattr(svg.width, "px")
            else float(svg.width)
        )
        height_px: float = (
            float(svg.height.px)
            if hasattr(svg.height, "px")
            else float(svg.height)
        )

        if width_px <= 1e-9 or height_px <= 1e-9:
            return None

        viewbox: ViewBoxType = None
        if (
            svg.viewbox
            and svg.viewbox.x is not None
            and svg.viewbox.y is not None
            and svg.viewbox.width is not None
            and svg.viewbox.height is not None
        ):
            viewbox = (
                svg.viewbox.x,
                svg.viewbox.y,
                svg.viewbox.width,
                svg.viewbox.height,
            )

        msg = (
            "Normalizing vectors using final pixel dimensions from "
            f"svgelements: {width_px:.3f}px x {height_px:.3f}px"
        )
        logger.debug(msg)
        return width_px, height_px, viewbox

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
