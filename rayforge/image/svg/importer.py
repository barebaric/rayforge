from __future__ import annotations
import io
import math
import logging
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from xml.etree import ElementTree as ET
from svgelements import (
    SVG,
    Arc,
    Close,
    CubicBezier,
    Line,
    Move,
    Path as SvgPath,
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
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
)
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

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.trimmed_data: Optional[bytes] = None
        self.svg: Optional[SVG] = None
        self.traced_artefacts: Dict[str, Any] = {}

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

    def _prepare_trimmed_data(self) -> None:
        """
        Prepares the trimmed SVG data by applying analytical trimming to the
        raw data. Populates self.trimmed_data.
        """
        self.trimmed_data = self._analytical_trim(self.raw_data)

    def create_source_asset(
        self, parse_result: ParsingResult, spec: VectorizationSpec
    ) -> SourceAsset:
        """
        Creates a SourceAsset for SVG import.
        """
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
        )
        source.base_render_data = self.trimmed_data

        # Populate dimensions from parse result
        _, _, w_px, h_px = parse_result.page_bounds
        source.width_px = int(w_px)
        source.height_px = int(h_px)
        source.width_mm = w_px * parse_result.native_unit_to_mm
        source.height_mm = h_px * parse_result.native_unit_to_mm

        # Populate metadata
        metadata: Dict[str, Any] = {}
        try:
            # Get size of original, untrimmed SVG
            untrimmed_size = get_natural_size(source.original_data)
            if untrimmed_size:
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

            metadata["is_vector"] = not isinstance(spec, TraceSpec)
            source.metadata.update(metadata)
        except Exception as e:
            logger.warning(f"Could not calculate SVG metadata: {e}")

        return source

    def parse(self) -> Optional[ParsingResult]:
        """
        Parses the SVG data into a ParsingResult.

        This method performs analytical trimming, parses the result into
        svgelements (creating self.svg), and generates the ParsingResult.
        """
        # Step 1: Prepare trimmed data
        self._prepare_trimmed_data()
        if not self.trimmed_data:
            logger.error("Failed to prepare trimmed SVG data.")
            return None

        # Step 2: Parse SVG into svgelements object
        svg = self._parse_svg_data(self.trimmed_data)
        if svg is None:
            return None
        self.svg = svg

        # Early exit check: If SVG has no explicit dimensions, we must check if
        # it has any geometry before proceeding.
        has_explicit_dims = svg.values is not None and (
            "width" in svg.values or "height" in svg.values
        )
        if not has_explicit_dims:
            geo = self._convert_svg_to_geometry(svg)
            if geo.is_empty():
                return None

        # Step 3: Extract top-level parsing facts from the parsed SVG object.
        facts = self._get_svg_parsing_facts(svg)
        if not facts:
            return None
        width_px, height_px, viewbox = facts

        # Step 4: Determine physical dimensions and scale factor.
        final_dims_mm = get_natural_size(self.trimmed_data)
        if not final_dims_mm:
            final_dims_mm = (width_px * MM_PER_PX, height_px * MM_PER_PX)
        unit_to_mm = final_dims_mm[0] / width_px if width_px > 0 else 1.0

        # Step 5: Determine the absolute page bounds in pixels.
        page_bounds_px: Tuple[float, float, float, float]
        if viewbox:
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
            page_bounds_px = (0.0, 0.0, width_px, height_px)
        page_x, page_y = page_bounds_px[0], page_bounds_px[1]

        # Step 6: Get untrimmed page bounds for positioning reference.
        untrimmed_page_bounds_px: Optional[
            Tuple[float, float, float, float]
        ] = None
        untrimmed_size_mm = get_natural_size(self.raw_data)
        if untrimmed_size_mm and unit_to_mm > 0:
            untrimmed_w_px = untrimmed_size_mm[0] / unit_to_mm
            untrimmed_h_px = untrimmed_size_mm[1] / unit_to_mm
            untrimmed_page_bounds_px = (0, 0, untrimmed_w_px, untrimmed_h_px)

        # Step 7: Parse geometry for all layers to calculate their bounds.
        raw_geometries_by_layer: Dict[str, Geometry] = {}
        all_layer_ids = [
            elem.id for elem in svg if isinstance(elem, Group) and elem.id
        ]
        if not all_layer_ids:
            all_layer_ids = ["__default__"]
            raw_geometries_by_layer["__default__"] = (
                self._convert_svg_to_geometry(svg)
            )
        else:
            for lid in all_layer_ids:
                temp_geo = Geometry()
                for element in svg:
                    if isinstance(element, Group) and element.id == lid:
                        for shape in element:
                            try:
                                path = SvgPath(shape)
                                self._add_path_to_geometry(path, temp_geo)
                            except (AttributeError, TypeError):
                                pass
                if not temp_geo.is_empty():
                    raw_geometries_by_layer[lid] = temp_geo

        # Step 8: Calculate content bounds and assemble LayerGeometry objects.
        layer_geometries: List[LayerGeometry] = []
        for layer_id, geo in raw_geometries_by_layer.items():
            if not geo.is_empty():
                min_x, min_y, max_x, max_y = geo.rect()
                w = max_x - min_x
                h = max_y - min_y
                abs_content_bounds = (page_x + min_x, page_y + min_y, w, h)
                layer_geometries.append(
                    LayerGeometry(
                        layer_id=layer_id,
                        content_bounds=abs_content_bounds,
                    )
                )

        # Step 9: Assemble and return the final ParsingResult.
        return ParsingResult(
            page_bounds=page_bounds_px,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_page_bounds=untrimmed_page_bounds_px,
            geometry_is_relative_to_bounds=True,
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
        spec = vectorization_spec or PassthroughSpec()

        # Phase 2: Parse
        parse_result = self.parse()
        if not parse_result:
            return None

        # Create Source Asset
        source = self.create_source_asset(parse_result, spec)

        # Phase 3: Vectorize
        vec_result = self.vectorize(parse_result, spec)
        if vec_result is None:
            return None

        # If tracing occurred, update the source asset with the rendered png
        if self.traced_artefacts:
            source.base_render_data = self.traced_artefacts.get("png_data")
            source.width_px = self.traced_artefacts.get("width_px")
            source.height_px = self.traced_artefacts.get("height_px")

        logger.debug(
            "Phase 4: Calculating layout plan with NormalizationEngine."
        )
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)

        if not plan:
            logger.info(
                "SVG import resulted in empty layout plan. No items created."
            )
            items: List["DocItem"] = []
        else:
            geometries: Dict[Optional[str], Geometry]
            if (
                not isinstance(spec, TraceSpec)
                and len(plan) == 1
                and plan[0].layer_id is None
            ):
                if not self.svg:
                    items = []
                else:
                    geometries = {
                        None: self._convert_svg_to_geometry(
                            self.svg, translate_to_origin=True
                        )
                    }
                    logger.debug("Phase 5: Assembling DocItems.")
                    assembler = ItemAssembler()
                    items = assembler.create_items(
                        source_asset=source,
                        layout_plan=plan,
                        spec=spec,
                        source_name=self.source_file.stem,
                        geometries=geometries,
                        layer_manifest=extract_layer_manifest(self.raw_data),
                        page_bounds=parse_result.page_bounds,
                    )
            else:
                geometries = vec_result.geometries_by_layer

                logger.debug("Phase 5: Assembling DocItems.")
                assembler = ItemAssembler()
                items = assembler.create_items(
                    source_asset=source,
                    layout_plan=plan,
                    spec=spec,
                    source_name=self.source_file.stem,
                    geometries=geometries,
                    layer_manifest=extract_layer_manifest(self.raw_data),
                    page_bounds=vec_result.source_parse_result.page_bounds,
                )

        # If items is an empty list, it's a valid file with no content.
        # Return a payload with the source but no items.
        return ImportPayload(source=source, items=items)

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

    def _vectorize_trace(
        self,
        result: ParsingResult,
        spec: TraceSpec,
    ) -> Optional[VectorizationResult]:
        """
        Renders the SVG to a bitmap, traces it, and returns a
        VectorizationResult containing the traced geometry.
        """
        page_bounds = (
            result.untrimmed_page_bounds
            if result.untrimmed_page_bounds
            else result.page_bounds
        )
        native_unit_to_mm = result.native_unit_to_mm

        w_px_orig = page_bounds[2]
        h_px_orig = page_bounds[3]
        w_mm = w_px_orig * native_unit_to_mm
        h_mm = h_px_orig * native_unit_to_mm

        if w_mm <= 0 or h_mm <= 0:
            logger.warning("Cannot trace SVG: failed to determine size.")
            return None

        aspect = w_mm / h_mm if h_mm > 0 else 1.0
        TARGET_DIM = math.sqrt(VTRACER_PIXEL_LIMIT)

        if aspect >= 1.0:
            w_px = int(TARGET_DIM)
            h_px = int(TARGET_DIM / aspect)
        else:
            h_px = int(TARGET_DIM)
            w_px = int(TARGET_DIM * aspect)
        w_px, h_px = max(1, w_px), max(1, h_px)

        logger.debug(
            f"Rendering SVG: w_px={w_px}, h_px={h_px}, aspect={aspect}"
        )

        vips_image = SVG_RENDERER.render_base_image(
            self.raw_data, width=w_px, height=h_px
        )

        if vips_image:
            logger.debug(
                f"Rendered image: width={vips_image.width}, "
                f"height={vips_image.height}"
            )
        if not vips_image:
            logger.error("Failed to render SVG to vips image for tracing.")
            return None

        if w_mm > 0 and h_mm > 0:
            xres = w_px / w_mm
            yres = h_px / h_mm
            vips_image = vips_image.copy(xres=xres, yres=yres)

        png_data = vips_image.pngsave_buffer()
        logger.debug(f"Storing traced artefacts: len={len(png_data)}")
        self.traced_artefacts["png_data"] = png_data
        self.traced_artefacts["width_px"] = vips_image.width
        self.traced_artefacts["height_px"] = vips_image.height

        normalized_vips = image_util.normalize_to_rgba(vips_image)
        if not normalized_vips:
            return None
        surface = image_util.vips_rgba_to_cairo_surface(normalized_vips)

        logger.debug("Phase 3: Vectorizing (tracing) geometry.")
        geometries = trace_surface(surface, spec)

        combined_geo = Geometry()
        if geometries:
            for geo in geometries:
                geo.close_gaps()
                combined_geo.extend(geo)

        pristine_geo = combined_geo.copy()

        rendered_width = w_px
        rendered_height = h_px
        mm_per_px_x, mm_per_px_y = image_util.get_mm_per_pixel(vips_image)
        trace_parse_result = ParsingResult(
            page_bounds=(
                0.0,
                0.0,
                float(rendered_width),
                float(rendered_height),
            ),
            native_unit_to_mm=mm_per_px_x,
            is_y_down=True,
            layers=[],
            geometry_is_relative_to_bounds=False,
        )
        logger.debug(
            f"trace_parse_result.page_bounds={trace_parse_result.page_bounds}"
        )

        return VectorizationResult(
            geometries_by_layer={None: pristine_geo},
            source_parse_result=trace_parse_result,
        )

    def _vectorize_direct(
        self,
        result: ParsingResult,
        spec: PassthroughSpec,
    ) -> VectorizationResult:
        """Extracts vector geometry from SVG for direct import."""
        if not self.svg:
            raise ValueError("self.svg is not set")

        all_layer_ids = [layer.layer_id for layer in result.layers]
        geometries_by_layer = self._parse_geometry_by_layer(
            self.svg, all_layer_ids
        )

        if not geometries_by_layer:
            layer_id = result.layers[0].layer_id if result.layers else None
            geometries_by_layer[layer_id] = self._convert_svg_to_geometry(
                self.svg, translate_to_origin=True
            )

        return VectorizationResult(
            geometries_by_layer=geometries_by_layer,
            source_parse_result=result,
        )

    def vectorize(
        self,
        result: ParsingResult,
        spec: VectorizationSpec,
    ) -> Optional[VectorizationResult]:
        """Unified vectorization method that dispatches based on spec type."""
        self.traced_artefacts = {}  # Reset
        if isinstance(spec, TraceSpec):
            return self._vectorize_trace(result, spec)
        if not isinstance(spec, PassthroughSpec):
            spec = PassthroughSpec()
        return self._vectorize_direct(result, spec)

    def _parse_geometry_by_layer(
        self, svg: SVG, layer_ids: List[str]
    ) -> Dict[Optional[str], Geometry]:
        """
        Parses an SVG object and returns a dictionary mapping layer IDs to
        their corresponding Geometry. Each geometry is translated to its
        own local origin (0,0) to fulfill the pipeline contract.
        """
        layer_geoms: Dict[Optional[str], Geometry] = {}

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
            if lid and lid in layer_ids:
                layer_geo = Geometry()
                for shape in _get_all_shapes(element):
                    try:
                        path = SvgPath(shape)
                        self._add_path_to_geometry(path, layer_geo)
                    except (AttributeError, TypeError):
                        pass

                if not layer_geo.is_empty():
                    # Translate the extracted geometry
                    # so its own bounding box starts at (0,0). This makes it
                    # truly relative to its content bounds.
                    min_x, min_y, _, _ = layer_geo.rect()
                    translate_matrix = Matrix.translation(-min_x, -min_y)
                    layer_geo.transform(translate_matrix.to_4x4_numpy())
                    layer_geoms[lid] = layer_geo

        return layer_geoms

    def _parse_svg_data(self, data: bytes) -> Optional[SVG]:
        """Parses SVG byte data into an svgelements.SVG object."""
        try:
            svg_stream = io.BytesIO(data)
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

    def _convert_svg_to_geometry(
        self, svg: SVG, translate_to_origin: bool = False
    ) -> Geometry:
        """
        Converts an SVG object into a Geometry object in pixel coordinates.
        Preserves curves as Béziers.
        Optionally translates the entire geometry to its own origin.
        """
        geo = Geometry()

        for shape in svg.elements():
            try:
                path = SvgPath(shape)
                path.reify()  # Apply transforms
                self._add_path_to_geometry(path, geo)
            except (AttributeError, TypeError):
                continue  # Skip non-shape elements like <defs>

        if translate_to_origin and not geo.is_empty():
            min_x, min_y, _, _ = geo.rect()
            translate_matrix = Matrix.translation(-min_x, -min_y)
            geo.transform(translate_matrix.to_4x4_numpy())

        return geo

    def _add_path_to_geometry(self, path: SvgPath, geo: Geometry) -> None:
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
