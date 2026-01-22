from __future__ import annotations
import io
import logging
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
from xml.etree import ElementTree as ET

from svgelements import (
    SVG,
    Path as SvgPath,
    Close,
    Move,
    Line,
    CubicBezier,
    QuadraticBezier,
    Arc,
)

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ..base_importer import (
    Importer,
)
from ..structures import (
    ParsingResult,
    ImportManifest,
    LayerInfo,
)
from .renderer import SVG_RENDERER
from .svgutil import (
    PPI,
    MM_PER_PX,
    get_natural_size,
    extract_layer_manifest,
)

logger = logging.getLogger(__name__)

ViewBoxType = Optional[Tuple[float, float, float, float]]
ParsingFactsType = Optional[Tuple[float, float, ViewBoxType]]


class SvgImporterBase(Importer):
    """
    Base class for SVG importers containing shared logic for:
    - Scanning metadata
    - Analytical trimming
    - Parsing SVG dimensions and units
    - Converting SVG paths to Geometry (for bounds/trimming)
    """

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self.trimmed_data: Optional[bytes] = None
        self.svg: Optional[SVG] = None

    def scan(self) -> ImportManifest:
        """Shared scan logic."""
        layers = []
        size_mm = None
        try:
            # Check for basic XML validity first to ensure we can catch
            # malformed files and warn the user.
            try:
                ET.fromstring(self.raw_data)
            except ET.ParseError as e:
                raise ET.ParseError(f"XML Parse Error: {e}")

            size_mm = get_natural_size(self.raw_data)
            layer_data = extract_layer_manifest(self.raw_data)
            layers = [
                LayerInfo(
                    id=layer["id"],
                    name=layer["name"],
                    feature_count=layer.get("count"),
                )
                for layer in layer_data
            ]
        except ET.ParseError as e:
            logger.warning(f"SVG scan failed for {self.source_file.name}: {e}")
            self.add_error(f"Could not parse SVG. File may be corrupt: {e}")
        except Exception as e:
            logger.error(
                f"Unexpected error during SVG scan for "
                f"{self.source_file.name}: {e}",
                exc_info=True,
            )
            self.add_error(f"Unexpected error while scanning SVG: {e}")

        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=size_mm,
            warnings=self._warnings,
            errors=self._errors,
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """Shared SourceAsset creation logic."""
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
        )
        source.base_render_data = self.trimmed_data

        # Get pixel dimensions for rendering from the parsed SVG object,
        # which is based on the trimmed_data. This is the authoritative source
        # for the trimmed pixel dimensions.
        if self.svg:
            facts = self._get_svg_parsing_facts(self.svg)
            if facts:
                w_px_float, h_px_float, viewbox = facts
                source.width_px = int(w_px_float)
                source.height_px = int(h_px_float)
                if viewbox:
                    source.metadata["viewbox"] = viewbox

        # The physical mm size comes from the layout, which is correct.
        _ignored1, _ignored2, w_native, h_native = parse_result.document_bounds
        source.width_mm = w_native * parse_result.native_unit_to_mm
        source.height_mm = h_native * parse_result.native_unit_to_mm

        metadata: Dict[str, Any] = {}
        try:
            untrimmed_size = get_natural_size(source.original_data)
            if untrimmed_size:
                metadata["untrimmed_width_mm"] = untrimmed_size[0]
                metadata["untrimmed_height_mm"] = untrimmed_size[1]

            if source.base_render_data:
                trimmed_size = get_natural_size(source.base_render_data)
                if trimmed_size:
                    metadata["trimmed_width_mm"] = trimmed_size[0]
                    metadata["trimmed_height_mm"] = trimmed_size[1]

                # Extract viewbox from trimmed data
                try:
                    root = ET.fromstring(source.base_render_data)
                    vb_str = root.get("viewBox")
                    if vb_str:
                        metadata["viewbox"] = tuple(map(float, vb_str.split()))
                except ET.ParseError:
                    pass

            source.metadata.update(metadata)
        except (ValueError, ET.ParseError):
            logger.warning("Could not calculate SVG metadata.", exc_info=True)
            self.add_warning(_("Could not calculate SVG metadata."))

        return source

    def _calculate_parsing_basics(
        self,
    ) -> Optional[
        Tuple[
            SVG,
            Tuple[float, float, float, float],
            float,
            Optional[Tuple[float, float, float, float]],
            Tuple[float, float, float, float],
        ]
    ]:
        """
        Common parsing logic. Returns:
        (svg_object, document_bounds, unit_to_mm, untrimmed_document_bounds,
         world_frame_of_reference)
        or None if parsing fails.

        Note: document_bounds are in Native Units (ViewBox units if available,
        otherwise Pixels).
        """
        self.trimmed_data = self._analytical_trim(self.raw_data)
        if not self.trimmed_data:
            logger.error("Failed to prepare trimmed SVG data.")
            self.add_error(_("Failed to prepare trimmed SVG data."))
            return None

        svg = self._parse_svg_data(self.trimmed_data)
        if svg is None:
            # Error already added in _parse_svg_data
            return None
        self.svg = svg

        # Check dimensions
        has_explicit_dims = svg.values is not None and (
            "width" in svg.values or "height" in svg.values
        )
        if not has_explicit_dims:
            geo = self._convert_svg_to_geometry(svg)
            if geo.is_empty():
                self.add_error(_("SVG contains no geometry or dimensions."))
                return None

        facts = self._get_svg_parsing_facts(svg)
        if not facts:
            self.add_error(_("Could not determine valid SVG dimensions."))
            return None
        width_px, height_px, viewbox = facts

        # Get the physical size of the trimmed content
        final_dims_mm = get_natural_size(self.trimmed_data)
        if not final_dims_mm:
            final_dims_mm = (width_px * MM_PER_PX, height_px * MM_PER_PX)

        if viewbox:
            # If ViewBox exists, we use ViewBox units as the Native Units.
            vb_x, vb_y, vb_w, vb_h = viewbox
            # Return absolute bounds of the trimmed ViewBox. This is critical.
            document_bounds = (vb_x, vb_y, vb_w, vb_h)
            unit_to_mm = final_dims_mm[0] / vb_w if vb_w > 0 else 1.0
        else:
            # No ViewBox: Native Units are Pixels.
            document_bounds = (0.0, 0.0, width_px, height_px)
            unit_to_mm = final_dims_mm[0] / width_px if width_px > 0 else 1.0

        # Calculate untrimmed bounds in the same Native Units
        untrimmed_document_bounds: Optional[
            Tuple[float, float, float, float]
        ] = None
        untrimmed_size_mm = get_natural_size(self.raw_data)

        if untrimmed_size_mm and unit_to_mm > 0:
            untrimmed_w = untrimmed_size_mm[0] / unit_to_mm
            untrimmed_h = untrimmed_size_mm[1] / unit_to_mm
            untrimmed_document_bounds = (0, 0, untrimmed_w, untrimmed_h)

        # Calculate the authoritative world frame of reference (mm, Y-Up)
        ref_bounds_native = untrimmed_document_bounds or document_bounds
        ref_x, _ignored3, ref_w, ref_h = ref_bounds_native
        w_mm = ref_w * unit_to_mm
        h_mm = ref_h * unit_to_mm
        x_mm = ref_x * unit_to_mm
        y_mm = 0.0  # The world frame's origin is at its bottom-left.

        world_frame = (x_mm, y_mm, w_mm, h_mm)

        return (
            svg,
            document_bounds,
            unit_to_mm,
            untrimmed_document_bounds,
            world_frame,
        )

    # --- Low-level Helpers ---

    def _analytical_trim(self, data: bytes) -> bytes:
        """Trims the SVG using vector geometry bounds."""
        try:
            svg = SVG.parse(io.BytesIO(data), ppi=PPI)
            root = ET.fromstring(data)

            # 1. Get geometry bounds in the SVG's native user coordinate
            # system.
            # This requires converting from svgelements' pixel-based coords.
            geo_px = self._convert_svg_to_geometry(svg)
            if geo_px.is_empty():
                return data

            # Get pixel and user unit dimensions to calculate the scale
            facts = self._get_svg_parsing_facts(svg)
            if not facts:
                return data
            orig_w_px, orig_h_px, viewbox = facts

            if viewbox:
                vb_x, vb_y, orig_vb_w, orig_vb_h = viewbox
            else:
                vb_x, vb_y, orig_vb_w, orig_vb_h = 0, 0, orig_w_px, orig_h_px

            scale_x = orig_vb_w / orig_w_px if orig_w_px > 0 else 1.0
            scale_y = orig_vb_h / orig_h_px if orig_h_px > 0 else 1.0

            from ...core.matrix import Matrix

            scale_matrix = Matrix.scale(scale_x, scale_y)
            translate_matrix = Matrix.translation(vb_x, vb_y)
            final_transform = translate_matrix @ scale_matrix

            geo_user_units = geo_px.copy()
            geo_user_units.transform(final_transform.to_4x4_numpy())

            min_x, min_y, max_x, max_y = geo_user_units.rect()

            # 3. Calculate new viewBox with padding to prevent clipping
            width = max_x - min_x
            height = max_y - min_y
            padding = max(width, height) * 0.01

            new_vb_x = min_x - padding
            new_vb_y = min_y - padding
            new_vb_w = width + (2 * padding)
            new_vb_h = height + (2 * padding)

            if new_vb_w <= 1e-6 or new_vb_h <= 1e-6:
                return data

            # 4. Calculate the original pixels-per-user-unit scale.
            scale_x_render = orig_w_px / orig_vb_w if orig_vb_w > 0 else 1.0
            scale_y_render = orig_h_px / orig_vb_h if orig_vb_h > 0 else 1.0

            # 5. Calculate new pixel dimensions for the trimmed SVG.
            new_w_px = new_vb_w * scale_x_render
            new_h_px = new_vb_h * scale_y_render

            # 6. Build the new SVG, using pixel units for consistency.
            new_vb_str = f"{new_vb_x:g} {new_vb_y:g} {new_vb_w:g} {new_vb_h:g}"
            root.set("viewBox", new_vb_str)
            root.set("width", f"{new_w_px:.4f}px")
            root.set("height", f"{new_h_px:.4f}px")

            if "preserveAspectRatio" in root.attrib:
                del root.attrib["preserveAspectRatio"]

            return ET.tostring(root, encoding="utf-8")

        except (ET.ParseError, ValueError, TypeError) as e:
            # Catch specific, expected errors instead of a generic Exception.
            logger.warning(f"Analytical trim failed: {e}")
            self.add_warning(f"Optimization (trimming) failed: {e}")
            return data

    def _parse_svg_data(self, data: bytes) -> Optional[SVG]:
        try:
            svg_stream = io.BytesIO(data)
            return SVG.parse(svg_stream, ppi=PPI)
        except (ET.ParseError, ValueError, TypeError) as e:
            logger.error(f"Failed to parse SVG for direct import: {e}")
            self.add_error(_(f"Failed to parse SVG structure: {e}"))
            return None

    def _get_svg_parsing_facts(self, svg: SVG) -> ParsingFactsType:
        if svg.width is None or svg.height is None:
            return None

        # Robustly get pixel dimensions, as svgelements can return a float
        width_px: float = float(getattr(svg.width, "px", svg.width))
        height_px: float = float(getattr(svg.height, "px", svg.height))

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

        return width_px, height_px, viewbox

    def _convert_svg_to_geometry(
        self, svg: SVG, translate_to_origin: bool = False
    ) -> Geometry:
        geo = Geometry()
        for shape in svg.elements():
            try:
                path = SvgPath(shape)
                path.reify()
                self._add_path_to_geometry(path, geo)
            except (AttributeError, TypeError):
                continue

        if translate_to_origin and not geo.is_empty():
            min_x, min_y, _, _ = geo.rect()
            from ...core.matrix import Matrix

            translate_matrix = Matrix.translation(-min_x, -min_y)
            geo.transform(translate_matrix.to_4x4_numpy())

        return geo

    def _add_path_to_geometry(self, path: SvgPath, geo: Geometry) -> None:
        for seg in path:
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
