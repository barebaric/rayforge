from __future__ import annotations

import logging
from gettext import gettext as _
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from xml.etree import ElementTree as ET

from raygeo.geo import Geometry
from raygeo.geo.types import Rect
from raygeo.svg import extract_svg_metadata, svg_string_to_geometry

from ...core.matrix import Matrix
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import PassthroughSpec
from ..base_importer import (
    Importer,
)
from ..structures import (
    ImportManifest,
    LayerInfo,
    ParsingResult,
)
from .renderer import SVG_RENDERER
from .svgutil import (
    PPI,
    extract_layer_manifest,
    get_natural_size,
    is_unitless_svg,
)

logger = logging.getLogger(__name__)


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

    def _get_ppi(self) -> float:
        if self._vectorization_spec and hasattr(
            self._vectorization_spec, "ppi"
        ):
            return self._vectorization_spec.ppi
        return PPI

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
            is_unitless=is_unitless_svg(self.raw_data),
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        """Shared SourceAsset creation logic."""
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=SVG_RENDERER,
            thumbnail_data=self._render_thumbnail_from_renderer(
                SVG_RENDERER, self.trimmed_data
            ),
        )
        source.base_render_data = self.trimmed_data

        # Get pixel dimensions for rendering from the parsed SVG object,
        # which is based on the trimmed_data. This is the authoritative source
        # for the trimmed pixel dimensions.
        if self.trimmed_data:
            facts = self._get_svg_parsing_facts(self.trimmed_data)
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
            ppi = self._get_ppi()
            untrimmed_size = get_natural_size(source.original_data, ppi=ppi)
            if untrimmed_size:
                metadata["untrimmed_width_mm"] = untrimmed_size[0]
                metadata["untrimmed_height_mm"] = untrimmed_size[1]

            if source.base_render_data:
                trimmed_size = get_natural_size(
                    source.base_render_data, ppi=ppi
                )
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
        except ValueError:
            logger.warning("Could not calculate SVG metadata.", exc_info=True)
            self.add_warning(_("Could not calculate SVG metadata."))

        return source

    def _calculate_parsing_basics(
        self,
    ) -> Optional[
        Tuple[
            Rect,
            float,
            Optional[Rect],
            Rect,
        ]
    ]:
        """
        Common parsing logic. Returns:
        (document_bounds, unit_to_mm, untrimmed_document_bounds,
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

        # Check dimensions: if no viewBox and no explicit width/height,
        # verify there's actual geometry.
        try:
            check_meta = extract_svg_metadata(
                self.trimmed_data.decode("utf-8")
            )
        except ValueError:
            check_meta = None
        has_viewbox = check_meta is not None and check_meta.viewbox is not None
        has_explicit_dims = check_meta is not None and (
            check_meta.width is not None or check_meta.height is not None
        )
        if not has_viewbox and not has_explicit_dims:
            geo = self._convert_svg_to_geometry(self.trimmed_data)
            if geo.is_empty():
                self.add_error(_("SVG contains no geometry or dimensions."))
                return None

        facts = self._get_svg_parsing_facts(self.trimmed_data)
        if not facts:
            self.add_error(_("Could not determine valid SVG dimensions."))
            return None
        width_px, height_px, viewbox = facts

        # Get the physical size of the trimmed content
        ppi = self._get_ppi()
        mm_per_px = 25.4 / ppi
        final_dims_mm = get_natural_size(self.trimmed_data, ppi=ppi)
        if not final_dims_mm:
            final_dims_mm = (width_px * mm_per_px, height_px * mm_per_px)

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
        untrimmed_document_bounds: Optional[Rect] = None

        # First, try to get the authoritative untrimmed viewbox by parsing
        # the original, untrimmed SVG data. This is the correct frame of
        # reference for positioning.
        untrimmed_facts = self._get_svg_parsing_facts(self.raw_data)
        if untrimmed_facts:
            _w, _h, untrimmed_vb = untrimmed_facts
            if untrimmed_vb:
                untrimmed_document_bounds = untrimmed_vb
                logger.debug(f"Found untrimmed viewBox: {untrimmed_vb}")
            else:
                # Fallback for SVGs without a viewbox, use pixel dims
                untrimmed_document_bounds = (
                    0.0,
                    0.0,
                    float(_w),
                    float(_h),
                )
            logger.debug(
                f"Using untrimmed viewBox: {untrimmed_document_bounds}"
            )

        # If parsing failed, fall back to calculating from physical size
        if not untrimmed_document_bounds:
            untrimmed_size_mm = get_natural_size(self.raw_data, ppi=ppi)
            if untrimmed_size_mm and unit_to_mm > 0:
                untrimmed_w = untrimmed_size_mm[0] / unit_to_mm
                untrimmed_h = untrimmed_size_mm[1] / unit_to_mm
                untrimmed_document_bounds = (
                    0,
                    0,
                    untrimmed_w,
                    untrimmed_h,
                )
                logger.debug(
                    "Calculated untrimmed bounds from physical size: "
                    f"{untrimmed_document_bounds}"
                )

        # Calculate the authoritative world frame of reference (mm, Y-Up)
        ref_bounds_native = untrimmed_document_bounds or document_bounds
        ref_x, _ignored3, ref_w, ref_h = ref_bounds_native
        w_mm = ref_w * unit_to_mm
        h_mm = ref_h * unit_to_mm
        x_mm = ref_x * unit_to_mm
        y_mm = 0.0  # The world frame's origin is at its bottom-left.

        world_frame = (x_mm, y_mm, w_mm, h_mm)

        return (
            document_bounds,
            unit_to_mm,
            untrimmed_document_bounds,
            world_frame,
        )

    # --- Low-level Helpers ---

    def _analytical_trim(self, data: bytes) -> bytes:
        """Trims the SVG using vector geometry bounds."""
        try:
            root = ET.fromstring(data)

            # 1. Get geometry bounds in the SVG's native user coordinate
            # system.
            geo = self._convert_svg_to_geometry(data)
            if geo.is_empty():
                return data

            # Get pixel and user unit dimensions to calculate the scale
            facts = self._get_svg_parsing_facts(data)
            if not facts:
                return data
            orig_w_px, orig_h_px, viewbox = facts

            if viewbox:
                vb_x, vb_y, orig_vb_w, orig_vb_h = viewbox
            else:
                vb_x, vb_y, orig_vb_w, orig_vb_h = 0, 0, orig_w_px, orig_h_px

            logger.debug(
                f"_analytical_trim: OrigPx={orig_w_px}x{orig_h_px}, "
                f"VB=({vb_x}, {vb_y}, {orig_vb_w}, {orig_vb_h})"
            )

            min_x, min_y, max_x, max_y = geo.rect()

            logger.debug(
                f"_analytical_trim: Content Bounds (User Units): "
                f"min_x={min_x:.4f}, min_y={min_y:.4f}, "
                f"max_x={max_x:.4f}, max_y={max_y:.4f}"
            )

            # 3. Calculate new viewBox with padding to prevent clipping
            width = max_x - min_x
            height = max_y - min_y
            trim_padding = 0.01  # Default 1%
            if isinstance(self._vectorization_spec, PassthroughSpec):
                trim_padding = self._vectorization_spec.trim_padding
            padding = max(width, height) * trim_padding

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

        except (ValueError, ET.ParseError) as e:
            logger.warning(f"Analytical trim failed: {e}")
            self.add_warning(f"Optimization (trimming) failed: {e}")
            return data

    def _get_svg_parsing_facts(
        self, data: bytes
    ) -> Optional[Tuple[float, float, Optional[Rect]]]:
        try:
            meta = extract_svg_metadata(data.decode())
        except ValueError:
            return None

        ppi = self._get_ppi()
        w_px = meta.width_px(ppi)
        h_px = meta.height_px(ppi)
        if w_px is None or h_px is None:
            return None
        if w_px <= 1e-9 or h_px <= 1e-9:
            return None

        return w_px, h_px, meta.viewbox

    def _convert_svg_to_geometry(
        self, data: bytes, translate_to_origin: bool = False
    ) -> Geometry:
        try:
            geo = svg_string_to_geometry(data.decode("utf-8"), 1.0, 1.0)
        except ValueError:
            geo = Geometry()

        if translate_to_origin and not geo.is_empty():
            min_x, min_y, _, _ = geo.rect()
            translate_matrix = Matrix.translation(-min_x, -min_y)
            geo.transform(translate_matrix.to_4x4_numpy())

        return geo
