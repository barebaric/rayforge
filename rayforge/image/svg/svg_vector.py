from __future__ import annotations
import logging
from typing import List, Optional, Dict
from svgelements import Group, Path as SvgPath, SVG
from ...core.geo import Geometry
from ...core.matrix import Matrix
from ...core.vectorization_spec import (
    PassthroughSpec,
    VectorizationSpec,
)
from ..base_importer import ImporterFeature
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
)
from .svgutil import extract_layer_manifest
from .svg_base import SvgImporterBase

logger = logging.getLogger(__name__)


class SvgVectorImporter(SvgImporterBase):
    """
    Imports SVG files by parsing vector paths directly.
    """

    label = "SVG (Vector Strategy)"
    mime_types = ()
    extensions = ()
    features = {
        ImporterFeature.DIRECT_VECTOR,
        ImporterFeature.LAYER_SELECTION,
    }

    def parse(self) -> Optional[ParsingResult]:
        # 1. Use base class to get dimensions, units, and the parsed SVG object
        basics = self._calculate_parsing_basics()
        if not basics:
            return None

        # Unpack. Both trimmed and untrimmed bounds are now available.
        svg_obj, page_bounds, unit_to_mm, untrimmed_page_bounds = basics

        # 2. Extract layers geometry. It will be in pixel coordinates.
        geometries_by_layer_px = self._parse_geometry_by_layer(
            svg_obj,
            [
                elem.id
                for elem in svg_obj
                if isinstance(elem, Group) and elem.id
            ],
        )

        # 3. Convert geometry from pixel-space to user-unit-space.
        geometries_by_layer_user = self._convert_pixel_geo_to_user_geo(
            svg_obj, geometries_by_layer_px
        )

        # 4. Calculate content bounds from the correctly-scaled geometry.
        assert self.trimmed_data is not None
        layer_manifest = extract_layer_manifest(self.trimmed_data)
        layer_names_by_id = {
            layer["id"]: layer["name"] for layer in layer_manifest
        }

        layer_geometries: List[LayerGeometry] = []
        for layer_id, geo in geometries_by_layer_user.items():
            if not geo.is_empty() and layer_id is not None:
                min_x, min_y, max_x, max_y = geo.rect()
                w = max_x - min_x
                h = max_y - min_y
                abs_content_bounds = (min_x, min_y, w, h)

                layer_geometries.append(
                    LayerGeometry(
                        layer_id=layer_id,
                        name=layer_names_by_id.get(layer_id, layer_id),
                        content_bounds=abs_content_bounds,
                    )
                )

        return ParsingResult(
            page_bounds=page_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_page_bounds=untrimmed_page_bounds,
            geometry_is_relative_to_bounds=False,
            is_cropped_to_content=True,
        )

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """
        Extracts vector geometry from SVG for direct import.
        """
        if not isinstance(spec, PassthroughSpec):
            spec = PassthroughSpec()

        if not self.svg:
            raise ValueError("self.svg is not set")

        all_layer_ids = [layer.layer_id for layer in parse_result.layers]

        target_layer_ids = (
            spec.active_layer_ids
            if spec.active_layer_ids is not None
            else all_layer_ids
        )

        geometries_by_layer_px = self._parse_geometry_by_layer(
            self.svg, target_layer_ids
        )

        # If layer-specific parsing found nothing (SVG has no layers),
        # fall back to parsing the whole SVG into a default 'None' layer.
        # This ensures geometry is always extracted.
        if not geometries_by_layer_px:
            logger.debug(
                "No layer-specific geometry found, parsing entire SVG as "
                "default."
            )
            geo = self._convert_svg_to_geometry(
                self.svg, translate_to_origin=False
            )
            if not geo.is_empty():
                geometries_by_layer_px[None] = geo

        geometries_by_layer_user = self._convert_pixel_geo_to_user_geo(
            self.svg, geometries_by_layer_px
        )

        return VectorizationResult(
            geometries_by_layer=geometries_by_layer_user,
            source_parse_result=parse_result,
        )

    def _convert_pixel_geo_to_user_geo(
        self, svg: SVG, pixel_geometries: Dict[Optional[str], Geometry]
    ) -> Dict[Optional[str], Geometry]:
        """
        Transforms a dictionary of Geometries from the pixel-based coordinate
        system of svgelements to the SVG's native user unit system.
        """
        width_val = getattr(svg.width, "px", svg.width)
        height_val = getattr(svg.height, "px", svg.height)
        w_px = float(width_val) if width_val is not None else 1.0
        h_px = float(height_val) if height_val is not None else 1.0

        if svg.viewbox:
            vb_x = float(svg.viewbox.x) if svg.viewbox.x is not None else 0.0
            vb_y = float(svg.viewbox.y) if svg.viewbox.y is not None else 0.0
            vb_w = (
                float(svg.viewbox.width)
                if svg.viewbox.width is not None
                else w_px
            )
            vb_h = (
                float(svg.viewbox.height)
                if svg.viewbox.height is not None
                else h_px
            )
        else:
            vb_x, vb_y, vb_w, vb_h = 0.0, 0.0, w_px, h_px

        scale_x = vb_w / w_px if w_px > 0 else 1.0
        scale_y = vb_h / h_px if h_px > 0 else 1.0

        transform = Matrix.translation(vb_x, vb_y) @ Matrix.scale(
            scale_x, scale_y
        )

        user_geometries = {}
        for layer_id, geo_px in pixel_geometries.items():
            geo_user = geo_px.copy()
            geo_user.transform(transform.to_4x4_numpy())
            user_geometries[layer_id] = geo_user

        return user_geometries

    def _parse_geometry_by_layer(
        self, svg: SVG, layer_ids: List[str]
    ) -> Dict[Optional[str], Geometry]:
        """
        Parses an SVG object and returns a dictionary mapping layer IDs to
        their corresponding Geometry in PIXEL coordinates. The caller is
        responsible for transforming to user units.
        """
        layer_geoms: Dict[Optional[str], Geometry] = {}

        has_explicit_layers = any(isinstance(e, Group) and e.id for e in svg)

        if not has_explicit_layers:
            return {}

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
                    layer_geoms[lid] = layer_geo

        return layer_geoms
