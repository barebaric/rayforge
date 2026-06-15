from __future__ import annotations

import logging
from typing import Dict, List, Optional

from raygeo.geo import Geometry
from raygeo.svg import (
    svg_string_to_geometries,
    svg_string_to_geometries_by_layer,
)

from ...core.vectorization_spec import (
    PassthroughSpec,
    VectorizationSpec,
)
from ..base_importer import ImporterFeature
from ..engine import NormalizationEngine
from ..structures import (
    LayerGeometry,
    ParsingResult,
    VectorizationResult,
)
from .svg_base import SvgImporterBase
from .svgutil import extract_layer_manifest

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
        (
            document_bounds,
            unit_to_mm,
            untrimmed_document_bounds,
            world_frame,
        ) = basics

        # 2. Extract layer geometry from trimmed data.
        assert self.trimmed_data is not None
        svg_str = self.trimmed_data.decode("utf-8")
        layers_raw = svg_string_to_geometries_by_layer(svg_str, 1.0, 1.0)

        # 3. Build layer geometries with names from manifest.
        layer_manifest = extract_layer_manifest(self.trimmed_data)
        layer_names_by_id = {
            layer["id"]: layer["name"] for layer in layer_manifest
        }

        layer_geometries: List[LayerGeometry] = []
        for layer_id, geo_list in layers_raw:
            geo = Geometry()
            for g in geo_list:
                geo.extend(g)
            if not geo.is_empty():
                layer_name = layer_names_by_id.get(layer_id, layer_id)
                min_x, min_y, max_x, max_y = geo.rect()
                w = max_x - min_x
                h = max_y - min_y
                abs_content_bounds = (min_x, min_y, w, h)

                layer_geometries.append(
                    LayerGeometry(
                        layer_id=layer_id,
                        name=layer_name,
                        content_bounds=abs_content_bounds,
                    )
                )

        # Create temporary result to calculate background transform
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

        return ParsingResult(
            document_bounds=document_bounds,
            native_unit_to_mm=unit_to_mm,
            is_y_down=True,
            layers=layer_geometries,
            untrimmed_document_bounds=untrimmed_document_bounds,
            geometry_is_relative_to_bounds=False,
            is_cropped_to_content=True,
            world_frame_of_reference=world_frame,
            background_world_transform=bg_item.world_matrix,
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

        assert self.trimmed_data is not None

        svg_str = self.trimmed_data.decode("utf-8")
        all_layer_ids = [layer.layer_id for layer in parse_result.layers]

        target_layer_ids = (
            spec.active_layer_ids
            if spec.active_layer_ids is not None
            else all_layer_ids
        )

        # Extract per-layer geometries via raygeo (already in user space).
        layers_raw = svg_string_to_geometries_by_layer(svg_str, 1.0, 1.0)
        geometries_by_layer: Dict[Optional[str], Geometry] = {}
        for layer_id, geo_list in layers_raw:
            if layer_id in target_layer_ids:
                geo = Geometry()
                for g in geo_list:
                    geo.extend(g)
                if not geo.is_empty():
                    geometries_by_layer[layer_id] = geo

        # If no layers found, fall back to the whole SVG.
        if not geometries_by_layer:
            logger.debug(
                "No layer-specific geometry found, parsing entire SVG as "
                "default."
            )
            geos = svg_string_to_geometries(svg_str, 1.0, 1.0)
            if geos:
                geo = Geometry()
                for g in geos:
                    geo.extend(g)
                if not geo.is_empty():
                    geometries_by_layer[None] = geo

        return VectorizationResult(
            geometries_by_layer=geometries_by_layer,
            source_parse_result=parse_result,
        )
