import io
import logging
from typing import Optional, List, Dict, Tuple
import ezdxf
import ezdxf.math
from ezdxf import bbox
from ezdxf.lldxf.const import DXFStructureError
from ezdxf.addons import text2path
from ezdxf.path import Command
from pathlib import Path

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import VectorizationSpec, PassthroughSpec
from ..assembler import ItemAssembler
from ..base_importer import (
    Importer,
    ImportPayload,
    ImporterFeature,
    ImportManifest,
    LayerInfo,
)
from ..engine import NormalizationEngine
from ..structures import ParsingResult, LayerGeometry, VectorizationResult
from .renderer import DXF_RENDERER

logger = logging.getLogger(__name__)

# Mapping of DXF units to millimeters
units_to_mm = {
    0: 1.0,
    1: 25.4,
    2: 304.8,
    4: 1.0,
    5: 10.0,
    6: 1000.0,
    8: 0.0254,
    9: 0.0254,
    10: 914.4,
}


class DxfImporter(Importer):
    label = "DXF files (2D)"
    mime_types = ("image/vnd.dxf",)
    extensions = (".dxf",)
    features = {ImporterFeature.DIRECT_VECTOR, ImporterFeature.LAYER_SELECTION}

    def __init__(self, data: bytes, source_file: Optional[Path] = None):
        super().__init__(data, source_file)
        self._dxf_doc: Optional[ezdxf.document.Drawing] = None  # type: ignore
        self._geometries_by_layer: Dict[Optional[str], Geometry] = {}

    def scan(self) -> ImportManifest:
        """
        Scans the DXF file for layer names and overall dimensions.
        """
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
        except DXFStructureError as e:
            logger.warning(f"DXF scan failed for {self.source_file.name}: {e}")
            return ImportManifest(
                title=self.source_file.name,
                warnings=["File appears to be a corrupt or unsupported DXF."],
            )
        except Exception as e:
            logger.error(
                f"Unexpected error during DXF scan for "
                f"{self.source_file.name}: {e}",
                exc_info=True,
            )
            return ImportManifest(
                title=self.source_file.name,
                warnings=[
                    "An unexpected error occurred while scanning the DXF file."
                ],
            )

        manifest_data = self._get_layer_manifest(doc)
        layers = [LayerInfo(id=m["id"], name=m["name"]) for m in manifest_data]
        bounds = self._get_bounds_mm(doc)
        size_mm = (bounds[2], bounds[3]) if bounds else None

        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=size_mm,
        )

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportPayload]:
        logger.debug("Starting DXF import process.")

        # Phase 2: Parsing DXF to native geometry.
        parse_result = self.parse()
        if not parse_result or not self._dxf_doc:
            logger.error("DXF Importer: Failed to parse DXF file.")
            return None

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=DXF_RENDERER,
            metadata={"is_vector": True},
        )
        spec = vectorization_spec or PassthroughSpec()

        if not parse_result.layers:
            logger.warning("DXF contains no valid geometry to import.")
            return ImportPayload(source=source, items=[])

        logger.debug("Phase 3: Vectorizing (packaging) parsed data.")
        vec_result = self._vectorize(parse_result, self._geometries_by_layer)

        logger.debug(
            "Phase 4: Calculating layout plan with NormalizationEngine."
        )
        engine = NormalizationEngine()
        plan = engine.calculate_layout(vec_result, spec)
        if not plan:
            logger.warning("Layout plan is empty; no items will be created.")
            return ImportPayload(source=source, items=[])

        logger.debug(f"Layout plan created with {len(plan)} item(s).")
        geometries: Dict[Optional[str], Geometry]
        if len(plan) == 1 and plan[0].layer_id is None:
            logger.debug("Merging all layer geometries into one.")
            merged_geo = Geometry()
            for geo in self._geometries_by_layer.values():
                merged_geo.extend(geo)
            geometries = {None: merged_geo}
        else:
            geometries = self._geometries_by_layer

        logger.debug("Phase 5: Assembling DocItems.")
        assembler = ItemAssembler()
        items = assembler.create_items(
            source_asset=source,
            layout_plan=plan,
            spec=spec,
            source_name=self.source_file.stem,
            geometries=geometries,
            layer_manifest=self._get_layer_manifest(self._dxf_doc),
        )
        logger.debug(f"Assembled {len(items)} top-level item(s).")

        return ImportPayload(source=source, items=items)

    def _vectorize(
        self,
        parse_result: ParsingResult,
        geometries_by_layer: Dict[Optional[str], Geometry],
    ) -> VectorizationResult:
        """Phase 3: Package parsed data for the layout engine."""
        return VectorizationResult(
            geometries_by_layer=geometries_by_layer,
            source_parse_result=parse_result,
        )

    def _get_layer_manifest(self, doc) -> List[Dict[str, str]]:
        return [
            {"id": layer.dxf.name, "name": layer.dxf.name}
            for layer in doc.layers
            if layer.dxf.name.lower() != "defpoints"
        ]

    def parse(self) -> Optional[ParsingResult]:
        """Phase 2: Parse DXF file into geometric facts."""
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
            self._dxf_doc = doc
        except DXFStructureError:
            self._dxf_doc = None
            return None

        doc_bounds = self._get_bounds_native(self._dxf_doc)
        if not doc_bounds:
            doc_bounds = (0.0, 0.0, 0.0, 0.0)
            logger.warning("DXF appears to be empty, using zero bounds.")

        result = ParsingResult(
            page_bounds=doc_bounds,
            native_unit_to_mm=self._get_scale_to_mm(self._dxf_doc),
            is_y_down=False,
            layers=[],
        )

        geometries_by_layer = self._extract_geometries(
            self._dxf_doc, transform=None
        )
        self._geometries_by_layer = geometries_by_layer
        logger.debug(
            f"Extracted geometries from {len(geometries_by_layer)} layers."
        )

        for layer_name, geo in geometries_by_layer.items():
            if layer_name is None:
                continue
            min_x, min_y, max_x, max_y = geo.rect()
            w = max_x - min_x
            h = max_y - min_y
            content_bounds = (min_x, min_y, w, h)
            result.layers.append(
                LayerGeometry(
                    layer_id=layer_name, content_bounds=content_bounds
                )
            )

        return result

    def _extract_geometries(
        self, doc, transform: Optional[ezdxf.math.Matrix44]
    ) -> Dict[Optional[str], Geometry]:
        layer_map: Dict[str, List] = {}
        for entity in doc.modelspace():
            layer = entity.dxf.layer
            if layer not in layer_map:
                layer_map[layer] = []
            layer_map[layer].append(entity)

        geometries_by_layer: Dict[Optional[str], Geometry] = {}
        for layer_name, entities in layer_map.items():
            layer_geo = Geometry()
            for entity in entities:
                if entity.dxftype() == "INSERT":
                    continue
                self._entity_to_native_geo(
                    layer_geo,
                    entity,
                    doc,
                    transform=transform,
                    tolerance_mm=0.01,
                )
            if not layer_geo.is_empty():
                geometries_by_layer[layer_name] = layer_geo
        return geometries_by_layer

    def _entity_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        handler_map = {
            "LINE": self._line_to_native_geo,
            "CIRCLE": self._circle_to_native_geo,
            "ARC": self._arc_to_native_geo,
            "LWPOLYLINE": self._poly_approx_to_native_geo,
            "ELLIPSE": self._poly_approx_to_native_geo,
            "SPLINE": self._poly_approx_to_native_geo,
            "POLYLINE": self._poly_approx_to_native_geo,
            "HATCH": self._hatch_to_native_geo,
            "TEXT": self._text_to_native_geo,
            "MTEXT": self._text_to_native_geo,
        }
        handler = handler_map.get(entity.dxftype())
        if handler:
            handler(geo, entity, doc, transform, tolerance_mm)
        else:
            logger.warning(
                f"Unsupported DXF entity type: {entity.dxftype()}. Skipping."
            )

    def _get_scale_to_mm(self, doc, default: float = 1.0) -> float:
        insunits = doc.header.get("$INSUNITS", 0)
        return units_to_mm.get(insunits, default) or default

    def _get_bounds_native(
        self, doc
    ) -> Optional[Tuple[float, float, float, float]]:
        entity_bbox = bbox.extents(doc.modelspace(), fast=True)
        if not entity_bbox.has_data:
            return None
        min_p, max_p = entity_bbox.extmin, entity_bbox.extmax
        return (min_p.x, min_p.y, (max_p.x - min_p.x), (max_p.y - min_p.y))

    def _get_bounds_mm(
        self, doc
    ) -> Optional[Tuple[float, float, float, float]]:
        entity_bbox = bbox.extents(doc.modelspace(), fast=True)
        if not entity_bbox.has_data:
            return None
        min_p, max_p = entity_bbox.extmin, entity_bbox.extmax
        scale = self._get_scale_to_mm(doc)
        return (
            min_p.x * scale,
            min_p.y * scale,
            (max_p.x - min_p.x) * scale,
            (max_p.y - min_p.y) * scale,
        )

    def _line_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        points = [entity.dxf.start, entity.dxf.end]
        if transform:
            points = list(transform.transform_vertices(points))
        start_vec, end_vec = points
        geo.move_to(start_vec.x, start_vec.y, start_vec.z)
        geo.line_to(end_vec.x, end_vec.y, end_vec.z)

    def _circle_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        temp_entity = entity.copy()
        if transform:
            try:
                temp_entity.transform(transform)
            except (NotImplementedError, AttributeError):
                self._poly_approx_to_native_geo(
                    geo, entity, doc, transform, tolerance_mm
                )
                return
        if temp_entity.dxftype() == "ELLIPSE":
            self._poly_approx_to_native_geo(
                geo, temp_entity, doc, None, tolerance_mm
            )
            return
        center, radius = temp_entity.dxf.center, temp_entity.dxf.radius
        start_point = (center.x + radius, center.y, center.z)
        mid_point = (center.x - radius, center.y, center.z)
        geo.move_to(start_point[0], start_point[1], start_point[2])
        geo.arc_to_as_bezier(
            mid_point[0], mid_point[1], -radius, 0, clockwise=False, z=center.z
        )
        geo.arc_to_as_bezier(
            start_point[0],
            start_point[1],
            radius,
            0,
            clockwise=False,
            z=center.z,
        )

    def _arc_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        self._poly_approx_to_native_geo(
            geo, entity, doc, transform, tolerance_mm
        )

    def _poly_approx_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        try:
            path_obj = ezdxf.path.make_path(  # type: ignore
                entity, flattening=tolerance_mm / 4.0
            )
            if transform:
                path_obj = path_obj.transform(transform)
            self._consume_native_path(geo, path_obj)
        except ezdxf.path.EmptyPathError:  # type: ignore
            logger.debug(
                f"Skipping empty path from entity {entity.dxftype()}."
            )
        except Exception as e:
            logger.error(
                f"Failed to convert entity {entity.dxftype()} to path: {e}",
                exc_info=True,
            )

    def _hatch_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        try:
            for path in entity.paths:
                path_obj = path.to_path()
                if transform:
                    path_obj = path_obj.transform(transform)
                self._consume_native_path(geo, path_obj)
        except Exception as e:
            logger.error(f"Failed to process HATCH entity: {e}", exc_info=True)

    def _text_to_native_geo(
        self, geo, entity, doc, transform, tolerance_mm: float = 0.01
    ):
        try:
            paths = text2path.make_paths_from_entity(entity)
            for path in paths:
                if transform:
                    path = path.transform(transform)
                self._consume_native_path(geo, path)
        except Exception as e:
            logger.error(f"Failed to convert TEXT entity: {e}", exc_info=True)

    def _consume_native_path(self, geo: Geometry, path):
        if not path:
            return
        start_vec = path.start
        geo.move_to(start_vec.x, start_vec.y, start_vec.z)
        current_x, current_y = start_vec.x, start_vec.y
        for cmd in path.commands():
            end_x, end_y, end_z = cmd.end.x, cmd.end.y, cmd.end.z
            if cmd.type == Command.MOVE_TO:
                geo.move_to(end_x, end_y, end_z)
            elif cmd.type == Command.LINE_TO:
                geo.line_to(end_x, end_y, end_z)
            elif cmd.type == Command.CURVE3_TO:
                ctrl_x, ctrl_y = cmd.ctrl.x, cmd.ctrl.y
                c1x, c1y = (
                    current_x + 2 / 3 * (ctrl_x - current_x),
                    current_y + 2 / 3 * (ctrl_y - current_y),
                )
                c2x, c2y = (
                    end_x + 2 / 3 * (ctrl_x - end_x),
                    end_y + 2 / 3 * (ctrl_y - end_y),
                )
                geo.bezier_to(end_x, end_y, c1x, c1y, c2x, c2y, end_z)
            elif cmd.type == Command.CURVE4_TO:
                c1x, c1y = cmd.ctrl1.x, cmd.ctrl1.y
                c2x, c2y = cmd.ctrl2.x, cmd.ctrl2.y
                geo.bezier_to(end_x, end_y, c1x, c1y, c2x, c2y, end_z)
            current_x, current_y = end_x, end_y
