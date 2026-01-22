import io
import logging
import math
from typing import Optional, List, Dict, Tuple, DefaultDict, Iterable
from pathlib import Path
from collections import defaultdict
from dataclasses import replace

import ezdxf
import ezdxf.math
from ezdxf import bbox
from ezdxf.lldxf.const import DXFStructureError
from ezdxf.addons import text2path
from ezdxf.path import Command

from ...core.geo import Geometry
from ...core.source_asset import SourceAsset
from ...core.vectorization_spec import VectorizationSpec, PassthroughSpec
from ..base_importer import (
    Importer,
    ImporterFeature,
)
from ..engine import NormalizationEngine
from ..structures import (
    ParsingResult,
    LayerGeometry,
    VectorizationResult,
    ImportManifest,
    LayerInfo,
)
from .renderer import DXF_RENDERER

logger = logging.getLogger(__name__)

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
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
        except DXFStructureError as e:
            logger.warning(f"DXF scan failed: {e}")
            self.add_error(_(f"DXF file structure is invalid: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )
        except Exception as e:
            logger.error(f"DXF scan error: {e}", exc_info=True)
            self.add_error(_(f"Unexpected error while scanning DXF: {e}"))
            return ImportManifest(
                title=self.source_file.name, errors=self._errors
            )

        manifest_data = self._get_layer_manifest(doc)
        layers = [LayerInfo(id=m["id"], name=m["name"]) for m in manifest_data]
        bounds = self._get_bounds_mm(doc)
        size_mm = (bounds[2], bounds[3]) if bounds else None

        return ImportManifest(
            title=self.source_file.name,
            layers=layers,
            natural_size_mm=size_mm,
            warnings=self._warnings,
            errors=self._errors,
        )

    def create_source_asset(self, parse_result: ParsingResult) -> SourceAsset:
        _, _, w, h = parse_result.document_bounds
        width_mm = w * parse_result.native_unit_to_mm
        height_mm = h * parse_result.native_unit_to_mm

        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=DXF_RENDERER,
            metadata={"is_vector": True},
            width_mm=width_mm,
            height_mm=height_mm,
        )
        return source

    def vectorize(
        self,
        parse_result: ParsingResult,
        spec: VectorizationSpec,
    ) -> VectorizationResult:
        """
        Prepares the final vector geometry based on the user's specification.
        This method is "spec-aware" and handles the merging of geometries
        if requested.
        """
        split_layers = False
        active_layers_set = None
        if isinstance(spec, PassthroughSpec):
            split_layers = spec.create_new_layers
            if spec.active_layer_ids:
                active_layers_set = set(spec.active_layer_ids)

        # Filter geometries based on the active layers in the spec
        geometries_to_process: Dict[Optional[str], Geometry]
        if active_layers_set:
            geometries_to_process = {
                layer_id: geo
                for layer_id, geo in self._geometries_by_layer.items()
                if layer_id in active_layers_set
            }
        else:
            geometries_to_process = self._geometries_by_layer

        final_geometries: Dict[Optional[str], Geometry]
        if split_layers:
            # For a "split" strategy, return the dictionary of individual
            # layer geometries.
            final_geometries = geometries_to_process
        else:
            # For a "merge" strategy, combine all active geometries into a
            # single Geometry object under the `None` key.
            merged_geo = Geometry()
            for geo in geometries_to_process.values():
                merged_geo.extend(geo)
            final_geometries = {None: merged_geo}

        # Hack: Updating the parsing result bounds if layers were filtered
        # is a violation of the pipelines sequential nature. Ideally this
        # updated window would be communicated back to the layout engine in a
        # cleaner way. However, for now this ensures that the preview renderer
        # and layout engine remain in sync when layers are filtered out.

        # If we have filtered layers, we must recalculate the bounds in the
        # parsing result. Otherwise, the preview renderer (which renders based
        # on the filtered geometry) and the layout engine (which uses the
        # original whole-doc parsing result for the background image) will
        # disagree, causing the vector overlay to drift from the image.
        final_parse_result = parse_result
        if active_layers_set:
            union_bounds = self._calculate_geometry_union(
                final_geometries.values()
            )
            if union_bounds:
                final_parse_result = self._update_parse_result_bounds(
                    parse_result, union_bounds
                )

        return VectorizationResult(
            geometries_by_layer=final_geometries,
            source_parse_result=final_parse_result,
        )

    def _calculate_geometry_union(
        self, geometries: Iterable[Geometry]
    ) -> Optional[Tuple[float, float, float, float]]:
        """Calculates the bounding box of a collection of geometries."""
        min_x, min_y, max_x, max_y = (
            float("inf"),
            float("inf"),
            float("-inf"),
            float("-inf"),
        )
        has_content = False

        for geo in geometries:
            if not geo or geo.is_empty():
                continue
            gx1, gy1, gx2, gy2 = geo.rect()
            if gx1 < min_x:
                min_x = gx1
            if gy1 < min_y:
                min_y = gy1
            if gx2 > max_x:
                max_x = gx2
            if gy2 > max_y:
                max_y = gy2
            has_content = True

        if not has_content:
            return None

        return (min_x, min_y, max_x - min_x, max_y - min_y)

    def _update_parse_result_bounds(
        self,
        original: ParsingResult,
        new_bounds: Tuple[float, float, float, float],
    ) -> ParsingResult:
        """
        Creates a new ParsingResult with updated bounds and transform matrices
        to reflect a subset of the original document.
        """
        x, y, w, h = new_bounds
        scale = original.native_unit_to_mm

        # Recalculate world frame of reference (mm)
        new_world_frame = (x * scale, y * scale, w * scale, h * scale)

        # Create a temp result to allow the Engine to calculate the matrix
        # mapping the new bounds to the new world frame.
        # Note: We must preserve is_y_down=False for DXF (Y-Up).
        temp_result = replace(
            original,
            document_bounds=new_bounds,
            world_frame_of_reference=new_world_frame,
        )

        bg_item = NormalizationEngine.calculate_layout_item(
            new_bounds, temp_result
        )

        return replace(
            original,
            document_bounds=new_bounds,
            world_frame_of_reference=new_world_frame,
            background_world_transform=bg_item.world_matrix,
        )

    def _get_layer_manifest(self, doc) -> List[Dict[str, str]]:
        return [
            {"id": layer.dxf.name, "name": layer.dxf.name}
            for layer in doc.layers
            if layer.dxf.name.lower() != "defpoints"
        ]

    def parse(self) -> Optional[ParsingResult]:
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
            self._dxf_doc = doc
        except DXFStructureError as e:
            self._dxf_doc = None
            self.add_error(_(f"DXF file is corrupt or invalid: {e}"))
            return None

        # 1. Bounds
        doc_bounds = self._get_bounds_native(self._dxf_doc)
        if not doc_bounds:
            doc_bounds = (0.0, 0.0, 0.0, 0.0)

        # 2. Extract with Flattening & Sorting
        geometries_by_layer = self._extract_geometries(self._dxf_doc)
        self._geometries_by_layer = geometries_by_layer

        # 3. Consolidate (Adaptive Tolerance)
        w, h = doc_bounds[2], doc_bounds[3]
        diag = math.hypot(w, h)
        adaptive_tolerance = max(0.01, diag / 20000.0)

        for layer_name, geo in geometries_by_layer.items():
            if geo and not geo.is_empty():
                geo.close_gaps(tolerance=adaptive_tolerance)

        # 4. Create temporary result to calculate transforms
        native_unit_to_mm = self._get_scale_to_mm(self._dxf_doc)
        temp_result = ParsingResult(
            document_bounds=doc_bounds,
            native_unit_to_mm=native_unit_to_mm,
            is_y_down=False,
            layers=[],
            # Dummy values, will be replaced
            world_frame_of_reference=(0, 0, 0, 0),
            background_world_transform=None,  # type: ignore
        )

        # 5. Calculate authoritative frames using centralized logic
        x, y, w, h = doc_bounds
        world_frame = (
            x * native_unit_to_mm,
            y * native_unit_to_mm,
            w * native_unit_to_mm,
            h * native_unit_to_mm,
        )
        bg_layout_item = NormalizationEngine.calculate_layout_item(
            doc_bounds, temp_result
        )

        # 6. Final Result
        result = ParsingResult(
            document_bounds=doc_bounds,
            native_unit_to_mm=native_unit_to_mm,
            is_y_down=False,
            layers=[],
            world_frame_of_reference=world_frame,
            background_world_transform=bg_layout_item.world_matrix,
        )

        for layer_name, geo in geometries_by_layer.items():
            if layer_name is None or geo.is_empty():
                continue

            min_x, min_y, max_x, max_y = geo.rect()
            w = max_x - min_x
            h = max_y - min_y

            result.layers.append(
                LayerGeometry(
                    layer_id=layer_name,
                    name=layer_name,
                    content_bounds=(min_x, min_y, w, h),
                )
            )

        return result

    def _extract_geometries(self, doc) -> Dict[Optional[str], Geometry]:
        """
        Recursively extracts, flattens, sorts, and consumes DXF entities.
        Sorting is critical for creating continuous paths from scrambled
        modelspace entities.
        """
        raw_paths_by_layer: DefaultDict[str, List] = defaultdict(list)

        def process_entity(entity, parent_transform: ezdxf.math.Matrix44):
            # Transform Composition
            if hasattr(entity, "matrix44"):
                transform = parent_transform @ entity.matrix44()
            else:
                transform = parent_transform

            if entity.dxftype() == "INSERT":
                # Recurse into blocks (Flattening)
                block_name = entity.dxf.name
                if block_name in doc.blocks:
                    block_def = doc.blocks.get(block_name)
                    for sub_entity in block_def:
                        process_entity(sub_entity, transform)
                return

            layer_name = entity.dxf.layer
            if layer_name.lower() == "defpoints":
                return

            try:
                # Convert entities to ezdxf Paths
                if entity.dxftype() in ("TEXT", "MTEXT"):
                    paths = text2path.make_paths_from_entity(entity)
                    for path in paths:
                        raw_paths_by_layer[layer_name].append(
                            path.transform(transform)
                        )
                else:
                    path = ezdxf.path.make_path(entity)  # type: ignore
                    raw_paths_by_layer[layer_name].append(
                        path.transform(transform)
                    )
            except Exception:
                pass

        # 1. Collect all paths (Disordered)
        identity = ezdxf.math.Matrix44()
        for entity in doc.modelspace():
            process_entity(entity, identity)

        # 2. Sort/Chain paths and Consume
        final_geometries = {}
        for layer_name, paths in raw_paths_by_layer.items():
            if not paths:
                continue

            # Sort paths so end[i] approx== start[i+1]
            sorted_paths = self._chain_paths(paths)

            geo = Geometry()
            for path in sorted_paths:
                self._consume_native_path(geo, path)

            if not geo.is_empty():
                final_geometries[layer_name] = geo

        return final_geometries

    def _chain_paths(self, paths: List) -> List:
        """
        Greedy sorting of paths to restore continuity.
        Groups paths that share endpoints into contiguous chains.
        """
        if not paths:
            return []

        # Index start points for O(1) lookups
        # Precision: 3 decimal places ensures visual continuity matches
        start_map: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)

        for i, p in enumerate(paths):
            s = p.start
            key = (int(s.x * 1000), int(s.y * 1000))
            start_map[key].append(i)

        ordered = []
        visited = [False] * len(paths)

        for i in range(len(paths)):
            if visited[i]:
                continue

            # Start a new chain
            current_idx = i
            while True:
                visited[current_idx] = True
                p = paths[current_idx]
                ordered.append(p)

                # Look for a path starting where this one ends
                e = p.end
                end_key = (int(e.x * 1000), int(e.y * 1000))

                candidates = start_map.get(end_key)
                next_idx = -1

                if candidates:
                    for c_idx in candidates:
                        if not visited[c_idx]:
                            next_idx = c_idx
                            break

                if next_idx != -1:
                    current_idx = next_idx
                else:
                    # Chain broken
                    break

        return ordered

    def _consume_native_path(self, geo: Geometry, path):
        if not path:
            return

        start = path.start

        # Check continuity with the *internal* geometry cursor
        is_continuous = False
        if not geo.is_empty():
            lx, ly, lz = geo._get_last_point()
            if (lx - start.x) ** 2 + (ly - start.y) ** 2 + (
                lz - start.z
            ) ** 2 < 1e-8:
                is_continuous = True

        if not is_continuous:
            geo.move_to(start.x, start.y, start.z)

        for cmd in path.commands():
            end = cmd.end
            if cmd.type == Command.LINE_TO:
                geo.line_to(end.x, end.y, end.z)
            elif cmd.type == Command.CURVE4_TO:
                c1, c2 = cmd.ctrl1, cmd.ctrl2
                geo.bezier_to(end.x, end.y, c1.x, c1.y, c2.x, c2.y, end.z)
            elif cmd.type == Command.CURVE3_TO:
                start_x, start_y, _ = geo._get_last_point()
                ctrl = cmd.ctrl
                c1x = start_x + (2 / 3) * (ctrl.x - start_x)
                c1y = start_y + (2 / 3) * (ctrl.y - start_y)
                c2x = end.x + (2 / 3) * (ctrl.x - end.x)
                c2y = end.y + (2 / 3) * (ctrl.y - end.y)
                geo.bezier_to(end.x, end.y, c1x, c1y, c2x, c2y, end.z)
            elif cmd.type == Command.MOVE_TO:
                # Check internal continuity of the path object itself
                cx, cy, cz = geo._get_last_point()
                if (cx - end.x) ** 2 + (cy - end.y) ** 2 + (
                    cz - end.z
                ) ** 2 > 1e-8:
                    geo.move_to(end.x, end.y, end.z)

    def _get_scale_to_mm(self, doc, default: float = 1.0) -> float:
        insunits = doc.header.get("$INSUNITS", 0)
        return units_to_mm.get(insunits, default) or default

    def _get_bounds_native(self, doc):
        entity_bbox = bbox.extents(doc.modelspace(), fast=True)
        if not entity_bbox.has_data:
            return None
        min_p, max_p = entity_bbox.extmin, entity_bbox.extmax
        return (min_p.x, min_p.y, (max_p.x - min_p.x), (max_p.y - min_p.y))

    def _get_bounds_mm(self, doc):
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
