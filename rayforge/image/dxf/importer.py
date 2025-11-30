import io
import logging
from copy import deepcopy
from typing import Iterable, Optional, List, Dict, Tuple
import ezdxf
import ezdxf.math
from ezdxf import bbox
from ezdxf.lldxf.const import DXFStructureError
from ezdxf.addons import text2path

from ...core.geo import Geometry
from ...core.group import Group
from ...core.workpiece import WorkPiece
from ...core.matrix import Matrix
from ...core.item import DocItem
from ...core.vectorization_spec import VectorizationSpec
from ...core.source_asset import SourceAsset
from ...core.source_asset_segment import SourceAssetSegment
from ...core.vectorization_spec import PassthroughSpec
from ..base_importer import Importer, ImportPayload
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

    def get_doc_items(
        self, vectorization_spec: Optional[VectorizationSpec] = None
    ) -> Optional[ImportPayload]:
        # DXF is a vector format, so the vectorization_spec is ignored.
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
        except DXFStructureError:
            return None

        bounds = self._get_bounds_mm(doc)

        # Create the SourceAsset. It's valid even for an empty file.
        source = SourceAsset(
            source_file=self.source_file,
            original_data=self.raw_data,
            renderer=DXF_RENDERER,
            metadata={"is_vector": True},
        )

        if not bounds or not bounds[2] or not bounds[3]:
            return ImportPayload(source=source, items=[])

        _, _, width_mm, height_mm = bounds
        source.width_mm = width_mm
        source.height_mm = height_mm
        source.metadata["natural_size"] = (width_mm, height_mm)

        scale = self._get_scale_to_mm(doc)
        min_x_mm, min_y_mm, _, _ = bounds
        blocks_cache: Dict[str, List[DocItem]] = {}

        # Pre-parse all block definitions into DocItem templates.
        self._prepare_blocks_cache(
            doc, scale, min_x_mm, min_y_mm, source, blocks_cache
        )

        # Group entities by layer to support layer-based grouping
        layer_map: Dict[str, List] = {}
        for entity in doc.modelspace():
            layer = entity.dxf.layer
            if layer not in layer_map:
                layer_map[layer] = []
            layer_map[layer].append(entity)

        active_layers = []
        for layer_name, entities in layer_map.items():
            items = self._entities_to_doc_items(
                entities,
                doc,
                scale,
                min_x_mm,
                min_y_mm,
                source,
                blocks_cache,
            )
            if items:
                active_layers.append((layer_name, items))

        result_items: List[DocItem] = []

        # If only one single layer exist, it should not be grouped.
        if len(active_layers) == 1:
            result_items = active_layers[0][1]
        else:
            for layer_name, items in active_layers:
                group = Group(name=layer_name)
                group.set_children(items)
                result_items.append(group)

        return ImportPayload(source=source, items=result_items)

    def _prepare_blocks_cache(
        self,
        doc,
        scale: float,
        tx: float,
        ty: float,
        source: SourceAsset,
        blocks_cache: Dict[str, List[DocItem]],
    ):
        """Recursively parses all block definitions into lists of DocItems."""
        blocks_cache.clear()
        for block in doc.blocks:
            blocks_cache[block.name] = self._entities_to_doc_items(
                block,
                doc,
                scale,
                tx,
                ty,
                source,
                blocks_cache,
                ezdxf.math.Matrix44(),
            )

    def _entities_to_doc_items(
        self,
        entities: Iterable,
        doc,
        scale: float,
        tx: float,
        ty: float,
        source: SourceAsset,
        blocks_cache: Dict[str, List[DocItem]],
        parent_transform: Optional[ezdxf.math.Matrix44] = None,
    ) -> List[DocItem]:
        """
        Converts a list of DXF entities into a list of DocItems (WorkPieces
        and Groups).
        """
        result_items: List[DocItem] = []
        current_geo = Geometry()
        current_solids: List[List[Tuple[float, float]]] = []

        def flush_geo_to_workpiece():
            """
            Converts the accumulated Geometry and solid data into a single
            WorkPiece.
            """
            nonlocal current_geo, current_solids
            if current_geo.is_empty():
                return

            if source and current_solids:
                existing_solids = source.metadata.get("solids", [])
                existing_solids.extend(current_solids)
                source.metadata["solids"] = existing_solids

            min_x, min_y, max_x, max_y = current_geo.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

            # The geometry from the DXF is Y-up. We must convert it to a
            # normalized Y-down geometry for storage in the segment.
            segment_mask_geo = current_geo.copy()
            segment_mask_geo.close_gaps()

            # 1. Translate to origin (0,0 is bottom-left).
            translation_matrix = Matrix.translation(-min_x, -min_y)
            segment_mask_geo.transform(translation_matrix.to_4x4_numpy())

            # 2. Normalize to a 1x1 box. The geometry is now in a
            #    (0,0)-(1,1) box, but is still Y-up.
            if width > 0 and height > 0:
                norm_matrix = Matrix.scale(1.0 / width, 1.0 / height)
                segment_mask_geo.transform(norm_matrix.to_4x4_numpy())

            # 3. Flip the Y-axis to convert to the required Y-down format.
            #    This is a scale by -1 on Y, then a translation by +1 on Y.
            flip_matrix = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            segment_mask_geo.transform(flip_matrix.to_4x4_numpy())

            gen_config = SourceAssetSegment(
                source_asset_uid=source.uid,
                segment_mask_geometry=segment_mask_geo,
                vectorization_spec=PassthroughSpec(),
            )

            wp = WorkPiece(
                name=self.source_file.stem,
                source_segment=gen_config,
            )
            wp.natural_width_mm = width
            wp.natural_height_mm = height

            # Set the workpiece's matrix to position and scale it.
            # This matrix operates on the Y-up geometry that the
            # WorkPiece.boundaries property will provide.
            wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
                width, height
            )

            result_items.append(wp)
            current_geo = Geometry()
            current_solids = []

        for entity in entities:
            if entity.dxftype() == "INSERT":
                flush_geo_to_workpiece()
                block_items = blocks_cache.get(entity.dxf.name)
                if not block_items:
                    continue

                group = Group(name=entity.dxf.name)
                group.set_children(deepcopy(block_items))

                m = entity.matrix44()
                if parent_transform:
                    m = parent_transform @ m
                ux, uy, uz, pos = m.get_components()

                instance_matrix = Matrix(
                    [[ux.x, uy.x, pos.x], [ux.y, uy.y, pos.y], [0, 0, 1]]
                )

                global_transform = Matrix.translation(-tx, -ty) @ Matrix.scale(
                    scale, scale
                )
                group.matrix = global_transform @ instance_matrix

                result_items.append(group)

            elif entity.dxftype() == "SOLID":
                self._solid_to_geo_and_data(
                    current_geo,
                    current_solids,
                    entity,
                    scale,
                    tx,
                    ty,
                    parent_transform,
                )
            else:
                self._entity_to_geo(
                    current_geo, entity, doc, scale, tx, ty, parent_transform
                )

        flush_geo_to_workpiece()
        return result_items

    def _entity_to_geo(self, geo, entity, doc, scale, tx, ty, transform):
        """Dispatcher to call the correct handler for a given DXF entity."""
        handler_map = {
            "LINE": self._line_to_geo,
            "CIRCLE": self._poly_approx_to_geo,
            "LWPOLYLINE": self._lwpolyline_to_geo,
            "ARC": self._arc_to_geo,
            "ELLIPSE": self._poly_approx_to_geo,
            "SPLINE": self._poly_approx_to_geo,
            "POLYLINE": self._polyline_to_geo,
            "HATCH": self._hatch_to_geo,
            "TEXT": self._text_to_geo,
            "MTEXT": self._text_to_geo,
        }
        handler = handler_map.get(entity.dxftype())
        if handler:
            handler(geo, entity, scale, tx, ty, transform)
        else:
            logger.warning(
                f"Unsupported DXF entity type: {entity.dxftype()}. "
                "Skipping entity."
            )

    def _get_scale_to_mm(self, doc, default: float = 1.0) -> float:
        insunits = doc.header.get("$INSUNITS", 0)
        return units_to_mm.get(insunits, default) or default

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

    def _poly_to_geo(
        self,
        geo: Geometry,
        points: List[ezdxf.math.Vec3],
        is_closed: bool,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ) -> Optional[List[Tuple[float, float]]]:
        if not points:
            return None
        if transform:
            points = list(transform.transform_vertices(points))
        if not points:
            return None
        scaled_points = [
            ((p.x * scale) - tx, (p.y * scale) - ty) for p in points
        ]
        geo.move_to(scaled_points[0][0], scaled_points[0][1])
        for x, y in scaled_points[1:]:
            geo.line_to(x, y)
        if is_closed:
            geo.line_to(scaled_points[0][0], scaled_points[0][1])
        return scaled_points

    def _solid_to_geo_and_data(
        self,
        geo: Geometry,
        solids_list: List[List[Tuple[float, float]]],
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        # A SOLID is a quadrilateral. Note the strange vertex order for DXF.
        points = [
            entity.dxf.vtx0,
            entity.dxf.vtx1,
            entity.dxf.vtx3,
            entity.dxf.vtx2,
        ]
        # Add the outline to geometry and get the final scaled points for
        # the fill
        scaled_points = self._poly_to_geo(
            geo, points, True, scale, tx, ty, transform
        )
        if scaled_points:
            solids_list.append(scaled_points)

    def _line_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        points = [entity.dxf.start, entity.dxf.end]
        self._poly_to_geo(geo, points, False, scale, tx, ty, transform)

    def _lwpolyline_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        points = [ezdxf.math.Vec3(p[0], p[1], 0) for p in entity.vertices()]
        self._poly_to_geo(geo, points, entity.closed, scale, tx, ty, transform)

    def _arc_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        start_point, end_point, center_point = (
            entity.start_point,
            entity.end_point,
            entity.dxf.center,
        )
        if transform:
            start_point, end_point, center_point = (
                transform.transform(start_point),
                transform.transform(end_point),
                transform.transform(center_point),
            )
        center_offset = center_point - start_point
        final_start_x, final_start_y = (
            (start_point.x * scale) - tx,
            (start_point.y * scale) - ty,
        )
        final_end_x, final_end_y = (
            (end_point.x * scale) - tx,
            (end_point.y * scale) - ty,
        )
        final_offset_i, final_offset_j = (
            center_offset.x * scale,
            center_offset.y * scale,
        )
        geo.move_to(final_start_x, final_start_y, start_point.z * scale)
        geo.arc_to(
            final_end_x,
            final_end_y,
            final_offset_i,
            final_offset_j,
            clockwise=entity.dxf.extrusion.z < 0,
            z=end_point.z * scale,
        )

    def _poly_approx_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        try:
            path_obj = ezdxf.path.make_path(entity)  # type: ignore
            points = list(path_obj.flattening(distance=0.01))
            is_closed = getattr(entity, "closed", False)
            self._poly_to_geo(geo, points, is_closed, scale, tx, ty, transform)
        except Exception:
            pass

    def _polyline_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        try:
            for v_entity in entity.virtual_entities():
                if v_entity.dxftype() == "LINE":
                    self._line_to_geo(geo, v_entity, scale, tx, ty, transform)
                elif v_entity.dxftype() == "ARC":
                    self._arc_to_geo(geo, v_entity, scale, tx, ty, transform)
        except Exception:
            self._poly_to_geo(
                geo,
                list(entity.points()),
                entity.is_closed,
                scale,
                tx,
                ty,
                transform,
            )

    def _hatch_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        try:
            for path in entity.paths:
                for v_entity in path.virtual_entities():
                    if v_entity.dxftype() == "LINE":
                        self._line_to_geo(
                            geo, v_entity, scale, tx, ty, transform
                        )
                    elif v_entity.dxftype() == "ARC":
                        self._arc_to_geo(
                            geo, v_entity, scale, tx, ty, transform
                        )
                    elif v_entity.dxftype() in ("SPLINE", "ELLIPSE"):
                        self._poly_approx_to_geo(
                            geo, v_entity, scale, tx, ty, transform
                        )
        except Exception:
            pass

    def _text_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        try:
            for path in text2path.make_paths_from_entity(entity):
                points = list(path.flattening(distance=0.01))
                self._poly_to_geo(geo, points, False, scale, tx, ty, transform)
        except Exception:
            pass
