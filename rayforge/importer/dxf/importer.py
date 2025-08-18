import io
import json
import logging
from copy import deepcopy
from typing import Iterable, Optional, List, Dict, Tuple
import ezdxf
import ezdxf.math
from ezdxf import bbox
from ezdxf.lldxf.const import DXFStructureError
from ezdxf.addons import text2path

from ...core.ops import Ops
from ...core.group import Group
from ...core.workpiece import WorkPiece
from ...core.matrix import Matrix
from ...core.item import DocItem
from ..base_importer import Importer
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

    def __init__(self, data: bytes, source_file=None):
        super().__init__(data, source_file)
        self._blocks_cache: Dict[str, List[DocItem]] = {}

    def get_doc_items(self) -> Optional[List[DocItem]]:
        try:
            data_str = self.raw_data.decode("utf-8", errors="replace")
            normalized_str = data_str.replace("\r\n", "\n")
            doc = ezdxf.read(io.StringIO(normalized_str))  # type: ignore
        except DXFStructureError:
            return []

        bounds = self._get_bounds_mm(doc)
        if not bounds or not bounds[2] or not bounds[3]:
            return []

        scale = self._get_scale_to_mm(doc)
        min_x_mm, min_y_mm, _, _ = bounds

        # Pre-parse all block definitions into DocItem templates.
        self._prepare_blocks_cache(doc, scale, min_x_mm, min_y_mm)
        # Parse the main modelspace into a list of DocItems.
        return self._entities_to_doc_items(
            doc.modelspace(), doc, scale, min_x_mm, min_y_mm
        )

    def _prepare_blocks_cache(self, doc, scale: float, tx: float, ty: float):
        """Recursively parses all block definitions into lists of DocItems."""
        self._blocks_cache.clear()
        for block in doc.blocks:
            # Note: We pass a base transform to handle nested blocks correctly.
            self._blocks_cache[block.name] = self._entities_to_doc_items(
                block, doc, scale, tx, ty, ezdxf.math.Matrix44()
            )

    def _entities_to_doc_items(
        self,
        entities: Iterable,
        doc,
        scale: float,
        tx: float,
        ty: float,
        parent_transform: Optional[ezdxf.math.Matrix44] = None,
    ) -> List[DocItem]:
        """
        Converts a list of DXF entities into a list of DocItems (WorkPieces
        and Groups).
        """
        result_items: List[DocItem] = []
        current_ops = Ops()
        current_solids: List[List[Tuple[float, float]]] = []

        def flush_ops_to_workpiece():
            """
            Converts the accumulated Ops and solid data into a single,
            normalized WorkPiece.
            """
            nonlocal current_ops, current_solids
            if current_ops.is_empty():
                return

            min_x, min_y, max_x, max_y = current_ops.rect()

            # Normalize ops and solids to have their origin at (0,0)
            normalized_ops = current_ops.copy()
            normalized_ops.translate(-min_x, -min_y)

            normalized_solids = [
                [(p[0] - min_x, p[1] - min_y) for p in solid]
                for solid in current_solids
            ]

            # Serialize solid data for the renderer
            wp_data = None
            if normalized_solids:
                wp_data = json.dumps({"solids": normalized_solids}).encode(
                    "utf-8"
                )

            wp = WorkPiece(
                source_file=self.source_file,
                renderer=DXF_RENDERER,
                source_ops=normalized_ops,
                data=wp_data,  # Store the renderer-specific data
            )

            # Set the workpiece's matrix to position and scale it correctly.
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)
            wp.matrix = Matrix.translation(min_x, min_y) @ Matrix.scale(
                width, height
            )

            result_items.append(wp)
            current_ops = Ops()
            current_solids = []

        for entity in entities:
            if entity.dxftype() == "INSERT":
                # An INSERT entity ends the current Ops run and creates a
                # Group.
                flush_ops_to_workpiece()
                block_items = self._blocks_cache.get(entity.dxf.name)
                if not block_items:
                    continue

                group = Group(name=entity.dxf.name)
                group.set_children(deepcopy(block_items))

                # Calculate the transformation for this block instance.
                m = entity.matrix44()
                if parent_transform:
                    m = parent_transform @ m
                ux, uy, uz, pos = m.get_components()

                # Convert the 4x4 ezdxf matrix to our 3x3 document matrix.
                # This matrix is in original DXF units and coordinates.
                instance_matrix = Matrix(
                    [[ux.x, uy.x, pos.x], [ux.y, uy.y, pos.y], [0, 0, 1]]
                )

                # Apply the global scale and translation to bring it into
                # the final document coordinate system.
                global_transform = Matrix.translation(-tx, -ty) @ Matrix.scale(
                    scale, scale
                )
                group.matrix = global_transform @ instance_matrix

                result_items.append(group)

            elif entity.dxftype() == "SOLID":
                # A SOLID contributes to both Ops (outline) and data (fill)
                self._solid_to_ops_and_data(
                    current_ops,
                    current_solids,
                    entity,
                    scale,
                    tx,
                    ty,
                    parent_transform,
                )
            else:
                # Append vector data from other entities to the current Ops.
                self._entity_to_ops(
                    current_ops, entity, doc, scale, tx, ty, parent_transform
                )

        flush_ops_to_workpiece()  # Flush any remaining Ops.
        return result_items

    def _entity_to_ops(self, ops, entity, doc, scale, tx, ty, transform):
        """Dispatcher to call the correct handler for a given DXF entity."""
        handler_map = {
            "LINE": self._line_to_ops,
            "CIRCLE": self._poly_approx_to_ops,
            "LWPOLYLINE": self._lwpolyline_to_ops,
            "ARC": self._arc_to_ops,
            "ELLIPSE": self._poly_approx_to_ops,
            "SPLINE": self._poly_approx_to_ops,
            "POLYLINE": self._polyline_to_ops,
            "HATCH": self._hatch_to_ops,
            "TEXT": self._text_to_ops,
            "MTEXT": self._text_to_ops,
        }
        handler = handler_map.get(entity.dxftype())
        if handler:
            handler(ops, entity, scale, tx, ty, transform)
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

    def _poly_to_ops(
        self,
        ops: Ops,
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
        ops.move_to(scaled_points[0][0], scaled_points[0][1])
        for x, y in scaled_points[1:]:
            ops.line_to(x, y)
        if is_closed:
            ops.line_to(scaled_points[0][0], scaled_points[0][1])
        return scaled_points

    def _solid_to_ops_and_data(
        self,
        ops: Ops,
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
        # Add the outline to ops and get the final scaled points for the fill
        scaled_points = self._poly_to_ops(
            ops, points, True, scale, tx, ty, transform
        )
        if scaled_points:
            solids_list.append(scaled_points)

    def _line_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        points = [entity.dxf.start, entity.dxf.end]
        self._poly_to_ops(ops, points, False, scale, tx, ty, transform)

    def _lwpolyline_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        points = [ezdxf.math.Vec3(p[0], p[1], 0) for p in entity.vertices()]
        self._poly_to_ops(ops, points, entity.closed, scale, tx, ty, transform)

    def _arc_to_ops(
        self,
        ops: Ops,
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
        ops.move_to(final_start_x, final_start_y, start_point.z * scale)
        ops.arc_to(
            final_end_x,
            final_end_y,
            final_offset_i,
            final_offset_j,
            clockwise=entity.dxf.extrusion.z < 0,
            z=end_point.z * scale,
        )

    def _poly_approx_to_ops(
        self,
        ops: Ops,
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
            self._poly_to_ops(ops, points, is_closed, scale, tx, ty, transform)
        except Exception:
            pass

    def _polyline_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        try:
            for v_entity in entity.virtual_entities():
                if v_entity.dxftype() == "LINE":
                    self._line_to_ops(ops, v_entity, scale, tx, ty, transform)
                elif v_entity.dxftype() == "ARC":
                    self._arc_to_ops(ops, v_entity, scale, tx, ty, transform)
        except Exception:
            self._poly_to_ops(
                ops,
                list(entity.points()),
                entity.is_closed,
                scale,
                tx,
                ty,
                transform,
            )

    def _hatch_to_ops(
        self,
        ops: Ops,
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
                        self._line_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
                    elif v_entity.dxftype() == "ARC":
                        self._arc_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
                    elif v_entity.dxftype() in ("SPLINE", "ELLIPSE"):
                        self._poly_approx_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
        except Exception:
            pass

    def _text_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        try:
            for path in text2path.make_paths_from_entity(entity):
                points = list(path.flattening(distance=0.01))
                self._poly_to_ops(ops, points, False, scale, tx, ty, transform)
        except Exception:
            pass
