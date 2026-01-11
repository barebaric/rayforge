import io
import logging
import math
from copy import deepcopy
from typing import Iterable, Optional, List, Dict, Tuple
import ezdxf
import ezdxf.math
from ezdxf import bbox
from ezdxf.lldxf.const import DXFStructureError
from ezdxf.addons import text2path
from ezdxf.path import Command
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
            logger.error(
                "DXF Importer: Failed to parse DXF file due to "
                "structure error."
            )
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
            logger.warning("DXF Importer: No valid bounds found in the file.")
            return ImportPayload(source=source, items=[])

        _, _, width_mm, height_mm = bounds
        source.width_mm = width_mm
        source.height_mm = height_mm
        source.metadata["natural_size"] = (width_mm, height_mm)

        scale = self._get_scale_to_mm(doc)
        min_x_mm, min_y_mm, _, _ = bounds

        # Calculate adaptive tolerance based on diagonal size
        # Baseline 0.01mm for high precision, scaled up for large files.
        # e.g., 2m diagonal -> 0.1mm tolerance.
        diag_mm = math.hypot(width_mm, height_mm)
        tolerance_mm = max(0.01, diag_mm / 20000.0)
        logger.debug(
            f"DXF Importer: Using adaptive tolerance of {tolerance_mm:.4f} mm."
        )

        blocks_cache: Dict[str, List[DocItem]] = {}

        # Pre-parse all block definitions into DocItem templates.
        self._prepare_blocks_cache(
            doc,
            scale,
            min_x_mm,
            min_y_mm,
            source,
            blocks_cache,
            tolerance_mm,
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
            logger.debug(f"Processing layer: '{layer_name}'")
            items = self._entities_to_doc_items(
                entities,
                doc,
                scale,
                min_x_mm,
                min_y_mm,
                source,
                blocks_cache,
                parent_transform=None,
                tolerance_mm=tolerance_mm,
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
        tolerance_mm: float,
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
                tolerance_mm,
                ignore_solids=True,  # Suppress global solid harvest in blocks
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
        tolerance_mm: float = 0.01,
        ignore_solids: bool = False,
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

            # Only harvest solids if we are in the main modelspace context.
            # Block definitions should not pollute the global source metadata.
            if source and current_solids and not ignore_solids:
                existing_solids = source.metadata.get("solids", [])
                existing_solids.extend(current_solids)
                source.metadata["solids"] = existing_solids

            pristine_geo = current_geo.copy()
            pristine_geo.close_gaps()

            min_x, min_y, max_x, max_y = pristine_geo.rect()
            width = max(max_x - min_x, 1e-9)
            height = max(max_y - min_y, 1e-9)

            logger.debug(
                f"Flushing geometry to WorkPiece. Bounds: "
                f"w={width:.2f}, h={height:.2f}"
            )

            # Create a matrix that transforms the pristine geometry
            # (in mm, Y-up) into a normalized (0-1, Y-down) coordinate space.
            translate_to_origin = Matrix.translation(-min_x, -min_y)
            scale_to_unit = Matrix.scale(1.0 / width, 1.0 / height)
            flip_y = Matrix.translation(0, 1) @ Matrix.scale(1, -1)
            normalization_matrix = flip_y @ scale_to_unit @ translate_to_origin

            gen_config = SourceAssetSegment(
                source_asset_uid=source.uid,
                vectorization_spec=PassthroughSpec(),
                pristine_geometry=pristine_geo,
                normalization_matrix=normalization_matrix,
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
                    current_geo,
                    entity,
                    doc,
                    scale,
                    tx,
                    ty,
                    parent_transform,
                    tolerance_mm,
                )

        flush_geo_to_workpiece()
        return result_items

    def _entity_to_geo(
        self,
        geo,
        entity,
        doc,
        scale,
        tx,
        ty,
        transform,
        tolerance_mm: float = 0.01,
    ):
        """Dispatcher to call the correct handler for a given DXF entity."""
        handler_map = {
            "LINE": self._line_to_geo,
            "CIRCLE": self._circle_to_geo,
            "ARC": self._arc_to_geo,
            "LWPOLYLINE": self._poly_approx_to_geo,
            "ELLIPSE": self._poly_approx_to_geo,
            "SPLINE": self._poly_approx_to_geo,
            "POLYLINE": self._poly_approx_to_geo,
            "HATCH": self._hatch_to_geo,
            "TEXT": self._text_to_geo,
            "MTEXT": self._text_to_geo,
        }
        handler = handler_map.get(entity.dxftype())
        if handler:
            handler(geo, entity, scale, tx, ty, transform, tolerance_mm)
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
        # Use poly_approx to draw the outline
        self._poly_approx_to_geo(geo, entity, scale, tx, ty, transform)

        # For the solid fill data, we need the transformed 2D points
        points = [
            entity.dxf.vtx0,
            entity.dxf.vtx1,
            entity.dxf.vtx3,
            entity.dxf.vtx2,
        ]
        if transform:
            points = list(transform.transform_vertices(points))

        scaled_points = [
            ((p.x * scale) - tx, (p.y * scale) - ty) for p in points
        ]
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
        tolerance_mm: float = 0.01,
    ):
        """
        Converts a LINE entity directly to geometry commands without
        approximation.
        """
        points = [entity.dxf.start, entity.dxf.end]
        if transform:
            points = list(transform.transform_vertices(points))

        start_vec, end_vec = points

        # Apply global scale and translation
        start_x_mm = start_vec.x * scale - tx
        start_y_mm = start_vec.y * scale - ty
        start_z_mm = start_vec.z * scale

        end_x_mm = end_vec.x * scale - tx
        end_y_mm = end_vec.y * scale - ty
        end_z_mm = end_vec.z * scale

        # Check for continuity with the last point in the geometry
        is_continuous = False
        if not geo.is_empty():
            last_point = geo._get_last_point()
            # Use a small tolerance for floating point comparison
            dist_sq = (
                (last_point[0] - start_x_mm) ** 2
                + (last_point[1] - start_y_mm) ** 2
                + (last_point[2] - start_z_mm) ** 2
            )
            if dist_sq < 1e-6:
                is_continuous = True

        # If the path is not continuous, start a new subpath
        if not is_continuous:
            geo.move_to(start_x_mm, start_y_mm, start_z_mm)

        # Add the line segment
        geo.line_to(end_x_mm, end_y_mm, end_z_mm)

    def _circle_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
        tolerance_mm: float = 0.01,
    ):
        """Handles CIRCLE entities by creating two 180-degree arcs."""
        # Copy the entity to avoid modifying the original in a block def
        temp_entity = entity.copy()
        if transform:
            try:
                temp_entity.transform(transform)
            except (NotImplementedError, AttributeError):
                # Some entities might not support transformation directly
                self._poly_approx_to_geo(
                    geo, entity, scale, tx, ty, transform, tolerance_mm
                )
                return

        # If a non-uniform scale was applied, it becomes an ellipse
        if temp_entity.dxftype() == "ELLIPSE":
            self._poly_approx_to_geo(
                geo, temp_entity, scale, tx, ty, None, tolerance_mm
            )
            return

        center = temp_entity.dxf.center
        radius = temp_entity.dxf.radius

        # Apply global scale and translation
        cx_mm = center.x * scale - tx
        cy_mm = center.y * scale - ty
        z_mm = center.z * scale
        r_mm = radius * scale

        # Define circle as two 180-degree arcs. Start at 3 o'clock.
        start_point = (cx_mm + r_mm, cy_mm, z_mm)
        mid_point = (cx_mm - r_mm, cy_mm, z_mm)

        geo.move_to(start_point[0], start_point[1], start_point[2])
        # First semi-circle (CCW by default in DXF)
        geo.arc_to_as_bezier(
            mid_point[0], mid_point[1], -r_mm, 0, clockwise=False, z=z_mm
        )
        # Second semi-circle
        geo.arc_to_as_bezier(
            start_point[0], start_point[1], r_mm, 0, clockwise=False, z=z_mm
        )

    def _arc_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
        tolerance_mm: float = 0.01,
    ):
        self._poly_approx_to_geo(
            geo, entity, scale, tx, ty, transform, tolerance_mm
        )

    def _consume_path(
        self,
        geo: Geometry,
        path,
        scale: float,
        tx: float,
        ty: float,
    ):
        """
        Consumes an ezdxf.path.Path object and adds it to the Geometry.
        This is the core path construction logic.
        """
        if not path:
            return

        all_commands = list(path.commands())
        if not all_commands:
            return

        # Initialize current point from the last point in the geometry, if any
        last_geo_point = geo._get_last_point() if not geo.is_empty() else None

        # The 'start' of the path is the start of the first command.
        start_vec = path.start * scale
        current_x, current_y, current_z = (
            start_vec.x - tx,
            start_vec.y - ty,
            start_vec.z,
        )

        is_continuous = False
        if last_geo_point:
            dist_sq = (
                (last_geo_point[0] - current_x) ** 2
                + (last_geo_point[1] - current_y) ** 2
                + (last_geo_point[2] - current_z) ** 2
            )
            if dist_sq < 1e-6:
                is_continuous = True

        if not is_continuous:
            geo.move_to(current_x, current_y, current_z)

        for i, cmd in enumerate(all_commands):
            end_vec = cmd.end * scale
            end_x, end_y, end_z = end_vec.x - tx, end_vec.y - ty, end_vec.z

            if cmd.type == Command.MOVE_TO:
                # This command type should not appear after the first one
                # in a well-formed sub-path, but we handle it defensively
                # by moving the geo's cursor.
                logger.debug(
                    f"[Cmd {i}] Explicit MOVE_TO: ({end_x:.2f}, {end_y:.2f})"
                )
                geo.move_to(end_x, end_y, end_z)
            elif cmd.type == Command.LINE_TO:
                geo.line_to(end_x, end_y, end_z)
            elif cmd.type == Command.CURVE3_TO:
                ctrl = cmd.ctrl * scale
                ctrl_x, ctrl_y = ctrl.x - tx, ctrl.y - ty
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
                ctrl1 = cmd.ctrl1 * scale
                c1x, c1y = ctrl1.x - tx, ctrl1.y - ty
                ctrl2 = cmd.ctrl2 * scale
                c2x, c2y = ctrl2.x - tx, ctrl2.y - ty
                geo.bezier_to(end_x, end_y, c1x, c1y, c2x, c2y, end_z)

            # Update the current point for the next command in the loop
            current_x, current_y, current_z = end_x, end_y, end_z

    def _poly_approx_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
        tolerance_mm: float = 0.01,
    ):
        """
        Converts entities to Geometry using ezdxf's path interface.
        """
        try:
            # Use `flattening` to control the linearization of curves.
            # A small value ensures curves are converted to many small lines,
            # which our `arc_to_as_bezier` can reconstruct. A value of 0
            # might use the default, so a small explicit value is better.
            path_obj = ezdxf.path.make_path(  # type: ignore
                entity, flattening=tolerance_mm / 4.0
            )
            if transform:
                path_obj = path_obj.transform(transform)

            self._consume_path(geo, path_obj, scale, tx, ty)
        except ezdxf.path.EmptyPathError:  # type: ignore
            logger.debug(
                f"Skipping empty path from entity {entity.dxftype()}."
            )
        except Exception as e:
            logger.error(
                f"Failed to convert entity {entity.dxftype()} to path: {e}",
                exc_info=True,
            )

    def _hatch_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
        tolerance_mm: float = 0.01,
    ):
        try:
            # Hatches are complex; we convert each of their boundary paths.
            for path in entity.paths:
                # The path from a hatch is already a path object, so we
                # consume it.
                path_obj = path.to_path()
                if transform:
                    path_obj = path_obj.transform(transform)
                self._consume_path(geo, path_obj, scale, tx, ty)
        except Exception as e:
            logger.error(f"Failed to process HATCH entity: {e}", exc_info=True)

    def _text_to_geo(
        self,
        geo: Geometry,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
        tolerance_mm: float = 0.01,
    ):
        try:
            paths = text2path.make_paths_from_entity(entity)
            for path in paths:
                if transform:
                    path = path.transform(transform)
                self._consume_path(geo, path, scale, tx, ty)
        except Exception as e:
            logger.error(f"Failed to convert TEXT entity: {e}", exc_info=True)
