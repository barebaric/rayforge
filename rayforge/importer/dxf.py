import io
import math
import logging
import ezdxf
from ezdxf import bbox
from ezdxf import DXFStructureError  # type: ignore[reportPrivateImportUsage]
import ezdxf.math
from ezdxf.addons import text2path
import warnings

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
import xml.etree.ElementTree as ET
from typing import Generator, Optional, Tuple, Iterable, List

import cairo
from .svg import SvgImporter
from .base import Importer
from ..core.ops import Ops

logger = logging.getLogger(__name__)

# Conversion factors from DXF drawing units to millimeters.
# See DXF documentation for header variable $INSUNITS.
units_to_mm = {
    0: 1.0,  # Unitless
    1: 25.4,  # Inches
    2: 304.8,  # Feet
    4: 1.0,  # Millimeters
    5: 10.0,  # Centimeters
    6: 1000.0,  # Meters
    8: 0.0254,  # Mils
    9: 0.0254,  # Microinches
    10: 914.4,  # Yards
}


class DxfImporter(Importer):
    label = "DXF files (2D)"
    mime_types = ("image/vnd.dxf",)
    extensions = (".dxf",)

    def __init__(self, data: bytes):
        """
        Initializes the importer by performing an immediate, synchronous
        conversion of the DXF data to an in-memory SVG representation.
        All subsequent rendering operations are delegated to an internal
        SVGImporter instance.
        """
        self.raw_data = data
        self._ops_cache: Optional[Ops] = None
        svg_data = self._convert_dxf_to_svg(data)
        self._svg_importer = SvgImporter(svg_data)

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        return self._svg_importer.get_natural_size(px_factor)

    def get_aspect_ratio(self) -> float:
        return self._svg_importer.get_aspect_ratio()

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        return self._svg_importer.render_to_pixels(width, height)

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        return self._svg_importer._render_to_vips_image(width, height)

    def render_chunk(
        self,
        width_px: int,
        height_px: int,
        max_chunk_width: Optional[int] = None,
        max_chunk_height: Optional[int] = None,
        max_memory_size: Optional[int] = None,
        overlap_x: int = 1,
        overlap_y: int = 0,
    ) -> Generator[Tuple[cairo.ImageSurface, Tuple[float, float]], None, None]:
        return self._svg_importer.render_chunk(
            width_px,
            height_px,
            max_chunk_width,
            max_chunk_height,
            max_memory_size,
            overlap_x,
            overlap_y,
        )

    def get_vector_ops(self) -> "Optional[Ops]":
        """
        Parses the DXF data directly and converts its geometric entities
        into a sequence of vector operations (Ops).

        This provides a high-fidelity, un-rendered representation of the
        source vectors, suitable for direct use in cutting or engraving jobs.
        The coordinate system is canonical (Y-up), with the origin at the
        bottom-left of the drawing's bounding box.
        """
        if self._ops_cache is not None:
            return self._ops_cache

        try:
            data_str = self.raw_data.decode("utf-8")
        except UnicodeDecodeError:
            data_str = self.raw_data.decode("ascii", errors="replace")
        data_str = data_str.replace("\r\n", "\n")

        try:
            doc = ezdxf.read(io.StringIO(data_str))  # type: ignore
        except DXFStructureError:
            self._ops_cache = Ops()
            return self._ops_cache

        bounds = self._get_bounds_mm(doc)
        if not bounds or not bounds[2] or not bounds[3]:
            self._ops_cache = Ops()
            return self._ops_cache

        ops = Ops()
        scale = self._get_scale_to_mm(doc)
        min_x_mm, min_y_mm, _, _ = bounds

        self._entities_to_ops(
            ops, doc.modelspace(), doc, scale, min_x_mm, min_y_mm
        )

        self._ops_cache = ops
        return self._ops_cache

    def _entities_to_ops(
        self,
        ops: Ops,
        entities: Iterable["ezdxf.entities.DXFEntity"],  # type: ignore
        doc: "ezdxf.document.Drawing",  # type: ignore
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        """Recursively processes entities and converts them to Ops."""
        for entity in entities:
            dxftype = entity.dxftype()
            if dxftype == "LINE":
                self._line_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "CIRCLE":
                self._poly_approx_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "LWPOLYLINE":
                self._lwpolyline_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "ARC":
                self._arc_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "ELLIPSE":
                self._poly_approx_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "SPLINE":
                self._poly_approx_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "POLYLINE":
                self._polyline_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "HATCH":
                self._hatch_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype in ("TEXT", "MTEXT"):
                self._text_to_ops(ops, entity, scale, tx, ty, transform)
            elif dxftype == "INSERT":
                self._insert_to_ops(ops, entity, doc, scale, tx, ty, transform)
            else:
                logger.warning(
                    f"DXF with unsupported entity {dxftype}, skipping"
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
    ):
        """Converts a list of points into move_to/line_to operations."""
        if not points:
            return

        if transform:
            points = list(transform.transform_vertices(points))

        if not points:
            return

        # Translate points to place the drawing's bottom-left at (0,0)
        scaled_points = [
            ((p.x * scale) - tx, (p.y * scale) - ty) for p in points
        ]

        ops.move_to(scaled_points[0][0], scaled_points[0][1])
        for x, y in scaled_points[1:]:
            ops.line_to(x, y)

        if is_closed:
            ops.line_to(scaled_points[0][0], scaled_points[0][1])

    def _line_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform=None,
    ):
        logger.debug(f"Processing LINE entity ({entity.dxf.handle})")
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
        logger.debug(
            f"Processing LWPOLYLINE entity ({entity.dxf.handle}) with "
            f"{len(list(entity.vertices()))} vertices"
        )
        # Convert 2D vertices to Vec3 for consistent transformation
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
        """Converts a DXF ARC entity to a move_to and an arc_to command."""
        logger.debug(f"Processing ARC entity ({entity.dxf.handle})")
        start_point = entity.start_point
        end_point = entity.end_point
        center_point = entity.dxf.center

        if transform:
            start_point = transform.transform(start_point)
            end_point = transform.transform(end_point)
            center_point = transform.transform(center_point)

        # The ops.arc_to command requires the center offset relative to
        # the arc's start point.
        center_offset = center_point - start_point

        # Scale and translate all coordinates to the final output space
        final_start_x = (start_point.x * scale) - tx
        final_start_y = (start_point.y * scale) - ty
        final_end_x = (end_point.x * scale) - tx
        final_end_y = (end_point.y * scale) - ty
        # Offsets are scaled, but not translated
        final_offset_i = center_offset.x * scale
        final_offset_j = center_offset.y * scale

        # Move to the start of the arc
        ops.move_to(final_start_x, final_start_y, start_point.z * scale)

        # DXF arcs are always counter-clockwise. Cairo's `arc()` is also CCW,
        # which corresponds to `clockwise=False` in our Ops/CairoEncoder.
        ops.arc_to(
            final_end_x,
            final_end_y,
            final_offset_i,
            final_offset_j,
            clockwise=False,
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
        """Approximates an entity (Circle, Ellipse, Spline) with a polyline."""
        logger.debug(
            f"Approximating {entity.dxftype()} entity ({entity.dxf.handle})"
            " with a polyline"
        )
        try:
            path_obj = ezdxf.path.make_path(entity)  # type: ignore
            # Use a flattening distance of 0.01 drawing units for high quality
            points = list(path_obj.flattening(distance=0.01))
            logger.debug(f"  ...approximated to {len(points)} points.")

            is_closed = getattr(entity, "closed", False)
            self._poly_to_ops(ops, points, is_closed, scale, tx, ty, transform)
        except Exception as e:
            logger.warning(
                f"Could not approximate {entity.dxftype()} "
                f" ({entity.dxf.handle}). Error: {e}"
            )

    def _polyline_to_ops(
        self,
        ops: Ops,
        entity,
        scale: float,
        tx: float,
        ty: float,
        transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        """
        Decomposes a complex POLYLINE entity into its constituent virtual
        entities (Lines and Arcs) and processes them.
        """
        logger.debug(f"Decomposing POLYLINE entity ({entity.dxf.handle})")
        try:
            for v_entity in entity.virtual_entities():
                # Dispatch to the appropriate handler, passing the transform
                # context
                dxftype = v_entity.dxftype()
                if dxftype == "LINE":
                    self._line_to_ops(ops, v_entity, scale, tx, ty, transform)
                elif dxftype == "ARC":
                    self._arc_to_ops(ops, v_entity, scale, tx, ty, transform)
        except Exception as e:
            # Fallback to simple point connection for problematic polylines
            logger.warning(
                f"Could not decompose POLYLINE ({entity.dxf.handle}), falling"
                f" back to points. Error: {e}"
            )
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
        """
        Decomposes a HATCH entity into its boundary path entities
        and processes them.
        """
        logger.debug(f"Decomposing HATCH entity ({entity.dxf.handle})")
        try:
            # Hatches can be complex. We iterate through each boundary path.
            for path in entity.paths:
                # And for each path, we get its constituent virtual entities.
                for v_entity in path.virtual_entities():
                    # Dispatch to the appropriate handler, passing the
                    # transform context
                    dxftype = v_entity.dxftype()
                    if dxftype == "LINE":
                        self._line_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
                    elif dxftype == "ARC":
                        self._arc_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
                    # Hatch boundaries can also be splines or ellipses
                    elif dxftype in ("SPLINE", "ELLIPSE"):
                        self._poly_approx_to_ops(
                            ops, v_entity, scale, tx, ty, transform
                        )
        except Exception as e:
            # It's possible for virtual_entities to fail on corrupt data.
            logger.warning(
                f"Could not decompose HATCH ({entity.dxf.handle}). Error: {e}"
            )
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
        """Decomposes a TEXT or MTEXT entity into vector paths."""
        logger.debug(
            f"Processing {entity.dxftype()} entity ({entity.dxf.handle})"
        )
        try:
            # Use ezdxf's addon to convert text to paths.
            # This uses pre-packaged Hershey fonts, which is robust.
            text_paths = text2path.make_paths_from_entity(entity)
            for path in text_paths:
                # Approximate the path with a fine-grained polyline
                # A flattening distance of 0.01mm is very high quality
                points = list(path.flattening(distance=0.01))
                # The path from text2path is a simple polyline, not closed.
                self._poly_to_ops(ops, points, False, scale, tx, ty, transform)
        except Exception as e:
            # This can fail for empty or malformed text entities
            logger.warning(
                f"Could not process {entity.dxftype()} entity"
                f" ({entity.dxf.handle}). Error: {e}"
            )

    def _insert_to_ops(
        self,
        ops: Ops,
        entity,
        doc: "ezdxf.document.Drawing",  # type: ignore
        scale: float,
        tx: float,
        ty: float,
        parent_transform: Optional[ezdxf.math.Matrix44] = None,
    ):
        """Handles block references (INSERT entities) by recursion."""
        logger.debug(
            f"Processing INSERT entity ({entity.dxf.handle}), "
            f"block: '{entity.dxf.name}'"
        )
        block = doc.blocks.get(entity.dxf.name)
        if not block:
            return

        # Get the transformation matrix for this specific block insert
        insert_matrix = entity.matrix44()

        # Chain it with the parent's transform if we are inside a nested block
        if parent_transform:
            transform_matrix = parent_transform @ insert_matrix
        else:
            transform_matrix = insert_matrix

        # Recursively process entities within the block definition,
        # passing down the combined transformation matrix.
        self._entities_to_ops(ops, block, doc, scale, tx, ty, transform_matrix)

    def _get_scale_to_mm(self, doc, default: float = 1.0):
        insunits = doc.header.get("$INSUNITS", 0)
        if insunits not in units_to_mm:
            return default
        return units_to_mm.get(insunits, default) or default

    def _get_bounds_px(self, doc):
        """Calculates the bounding box of the modelspace in drawing units."""
        msp = doc.modelspace()
        entity_bbox = bbox.extents(msp, fast=True)
        if not entity_bbox.has_data:
            return None

        min_x, min_y, _ = entity_bbox.extmin
        max_x, max_y, _ = entity_bbox.extmax
        return min_x, min_y, (max_x - min_x), (max_y - min_y)

    def _get_bounds_mm(self, doc):
        """Calculates the bounding box and converts it to millimeters."""
        bounds = self._get_bounds_px(doc)
        if bounds is None:
            return None
        min_x, min_y, width, height = bounds

        scale = self._get_scale_to_mm(doc)
        return min_x * scale, min_y * scale, width * scale, height * scale

    def _convert_dxf_to_svg(self, dxf_data: bytes) -> bytes:
        """
        Parses DXF data and converts its geometric entities into an
        SVG byte string.
        """
        if not isinstance(dxf_data, bytes):
            raise TypeError("Input must be bytes")

        try:
            # Standard-compliant decoding
            data_str = dxf_data.decode("utf-8")
        except UnicodeDecodeError:
            # Fallback for older DXF files with non-standard encodings
            data_str = dxf_data.decode("ascii", errors="replace")
        # Normalize line endings
        data_str = data_str.replace("\r\n", "\n")

        try:
            doc = ezdxf.read(  # pyright: ignore[reportPrivateImportUsage]
                io.StringIO(data_str)
            )
        except DXFStructureError as e:
            raise ValueError(f"Invalid DXF data: {e}")

        bounds = self._get_bounds_mm(doc)
        if not bounds or not bounds[2] or not bounds[3]:
            # Return an empty SVG if the DXF is empty or has no size
            return b'<svg xmlns="http://www.w3.org/2000/svg"/>'

        min_x_mm, min_y_mm, width_mm, height_mm = bounds
        scale_to_mm = self._get_scale_to_mm(doc)

        # Create the root SVG element
        svg = ET.Element("svg", xmlns="http://www.w3.org/2000/svg")
        svg.set("viewBox", f"0 0 {width_mm} {height_mm}")
        svg.set("width", f"{width_mm}mm")
        svg.set("height", f"{height_mm}mm")

        # Create a group to handle the coordinate system transform
        # (DXF origin is often arbitrary, SVG is top-left)
        group = ET.SubElement(svg, "g")
        # This transform flips the Y-axis and shifts the origin
        transform = f"matrix(1 0 0 -1 {-min_x_mm} {min_y_mm + height_mm})"
        group.set("transform", transform)

        msp = doc.modelspace()
        for entity in msp:
            self._process_entity(group, entity, doc, scale=scale_to_mm)

        return ET.tostring(svg, encoding="utf-8")

    def _process_entity(self, parent, entity, doc, scale):
        """Processes a single DXF entity and converts it to an SVG element."""
        dxftype = entity.dxftype()
        if dxftype == "LINE":
            self._add_line(parent, entity, scale)
        elif dxftype == "CIRCLE":
            self._add_circle(parent, entity, scale)
        elif dxftype == "LWPOLYLINE":
            self._add_lwpolyline(parent, entity, scale)
        elif dxftype == "ARC":
            self._add_arc(parent, entity, scale)
        elif dxftype in ("TEXT", "MTEXT"):
            self._add_text(parent, entity, scale)
        elif dxftype == "ELLIPSE":
            self._add_ellipse(parent, entity, scale)
        elif dxftype == "SPLINE":
            self._add_spline(parent, entity, scale)
        elif dxftype == "POLYLINE":
            self._add_polyline(parent, entity, doc, scale)
        elif dxftype == "HATCH":
            self._add_hatch(parent, entity, doc, scale)
        elif dxftype == "INSERT":
            self._add_insert(parent, entity, doc, scale)

    def _add_line(self, parent, entity, scale):
        elem = ET.SubElement(parent, "line")
        elem.set("x1", str(entity.dxf.start.x * scale))
        elem.set("y1", str(entity.dxf.start.y * scale))
        elem.set("x2", str(entity.dxf.end.x * scale))
        elem.set("y2", str(entity.dxf.end.y * scale))
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")

    def _add_circle(self, parent, entity, scale):
        elem = ET.SubElement(parent, "circle")
        elem.set("cx", str(entity.dxf.center.x * scale))
        elem.set("cy", str(entity.dxf.center.y * scale))
        elem.set("r", str(entity.dxf.radius * scale))
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

    def _add_lwpolyline(self, parent, entity, scale):
        points = list(entity.vertices())
        if not points:
            return
        scaled_points = [(p[0] * scale, p[1] * scale) for p in points]
        d = f"M {scaled_points[0][0]},{scaled_points[0][1]}"
        for point in scaled_points[1:]:
            d += f" L {point[0]},{point[1]}"
        if entity.closed:
            d += " Z"
        elem = ET.SubElement(parent, "path")
        elem.set("d", d)
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

    def _add_arc(self, parent, entity, scale):
        center_x = entity.dxf.center.x * scale
        center_y = entity.dxf.center.y * scale
        radius = entity.dxf.radius * scale
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)

        start_x = center_x + radius * math.cos(start_angle)
        start_y = center_y + radius * math.sin(start_angle)
        end_x = center_x + radius * math.cos(end_angle)
        end_y = center_y + radius * math.sin(end_angle)

        # Determine arc flags for SVG path
        angular_dist = (end_angle - start_angle) % (2 * math.pi)
        large_arc = "1" if angular_dist > math.pi else "0"
        sweep_flag = "1"  # DXF arcs are counter-clockwise

        d = (
            f"M {start_x} {start_y} "
            f"A {radius} {radius} 0 {large_arc} {sweep_flag} {end_x} {end_y}"
        )

        elem = ET.SubElement(parent, "path")
        elem.set("d", d)
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

    def _add_text(self, parent, entity, scale):
        """
        Converts a TEXT or MTEXT entity to an SVG <path> element by
        decomposing its characters into vector outlines.
        """
        try:
            text_paths = text2path.make_paths_from_entity(entity)
            for path_obj in text_paths:
                # Use flattening to convert beziers to polylines
                points = list(path_obj.flattening(distance=0.01))
                if not points:
                    continue
                scaled_points = [(p.x * scale, p.y * scale) for p in points]
                d_str = "M " + " L ".join(f"{x},{y}" for x, y in scaled_points)

                elem = ET.SubElement(parent, "path")
                elem.set("d", d_str)
                elem.set("stroke", "black")
                elem.set("stroke-width", "0.1mm")
                elem.set("fill", "none")
        except Exception as e:
            logger.warning(
                f"Could not render {entity.dxftype()} to SVG. Error: {e}"
            )

    def _add_ellipse(self, parent, entity, scale):
        center_x = entity.dxf.center.x * scale
        center_y = entity.dxf.center.y * scale
        major_x = entity.dxf.major_axis.x * scale
        major_y = entity.dxf.major_axis.y * scale

        rx = math.hypot(major_x, major_y)
        ry = rx * entity.dxf.ratio
        angle = math.degrees(math.atan2(major_y, major_x))

        elem = ET.SubElement(parent, "ellipse")
        elem.set("cx", str(center_x))
        elem.set("cy", str(center_y))
        elem.set("rx", str(rx))
        elem.set("ry", str(ry))
        elem.set("transform", f"rotate({angle} {center_x} {center_y})")
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

    def _add_spline(self, parent, entity, scale):
        """
        Converts a SPLINE entity to an SVG path by approximating the
        B-spline curve using the robust flattening method.
        """
        try:
            # Use ezdxf's built-in tool to get an approximated polyline.
            # 20 segments per span should provide good quality.
            path_obj = ezdxf.path.make_path(entity)  # type: ignore
            points = list(path_obj.flattening(distance=0.01))
        except Exception:
            logger.exception("Could not convert SPLINE to SVG path")
            return

        if not points:
            return

        scaled_points = [(p.x * scale, p.y * scale) for p in points]
        d_str = "M " + " L ".join(f"{x},{y}" for x, y in scaled_points)
        if getattr(entity, "closed", False):
            d_str += " Z"

        elem = ET.SubElement(parent, "path")
        elem.set("d", d_str)
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

    def _add_polyline(self, parent, entity, doc, scale):
        """Decomposes a complex POLYLINE entity for SVG rendering."""
        try:
            for v_entity in entity.virtual_entities():
                self._process_entity(parent, v_entity, doc, scale)
        except Exception as e:
            logger.warning(f"Could not decompose POLYLINE for SVG. Error: {e}")

    def _add_hatch(self, parent, entity, doc, scale):
        """Decomposes a HATCH's boundaries for SVG rendering."""
        try:
            for path_boundary in entity.paths:
                for v_entity in path_boundary.virtual_entities():
                    self._process_entity(parent, v_entity, doc, scale)
        except Exception as e:
            logger.warning(f"Could not decompose HATCH for SVG. Error: {e}")

    def _add_insert(self, parent, entity, doc, scale):
        """Handles block references (INSERT entities)."""
        block = doc.blocks.get(entity.dxf.name)
        if not block:
            return

        insert_x = entity.dxf.insert.x * scale
        insert_y = entity.dxf.insert.y * scale
        scale_x = entity.dxf.xscale
        scale_y = entity.dxf.yscale
        rotation = -entity.dxf.rotation

        # Create a group for the block with its own transform
        g = ET.SubElement(parent, "g")
        transform = (
            f"translate({insert_x} {insert_y}) "
            f"rotate({rotation}) "
            f"scale({scale_x} {scale_y})"
        )
        g.set("transform", transform)

        # Recursively process entities within the block
        for e in block:
            self._process_entity(g, e, doc, scale)
