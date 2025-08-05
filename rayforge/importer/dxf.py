import io
import math
import ezdxf
from ezdxf import bbox
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips
import xml.etree.ElementTree as ET
from typing import Generator, Optional, Tuple
import cairo
from .svg import SVGRenderer
from .renderer import Renderer

# Conversion factors from DXF drawing units to millimeters.
# See DXF documentation for header variable $INSUNITS.
units_to_mm = {
    0: 1.0,      # Unitless
    1: 25.4,     # Inches
    2: 304.8,    # Feet
    4: 1.0,      # Millimeters
    5: 10.0,     # Centimeters
    6: 1000.0,   # Meters
    8: 0.0254,   # Mils
    9: 0.0254,   # Microinches
    10: 914.4,   # Yards
}


class DXFRenderer(Renderer):
    label = "DXF files (2D)"
    mime_types = ("image/vnd.dxf",)
    extensions = (".dxf",)

    def __init__(self, data: bytes):
        """
        Initializes the renderer by performing an immediate, synchronous
        conversion of the DXF data to an in-memory SVG representation.
        All subsequent rendering operations are delegated to an internal
        SVGRenderer instance.
        """
        svg_data = self._convert_dxf_to_svg(data)
        self._svg_renderer = SVGRenderer(svg_data)

    def get_natural_size(
        self, px_factor: float = 0.0
    ) -> Tuple[Optional[float], Optional[float]]:
        return self._svg_renderer.get_natural_size(px_factor)

    def get_aspect_ratio(self) -> float:
        return self._svg_renderer.get_aspect_ratio()

    def render_to_pixels(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        return self._svg_renderer.render_to_pixels(width, height)

    def _render_to_vips_image(
        self, width: int, height: int
    ) -> Optional[pyvips.Image]:
        return self._svg_renderer._render_to_vips_image(width, height)

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
        return self._svg_renderer.render_chunk(
            width_px,
            height_px,
            max_chunk_width,
            max_chunk_height,
            max_memory_size,
            overlap_x,
            overlap_y,
        )

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
        if isinstance(dxf_data, bytes):
            try:
                # Standard-compliant decoding
                data_str = dxf_data.decode("utf-8")
            except UnicodeDecodeError:
                # Fallback for older DXF files with non-standard encodings
                data_str = dxf_data.decode("ascii", errors="replace")
            # Normalize line endings
            data_str = data_str.replace("\r\n", "\n")
        else:
            raise TypeError("Input must be bytes")

        try:
            doc = ezdxf.read(io.StringIO(data_str))
        except ezdxf.DXFStructureError as e:
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
        elif dxftype == "TEXT":
            self._add_text(parent, entity, scale)
        elif dxftype == "ELLIPSE":
            self._add_ellipse(parent, entity, scale)
        elif dxftype == "SPLINE":
            self._add_spline(parent, entity, scale)
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
        elem = ET.SubElement(parent, "text")
        insert_x = entity.dxf.insert.x * scale
        insert_y = entity.dxf.insert.y * scale
        elem.set("x", str(insert_x))
        elem.set("y", str(insert_y))
        elem.set(
            "transform",
            f"rotate({-entity.dxf.rotation} {insert_x} {insert_y})",
        )
        elem.set("font-size", f"{entity.dxf.height * scale}mm")
        elem.set("fill", "black")
        elem.text = entity.dxf.text

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
        B-spline curve.
        """
        # Use ezdxf's built-in `approximate` method
        # to generate a visually correct polyline from the curve.
        try:
            # Use ezdxf's built-in tool to get an approximated polyline.
            # 20 segments per span should provide good quality.
            points = list(entity.approximate(segments=20))
        except Exception:
            # Fallback for splines that can't be approximated or if the
            # ezdxf version is too old. This mimics the old, less accurate
            # behavior of just connecting the points.
            if entity.dxf.n_fit_points > 0:
                points = entity.fit_points
            else:
                points = entity.control_points

        if not points:
            return

        scaled_points = [(p[0] * scale, p[1] * scale) for p in points]
        if not scaled_points:
            return

        d = "M " + " L ".join(f"{x},{y}" for x, y in scaled_points)
        if entity.is_closed:
            d += " Z"
        elem = ET.SubElement(parent, "path")
        elem.set("d", d)
        elem.set("stroke", "black")
        elem.set("stroke-width", "0.1mm")
        elem.set("fill", "none")

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
