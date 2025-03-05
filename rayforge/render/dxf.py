import io
import math
import ezdxf
from ezdxf import bbox
import xml.etree.ElementTree as ET
from .svg import SVGRenderer

units_to_mm = {
    0: 1.0,      # Unitless (assume 1:1)
    1: 25.4,     # Inches → mm
    2: 304.8,    # Feet → mm
    4: 1.0,      # Millimeters
    5: 10.0,     # Centimeters → mm
    6: 1000.0,   # Meters → mm
    8: 0.0254,   # Microinches → mm
    9: 0.0254,   # Mils → mm
    10: 914.4,   # Yards → mm
}


def get_scale_to_mm(doc, default=None):
    insunits = doc.header.get("$INSUNITS", 0)  # Default to 0 (undefined)
    if insunits not in units_to_mm:
        return default
    return units_to_mm.get(insunits, default) or default


def get_bounds_px(doc):
    """
    Return x, y, w, h
    """
    msp = doc.modelspace()
    entity_bbox = bbox.extents(msp)
    if not entity_bbox.has_data:
        return None

    min_x, min_y, _ = entity_bbox.extmin
    max_x, max_y, _ = entity_bbox.extmax
    return min_x, min_y, (max_x-min_x), (max_y-min_y)


def get_bounds_mm(doc):
    """
    Return x, y, w, h
    """
    bounds = get_bounds_px(doc)
    if bounds is None:
        return None
    min_x, min_y, width, height = bounds

    scale = get_scale_to_mm(doc)
    if scale is None:
        return None

    return min_x*scale, min_y*scale, width*scale, height*scale


class DXFRenderer(SVGRenderer):
    label = 'DFX files (2d)'
    mime_types = ('image/vnd.dxf',)
    extensions = ('.dxf',)

    @classmethod
    def prepare(cls, data):
        # Handle binary input by decoding explicitly
        if isinstance(data, bytes):
            # Decode with UTF-8, replacing errors, and normalize line endings
            try:
                data = data.decode('utf-8')
            except UnicodeDecodeError:
                print("UTF-8 decoding failed, falling back to ASCII")
                data = data.decode('ascii', errors='replace')
            data = data.replace('\r\n', '\n')
        elif not isinstance(data, str):
            raise TypeError("Input must be bytes or str")

        # Parse the DXF data
        try:
            doc = ezdxf.read(io.StringIO(data))
        except ezdxf.DXFStructureError as e:
            raise ValueError(f"Invalid DXF data: {e}")

        return cls.convert_dxf_to_svg(doc)

    @classmethod
    def convert_dxf_to_svg(cls, doc):
        bounds = get_bounds_mm(doc)
        if not bounds:
            return b'<svg xmlns="http://www.w3.org/2000/svg"/>'

        min_x, min_y, width, height = bounds
        scale_to_mm = get_scale_to_mm(doc) or 1.0
        svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg")
        svg.set('viewBox', f"0 0 {width} {height}")
        svg.set('width', f"{width}mm")
        svg.set('height', f"{height}mm")

        group = ET.SubElement(svg, 'g')
        group_transform = f"matrix(1 0 0 -1 {-min_x} {min_y + height})"
        group.set('transform', group_transform)

        msp = doc.modelspace()
        for entity in msp:
            cls.process_entity(group, entity, doc, scale=scale_to_mm)

        return ET.tostring(svg, encoding='utf-8')

    @classmethod
    def create_minimal_svg(cls):
        svg = ET.Element('svg', xmlns="http://www.w3.org/2000/svg")
        svg.set('width', '100mm')
        svg.set('height', '100mm')
        svg.set('viewBox', '0 0 100 100')
        return ET.tostring(svg, encoding='utf-8')

    @classmethod
    def process_entity(cls, parent, entity, doc, scale):
        dxftype = entity.dxftype()
        if dxftype == 'LINE':
            cls.add_line(parent, entity, scale)
        elif dxftype == 'CIRCLE':
            cls.add_circle(parent, entity, scale)
        elif dxftype == 'LWPOLYLINE':
            cls.add_lwpolyline(parent, entity, scale)
        elif dxftype == 'ARC':
            cls.add_arc(parent, entity, scale)
        elif dxftype == 'TEXT':
            cls.add_text(parent, entity, scale)
        elif dxftype == 'ELLIPSE':
            cls.add_ellipse(parent, entity, scale)
        elif dxftype == 'SPLINE':
            cls.add_spline(parent, entity, scale)
        elif dxftype == 'INSERT':
            cls.add_insert(parent, entity, doc, scale)
        else:
            print(f"Unsupported entity type: {dxftype}")

    @classmethod
    def add_line(cls, parent, entity, scale):
        elem = ET.SubElement(parent, 'line')
        elem.set('x1', str(entity.dxf.start.x * scale))
        elem.set('y1', str(entity.dxf.start.y * scale))
        elem.set('x2', str(entity.dxf.end.x * scale))
        elem.set('y2', str(entity.dxf.end.y * scale))
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')

    @classmethod
    def add_circle(cls, parent, entity, scale):
        elem = ET.SubElement(parent, 'circle')
        elem.set('cx', str(entity.dxf.center.x * scale))
        elem.set('cy', str(entity.dxf.center.y * scale))
        elem.set('r', str(entity.dxf.radius * scale))
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')
        elem.set('fill', 'none')

    @classmethod
    def add_lwpolyline(cls, parent, entity, scale):
        points = list(entity.vertices())
        if not points:
            return
        scaled_points = [(p[0] * scale, p[1] * scale) for p in points]
        d = f"M {scaled_points[0][0]},{scaled_points[0][1]}"
        for point in scaled_points[1:]:
            d += f" L {point[0]},{point[1]}"
        if entity.closed:
            d += " Z"
        elem = ET.SubElement(parent, 'path')
        elem.set('d', d)
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')
        elem.set('fill', 'none')

    @classmethod
    def add_arc(cls, parent, entity, scale):
        center = (entity.dxf.center.x * scale,
                  entity.dxf.center.y * scale)
        r = entity.dxf.radius * scale
        start_angle = math.radians(entity.dxf.start_angle)
        end_angle = math.radians(entity.dxf.end_angle)
        start_x = center[0] + r * math.cos(start_angle)
        start_y = center[1] + r * math.sin(start_angle)
        end_x = center[0] + r * math.cos(end_angle)
        end_y = center[1] + r * math.sin(end_angle)
        arc = 1 if (end_angle-start_angle) % (2*math.pi) > math.pi else 0
        d = f"M {start_x} {start_y} A {r} {r} 0 {arc} 0 {end_x} {end_y}"
        elem = ET.SubElement(parent, 'path')
        elem.set('d', d)
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')
        elem.set('fill', 'none')

    @classmethod
    def add_text(cls, parent, entity, scale):
        elem = ET.SubElement(parent, 'text')
        elem.set('x', str(entity.dxf.insert.x * scale))
        elem.set('y', str(entity.dxf.insert.y * scale))
        elem.set('transform',
                 f"rotate({-entity.dxf.rotation} "
                 f"{entity.dxf.insert.x * scale} "
                 f"{entity.dxf.insert.y * scale})")
        elem.set('font-size', f"{entity.dxf.height * scale}mm")
        elem.set('fill', 'black')
        elem.text = entity.dxf.text

    @classmethod
    def add_ellipse(cls, parent, entity, scale):
        center = (entity.dxf.center.x * scale,
                  entity.dxf.center.y * scale)
        major = (entity.dxf.major_axis[0] * scale,
                 entity.dxf.major_axis[1] * scale)
        ratio = entity.dxf.ratio
        angle = math.degrees(math.atan2(major[1], major[0]))
        rx = math.hypot(major[0], major[1])
        ry = rx * ratio
        elem = ET.SubElement(parent, 'ellipse')
        elem.set('cx', str(center[0]))
        elem.set('cy', str(center[1]))
        elem.set('rx', str(rx))
        elem.set('ry', str(ry))
        elem.set('transform', f"rotate({angle} {center[0]} {center[1]})")
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')
        elem.set('fill', 'none')

    @classmethod
    def add_spline(cls, parent, entity, scale):
        # Control points are numpy arrays [x, y, z]
        points = [(p[0] * scale, p[1] * scale) for p in entity.control_points]
        d = "M " + " L ".join(f"{x},{y}" for x, y in points)
        elem = ET.SubElement(parent, 'path')
        elem.set('d', d)
        elem.set('stroke', 'black')
        elem.set('stroke-width', '0.1mm')
        elem.set('fill', 'none')

    @classmethod
    def add_insert(cls, parent, entity, doc, scale):
        block = doc.blocks[entity.dxf.name]
        x = entity.dxf.insert.x * scale
        y = entity.dxf.insert.y * scale
        scale_x = entity.dxf.xscale * scale
        scale_y = entity.dxf.yscale * scale
        rotation = -entity.dxf.rotation
        g = ET.SubElement(parent, 'g')
        transform = (f"translate({x} {y}) "
                     f"rotate({rotation}) "
                     f"scale({scale_x} {scale_y})")
        g.set('transform', transform)
        for e in block:
            cls.process_entity(g, e, doc, scale)
