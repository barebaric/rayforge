import io
import math
import cairo
import ezdxf
from ezdxf import bbox
from .renderer import Renderer


units_to_mm = {
    0: None,     # Unitless
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


def draw_line(ctx, entity):
    """Draw a LINE entity."""
    start = entity.dxf.start
    end = entity.dxf.end
    ctx.move_to(start.x, start.y)
    ctx.line_to(end.x, end.y)
    ctx.stroke()


def draw_circle(ctx, entity):
    """Draw a CIRCLE entity."""
    center = entity.dxf.center
    radius = entity.dxf.radius
    ctx.arc(center.x, center.y, radius, 0, 2*math.pi)
    ctx.stroke()


def draw_lwpolyline(ctx, entity, factor):
    """Draw an LWPOLYLINE entity."""
    points = list(entity.vertices())  # Get vertices as tuples
    if len(points) == 0:
        return
    ctx.move_to(points[0][0] * factor, points[0][1] * factor)
    for point in points[1:]:
        ctx.line_to(point[0] * factor, point[1] * factor)
    if entity.closed:
        ctx.close_path()
    ctx.stroke()


def draw_arc(ctx, entity):
    """Draw an ARC entity."""
    center = entity.dxf.center
    radius = entity.dxf.radius
    start_angle = math.radians(entity.dxf.start_angle)
    end_angle = math.radians(entity.dxf.end_angle)
    ctx.arc(center.x, center.y, radius, start_angle, end_angle)
    ctx.stroke()


def draw_text(ctx, entity):
    """Draw a TEXT entity."""
    ctx.save()  # Save the current state of the context
    insert = entity.dxf.insert
    text = entity.dxf.text
    height = entity.dxf.height
    rotation = math.radians(entity.dxf.rotation)

    # Set font size and rotation
    ctx.set_font_size(height)
    ctx.translate(insert.x, insert.y)
    ctx.rotate(rotation)

    # Draw text
    ctx.move_to(0, 0)
    ctx.show_text(text)
    ctx.restore()  # Restore the original state


def draw_ellipse(ctx, entity):
    """Draw an ELLIPSE entity."""
    ctx.save()
    center = entity.dxf.center
    major_axis = entity.dxf.major_axis
    ratio = entity.dxf.ratio
    start_angle = math.radians(entity.dxf.start_param)
    end_angle = math.radians(entity.dxf.end_param)

    # Calculate minor axis
    minor_axis = (major_axis[1], -major_axis[0])  # Rotate 90 degrees
    minor_axis = (minor_axis[0] * ratio, minor_axis[1] * ratio)

    # Apply transformation for ellipse
    ctx.translate(center.x, center.y)
    ctx.rotate(math.atan2(major_axis[1], major_axis[0]))
    ctx.scale(math.hypot(*major_axis), math.hypot(*minor_axis))

    # Draw ellipse
    ctx.arc(0, 0, 1, start_angle, end_angle)
    ctx.stroke()
    ctx.restore()


def draw_spline(ctx, entity):
    """Draw a SPLINE entity."""
    ctx.save()
    control_points = entity.control_points()
    if len(control_points) == 0:
        return

    # Move to the first control point
    ctx.move_to(control_points[0][0], control_points[0][1])

    # Draw a polyline approximation of the spline
    for point in control_points[1:]:
        ctx.line_to(point[0], point[1])
    ctx.stroke()
    ctx.restore()


def draw_insert(ctx, entity, doc):
    """Draw an INSERT entity (block reference)."""
    block = doc.blocks[entity.dxf.name]
    insert_point = entity.dxf.insert
    scale_x = entity.dxf.xscale
    scale_y = entity.dxf.yscale
    rotation = math.radians(entity.dxf.rotation)

    # Apply transformations for the block
    ctx.save()
    ctx.translate(insert_point.x, insert_point.y)
    ctx.rotate(rotation)
    ctx.scale(scale_x, scale_y)

    # Recursively render the block's entities
    for block_entity in block:
        match block_entity.dxftype():
            case 'LINE':
                draw_line(ctx, block_entity)
            case 'CIRCLE':
                draw_circle(ctx, block_entity)
            case 'LWPOLYLINE':
                # No scaling for nested entities
                draw_lwpolyline(ctx, block_entity, 1.0)
            case 'ARC':
                draw_arc(ctx, block_entity)
            case 'TEXT':
                draw_text(ctx, block_entity)
            case 'ELLIPSE':
                draw_ellipse(ctx, block_entity)
            case 'SPLINE':
                draw_spline(ctx, block_entity)
            case 'INSERT':
                draw_insert(ctx, block_entity, doc)  # Handle nested blocks
            case _:
                thetype = block_entity.dxftype()
                print(f"Unsupported nested entity type: {thetype}")

    ctx.restore()


class DXFRenderer(Renderer):
    label = 'DFX files (2d)'
    mime_types = ('image/vnd.dxf',)
    extensions = ('.dxf',)

    @classmethod
    def prepare(cls, data):
        return ezdxf.read(io.StringIO(data.decode("utf-8")))

    @classmethod
    def get_natural_size(cls, data, px_factor=0):
        """
        Returns the natural size of the document in mm as a tuple (w, h).
        This is BEFORE cropping the margins.
        """
        bounds = get_bounds_mm(data)
        if bounds is None:
            return None, None  # No known dimensions
        return bounds[2], bounds[3]

    @classmethod
    def get_aspect_ratio(cls, data):
        x, y, w, h = get_bounds_px(data)
        return w/h if h else 1

    @classmethod
    def render_workpiece(cls, data, width=None, height=None):
        msp = data.modelspace()
        factor = get_scale_to_mm(data, 1.0)   # default to 1mm = 1px

        # Set up Cairo transformations
        if width is None or height is None:
            _, _, width, height = get_bounds_px(data)
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
        ctx = cairo.Context(surface)

        # Scale and flip Y-axis due to DXF's coordinate system
        x, y, w, h = get_bounds_px(data)
        ctx.scale(width/w, -height/h)
        ctx.translate(-x, -y-h)

        # Set default drawing style (0.1 mm line width, black)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_line_width(1*factor)

        # Draw all entities
        for entity in msp:
            match entity.dxftype():
                case 'LINE':
                    draw_line(ctx, entity)
                case 'CIRCLE':
                    draw_circle(ctx, entity)
                case 'LWPOLYLINE':
                    draw_lwpolyline(ctx, entity, factor)
                case 'ARC':
                    draw_arc(ctx, entity)
                case 'TEXT':
                    draw_text(ctx, entity)
                case 'ELLIPSE':
                    draw_ellipse(ctx, entity)
                case 'SPLINE':
                    draw_spline(ctx, entity)
                case 'INSERT':
                    draw_insert(ctx, entity, data)
                case _:
                    print(f"Unsupported entity type: {entity.dxftype()}")

        return surface
