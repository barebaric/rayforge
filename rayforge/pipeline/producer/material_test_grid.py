import logging
import math
import cairo
import re
from enum import Enum
from typing import Tuple, Dict, Any, List, Optional, TYPE_CHECKING

from ...core.ops import Ops, SectionType
from ...core.geo.geometry import Geometry
from ...core.matrix import Matrix
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...shared.tasker.proxy import BaseExecutionContext


logger = logging.getLogger(__name__)

_TRACE_DPI = 300.0
_MM_PER_INCH = 25.4


def get_material_test_proportional_size(
    params: Dict[str, Any],
) -> Tuple[float, float]:
    """
    Calculates the natural size in mm for a material test grid.

    This is a standalone function to allow the ProceduralImporter to
    determine the initial WorkPiece size without instantiating the producer.

    Args:
        params: A dictionary of geometric parameters for the grid, including
          'grid_dimensions', 'shape_size', 'spacing', and 'include_labels'.

    Returns:
        A tuple of (width, height) in millimeters.
    """
    cols, rows = map(int, params.get("grid_dimensions", (5, 5)))
    shape_size = params.get("shape_size", 10.0)
    spacing = params.get("spacing", 2.0)
    include_labels = params.get("include_labels", True)

    base_margin_left = 15.0
    base_margin_top = 15.0
    width = (cols * shape_size) + ((cols - 1) * spacing)
    height = (rows * shape_size) + ((rows - 1) * spacing)
    if include_labels:
        width += base_margin_left
        height += base_margin_top
    return width, height


def draw_material_test_preview(
    ctx: cairo.Context, width: float, height: float, params: Dict[str, Any]
):
    """
    Stable, importable entry point for the generic procedural renderer.
    This function delegates the actual drawing to the producer class.
    """
    MaterialTestGridProducer.draw_preview(ctx, width, height, params)


class MaterialTestGridType(Enum):
    """Material test types."""

    CUT = "Cut"
    ENGRAVE = "Engrave"


class MaterialTestGridProducer(OpsProducer):
    """
    Generates a material test grid with varying speed and power settings.
    This producer creates both the final machine operations and a matching
    visual preview for the UI.
    """

    def __init__(
        self,
        test_type: MaterialTestGridType = MaterialTestGridType.CUT,
        speed_range: Tuple[float, float] = (100.0, 500.0),
        power_range: Tuple[float, float] = (10.0, 100.0),
        grid_dimensions: Tuple[int, int] = (5, 5),
        shape_size: float = 10.0,
        spacing: float = 2.0,
        include_labels: bool = True,
        label_power_percent: float = 10.0,
    ):
        super().__init__()
        if isinstance(test_type, str):
            self.test_type = MaterialTestGridType(test_type)
        else:
            self.test_type = test_type
        self.speed_range = speed_range
        self.power_range = power_range
        self.grid_dimensions = grid_dimensions
        self.shape_size = shape_size
        self.spacing = spacing
        self.include_labels = include_labels
        self.label_power_percent = label_power_percent

    @property
    def supports_power(self) -> bool:
        return False

    @property
    def supports_cut_speed(self) -> bool:
        return False

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        workpiece: Optional["WorkPiece"] = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        proxy: Optional["BaseExecutionContext"] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError(
                "MaterialTestGridProducer requires a workpiece context."
            )

        # Only run on the designated workpiece for this step.
        if settings:
            owner_uid = settings.get("generated_workpiece_uid")
            if owner_uid and owner_uid != workpiece.uid:
                return WorkPieceArtifact(
                    ops=Ops(),
                    is_scalable=False,
                    source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
                    source_dimensions=workpiece.size or (0, 0),
                    generation_size=workpiece.size,
                )

        width_mm, height_mm = workpiece.size
        params = self.to_dict()["params"]
        elements = self._calculate_abstract_layout(params, width_mm, height_mm)
        grid_elements = [el for el in elements if el["class"] == "grid-rect"]
        label_elements = [el for el in elements if "label" in el["class"]]

        main_ops = Ops()
        main_ops.set_laser(laser.uid)
        main_ops.ops_section_start(SectionType.VECTOR_OUTLINE, workpiece.uid)

        # Sort for risk, highest risk first (high speed -> low speed)
        grid_elements.sort(key=lambda e: (-e["speed"], e["power"]))

        for element in grid_elements:
            # Force laser state to OFF before setting new parameters.
            # This prevents the encoder from emitting a "LASER ON" command
            # while still at the previous location if the laser was previously
            # active.
            main_ops.set_power(0.0)
            main_ops.set_power(element["power"] / 100.0)
            main_ops.set_cut_speed(element["speed"])
            if self.test_type == MaterialTestGridType.ENGRAVE:
                line_spacing = laser.spot_size_mm[1]
                self._draw_filled_box(main_ops, line_spacing, **element)
            else:
                self._draw_rectangle(main_ops, **element)

        # Labels are always outlines, engraved at a configurable power.
        if label_elements:
            text_ops = self._vectorize_text_to_ops(params, width_mm, height_mm)
            # Ensure laser is off before switching to label settings
            main_ops.set_power(0.0)
            main_ops.set_power(self.label_power_percent / 100.0)
            main_ops.set_cut_speed(1000)
            main_ops.extend(text_ops)

        main_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        if not main_ops.is_empty():
            main_ops.scale(1, -1)
            main_ops.translate(0, height_mm)

        return WorkPieceArtifact(
            ops=main_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(width_mm, height_mm),
            generation_size=workpiece.size,
        )

    @classmethod
    def draw_preview(
        cls,
        ctx: cairo.Context,
        width_px: float,
        height_px: float,
        params: Dict[str, Any],
    ):
        """
        Draws a visual-only preview by dynamically calculating the layout in
        pixel space. This prevents stretching.
        """
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.set_source_rgb(0, 0, 0)

        elements = cls._calculate_abstract_layout(params, width_px, height_px)
        test_type_str = params.get("test_type", "Cut")
        test_type = MaterialTestGridType(test_type_str)

        for el in elements:
            if el["class"] == "grid-rect":
                ctx.set_line_width(0.2 * (width_px / 73.0))  # Scale line
                ctx.rectangle(el["x"], el["y"], el["width"], el["height"])
                if test_type == MaterialTestGridType.ENGRAVE:
                    ctx.set_source_rgb(0.5, 0.5, 0.5)
                    ctx.fill_preserve()
                ctx.set_source_rgb(0, 0, 0)
                ctx.stroke()
            elif "label" in el["class"]:
                cls._setup_text_path_on_context(ctx, **el)
                ctx.fill()

    @staticmethod
    def _calculate_abstract_layout(
        params: Dict[str, Any], target_width: float, target_height: float
    ) -> List[Dict]:
        """
        Shared logic to calculate the positions and properties of all grid
        elements based on a complete set of parameters. Works for both
        pixel and millimeter coordinate systems.
        """
        base_width, base_height = get_material_test_proportional_size(params)
        scale_x = target_width / base_width if base_width > 1e-9 else 1.0
        scale_y = target_height / base_height if base_height > 1e-9 else 1.0

        cols, rows = map(int, params.get("grid_dimensions", (5, 5)))
        min_speed, max_speed = params.get("speed_range", (100.0, 500.0))
        min_power, max_power = params.get("power_range", (10.0, 100.0))
        shape_size = params.get("shape_size", 10.0)
        spacing = params.get("spacing", 2.0)
        include_labels = params.get("include_labels", True)

        speed_step = (max_speed - min_speed) / (rows - 1) if rows > 1 else 0
        power_step = (max_power - min_power) / (cols - 1) if cols > 1 else 0

        margin_left, margin_top = (
            (15.0 * scale_x, 15.0 * scale_y) if include_labels else (0.0, 0.0)
        )
        shape_w, shape_h = shape_size * scale_x, shape_size * scale_y
        spacing_x, spacing_y = spacing * scale_x, spacing * scale_y

        elements = []
        for r in range(rows):
            for c in range(cols):
                elements.append(
                    {
                        "class": "grid-rect",
                        "x": margin_left + c * (shape_w + spacing_x),
                        "y": margin_top + r * (shape_h + spacing_y),
                        "width": shape_w,
                        "height": shape_h,
                        "speed": min_speed + r * speed_step,
                        "power": min_power + c * power_step,
                    }
                )

        if include_labels:
            # Make font size proportional to the available margin space, which
            # adapts to non-uniform scaling. Use the smaller margin to ensure
            # text always fits.
            font_size_axis = min(margin_left, margin_top) * 0.25
            font_size_grid = font_size_axis * 0.85

            # Add an absolute minimum size in target units (pixels or mm) to
            # prevent text from becoming illegible if squashed.
            min_abs_font_size = 2.0
            font_size_axis = max(font_size_axis, min_abs_font_size * 1.1)
            font_size_grid = max(font_size_grid, min_abs_font_size)

            grid_w = target_width - margin_left
            grid_h = target_height - margin_top

            # Position labels proportionally within their margin spaces.
            elements.extend(
                [
                    {
                        "x": margin_left + grid_w / 2,
                        "y": margin_top * 0.3,
                        "text": "Power (%)",
                        "class": "axis-label",
                        "font_size": font_size_axis,
                    },
                    {
                        "x": margin_left * 0.3,
                        "y": margin_top + grid_h / 2,
                        "text": "Speed (mm/min)",
                        "class": "axis-label",
                        "font_size": font_size_axis,
                        "transform": "rotate(-90)",
                    },
                ]
            )
            for c in range(cols):
                text = f"{int(min_power + c * power_step)}"
                elements.append(
                    {
                        "x": margin_left
                        + c * (shape_w + spacing_x)
                        + shape_w / 2,
                        "y": margin_top * 0.75,
                        "text": text,
                        "class": "grid-label",
                        "font_size": font_size_grid,
                    }
                )
            for r in range(rows):
                text = f"{int(min_speed + r * speed_step)}"
                elements.append(
                    {
                        "x": margin_left * 0.9,
                        "y": margin_top
                        + r * (shape_h + spacing_y)
                        + shape_h / 2,
                        "text": text,
                        "class": "grid-label",
                        "font_size": font_size_grid,
                        "align_h": "right",
                    }
                )
        return elements

    @staticmethod
    def _vectorize_text_to_ops(
        params: Dict[str, Any], width_mm: float, height_mm: float
    ) -> Ops:
        """
        Generates clean vector outlines for text by creating a single combined
        path in Cairo before converting to Geometry.
        """
        px_per_mm = _TRACE_DPI / _MM_PER_INCH
        width_px = int(width_mm * px_per_mm)
        height_px = int(height_mm * px_per_mm)

        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        ctx = cairo.Context(surface)

        label_elements = [
            el
            for el in MaterialTestGridProducer._calculate_abstract_layout(
                params, width_px, height_px
            )
            if "label" in el["class"]
        ]

        ctx.new_path()
        for el in label_elements:
            MaterialTestGridProducer._setup_text_path_on_context(ctx, **el)

        path_data = ctx.copy_path_flat()
        if not path_data:
            return Ops()

        geo = Geometry.from_cairo_path(path_data)
        if geo.is_empty():
            return Ops()

        scale_back = 1.0 / px_per_mm
        scaling_matrix = Matrix.scale(scale_back, scale_back)
        geo.transform(scaling_matrix.to_4x4_numpy())

        return Ops.from_geometry(geo)

    @staticmethod
    def _setup_text_path_on_context(ctx: cairo.Context, **el):
        """
        Configures a Cairo context and adds a text path to it. This is the
        shared logic for both preview and direct vectorization.
        """
        ctx.save()
        is_axis = el["class"] == "axis-label"
        align_h = el.get("align_h", "center")
        ctx.select_font_face(
            "Sans",
            cairo.FONT_SLANT_NORMAL,
            cairo.FONT_WEIGHT_BOLD if is_axis else cairo.FONT_WEIGHT_NORMAL,
        )
        ctx.set_font_size(el["font_size"])
        extents = ctx.text_extents(el["text"])
        ctx.translate(el["x"], el["y"])
        if "transform" in el:
            match = re.search(r"rotate\((.+?)\)", el["transform"])
            if match:
                ctx.rotate(math.radians(float(match.group(1))))

        if align_h == "right":
            x_offset = -(extents.x_bearing + extents.width)
        elif align_h == "left":
            x_offset = -extents.x_bearing
        else:  # "center"
            x_offset = -(extents.x_bearing + extents.width / 2)

        ctx.move_to(
            x_offset,
            -(extents.y_bearing + extents.height / 2),
        )
        ctx.text_path(el["text"])
        ctx.restore()

    @staticmethod
    def _draw_rectangle(ops: Ops, **el):
        x, y, w, h = el["x"], el["y"], el["width"], el["height"]
        ops.move_to(x, y, 0.0)
        ops.line_to(x + w, y, 0.0)
        ops.line_to(x + w, y + h, 0.0)
        ops.line_to(x, y + h, 0.0)
        ops.line_to(x, y, 0.0)

    @staticmethod
    def _draw_filled_box(ops: Ops, line_spacing: float, **el):
        """Generates a serpentine (back-and-forth) fill pattern."""
        x, y, w, h = el["x"], el["y"], el["width"], el["height"]
        if h < 1e-6:
            return

        num_lines = int(h / line_spacing)
        if num_lines < 1:
            # If the box is thinner than the line spacing, draw one line
            ops.move_to(x, y + h / 2, 0.0)
            ops.line_to(x + w, y + h / 2, 0.0)
            return

        y_step = h / num_lines
        ops.move_to(x, y, 0.0)
        last_x = x

        for i in range(num_lines + 1):
            cur_y = y + i * y_step
            # Move to the current Y level on the correct side
            ops.line_to(last_x, cur_y, 0.0)

            if i % 2 == 0:  # Even lines, go right
                ops.line_to(x + w, cur_y, 0.0)
                last_x = x + w
            else:  # Odd lines, go left
                ops.line_to(x, cur_y, 0.0)
                last_x = x

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": {
                "test_type": self.test_type.value,
                "speed_range": list(self.speed_range),
                "power_range": list(self.power_range),
                "grid_dimensions": list(self.grid_dimensions),
                "shape_size": self.shape_size,
                "spacing": self.spacing,
                "include_labels": self.include_labels,
                "label_power_percent": self.label_power_percent,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialTestGridProducer":
        params = data.get("params", {})
        return cls(**params)
