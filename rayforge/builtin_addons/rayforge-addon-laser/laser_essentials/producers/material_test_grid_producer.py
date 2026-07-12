import logging
import math
import re
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import cairo
from raygeo.geo import Geometry, Matrix
from raygeo.geo.shape.text import FontConfig, text_to_geometry
from raygeo.ops import Ops
from raygeo.ops.types import SectionType

from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece


_MM_PER_INCH = 25.4

logger = logging.getLogger(__name__)


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

    base_margin_left = min(shape_size * 1.5, 15.0)
    base_margin_top = min(shape_size * 1.5, 15.0)
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


class GridMode(Enum):
    """Defines which parameters vary on grid axes."""

    POWER_VS_SPEED = "Power vs Speed"
    POWER_VS_PASSES = "Power vs Passes"
    SPEED_VS_PASSES = "Speed vs Passes"


class MaterialTestGridProducer(OpsProducer):
    """
    Generates a material test grid with varying speed and power settings.
    This producer creates both the final machine operations and a matching
    visual preview for the UI.
    """

    def __init__(
        self,
        test_type: MaterialTestGridType = MaterialTestGridType.CUT,
        grid_mode: GridMode = GridMode.POWER_VS_SPEED,
        speed_range: Tuple[float, float] = (100.0, 500.0),
        power_range: Tuple[float, float] = (10.0, 100.0),
        passes_range: Tuple[int, int] = (1, 5),
        fixed_speed: float = 1000.0,
        fixed_power: float = 50.0,
        grid_dimensions: Tuple[int, int] = (5, 5),
        shape_size: float = 10.0,
        spacing: float = 2.0,
        include_labels: bool = True,
        label_power_percent: float = 10.0,
        label_speed: float = 1000.0,
        line_interval_mm: Optional[float] = None,
    ):
        super().__init__()
        if isinstance(test_type, str):
            self.test_type = MaterialTestGridType(test_type)
        else:
            self.test_type = test_type
        if isinstance(grid_mode, str):
            self.grid_mode = GridMode(grid_mode)
        else:
            self.grid_mode = grid_mode
        self.speed_range = speed_range
        self.power_range = power_range
        self.passes_range = passes_range
        self.fixed_speed = fixed_speed
        self.fixed_power = fixed_power
        self.grid_dimensions = grid_dimensions
        self.shape_size = shape_size
        self.spacing = spacing
        self.include_labels = include_labels
        self.label_power_percent = label_power_percent
        self.label_speed = label_speed
        self.line_interval_mm = line_interval_mm

    @property
    def show_recipe_settings(self) -> bool:
        return False

    def run(
        self,
        laser,
        surface,
        pixels_per_mm,
        *,
        generation_id: int,
        workpiece: Optional["WorkPiece"] = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
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
                    generation_id=generation_id,
                )

        width_mm, height_mm = workpiece.size
        params = self.to_dict()["params"]
        elements = self._calculate_abstract_layout(params, width_mm, height_mm)
        grid_elements = [el for el in elements if el["class"] == "grid-rect"]
        label_elements = [el for el in elements if "label" in el["class"]]

        main_ops = Ops()
        main_ops.set_head(laser.uid)

        # Labels are always outlines, engraved first at a configurable power.
        if label_elements:
            text_ops = self._vectorize_text_to_ops(params, width_mm, height_mm)
            main_ops.ops_section_start(
                SectionType.VECTOR_OUTLINE, workpiece.uid
            )
            main_ops.set_power(0.0)
            main_ops.set_power(self.label_power_percent / 100.0)
            main_ops.set_feed_rate(self.label_speed)
            main_ops.extend(text_ops)
            main_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        section_type = (
            SectionType.RASTER_FILL
            if self.test_type == MaterialTestGridType.ENGRAVE
            else SectionType.VECTOR_OUTLINE
        )
        main_ops.ops_section_start(section_type, workpiece.uid)

        # Sort for risk, highest risk first (high speed -> low speed)
        grid_elements.sort(
            key=lambda e: (-e["speed"], e["power"], e.get("passes", 1))
        )

        is_engrave = self.test_type == MaterialTestGridType.ENGRAVE
        line_spacing = 0.0
        if is_engrave:
            line_spacing = (
                self.line_interval_mm
                if self.line_interval_mm is not None
                else laser.spot_size_mm[1]
            )

        for element in grid_elements:
            # Force laser state to OFF before setting new parameters.
            # This prevents the encoder from emitting a "LASER ON" command
            # while still at the previous location if the laser was previously
            # active.
            main_ops.set_power(0.0)
            main_ops.set_power(element["power"] / 100.0)
            main_ops.set_feed_rate(element["speed"])
            passes = element.get("passes", 1)
            for _ in range(passes):
                if is_engrave:
                    self._draw_filled_box(main_ops, line_spacing, **element)
                else:
                    self._draw_rectangle(main_ops, **element)

        main_ops.ops_section_end(section_type)

        if not main_ops.is_empty():
            main_ops.scale(1, -1)
            main_ops.translate(0, height_mm)

        return WorkPieceArtifact(
            ops=main_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(width_mm, height_mm),
            generation_size=workpiece.size,
            generation_id=generation_id,
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
        Draws a visual-only preview in pixel space using the same
        text_to_geometry / Pango pipeline as _vectorize_text_to_ops, so the
        preview matches the generated Ops.
        """
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()
        ctx.set_source_rgb(0, 0, 0)

        elements = cls._calculate_abstract_layout(params, width_px, height_px)
        test_type_str = params.get("test_type", "Cut")
        test_type = MaterialTestGridType(test_type_str)

        for el in elements:
            if el["class"] == "grid-rect":
                ctx.set_line_width(0.2 * (width_px / 73.0))
                ctx.rectangle(el["x"], el["y"], el["width"], el["height"])
                if test_type == MaterialTestGridType.ENGRAVE:
                    ctx.set_source_rgb(0.5, 0.5, 0.5)
                    ctx.fill_preserve()
                ctx.set_source_rgb(0, 0, 0)
                ctx.stroke()
            elif "label" in el["class"]:
                cls._setup_text_path_on_context(ctx, width_px, **el)
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
        grid_mode = params.get("grid_mode", "Power vs Speed")
        passes_range = params.get("passes_range", (1, 5))
        fixed_speed = params.get("fixed_speed", 1000.0)
        fixed_power = params.get("fixed_power", 50.0)

        min_passes, max_passes = int(passes_range[0]), int(passes_range[1])

        if grid_mode == "Power vs Passes":
            col_range = (min_power, max_power)
            row_range = (float(min_passes), float(max_passes))
            col_label_text = "Power (%)"
            row_label_text = "Passes"
        elif grid_mode == "Speed vs Passes":
            col_range = (min_speed, max_speed)
            row_range = (float(min_passes), float(max_passes))
            col_label_text = "Speed (mm/min)"
            row_label_text = "Passes"
        else:
            col_range = (min_power, max_power)
            row_range = (min_speed, max_speed)
            col_label_text = "Power (%)"
            row_label_text = "Speed (mm/min)"

        col_step = (
            (col_range[1] - col_range[0]) / (cols - 1) if cols > 1 else 0
        )
        row_step = (
            (row_range[1] - row_range[0]) / (rows - 1) if rows > 1 else 0
        )

        base_margin = min(shape_size * 1.5, 15.0)
        margin_left, margin_top = (
            (base_margin * scale_x, base_margin * scale_y)
            if include_labels
            else (0.0, 0.0)
        )
        shape_w, shape_h = shape_size * scale_x, shape_size * scale_y
        spacing_x, spacing_y = spacing * scale_x, spacing * scale_y

        elements = []
        for r in range(rows):
            for c in range(cols):
                col_val = col_range[0] + c * col_step
                row_val = row_range[0] + r * row_step

                if grid_mode == "Power vs Passes":
                    speed = fixed_speed
                    power = col_val
                    passes = max(1, int(round(row_val)))
                elif grid_mode == "Speed vs Passes":
                    speed = col_val
                    power = fixed_power
                    passes = max(1, int(round(row_val)))
                else:
                    speed = row_val
                    power = col_val
                    passes = 1

                elements.append(
                    {
                        "class": "grid-rect",
                        "x": margin_left + c * (shape_w + spacing_x),
                        "y": margin_top + r * (shape_h + spacing_y),
                        "width": shape_w,
                        "height": shape_h,
                        "speed": speed,
                        "power": power,
                        "passes": passes,
                    }
                )

        if include_labels:
            # Make font size proportional to the available margin space, which
            # adapts to non-uniform scaling. Use the smaller margin to ensure
            # text always fits.
            font_size_axis = min(margin_left, margin_top) * 0.25
            font_size_grid = font_size_axis * 0.85

            grid_w = target_width - margin_left
            grid_h = target_height - margin_top

            axis_labels = [
                {
                    "x": margin_left + grid_w / 2,
                    "y": margin_top * 0.3,
                    "text": col_label_text,
                    "class": "axis-label",
                    "font_size": font_size_axis,
                },
                {
                    "x": margin_left * 0.3,
                    "y": margin_top + grid_h / 2,
                    "text": row_label_text,
                    "class": "axis-label",
                    "font_size": font_size_axis,
                    "transform": "rotate(-90)",
                },
            ]

            fixed_label_font_size = font_size_grid * 0.8
            fixed_label_offset = min(margin_left, margin_top) * 0.15
            if grid_mode == "Power vs Passes":
                axis_labels.append(
                    {
                        "x": fixed_label_offset,
                        "y": fixed_label_offset,
                        "text": f"Speed: {int(fixed_speed)} mm/min",
                        "class": "grid-label",
                        "font_size": fixed_label_font_size,
                        "align_h": "left",
                    }
                )
            elif grid_mode == "Speed vs Passes":
                axis_labels.append(
                    {
                        "x": fixed_label_offset,
                        "y": fixed_label_offset,
                        "text": f"Power: {int(fixed_power)}%",
                        "class": "grid-label",
                        "font_size": fixed_label_font_size,
                        "align_h": "left",
                    }
                )

            elements.extend(axis_labels)
            # Position labels proportionally within their margin spaces.
            for c in range(cols):
                col_val = col_range[0] + c * col_step
                text = f"{int(col_val)}"
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
                row_val = row_range[0] + r * row_step
                text = f"{int(round(row_val))}"
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
        Generates clean vector outlines for text using raygeo's swash-based
        text-to-geometry, then converts to Ops.
        """
        label_elements = [
            el
            for el in MaterialTestGridProducer._calculate_abstract_layout(
                params, width_mm, height_mm
            )
            if "label" in el["class"]
        ]

        if not label_elements:
            return Ops()

        result_geo = Geometry()

        for el in label_elements:
            is_axis = el["class"] == "axis-label"
            align_h = el.get("align_h", "center")
            font_size_mm = el["font_size"]

            if font_size_mm <= 0 or not el.get("text"):
                continue

            # Convert mm font size to points (1pt = 1/72 inch).
            font_size_pt = font_size_mm * 72.0 / _MM_PER_INCH

            font_config = FontConfig(
                family="sans-serif",
                size=font_size_pt,
                bold=is_axis,
            )

            geo = text_to_geometry(el["text"], font_config=font_config)
            if geo is None or geo.is_empty():
                logger.warning(
                    "Empty geometry for text label '%s' at font size %.1f",
                    el["text"],
                    font_size_mm,
                )
                continue

            # raygeo text_to_geometry returns geometry in mm space with
            # Y-UP (font coords). The caller applies scale(1,-1) +
            # translate(0, height) to the combined ops, so we convert
            # to Y-DOWN here so the final flip produces Y-UP.

            # Compute alignment offset from the bounding rect.
            xmin, ymin, xmax, ymax = geo.rect()
            text_width_mm = xmax - xmin

            if align_h == "right":
                x_offset = -text_width_mm
            elif align_h == "left":
                x_offset = 0.0
            else:
                x_offset = -text_width_mm / 2.0

            # raygeo output is Y-UP.  The caller does
            # scale(1,-1)+translate(0,height) on the combined ops
            # (grid + labels).  We flip the glyphs to Y-DOWN in
            # place (at origin) BEFORE translating, so the final
            # flip makes them Y-UP at the correct position.
            geo.transform(Matrix.scale(1, -1))
            y_offset = 0.0

            if "transform" in el:
                match = re.search(r"rotate\((.+?)\)", el["transform"])
                if match:
                    angle_deg = float(match.group(1))
                    combined = (
                        Matrix.translation(el["x"], el["y"])
                        @ Matrix.rotation(angle_deg)
                        @ Matrix.translation(x_offset, y_offset)
                    )
                    geo.transform(combined)
                else:
                    geo.transform(Matrix.translation(
                        el["x"] + x_offset, el["y"] + y_offset
                    ))
            else:
                geo.transform(Matrix.translation(
                    el["x"] + x_offset, el["y"] + y_offset
                ))

            result_geo.extend(geo)

        if result_geo.is_empty():
            if label_elements:
                logger.warning(
                    "All text labels produced empty geometry for the material "
                    "test grid. Check font configuration."
                )
            return Ops()

        return Ops.from_geometry(result_geo)

    @staticmethod
    def _setup_text_path_on_context(
        ctx: cairo.Context, preview_px: float, **el
    ):
        """
        Configures a Cairo context and adds a text path using raygeo's
        text_to_geometry.  The context is expected to be in pixel space
        (96 DPI, Y-down).
        """
        from rayforge.image.geo_renderer import geometry_to_cairo

        ctx.save()
        is_axis = el["class"] == "axis-label"
        align_h = el.get("align_h", "center")
        font_size_px = el["font_size"]
        text = el.get("text", "")

        if font_size_px <= 0 or not text:
            ctx.restore()
            return

        # Convert pixel font size to points (1pt = 1/72 inch at 96 DPI).
        font_size_pt = font_size_px * 72.0 / 96.0

        font_config = FontConfig(
            family="sans-serif",
            size=font_size_pt,
            bold=is_axis,
        )

        geo = text_to_geometry(text, font_config=font_config)
        if geo is None or geo.is_empty():
            ctx.restore()
            return

        # raygeo returns mm-space Y-UP. Convert to pixel-space Y-DOWN
        # at 96 DPI for the Cairo canvas (mirrors the Ops path).
        px_per_mm = 96.0 / 25.4
        geo.transform(Matrix.scale(px_per_mm, -px_per_mm))

        # Compute alignment offsets from geometry bounding rect.
        xmin, ymin, xmax, ymax = geo.rect()
        text_w = xmax - xmin

        if align_h == "right":
            x_offset = -text_w
        elif align_h == "left":
            x_offset = 0.0
        else:
            x_offset = -text_w / 2.0

        geo.transform(Matrix.translation(x_offset, 0.0))

        ctx.translate(el["x"], el["y"])
        if "transform" in el:
            match = re.search(r"rotate\((.+?)\)", el["transform"])
            if match:
                ctx.rotate(math.radians(float(match.group(1))))

        geometry_to_cairo(geo, ctx)
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
        """Generates individual scan lines for raster fill."""
        x, y, w, h = el["x"], el["y"], el["width"], el["height"]
        if h < 1e-6:
            return

        num_lines = int(h / line_spacing)
        if num_lines < 1:
            ops.move_to(x, y + h / 2, 0.0)
            ops.line_to(x + w, y + h / 2, 0.0)
            return

        y_step = h / num_lines
        for i in range(num_lines + 1):
            cur_y = y + i * y_step
            if i % 2 == 0:
                ops.move_to(x, cur_y, 0.0)
                ops.line_to(x + w, cur_y, 0.0)
            else:
                ops.move_to(x + w, cur_y, 0.0)
                ops.line_to(x, cur_y, 0.0)

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": {
                "test_type": self.test_type.value,
                "grid_mode": self.grid_mode.value,
                "speed_range": list(self.speed_range),
                "power_range": list(self.power_range),
                "passes_range": list(self.passes_range),
                "fixed_speed": self.fixed_speed,
                "fixed_power": self.fixed_power,
                "grid_dimensions": list(self.grid_dimensions),
                "shape_size": self.shape_size,
                "spacing": self.spacing,
                "include_labels": self.include_labels,
                "label_power_percent": self.label_power_percent,
                "label_speed": self.label_speed,
                "line_interval_mm": self.line_interval_mm,
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialTestGridProducer":
        params = data.get("params", {})
        return cls(**params)
