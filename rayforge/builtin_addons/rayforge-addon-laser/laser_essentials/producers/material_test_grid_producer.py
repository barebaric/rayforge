import logging
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

import cairo
import numpy as np
from raygeo.ops import Ops
from raygeo.ops.assembly.material_test_grid import (
    generate_material_test_grid,
    generate_material_test_grid_preview,
)
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
        test_type = (
            "cut" if self.test_type == MaterialTestGridType.CUT else "engrave"
        )
        line_interval = (
            self.line_interval_mm
            if self.line_interval_mm is not None
            else laser.spot_size_mm[1]
        )

        logger.info(
            "MaterialTestGridProducer: calling generate_material_test_grid "
            "for '%s' (size=%.1fx%.1f, cols=%d, rows=%d, mode=%s, "
            "include_labels=%s)",
            workpiece.name,
            width_mm,
            height_mm,
            self.grid_dimensions[0],
            self.grid_dimensions[1],
            test_type,
            self.include_labels,
        )
        result = generate_material_test_grid(
            size_mm=(width_mm, height_mm),
            cols=self.grid_dimensions[0],
            rows=self.grid_dimensions[1],
            min_speed=self.speed_range[0],
            max_speed=self.speed_range[1],
            min_power=self.power_range[0],
            max_power=self.power_range[1],
            min_passes=self.passes_range[0],
            max_passes=self.passes_range[1],
            fixed_speed=self.fixed_speed,
            fixed_power=self.fixed_power,
            shape_size=self.shape_size,
            spacing=self.spacing,
            line_interval_mm=line_interval,
            mode=test_type,
            grid_mode=self.grid_mode.value,
            include_labels=self.include_labels,
        )

        logger.info(
            "MaterialTestGridProducer: generate_material_test_grid returned "
            "%d ops for '%s'",
            result.ops.len(),
            workpiece.name,
        )
        main_ops = Ops()
        main_ops.set_head(laser.uid)
        main_ops.ops_section_start(SectionType.VECTOR_OUTLINE, workpiece.uid)
        main_ops.extend(result.ops)
        main_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

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
        Draws a visual-only preview by rendering the Ops via
        ``generate_material_test_grid_preview`` (raygeo), then blitting
        the resulting RGBA buffer to the Cairo context.
        """
        size_mm = get_material_test_proportional_size(params)
        dpi_x = width_px / size_mm[0] * _MM_PER_INCH
        dpi_y = height_px / size_mm[1] * _MM_PER_INCH
        dpi = (dpi_x + dpi_y) / 2.0
        img = generate_material_test_grid_preview(
            size_mm=size_mm,
            dpi=dpi,
            cols=params.get("grid_dimensions", (5, 5))[0],
            rows=params.get("grid_dimensions", (5, 5))[1],
            min_speed=params.get("speed_range", (100.0, 500.0))[0],
            max_speed=params.get("speed_range", (100.0, 500.0))[1],
            min_power=params.get("power_range", (10.0, 100.0))[0],
            max_power=params.get("power_range", (10.0, 100.0))[1],
            min_passes=params.get("passes_range", (1, 5))[0],
            max_passes=params.get("passes_range", (1, 5))[1],
            fixed_speed=params.get("fixed_speed", 1000.0),
            fixed_power=params.get("fixed_power", 50.0),
            shape_size=params.get("shape_size", 10.0),
            spacing=params.get("spacing", 2.0),
            mode=(
                "cut" if params.get("test_type", "Cut") == "Cut" else "engrave"
            ),
            grid_mode=params.get("grid_mode", "Power vs Speed"),
            include_labels=params.get("include_labels", True),
        )

        # Blit the numpy RGBA array onto the Cairo context.
        # Cairo FORMAT_ARGB32 on little-endian expects B, G, R, A byte
        # order — swap R and B channels.
        h, w = img.shape[:2]
        bgra = np.empty_like(img)
        bgra[:, :, 0] = img[:, :, 2]  # B = R
        bgra[:, :, 1] = img[:, :, 1]  # G = G
        bgra[:, :, 2] = img[:, :, 0]  # R = B
        bgra[:, :, 3] = img[:, :, 3]  # A = A
        surface = cairo.ImageSurface.create_for_data(
            np.ascontiguousarray(bgra),
            cairo.FORMAT_ARGB32,
            w,
            h,
        )
        ctx.set_source_surface(surface, 0, 0)
        ctx.paint()
        surface.finish()

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
