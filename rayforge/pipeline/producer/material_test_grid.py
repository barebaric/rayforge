"""
Material Test Grid Producer

Generates a grid of test shapes with varying speed and power settings for
material testing. Unlike typical producers, this generates ops directly from
parameters without requiring pixel data.
"""

from __future__ import annotations
import logging
from enum import Enum
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
    MoveToCommand,
    LineToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
)
import cairo

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class MaterialTestGridType(Enum):
    """Material test types."""
    CUT = "Cut"
    ENGRAVE = "Engrave"


class MaterialTestGridProducer(OpsProducer):
    """
    Generates a material test grid with varying speed and power settings.

    Each cell in the grid represents a unique speed/power combination.
    Cells are executed in risk-optimized order (highest speed first, then
    lowest power) to minimize material charring.
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
    ):
        """
        Initializes the MaterialTestGridProducer.

        Args:
            test_type: MaterialTestGridType.CUT or MaterialTestGridType.ENGRAVE
            speed_range: (min_speed, max_speed) in mm/min
            power_range: (min_power, max_power) in percentage (0-100)
            grid_dimensions: (columns, rows) for the grid
            shape_size: Size of each test square in mm
            spacing: Gap between squares in mm
            include_labels: Whether to add speed/power labels
        """
        super().__init__()
        # Handle string-to-enum conversion for deserialization
        if isinstance(test_type, str):
            self.test_type = MaterialTestGridType(test_type)
        elif isinstance(test_type, MaterialTestGridType):
            self.test_type = test_type
        else:
            raise TypeError(
                "test_type must be a MaterialTestGridType or a valid string"
            )
        self.speed_range = speed_range
        self.power_range = power_range
        self.grid_dimensions = grid_dimensions
        self.shape_size = shape_size
        self.spacing = spacing
        self.include_labels = include_labels

    @property
    def supports_power(self) -> bool:
        """The grid generator does not support a fixed power setting."""
        return False

    @property
    def supports_cut_speed(self) -> bool:
        """The grid generator does not support a fixed speed setting."""
        return False

    def run(
        self,
        laser,
        surface,  # Unused - no rendering needed
        pixels_per_mm,  # Unused
        *,
        workpiece: Optional[WorkPiece] = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        """
        Generates the material test ops.

        Args:
            laser: Laser config (used for max_power scaling)
            surface: Unused - no rendering needed
            pixels_per_mm: Unused
            workpiece: Used only for UID marking in ops sections
            settings: Unused - no runtime settings needed
            y_offset_mm: Unused

        Returns:
            PipelineArtifact containing the test grid ops
        """
        ops = Ops()

        if workpiece:
            ops.add(OpsSectionStartCommand(
                SectionType.VECTOR_OUTLINE, workpiece.uid
            ))

        # Calculate dimensions for Y-flip calculation
        cols, rows = self.grid_dimensions
        grid_width = cols * (self.shape_size + self.spacing) - self.spacing
        grid_height = rows * (self.shape_size + self.spacing) - self.spacing

        # Generate labels first (if enabled) so they're processed first
        # The label generator returns the margin sizes it needs
        # (x and y can differ)
        if self.include_labels:
            margin_x, margin_y = self._generate_labels(
                ops, grid_width, grid_height, laser
            )
            offset_x = margin_x
            offset_y = margin_y
        else:
            margin_x = 0.0
            margin_y = 0.0
            offset_x = 0.0
            offset_y = 0.0

        # Generate test squares in risk-sorted order
        test_elements = self._create_test_grid()

        # Sort by risk: highest speed (safest) first, then lowest power
        test_elements.sort(key=lambda e: (-e["speed"], e["power"]))

        for element in test_elements:
            power_normalized = element["power"] / 100.0
            ops.add(SetPowerCommand(power_normalized))
            ops.add(SetCutSpeedCommand(element["speed"]))
            # Use coordinates as-is (r=0 at y=0)
            if self.test_type == MaterialTestGridType.ENGRAVE:
                self._draw_filled_box(
                        ops,
                        element["x"] + offset_x,
                        element["y"] + offset_y,
                        element["width"],
                        element["height"],
                    )
            else:  # Cut mode
                self._draw_rectangle(
                    ops,
                    element["x"] + offset_x,
                    element["y"] + offset_y,
                    element["width"],
                    element["height"],
                )

        if workpiece:
            ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        # Calculate total dimensions (reuse grid dimensions and margins
        # from above)
        total_width = grid_width + margin_x
        total_height = grid_height + margin_y

        logger.info(
            (
                f"Generated material test grid: {cols}x{rows} cells, "
                f"{grid_width:.1f}x{grid_height:.1f} mm"
            )
        )

        return PipelineArtifact(
            ops=ops,
            is_scalable=True,  # Can be scaled mathematically
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=(total_width, total_height),
            generation_size=(total_width, total_height),
        )

    def _create_test_grid(self) -> list:
        """Creates the list of test element dictionaries."""
        cols, rows = self.grid_dimensions
        cols, rows = int(cols), int(rows)  # Ensure integers for range()
        min_speed, max_speed = self.speed_range
        min_power, max_power = self.power_range

        # Rows vary speed (Y-axis), columns vary power (X-axis)
        speed_step = (max_speed - min_speed) / (rows - 1) if rows > 1 else 0
        power_step = (max_power - min_power) / (cols - 1) if cols > 1 else 0

        elements = []
        for r in range(rows):
            for c in range(cols):
                current_speed = min_speed + r * speed_step
                current_power = min_power + c * power_step

                x_pos = c * (self.shape_size + self.spacing)
                y_pos = r * (self.shape_size + self.spacing)

                elements.append(
                    {
                        "type": "box",
                        "x": x_pos,
                        "y": y_pos,
                        "width": self.shape_size,
                        "height": self.shape_size,
                        "speed": current_speed,
                        "power": current_power,
                    }
                )

        return elements

    def _generate_labels(
        self,
        ops: Ops,
        grid_width: float,
        grid_height: float,
        laser,
    ) -> tuple[float, float]:
        """Generates axis labels and numeric annotations using shared
        layout.

        Args:
            ops: Ops container to add label commands to
            grid_width: Grid width (not including margins)
            grid_height: Grid height (not including margins)
            laser: Laser config for power scaling

        Returns:
            Tuple of (margin_x, margin_y) in mm - the margin sizes needed
            for labels
        """
        # Calculate font size and scaling
        font_size = 4.375
        font_scale = font_size / 2.5

        # Calculate margins based on font size
        # For now, use same margin for both axes (could differ in future)
        margin_x = 15.0 * font_scale
        margin_y = 15.0 * font_scale
        offset_x = margin_x
        offset_y = margin_y
        cols, rows = self.grid_dimensions
        cols, rows = int(cols), int(rows)  # Ensure integers for range()
        min_speed, max_speed = self.speed_range
        min_power, max_power = self.power_range

        speed_step = (
            (max_speed - min_speed) / (cols - 1) if cols > 1 else 0
        )
        power_step = (
            (max_power - min_power) / (rows - 1) if rows > 1 else 0
        )

        # Scale label power percentage to machine power range
        # Default to 1000 if laser is None (for testing/compatibility)
        max_power = laser.max_power if laser else 1000
        label_power_percent = 10.0
        label_power = (label_power_percent / 100.0) * max_power
        label_speed = 1000.0

        # Reduced from 5.0 to bring closer
        numeric_label_offset = 2.0 * font_scale
        axis_label_offset = 12.0 * font_scale

        # Main axis labels (swapped: X=power, Y=speed)
        x_axis_label = "power (%)"
        y_axis_label = "speed (mm/min)"

        # Create temp context for text extent measurements
        import math
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        temp_cr = cairo.Context(temp_surface)
        temp_cr.select_font_face(
            "Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )
        temp_cr.set_font_size(font_size)

        # X-axis label centered below
        extents = temp_cr.text_extents(x_axis_label)
        x_pos = grid_width / 2 - extents.width / 2 + offset_x
        y_pos = -(10.0 * font_scale) + offset_y
        self._text_to_ops(
            x_axis_label,
            x_pos,
            y_pos,
            font_size,
            "Sans",
            ops,
            label_power,
            label_speed,
        )

        # Numeric labels for (X-axis)
        for c in range(cols):
            current_power = min_power + c * power_step
            label_text = f"{int(current_power)}"
            extents = temp_cr.text_extents(label_text)
            # Center label horizontally in column
            x_pos = (
                c * (self.shape_size + self.spacing)
                + self.shape_size / 2 - extents.height / 2 + offset_x
            )
            # Position above grid, accounting for rotated text height
            y_pos = -numeric_label_offset + margin_y
            self._text_to_ops(
                label_text,
                x_pos,
                y_pos,
                font_size,
                "Sans",
                ops,
                label_power,
                label_speed,
                rotation=-math.pi / 2,  # -90 degrees (vertical)
            )

        # Y-axis label (left of grid, rotated 90 degrees counter-clockwise)
        # Position needs adjustment to account for rotation and text centering
        extents = temp_cr.text_extents(y_axis_label)

        x_pos = -axis_label_offset + offset_x
        # Y-axis label centered vertically
        grid_center_y = (
            (rows * (self.shape_size + self.spacing) - self.spacing) / 2
        )
        y_pos = grid_center_y + offset_y + extents.width / 2
        self._text_to_ops(
            y_axis_label,
            x_pos,
            y_pos,
            font_size,
            "Sans",
            ops,
            label_power,
            label_speed,
            rotation=-math.pi / 2,  # -90 degrees
        )

        # Numeric labels for speed (Y-axis) - swapped from power
        for r in range(rows):
            current_speed = min_speed + r * speed_step
            label_text = f"{int(current_speed)}"
            extents = temp_cr.text_extents(label_text)
            x_pos = -numeric_label_offset - extents.width + offset_x
            # Row center position
            row_center = (
                r * (self.shape_size + self.spacing) + self.shape_size / 2
            )
            y_pos = row_center + offset_y - extents.height / 2
            self._text_to_ops(
                label_text,
                x_pos,
                y_pos,
                font_size,
                "Sans",
                ops,
                label_power,
                label_speed,
            )

        return margin_x, margin_y

    def _text_to_ops(
        self,
        text: str,
        x: float,
        y: float,
        font_size: float,
        font_face: str,
        ops: Ops,
        power: float,
        speed: float,
        rotation: float = 0.0,
    ):
        """Converts text to vector ops using Cairo.

        Args:
            rotation: Rotation angle in radians (counter-clockwise)
        """
        import math

        # Create temporary surface for text path generation
        temp_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 1, 1)
        temp_cr = cairo.Context(temp_surface)
        temp_cr.select_font_face(
            font_face, cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL
        )
        temp_cr.set_font_size(font_size)

        # Generate text path
        temp_cr.new_path()
        temp_cr.move_to(0, 0)
        temp_cr.text_path(text)
        path_data = temp_cr.copy_path()

        ops.add(SetPowerCommand(power / 100.0))
        ops.add(SetCutSpeedCommand(int(speed)))

        # Scale factor to match show_text rendering size
        # Empirically determined to match the rendered text size
        path_scale = 1

        # Pre-compute rotation matrix if needed
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        # Convert path to ops
        # Track first point of current subpath for closing
        first_point = None
        current_point = None

        for path_type, points in path_data:  # type: ignore
            if path_type == cairo.PATH_MOVE_TO:
                px, py = points[0], points[1]
                # Scale the path coordinates
                px *= path_scale
                py *= path_scale
                # Flip Y coordinate (text is naturally upside down in geometry)
                py = -py
                # Apply rotation
                rx = px * cos_r - py * sin_r
                ry = px * sin_r + py * cos_r
                final_point = (x + rx, y + ry, 0.0)
                ops.add(MoveToCommand(final_point))
                first_point = final_point
                current_point = final_point
            elif path_type == cairo.PATH_LINE_TO:
                px, py = points[0], points[1]
                # Scale the path coordinates
                px *= path_scale
                py *= path_scale
                # Flip Y coordinate
                py = -py
                # Apply rotation
                rx = px * cos_r - py * sin_r
                ry = px * sin_r + py * cos_r
                final_point = (x + rx, y + ry, 0.0)
                ops.add(LineToCommand(final_point))
                current_point = final_point
            elif path_type == cairo.PATH_CURVE_TO:
                p1x, p1y, p2x, p2y, p3x, p3y = points
                # Scale the path coordinates
                p1x *= path_scale
                p1y *= path_scale
                p2x *= path_scale
                p2y *= path_scale
                p3x *= path_scale
                p3y *= path_scale
                # Flip Y coordinates
                p1y = -p1y
                p2y = -p2y
                p3y = -p3y
                # Apply rotation to all control points
                r1x = p1x * cos_r - p1y * sin_r
                r1y = p1x * sin_r + p1y * cos_r
                r2x = p2x * cos_r - p2y * sin_r
                r2y = p2x * sin_r + p2y * cos_r
                r3x = p3x * cos_r - p3y * sin_r
                r3y = p3x * sin_r + p3y * cos_r
                c1 = (x + r1x, y + r1y, 0.0)
                c2 = (x + r2x, y + r2y, 0.0)
                end = (x + r3x, y + r3y, 0.0)
                ops.bezier_to(c1, c2, end)
                current_point = end
            elif path_type == cairo.PATH_CLOSE_PATH:
                # Close the path by drawing a line back to the first point
                if first_point and current_point != first_point:
                    ops.add(LineToCommand(first_point))

    def _draw_rectangle(
        self, ops: Ops, x: float, y: float, width: float, height: float
    ):
        """Draws a rectangle outline to the ops."""
        ops.add(MoveToCommand((x, y, 0.0)))
        ops.add(LineToCommand((x + width, y, 0.0)))
        ops.add(LineToCommand((x + width, y + height, 0.0)))
        ops.add(LineToCommand((x, y + height, 0.0)))
        ops.add(LineToCommand((x, y, 0.0)))  # Close

    def _draw_filled_box(
        self, ops: Ops, x: float, y: float, width: float, height: float,
        line_spacing: float = 0.1
    ):
        """Draws a filled box with horizontal raster lines for engraving.

        Args:
            ops: The Ops object to add commands to
            x: X position of the box
            y: Y position of the box
            width: Width of the box
            height: Height of the box
            line_spacing: Spacing between raster lines in mm (default 0.1mm)
        """
        # Calculate number of lines needed
        num_lines = int(height / line_spacing) + 1

        # Draw horizontal lines from bottom to top
        for i in range(num_lines):
            y_pos = y + (i * line_spacing)
            if y_pos > y + height:
                y_pos = y + height

            # Alternate direction for efficiency (bidirectional raster)
            if i % 2 == 0:
                # Left to right
                ops.add(MoveToCommand((x, y_pos, 0.0)))
                ops.add(LineToCommand((x + width, y_pos, 0.0)))
            else:
                # Right to left
                ops.add(MoveToCommand((x + width, y_pos, 0.0)))
                ops.add(LineToCommand((x, y_pos, 0.0)))

    @property
    def requires_full_render(self) -> bool:
        """Material test doesn't need rendering."""
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
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
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MaterialTestGridProducer":
        """Deserializes a MaterialTestGridProducer from a dictionary."""
        params = data.get("params", {})
        return cls(
            test_type=MaterialTestGridType(
                params.get("test_type", MaterialTestGridType.CUT.value)
            ),
            speed_range=tuple(
                params.get("speed_range", (100.0, 500.0))
            ),
            power_range=tuple(
                params.get("power_range", (10.0, 100.0))
            ),
            grid_dimensions=tuple(
                params.get("grid_dimensions", (5, 5))
            ),
            shape_size=params.get("shape_size", 10.0),
            spacing=params.get("spacing", 2.0),
            include_labels=params.get("include_labels", True),
        )
