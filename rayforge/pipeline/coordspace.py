"""
Coordinate Space Classes.

This module defines explicit coordinate space types for handling
coordinate transformations throughout Rayforge.

Coordinate Spaces:
- WORLD: Canonical internal space (bottom-left origin, Y-up, X-right)
- MACHINE: Physical machine bed (origin and axis directions vary by config)
- WORKAREA: Usable area within machine bed (defined by margins)
- PIXEL: Raster images (top-left origin, Y-down)
- COMMAND: G-code output (relative to active WCS or workarea origin)
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING, Tuple

import numpy as np

if TYPE_CHECKING:
    from rayforge.machine.models.machine import Machine


class OriginCorner(Enum):
    """Origin corner for a coordinate system."""

    BOTTOM_LEFT = "bottom_left"
    BOTTOM_RIGHT = "bottom_right"
    TOP_LEFT = "top_left"
    TOP_RIGHT = "top_right"


class AxisDirection(Enum):
    """Direction of axis positive movement."""

    POSITIVE_RIGHT = auto()
    POSITIVE_LEFT = auto()
    POSITIVE_UP = auto()
    POSITIVE_DOWN = auto()


@dataclass(frozen=True)
class CoordinateSpace(ABC):
    """
    Base class for coordinate spaces.

    Defines the geometric properties of a coordinate system and provides
    transformation methods to convert between spaces.
    """

    origin: OriginCorner
    x_positive_direction: AxisDirection
    y_positive_direction: AxisDirection
    reverse_x: bool = False
    reverse_y: bool = False

    @property
    def x_reversed(self) -> bool:
        """True if X axis positive direction is left."""
        return self.x_positive_direction == AxisDirection.POSITIVE_LEFT

    @property
    def y_reversed(self) -> bool:
        """True if Y axis positive direction is down."""
        return self.y_positive_direction == AxisDirection.POSITIVE_DOWN

    def get_transform_to_world(
        self, extents: Tuple[float, float]
    ) -> np.ndarray:
        """
        Returns the 4x4 transformation matrix to convert from this space
        to world space (BOTTOM_LEFT origin, Y-up, X-right).

        This handles origin corner transformation based on axis directions,
        plus reverse_x/reverse_y sign flipping for machine coordinates.

        Args:
            extents: The (width, height) of the coordinate space.

        Returns:
            A 4x4 numpy array representing the transformation matrix.
        """
        width, height = extents

        origin_is_top = self.origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )
        origin_is_right = self.origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )

        # Build origin corner transformation
        origin_transform = np.identity(4, dtype=np.float64)

        # Y-axis transformation
        if origin_is_top:
            if self.y_reversed:
                # Top origin with Y-down
                origin_transform[1, 1] = -1.0
                origin_transform[1, 3] = height
            else:
                # Top origin with Y-up
                origin_transform[1, 3] = -height
        elif self.y_reversed:
            # Bottom origin with Y-down
            origin_transform[1, 1] = -1.0

        # X-axis transformation
        if origin_is_right:
            if self.x_reversed:
                # Right origin with X-left: x' = -x + width
                origin_transform[0, 0] = -1.0
                origin_transform[0, 3] = width
            else:
                # Right origin with X-right: x' = width - x
                origin_transform[0, 0] = -1.0
                origin_transform[0, 3] = width
        elif self.x_reversed:
            # Left origin with X-left
            origin_transform[0, 0] = -1.0

        return origin_transform

    def transform_point_to_world(
        self, x: float, y: float, extents: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Transform a point from this space to world space.

        Args:
            x: X coordinate in this space.
            y: Y coordinate in this space.
            extents: The (width, height) of the coordinate space.

        Returns:
            Tuple of (x, y) in world space.
        """
        matrix = self.get_transform_to_world(extents)
        point = np.array([x, y, 0.0, 1.0])
        result = matrix @ point
        return float(result[0]), float(result[1])


@dataclass(frozen=True)
class WorldSpace(CoordinateSpace):
    """
    The canonical world coordinate system.

    - Origin: Bottom-left
    - X: Right is positive
    - Y: Up is positive
    - Unit: Millimeters

    This is the internal representation used for workpiece positions
    and all geometric operations in the document model.
    """

    origin: OriginCorner = OriginCorner.BOTTOM_LEFT
    x_positive_direction: AxisDirection = AxisDirection.POSITIVE_RIGHT
    y_positive_direction: AxisDirection = AxisDirection.POSITIVE_UP


@dataclass(frozen=True)
class MachineSpace(CoordinateSpace):
    """
    The machine's native coordinate system.

    Configured based on machine settings (origin corner, axis directions).
    Used for G-code generation and machine communication.

    Attributes:
        extents: The (width, height) of the machine bed in mm.
        margins: The (left, top, right, bottom) margins in mm.
    """

    extents: Tuple[float, float] = (200.0, 200.0)
    margins: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @classmethod
    def from_machine(cls, machine: "Machine") -> "MachineSpace":
        """
        Create a MachineSpace from a Machine configuration.

        Args:
            machine: The machine to create the space from.

        Returns:
            A MachineSpace instance matching the machine's configuration.
        """
        from rayforge.machine.models.machine import Origin

        origin_map = {
            Origin.BOTTOM_LEFT: OriginCorner.BOTTOM_LEFT,
            Origin.BOTTOM_RIGHT: OriginCorner.BOTTOM_RIGHT,
            Origin.TOP_LEFT: OriginCorner.TOP_LEFT,
            Origin.TOP_RIGHT: OriginCorner.TOP_RIGHT,
        }

        origin_corner = origin_map.get(
            machine.origin, OriginCorner.BOTTOM_LEFT
        )

        y_down = machine.origin in (Origin.TOP_LEFT, Origin.TOP_RIGHT)
        x_right = machine.origin in (Origin.TOP_RIGHT, Origin.BOTTOM_RIGHT)

        # Axis direction is based on origin position only.
        # reverse_x/reverse_y are stored separately for controller
        # sign flip handling in _machine_coords_to_canvas() and encoder.
        x_dir = (
            AxisDirection.POSITIVE_LEFT
            if x_right
            else AxisDirection.POSITIVE_RIGHT
        )

        y_dir = (
            AxisDirection.POSITIVE_DOWN
            if y_down
            else AxisDirection.POSITIVE_UP
        )

        return cls(
            origin=origin_corner,
            x_positive_direction=x_dir,
            y_positive_direction=y_dir,
            extents=machine.axis_extents,
            margins=machine.work_margins,
            reverse_x=machine.reverse_x_axis,
            reverse_y=machine.reverse_y_axis,
        )

    @property
    def workarea_size(self) -> Tuple[float, float]:
        """Returns the (width, height) of the workarea in mm."""
        ml, mt, mr, mb = self.margins
        width, height = self.extents
        return width - ml - mr, height - mt - mb

    def get_workarea_origin_in_machine(
        self,
    ) -> Tuple[float, float]:
        """
        Returns the position of the workarea origin in machine coordinates.

        The workarea origin is at the corner specified by the machine's
        origin setting, offset by the margins.
        """
        ml, mt, mr, mb = self.margins
        width, height = self.extents

        origin_is_top = self.origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )
        origin_is_right = self.origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )

        if origin_is_right:
            x = width - mr
        else:
            x = ml

        if origin_is_top:
            y = height - mt
        else:
            y = mb

        return x, y

    def get_axis_label_origin(
        self,
        wcs_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Get the origin offset for axis labels.

        This computes the (x, y, z) offset that should be passed to the
        axis renderer for drawing grid labels.

        Args:
            wcs_offset: The (x, y, z) WCS offset.
            wcs_is_workarea_origin: If True, workarea origin is coordinate
                zero.

        Returns:
            Tuple of (x, y, z) origin offset for axis labels.
        """
        if wcs_is_workarea_origin:
            ml, mt, mr, mb = self.margins
            width, height = self.extents

            origin_is_right = self.origin in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            origin_is_top = self.origin in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )

            if origin_is_right:
                origin_x = mr
            else:
                origin_x = ml

            if origin_is_top:
                origin_y = mt
            else:
                origin_y = mb

            if self.reverse_x:
                origin_x = -origin_x
            if self.reverse_y:
                origin_y = -origin_y

            return (origin_x, origin_y, 0.0)
        else:
            return wcs_offset

    def to_command_coords(
        self,
        machine_x: float,
        machine_y: float,
        wcs_offset: Tuple[float, float, float],
        wcs_is_workarea_origin: bool = False,
    ) -> Tuple[float, float]:
        """
        Convert machine coordinates to command coordinates (G-code output).

        Command coordinates are relative to the active WCS or workarea origin.

        Args:
            machine_x: X coordinate in machine space.
            machine_y: Y coordinate in machine space.
            wcs_offset: The (x, y, z) WCS offset.
            wcs_is_workarea_origin: If True, workarea origin is
                coordinate zero.

        Returns:
            Tuple of (x, y) in command space.
        """
        if wcs_is_workarea_origin:
            workarea_origin = self.get_workarea_origin_in_machine()
            return (
                machine_x - workarea_origin[0],
                machine_y - workarea_origin[1],
            )
        else:
            return machine_x - wcs_offset[0], machine_y - wcs_offset[1]


@dataclass(frozen=True)
class WorkareaSpace(CoordinateSpace):
    """
    The workarea coordinate system.

    A subset of the machine bed defined by margins. This is where
    actual work happens.

    The workarea space has the same orientation as the machine space
    but is offset by the margins.
    """

    extents: Tuple[float, float] = (200.0, 200.0)

    @classmethod
    def from_machine(cls, machine: "Machine") -> "WorkareaSpace":
        """
        Create a WorkareaSpace from a Machine configuration.

        Args:
            machine: The machine to create the space from.

        Returns:
            A WorkareaSpace instance matching the machine's workarea.
        """
        machine_space = MachineSpace.from_machine(machine)
        workarea_size = machine_space.workarea_size

        return cls(
            origin=machine_space.origin,
            x_positive_direction=machine_space.x_positive_direction,
            y_positive_direction=machine_space.y_positive_direction,
            extents=workarea_size,
        )


@dataclass(frozen=True)
class PixelSpace(CoordinateSpace):
    """
    Pixel coordinate space for raster images.

    - Origin: Top-left
    - X: Right is positive
    - Y: Down is positive
    - Unit: Pixels

    Attributes:
        dimensions: The (width, height) in pixels.
    """

    dimensions: Tuple[int, int] = (100, 100)
    origin: OriginCorner = OriginCorner.TOP_LEFT
    x_positive_direction: AxisDirection = AxisDirection.POSITIVE_RIGHT
    y_positive_direction: AxisDirection = AxisDirection.POSITIVE_DOWN

    def to_millimeters(
        self,
        x: float,
        y: float,
        mm_dimensions: Tuple[float, float],
    ) -> Tuple[float, float]:
        """
        Convert pixel coordinates to millimeter coordinates.

        Converts from pixel space (top-left origin, Y-down) to
        world space (bottom-left origin, Y-up).

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            mm_dimensions: The (width, height) in millimeters.

        Returns:
            Tuple of (x, y) in millimeters (world space orientation).
        """
        px_width, px_height = self.dimensions
        mm_width, mm_height = mm_dimensions

        mm_x = (x / px_width) * mm_width
        mm_y = mm_height - (y / px_height) * mm_height

        return mm_x, mm_y

    def transform_point_to_world(
        self, x: float, y: float, extents: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Transform a point from pixel space to world space.

        For pixel space, this delegates to to_millimeters().

        Args:
            x: X coordinate in pixels.
            y: Y coordinate in pixels.
            extents: The (width, height) in millimeters.

        Returns:
            Tuple of (x, y) in world space.
        """
        return self.to_millimeters(x, y, extents)
