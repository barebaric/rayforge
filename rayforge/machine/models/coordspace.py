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
- PANEL: Presented view of the canvas (WORLD rotated by the panel
  orientation). The document model only knows WORLD; the canvas
  presents PANEL.
"""

from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo.types import Point, Point3D, Rect

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
        self, extents: tuple[float, float]
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
        self, x: float, y: float, extents: tuple[float, float]
    ) -> Point:
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
class MachineSpace(CoordinateSpace):
    """
    The machine's native coordinate system.

    Configured based on machine settings (origin corner, axis directions).
    Used for G-code generation and machine communication.

    Attributes:
        extents: The (width, height) of the machine bed in mm.
        margins: The (left, top, right, bottom) margins in mm.
    """

    extents: tuple[float, float] = (200.0, 200.0)
    margins: Rect = (0.0, 0.0, 0.0, 0.0)

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

    def get_world_to_machine_matrix(self) -> np.ndarray:
        """
        Returns the 4x4 transformation matrix to convert from world space
        to machine space for the encoding pipeline.

        This composes the origin corner transformation and the axis
        reversal sign flips.
        """
        matrix = self.get_transform_to_world(self.extents)

        if self.reverse_x or self.reverse_y:
            sign_flip = np.identity(4, dtype=np.float64)
            if self.reverse_x:
                sign_flip[0, 0] = -1.0
            if self.reverse_y:
                sign_flip[1, 1] = -1.0
            matrix = sign_flip @ matrix

        return matrix

    def get_machine_to_command_matrix(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> np.ndarray:
        """
        Returns the 4x4 transformation matrix to convert from machine
        space to command space (G-code output).

        Command coordinates are machine coordinates minus the command
        offset returned by get_command_offset(), so that the controller
        lands at the machine position when it adds the active WCS offset
        back on its end.
        """
        off_x, off_y, _ = self.get_command_offset(
            wcs_offset=wcs_offset,
            wcs_is_workarea_origin=wcs_is_workarea_origin,
        )
        matrix = np.identity(4, dtype=np.float64)
        matrix[0, 3] = -off_x
        matrix[1, 3] = -off_y
        return matrix

    def get_machine_to_world_matrix(self) -> np.ndarray:
        """
        Returns the inverse of get_world_to_machine_matrix().

        Used to convert points from machine space back to world space.
        """
        return np.linalg.inv(self.get_world_to_machine_matrix())

    def get_command_offset(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Point3D:
        """
        Calculates the offset to subtract from machine coordinates to obtain
        command coordinates (G-code output).
        """
        if wcs_is_workarea_origin:
            ml, mt, mr, mb = self.margins

            origin_is_right = self.origin in (
                OriginCorner.TOP_RIGHT,
                OriginCorner.BOTTOM_RIGHT,
            )
            origin_is_top = self.origin in (
                OriginCorner.TOP_LEFT,
                OriginCorner.TOP_RIGHT,
            )

            if origin_is_right:
                x_offset = -mr if self.reverse_x else mr
            else:
                x_offset = -ml if self.reverse_x else ml

            if origin_is_top:
                y_offset = -mt if self.reverse_y else mt
            else:
                y_offset = -mb if self.reverse_y else mb

            return (float(x_offset), float(y_offset), 0.0)
        else:
            return (wcs_offset[0], wcs_offset[1], 0.0)

    @property
    def workarea_size(self) -> tuple[float, float]:
        """Returns the (width, height) of the workarea in mm."""
        ml, mt, mr, mb = self.margins
        width, height = self.extents
        return width - ml - mr, height - mt - mb

    def get_workarea_world_rect(self) -> Rect:
        """
        Returns the work area boundary as a Rect in world space.
        """
        pos = self.get_workarea_origin_in_machine()
        w, h = self.workarea_size
        wx, wy = self.machine_item_to_world(pos, (w, h))
        return (wx, wy, w, h)

    def world_position_from_origin(
        self, ref_x: float, ref_y: float, size: tuple[float, float]
    ) -> Point:
        """
        Convert a reference position at the origin corner to world coords.

        Given a reference point at the machine's origin corner and an item
        size, returns the bottom-left position in world coordinates.
        This is useful for positioning items in world space when you have
        a reference point at the origin corner.

        Args:
            ref_x: X coordinate of reference point at origin corner (world).
            ref_y: Y coordinate of reference point at origin corner (world).
            size: (width, height) of the item.

        Returns:
            Tuple of (x, y) for bottom-left position in world coordinates.
        """
        width, height = size

        if self.origin == OriginCorner.BOTTOM_LEFT:
            return ref_x, ref_y
        elif self.origin == OriginCorner.TOP_LEFT:
            return ref_x, ref_y - height
        elif self.origin == OriginCorner.BOTTOM_RIGHT:
            return ref_x - width, ref_y
        else:  # TOP_RIGHT
            return ref_x - width, ref_y - height

    def get_workarea_origin_in_machine(
        self,
    ) -> Point:
        """
        Returns the position of the workarea origin in machine coordinates.

        The workarea origin is at the corner specified by the machine's
        origin setting, offset by the margins.
        """
        ml, mt, mr, mb = self.margins
        _width, _height = self.extents

        origin_is_top = self.origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )
        origin_is_right = self.origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )

        if origin_is_right:
            x = mr
        else:
            x = ml

        if origin_is_top:
            y = mt
        else:
            y = mb

        return x, y

    def get_axis_label_origin(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Point3D:
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
            _width, _height = self.extents

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

    def world_point_to_machine(self, x: float, y: float) -> Point:
        """
        Transform a point from world space to machine space.

        WORLD space: Bottom-Left (0,0), Y-up, X-right
        MACHINE space: Based on origin corner and axis reversal settings.

        Delegates to get_world_to_machine_matrix() so the scalar UI path
        and the matrix encoder path share a single source of truth.

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Tuple of (x, y) in machine space.
        """
        matrix = self.get_world_to_machine_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def machine_point_to_world(self, x: float, y: float) -> Point:
        """
        Transform a point from machine space to world space.

        Inverse of world_point_to_machine().

        Delegates to get_machine_to_world_matrix() so the scalar UI path
        and the matrix encoder path share a single source of truth.

        Args:
            x: X coordinate in machine space.
            y: Y coordinate in machine space.

        Returns:
            Tuple of (x, y) in world space.
        """
        matrix = self.get_machine_to_world_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def world_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """
        Convert item position from world space to machine space.

        Transforms the item's four bounding-box corners through
        world_point_to_machine and selects the corner nearest the machine
        origin (min, or max when the axis is reversed). Only this corner
        selection depends on the bounding box; the point transform itself
        delegates to the single matrix path.

        Args:
            pos: (x, y) position in world coordinates (top-left corner).
            size: (width, height) of the item.

        Returns:
            (x, y) position in machine coordinates.
        """
        wx, wy = pos
        w, h = size
        corners = (
            self.world_point_to_machine(wx, wy),
            self.world_point_to_machine(wx + w, wy),
            self.world_point_to_machine(wx, wy + h),
            self.world_point_to_machine(wx + w, wy + h),
        )
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        mx = max(xs) if self.reverse_x else min(xs)
        my = max(ys) if self.reverse_y else min(ys)
        return mx, my

    def machine_item_to_world(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """
        Convert item position from machine space to world space.

        The machine position refers to the corner nearest the machine
        origin; the opposite corner is reached by adding (or, when the
        axis is reversed, subtracting) the item size. All four bounding-
        box corners are then transformed through machine_point_to_world,
        and the world-space top-left is the per-axis minimum.

        Args:
            pos: (x, y) position in machine coordinates.
            size: (width, height) of the item in world space.

        Returns:
            (x, y) position in world coordinates.
        """
        mx, my = pos
        w, h = size
        if self.reverse_x:
            x_min, x_max = mx - w, mx
        else:
            x_min, x_max = mx, mx + w
        if self.reverse_y:
            y_min, y_max = my - h, my
        else:
            y_min, y_max = my, my + h
        corners = (
            self.machine_point_to_world(x_min, y_min),
            self.machine_point_to_world(x_max, y_min),
            self.machine_point_to_world(x_min, y_max),
            self.machine_point_to_world(x_max, y_max),
        )
        return min(c[0] for c in corners), min(c[1] for c in corners)
