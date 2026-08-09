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
from functools import cached_property
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


class WorkspaceOrientation(Enum):
    """How the native machine bed is presented in the workspace."""

    NATIVE = "native"
    ROTATED_LEFT = "rotated_left"
    ROTATED_RIGHT = "rotated_right"


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

    @property
    def workspace_origin(self) -> OriginCorner:
        """The origin corner as presented in the workspace.

        The base implementation is the identity: spaces without a
        presentation rotation show their native origin. MachineSpace
        overrides this to account for workspace_orientation.
        """
        return self.origin

    @property
    def workspace_x_negative(self) -> bool:
        """Whether displayed workspace X maps to a reversed axis."""
        return self.reverse_x

    @property
    def workspace_y_negative(self) -> bool:
        """Whether displayed workspace Y maps to a reversed axis."""
        return self.reverse_y

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
        workspace_orientation: How the native bed is rotated in the editor.
    """

    extents: tuple[float, float] = (200.0, 200.0)
    margins: Rect = (0.0, 0.0, 0.0, 0.0)
    workspace_orientation: WorkspaceOrientation = WorkspaceOrientation.NATIVE

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
        # reverse_x/reverse_y are stored separately for controller coordinate
        # conversion and encoding.
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
            workspace_orientation=machine.workspace_orientation,
        )

    @property
    def workspace_extents(self) -> tuple[float, float]:
        """Returns the dimensions presented in world/workspace space."""
        if self.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.extents
        return self.extents[1], self.extents[0]

    @property
    def workspace_margins(self) -> Rect:
        """Returns native edge margins rotated into workspace order."""
        left, top, right, bottom = self.margins
        if self.workspace_orientation == WorkspaceOrientation.ROTATED_LEFT:
            return top, right, bottom, left
        if self.workspace_orientation == WorkspaceOrientation.ROTATED_RIGHT:
            return bottom, left, top, right
        return self.margins

    @property
    def workspace_origin(self) -> OriginCorner:
        """Returns the native machine origin's visible workspace corner."""
        if self.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.origin
        if self.workspace_orientation == WorkspaceOrientation.ROTATED_LEFT:
            return {
                OriginCorner.BOTTOM_LEFT: OriginCorner.BOTTOM_RIGHT,
                OriginCorner.TOP_LEFT: OriginCorner.BOTTOM_LEFT,
                OriginCorner.TOP_RIGHT: OriginCorner.TOP_LEFT,
                OriginCorner.BOTTOM_RIGHT: OriginCorner.TOP_RIGHT,
            }[self.origin]
        return {
            OriginCorner.BOTTOM_LEFT: OriginCorner.TOP_LEFT,
            OriginCorner.TOP_LEFT: OriginCorner.TOP_RIGHT,
            OriginCorner.TOP_RIGHT: OriginCorner.BOTTOM_RIGHT,
            OriginCorner.BOTTOM_RIGHT: OriginCorner.BOTTOM_LEFT,
        }[self.origin]

    @property
    def workspace_x_negative(self) -> bool:
        """Whether displayed workspace X maps to a reversed native axis."""
        if self.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.reverse_x
        return self.reverse_y

    @property
    def workspace_y_negative(self) -> bool:
        """Whether displayed workspace Y maps to a reversed native axis."""
        if self.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.reverse_y
        return self.reverse_x

    @cached_property
    def _workspace_to_native_matrix(self) -> np.ndarray:
        """Cached rigid transform from workspace to native bed space.

        Safe to cache because the dataclass is frozen. Do not mutate the
        returned array; public callers get a copy via
        :meth:`get_workspace_to_native_matrix`.
        """
        width, height = self.extents
        matrix = np.identity(4, dtype=np.float64)
        if self.workspace_orientation == WorkspaceOrientation.ROTATED_LEFT:
            matrix[0, 0] = 0.0
            matrix[0, 1] = 1.0
            matrix[1, 0] = -1.0
            matrix[1, 1] = 0.0
            matrix[1, 3] = height
        elif self.workspace_orientation == WorkspaceOrientation.ROTATED_RIGHT:
            matrix[0, 0] = 0.0
            matrix[0, 1] = -1.0
            matrix[0, 3] = width
            matrix[1, 0] = 1.0
            matrix[1, 1] = 0.0
        return matrix

    @cached_property
    def _native_to_workspace_matrix(self) -> np.ndarray:
        """Cached inverse of the workspace-to-native transform."""
        return np.linalg.inv(self._workspace_to_native_matrix)

    def get_workspace_to_native_matrix(self) -> np.ndarray:
        """Return the rigid transform from workspace to native bed space."""
        return self._workspace_to_native_matrix.copy()

    def get_native_to_workspace_matrix(self) -> np.ndarray:
        """Return the rigid transform from native bed to workspace space."""
        return self._native_to_workspace_matrix.copy()

    def native_bed_point_to_workspace(self, x: float, y: float) -> Point:
        """Project a physical native-bed point into the workspace.

        Unlike :meth:`machine_point_to_world`, this applies only the
        presentation rotation. It is intended for physical bed geometry such
        as no-go zones, whose stored positions do not depend on the configured
        controller origin or axis direction.
        """
        point = np.array([x, y, 0.0, 1.0], dtype=np.float64)
        result = self._native_to_workspace_matrix @ point
        return float(result[0]), float(result[1])

    def workspace_point_to_native_bed(self, x: float, y: float) -> Point:
        """Project a workspace point into canonical physical-bed space."""
        point = np.array([x, y, 0.0, 1.0], dtype=np.float64)
        result = self._workspace_to_native_matrix @ point
        return float(result[0]), float(result[1])

    def native_bed_item_to_workspace(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> tuple[Point, tuple[float, float]]:
        """Project an axis-aligned native-bed rectangle into the workspace."""
        x, y = pos
        width, height = size
        corners = (
            self.native_bed_point_to_workspace(x, y),
            self.native_bed_point_to_workspace(x + width, y),
            self.native_bed_point_to_workspace(x, y + height),
            self.native_bed_point_to_workspace(x + width, y + height),
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        min_x, min_y = min(xs), min(ys)
        return (min_x, min_y), (max(xs) - min_x, max(ys) - min_y)

    def get_transform_from_world(self) -> np.ndarray:
        """
        Returns the inverse origin transformation matrix (world → machine).

        Full machine output conversion, including workspace orientation and
        controller axis reversals, is provided by
        :meth:`get_world_to_machine_matrix`.

        Returns:
            A 4x4 numpy array representing the inverse transformation matrix.
        """
        return np.linalg.inv(super().get_transform_to_world(self.extents))

    @cached_property
    def _world_to_machine_matrix(self) -> np.ndarray:
        """Cached full world-to-machine transform. Do not mutate."""
        native_world_to_machine = np.linalg.inv(
            super().get_transform_to_world(self.extents)
        )
        matrix = native_world_to_machine @ self._workspace_to_native_matrix

        if self.reverse_x or self.reverse_y:
            sign_flip = np.identity(4, dtype=np.float64)
            if self.reverse_x:
                sign_flip[0, 0] = -1.0
            if self.reverse_y:
                sign_flip[1, 1] = -1.0
            matrix = sign_flip @ matrix

        return matrix

    @cached_property
    def _machine_to_world_matrix(self) -> np.ndarray:
        """Cached inverse of the world-to-machine transform."""
        return np.linalg.inv(self._world_to_machine_matrix)

    def get_world_to_machine_matrix(self) -> np.ndarray:
        """
        Returns the full 4x4 transformation matrix to convert from world space
        to machine space for the encoding pipeline.

        This combines workspace rotation, the origin corner transformation,
        and the axis reversal sign flips.
        """
        return self._world_to_machine_matrix.copy()

    def get_machine_to_world_matrix(self) -> np.ndarray:
        """Return the full inverse native-machine to workspace transform."""
        return self._machine_to_world_matrix.copy()

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
        """Returns the workspace (width, height) of the workarea in mm.

        Each side is clamped to a minimum of 1mm, mirroring
        `Machine.work_area`. This property backs `workspace_work_area`,
        which replaced `Machine.work_area` at its call sites, so the two
        must agree: margins can reach or exceed the extents (device
        profiles pass YAML margins through unvalidated), and downstream
        layout and 3D viewport code assumes a positive work area.
        """
        ml, mt, mr, mb = self.workspace_margins
        width, height = self.workspace_extents
        return (
            max(1.0, width - ml - mr),
            max(1.0, height - mt - mb),
        )

    def get_workarea_world_rect(self) -> Rect:
        """
        Returns the work area boundary as a Rect in world space.
        """
        ml, _, _, mb = self.workspace_margins
        w, h = self.workarea_size
        return float(ml), float(mb), float(w), float(h)

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

        origin = self.workspace_origin
        if origin == OriginCorner.BOTTOM_LEFT:
            return ref_x, ref_y
        elif origin == OriginCorner.TOP_LEFT:
            return ref_x, ref_y - height
        elif origin == OriginCorner.BOTTOM_RIGHT:
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
            machine_x, machine_y = self.get_workarea_origin_in_machine()
        else:
            machine_x, machine_y = wcs_offset[0], wcs_offset[1]

        world_x, world_y = self.machine_point_to_world(machine_x, machine_y)
        width, height = self.workspace_extents
        origin = self.workspace_origin
        origin_is_right = origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )
        origin_is_top = origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )
        offset_x = width - world_x if origin_is_right else world_x
        offset_y = height - world_y if origin_is_top else world_y
        if self.workspace_x_negative:
            offset_x = -offset_x
        if self.workspace_y_negative:
            offset_y = -offset_y
        return float(offset_x), float(offset_y), 0.0

    def world_point_to_machine(self, x: float, y: float) -> Point:
        """
        Transform a point from world space to machine space.

        WORLD space: Bottom-Left (0,0), Y-up, X-right
        MACHINE space: Based on origin corner and axis reversal settings.

        This applies:
        1. Workspace-to-native bed rotation
        2. Native origin corner transformation
        3. Axis reversal sign flip (reverse_x/reverse_y)

        Args:
            x: X coordinate in world space.
            y: Y coordinate in world space.

        Returns:
            Tuple of (x, y) in machine space.
        """
        point = np.array([x, y, 0.0, 1.0], dtype=np.float64)
        result = self._world_to_machine_matrix @ point
        return float(result[0]), float(result[1])

    def machine_point_to_world(self, x: float, y: float) -> Point:
        """
        Transform a point from machine space to world space.

        Inverse of world_point_to_machine().

        This reverses the axis sign, native origin, and workspace rotation
        transformations applied by world_point_to_machine().

        Args:
            x: X coordinate in machine space.
            y: Y coordinate in machine space.

        Returns:
            Tuple of (x, y) in world space.
        """
        point = np.array([x, y, 0.0, 1.0], dtype=np.float64)
        result = self._machine_to_world_matrix @ point
        return float(result[0]), float(result[1])

    def world_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """
        Convert item position from world space to machine space.

        This handles the item's bounding box - for origins at right/bottom,
        the position refers to the top-left corner, which needs adjustment.

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
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
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

        This handles the item's bounding box - for origins at right/bottom,
        the position refers to the top-left corner, which needs adjustment.

        NOTE on spaces: `pos` is in MACHINE coordinates, but `size` is the
        item's (width, height) in WORLD/workspace space. With a rotated
        workspace orientation the two differ (width and height swap), so
        callers must not pass a machine-space size here.

        Args:
            pos: (x, y) position in machine coordinates.
            size: (width, height) of the item in world/workspace space.

        Returns:
            (x, y) position in world coordinates.
        """
        mx, my = pos
        w, h = size
        if self.workspace_orientation == WorkspaceOrientation.NATIVE:
            machine_w, machine_h = w, h
        else:
            machine_w, machine_h = h, w

        if self.reverse_x:
            x_min, x_max = mx - machine_w, mx
        else:
            x_min, x_max = mx, mx + machine_w
        if self.reverse_y:
            y_min, y_max = my - machine_h, my
        else:
            y_min, y_max = my, my + machine_h

        corners = (
            self.machine_point_to_world(x_min, y_min),
            self.machine_point_to_world(x_max, y_min),
            self.machine_point_to_world(x_min, y_max),
            self.machine_point_to_world(x_max, y_max),
        )
        return min(p[0] for p in corners), min(p[1] for p in corners)
