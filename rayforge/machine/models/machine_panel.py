"""
Display-facing projection of a machine's coordinate space.

MachineSpace (in rayforge.machine.models.coordspace) describes the
machine in native coordinates -- how the machine "speaks".
MachinePanel describes how the bed is *shown* after the workspace
rotation. Keeping these view-facing properties here avoids mixing
presentation concerns into the coordinate model itself.
"""

from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo.types import Point, Point3D, Rect

from .coordspace import (
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)

if TYPE_CHECKING:
    from .machine import Machine


class MachinePanel:
    """Display-facing projection of a machine's coordinate space.

    The machine is the source of truth in machine coordinates. This
    panel holds a reference to the :class:`Machine` and derives a
    :class:`MachineSpace` from it on demand, so that a rotated workspace
    presentation is reflected without the model itself carrying view
    logic.
    """

    def __init__(self, machine: "Machine"):
        self._machine = machine

    @property
    def machine(self) -> "Machine":
        return self._machine

    @property
    def space(self) -> MachineSpace:
        """Current coordinate-space projection for the machine."""
        return self._machine.get_coordinate_space()

    # -- Display properties --------------------------------------------

    @property
    def origin(self) -> OriginCorner:
        """The native origin corner as it appears after the rotation."""
        orientation = self.space.workspace_orientation
        if orientation == WorkspaceOrientation.NATIVE:
            return self.space.origin
        if orientation == WorkspaceOrientation.ROTATED_LEFT:
            return {
                OriginCorner.BOTTOM_LEFT: OriginCorner.BOTTOM_RIGHT,
                OriginCorner.TOP_LEFT: OriginCorner.BOTTOM_LEFT,
                OriginCorner.TOP_RIGHT: OriginCorner.TOP_LEFT,
                OriginCorner.BOTTOM_RIGHT: OriginCorner.TOP_RIGHT,
            }[self.space.origin]
        return {
            OriginCorner.BOTTOM_LEFT: OriginCorner.TOP_LEFT,
            OriginCorner.TOP_LEFT: OriginCorner.TOP_RIGHT,
            OriginCorner.TOP_RIGHT: OriginCorner.BOTTOM_RIGHT,
            OriginCorner.BOTTOM_RIGHT: OriginCorner.BOTTOM_LEFT,
        }[self.space.origin]

    @property
    def x_axis_right(self) -> bool:
        """True when the displayed origin sits on the right (X increases
        toward the left)."""
        return self.origin in (
            OriginCorner.TOP_RIGHT,
            OriginCorner.BOTTOM_RIGHT,
        )

    @property
    def y_axis_down(self) -> bool:
        """True when the displayed origin sits at the top (Y increases
        downward)."""
        return self.origin in (
            OriginCorner.TOP_LEFT,
            OriginCorner.TOP_RIGHT,
        )

    @property
    def x_axis_negative(self) -> bool:
        """Whether the displayed X axis reflects a reversed native axis.

        Rotation swaps which native axis the displayed X corresponds to,
        so under rotation this tracks the native Y reversal rather than
        the native X reversal.
        """
        if self.space.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.space.reverse_x
        return self.space.reverse_y

    @property
    def y_axis_negative(self) -> bool:
        """Whether the displayed Y axis reflects a reversed native axis."""
        if self.space.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self.space.reverse_y
        return self.space.reverse_x

    @property
    def workspace_orientation(self) -> WorkspaceOrientation:
        """The workspace orientation of the underlying machine."""
        return self.space.workspace_orientation

    # -- Composed transforms (delegate to space) -----------------------

    def get_world_to_machine_matrix(self) -> np.ndarray:
        """Full world-to-machine matrix, including orientation rotation."""
        return self.space.get_world_to_machine_matrix()

    def get_machine_to_world_matrix(self) -> np.ndarray:
        """Inverse of get_world_to_machine_matrix()."""
        return self.space.get_machine_to_world_matrix()

    def world_point_to_machine(self, x: float, y: float) -> Point:
        """Transform a point from world space to machine space."""
        return self.space.world_point_to_machine(x, y)

    def machine_point_to_world(self, x: float, y: float) -> Point:
        """Transform a point from machine space to world space."""
        return self.space.machine_point_to_world(x, y)

    def world_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from world space to machine space."""
        return self.space.world_item_to_machine(pos, size)

    def machine_item_to_world(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from machine space to world space."""
        return self.space.machine_item_to_world(pos, size)

    def get_workarea_world_rect(self) -> Rect:
        """Work area boundary as a Rect in world space."""
        return self.space.get_workarea_world_rect()

    @property
    def workarea_size(self) -> tuple[float, float]:
        """The (width, height) of the workarea in mm."""
        return self.space.workarea_size

    def world_position_from_origin(
        self,
        ref_x: float,
        ref_y: float,
        size: tuple[float, float],
    ) -> Point:
        """Convert a reference position at the origin corner to world
        coordinates."""
        return self.space.world_position_from_origin(ref_x, ref_y, size)

    def get_axis_label_origin(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Point3D:
        """Origin offset for axis labels."""
        return self.space.get_axis_label_origin(
            wcs_offset, wcs_is_workarea_origin
        )

    def get_workarea_origin_in_machine(self) -> Point:
        """Position of the workarea origin in machine coordinates."""
        return self.space.get_workarea_origin_in_machine()

    def get_command_offset(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Point3D:
        """Offset to subtract from machine coordinates to obtain command
        coordinates (G-code output)."""
        return self.space.get_command_offset(
            wcs_offset, wcs_is_workarea_origin
        )

    @property
    def reference_position_world(self) -> Point:
        """The reference origin position in world coordinates.

        Combines the machine's reference offset (WCS or workarea origin,
        in machine coordinates) with the machine→world transform.
        """
        offset_x, offset_y, _ = self._machine.get_reference_offset()
        return self.machine_point_to_world(offset_x, offset_y)
