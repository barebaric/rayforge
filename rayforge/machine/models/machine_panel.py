"""
Display-facing projection of a machine's coordinate space.

MachineSpace (in rayforge.machine.models.coordspace) describes the
machine in native coordinates -- how the machine "speaks".
MachinePanel describes how the bed is *shown* after the workspace
rotation. Keeping these view-facing properties here avoids mixing
presentation concerns into the coordinate model itself.
"""

from typing import TYPE_CHECKING

from .coordspace import (
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)

if TYPE_CHECKING:
    from .machine import Machine


class MachinePanel:
    """Display-facing projection of a machine's coordinate space.

    The machine is the source of truth in native bed coordinates.
    This panel holds a reference to the :class:`Machine` and derives
    a :class:`MachineSpace` from it on demand, so that a rotated
    workspace presentation is reflected without the model itself
    carrying view logic.
    """

    def __init__(self, machine: Machine):
        self._machine = machine

    @property
    def machine(self) -> Machine:
        return self._machine

    @property
    def space(self) -> MachineSpace:
        """Current coordinate-space projection for the machine."""
        return self._machine.get_coordinate_space()

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
