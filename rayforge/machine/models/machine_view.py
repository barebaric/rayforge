"""
Display-facing projection of a machine's coordinate space.

MachineSpace (in rayforge.machine.models.coordspace) describes the machine in
native coordinates -- how the machine "speaks". MachineView describes how
the bed is *shown* after the workspace rotation. Keeping these view-facing
properties here avoids mixing presentation concerns into the coordinate
model itself.
"""

from .coordspace import (
    MachineSpace,
    OriginCorner,
    WorkspaceOrientation,
)


class MachineView:
    """Display-facing projection of a MachineSpace into world space.

    The machine always reads native coordinates; these properties only
    drive rendering (axis labels, origin arrows, and the like). They are
    derived from a MachineSpace so that a rotated workspace presentation
    is reflected without the model itself carrying view logic.
    """

    def __init__(self, space: MachineSpace):
        self._space = space

    @property
    def origin(self) -> OriginCorner:
        """The native origin corner as it appears after the rotation."""
        orientation = self._space.workspace_orientation
        if orientation == WorkspaceOrientation.NATIVE:
            return self._space.origin
        if orientation == WorkspaceOrientation.ROTATED_LEFT:
            return {
                OriginCorner.BOTTOM_LEFT: OriginCorner.BOTTOM_RIGHT,
                OriginCorner.TOP_LEFT: OriginCorner.BOTTOM_LEFT,
                OriginCorner.TOP_RIGHT: OriginCorner.TOP_LEFT,
                OriginCorner.BOTTOM_RIGHT: OriginCorner.TOP_RIGHT,
            }[self._space.origin]
        return {
            OriginCorner.BOTTOM_LEFT: OriginCorner.TOP_LEFT,
            OriginCorner.TOP_LEFT: OriginCorner.TOP_RIGHT,
            OriginCorner.TOP_RIGHT: OriginCorner.BOTTOM_RIGHT,
            OriginCorner.BOTTOM_RIGHT: OriginCorner.BOTTOM_LEFT,
        }[self._space.origin]

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
        if self._space.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self._space.reverse_x
        return self._space.reverse_y

    @property
    def y_axis_negative(self) -> bool:
        """Whether the displayed Y axis reflects a reversed native axis."""
        if self._space.workspace_orientation == WorkspaceOrientation.NATIVE:
            return self._space.reverse_y
        return self._space.reverse_x
