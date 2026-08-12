"""
Display-facing projection of a machine's coordinate space.

MachineSpace (in rayforge.machine.models.coordspace) describes the
machine in native coordinates -- how the machine "speaks".
MachinePanel describes how the bed is *shown* after an optional
90-degree rotation. Keeping these view-facing properties here avoids
mixing presentation concerns into the coordinate model itself.
"""

from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo.types import Point, Point3D, Rect

from .coordspace import (
    MachineSpace,
    OriginCorner,
)

if TYPE_CHECKING:
    from .machine import Machine


class PanelOrientation(Enum):
    """How the native machine bed is presented on screen.

    The machine always reads and writes native coordinates; this only
    controls whether the bed is rotated 90 degrees when projected into
    world space (e.g. so a physically portrait bed can be edited in
    landscape). NATIVE leaves the mapping untouched.
    """

    NATIVE = "native"
    ROTATED_LEFT = "rotated_left"
    ROTATED_RIGHT = "rotated_right"


class MachinePanel:
    """Display-facing projection of a machine's coordinate space.

    The machine is the source of truth in machine coordinates. This
    panel holds a reference to the :class:`Machine` and derives a
    :class:`MachineSpace` from it on demand, then composes the panel
    orientation rotation on top of the native transform.
    """

    def __init__(self, machine: "Machine"):
        self._machine = machine
        self._orientation: PanelOrientation = PanelOrientation.NATIVE
        self._cached_extents = machine.axis_extents
        machine.changed.connect(self._on_machine_changed)

    @property
    def machine(self) -> "Machine":
        return self._machine

    @property
    def space(self) -> MachineSpace:
        """Native coordinate-space projection for the machine."""
        return self._machine.get_coordinate_space()

    # -- Orientation state --------------------------------------------

    @property
    def orientation(self) -> PanelOrientation:
        """The panel orientation (NATIVE / ROTATED_LEFT / ROTATED_RIGHT)."""
        return self._orientation

    def set_orientation(self, orientation: PanelOrientation) -> None:
        """Set how the native machine bed is presented on screen.

        All interactive changes of the orientation MUST go through this
        setter rather than assigning ``_orientation`` directly.

        Camera calibration (``camera.image_to_world``) is stored in
        presented coordinates because the camera pipeline consumes it
        directly in that space. Changing the presentation therefore
        re-projects the stored calibration points through the old and
        new transforms so an existing physical alignment stays valid.
        The rotation matrices contain only 0/+/-1 entries and exact
        translations, so repeated re-projection does not accumulate
        floating-point error.

        Deserialization (``Machine.from_dict``) assigns ``_orientation``
        directly instead, because persisted camera calibration was
        already saved in the matching orientation.
        """
        if self._orientation == orientation:
            return
        old_matrix = self._panel_to_native_matrix
        self._orientation = orientation
        new_inverse = np.linalg.inv(self._panel_to_native_matrix)
        self._reproject_cameras(old_matrix, new_inverse)
        self._machine.changed.send(self._machine)

    @property
    def supports_rotary(self) -> bool:
        """Whether rotary mapping can compose with this panel setup."""
        return self._orientation is PanelOrientation.NATIVE

    def _on_machine_changed(self, sender=None, **kwargs) -> None:
        """Watch for bed-dimension changes that require camera
        reprojection.

        A rotated presentation's translation depends on the native bed
        dimensions, so resizing the bed shifts where presented coordinates
        land.  Camera calibration (stored in presented coordinates) is
        re-projected so the physical alignment stays valid.
        """
        current = self._machine.axis_extents
        if current == self._cached_extents:
            return
        old_p2n = self._compute_p2n(self._orientation, self._cached_extents)
        new_n2p = np.linalg.inv(self._panel_to_native_matrix)
        self._cached_extents = current
        self._reproject_cameras(old_p2n, new_n2p)

    def _reproject_cameras(
        self,
        old_p2n: np.ndarray,
        new_n2p: np.ndarray,
    ) -> None:
        """Preserve physical camera calibration across orientation changes."""
        for camera in self._machine.cameras:
            if camera.image_to_world is None:
                continue
            image_points, world_points = camera.image_to_world
            reprojected = []
            for wx, wy in world_points:
                native = old_p2n @ np.array([wx, wy, 0.0, 1.0])
                new_world = new_n2p @ native
                reprojected.append((float(new_world[0]), float(new_world[1])))
            alignment_date = camera.alignment_date
            camera.image_to_world = (image_points, reprojected)
            camera.alignment_date = alignment_date

    # -- Rotation matrix ----------------------------------------------

    @staticmethod
    def _compute_p2n(
        orientation: PanelOrientation,
        extents: tuple[float, float],
    ) -> np.ndarray:
        """Rigid 90-degree rotation from the presented bed to the native
        machine bed.

        Identity for NATIVE. ROTATED_LEFT maps presented (x, y) to
        native (y, height - x); ROTATED_RIGHT maps it to native
        (width - y, x). Entries are only 0 / +/-1 plus exact
        translations.
        """
        width, height = extents
        matrix = np.identity(4, dtype=np.float64)
        if orientation == PanelOrientation.ROTATED_LEFT:
            matrix[0, 0] = 0.0
            matrix[0, 1] = 1.0
            matrix[1, 0] = -1.0
            matrix[1, 1] = 0.0
            matrix[1, 3] = height
        elif orientation == PanelOrientation.ROTATED_RIGHT:
            matrix[0, 0] = 0.0
            matrix[0, 1] = -1.0
            matrix[0, 3] = width
            matrix[1, 0] = 1.0
            matrix[1, 1] = 0.0
        return matrix

    @property
    def _panel_to_native_matrix(self) -> np.ndarray:
        return self._compute_p2n(self._orientation, self.space.extents)

    # -- Display properties -------------------------------------------

    @property
    def origin(self) -> OriginCorner:
        """The native origin corner as it appears after the rotation."""
        if self._orientation == PanelOrientation.NATIVE:
            return self.space.origin
        if self._orientation == PanelOrientation.ROTATED_LEFT:
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
        if self._orientation == PanelOrientation.NATIVE:
            return self.space.reverse_x
        return self.space.reverse_y

    @property
    def y_axis_negative(self) -> bool:
        """Whether the displayed Y axis reflects a reversed native axis."""
        if self._orientation == PanelOrientation.NATIVE:
            return self.space.reverse_y
        return self.space.reverse_x

    # -- Presented geometry -------------------------------------------

    @property
    def extents(self) -> tuple[float, float]:
        """The bed dimensions as presented on screen."""
        if self._orientation == PanelOrientation.NATIVE:
            return self.space.extents
        return self.space.extents[1], self.space.extents[0]

    @property
    def margins(self) -> Rect:
        """Native edge margins rotated into presented-edge order."""
        left, top, right, bottom = self.space.margins
        if self._orientation == PanelOrientation.ROTATED_LEFT:
            return top, right, bottom, left
        if self._orientation == PanelOrientation.ROTATED_RIGHT:
            return bottom, left, top, right
        return self.space.margins

    @property
    def workarea_size(self) -> tuple[float, float]:
        """The (width, height) of the workarea in presented space."""
        ml, mt, mr, mb = self.margins
        width, height = self.extents
        return width - ml - mr, height - mt - mb

    @property
    def extent_frame(self) -> Rect:
        """The full bed extent frame in presented coordinates.

        Positioned at (-margin_left, -margin_bottom) relative to the
        work-area origin, using the presented (rotated) margins and
        extents.
        """
        ml, mb = self.margins[0], self.margins[3]
        extent_w, extent_h = self.extents
        return (float(-ml), float(-mb), float(extent_w), float(extent_h))

    @property
    def has_custom_work_area(self) -> bool:
        """True when any edge margin is non-zero (rotation invariant)."""
        return self._machine.has_custom_work_area()

    # -- Composed transforms ------------------------------------------

    def get_world_to_machine_matrix(self) -> np.ndarray:
        """Full world-to-machine matrix, including panel rotation."""
        return (
            self.space.get_world_to_machine_matrix()
            @ self._panel_to_native_matrix
        )

    def get_machine_to_world_matrix(self) -> np.ndarray:
        """Inverse of get_world_to_machine_matrix()."""
        return np.linalg.inv(self.get_world_to_machine_matrix())

    def world_point_to_machine(self, x: float, y: float) -> Point:
        """Transform a point from world space to machine space."""
        matrix = self.get_world_to_machine_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def machine_point_to_world(self, x: float, y: float) -> Point:
        """Transform a point from machine space to world space."""
        matrix = self.get_machine_to_world_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def world_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from world space to machine space."""
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
        mx = max(xs) if self.space.reverse_x else min(xs)
        my = max(ys) if self.space.reverse_y else min(ys)
        return mx, my

    def machine_item_to_world(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from machine space to world space."""
        mx, my = pos
        w, h = size
        if self._orientation != PanelOrientation.NATIVE:
            w, h = h, w
        if self.space.reverse_x:
            x_min, x_max = mx - w, mx
        else:
            x_min, x_max = mx, mx + w
        if self.space.reverse_y:
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

    # -- Rect / position / label helpers ------------------------------

    def get_workarea_world_rect(self) -> Rect:
        """Work area boundary as a Rect in world space."""
        pos = self.space.get_workarea_origin_in_machine()
        w, h = self.workarea_size
        wx, wy = self.machine_item_to_world(pos, (w, h))
        return (wx, wy, w, h)

    def world_position_from_origin(
        self,
        ref_x: float,
        ref_y: float,
        size: tuple[float, float],
    ) -> Point:
        """Convert a reference position at the origin corner to world
        coordinates."""
        width, height = size

        origin = self.origin
        if origin == OriginCorner.BOTTOM_LEFT:
            return ref_x, ref_y
        elif origin == OriginCorner.TOP_LEFT:
            return ref_x, ref_y - height
        elif origin == OriginCorner.BOTTOM_RIGHT:
            return ref_x - width, ref_y
        else:  # TOP_RIGHT
            return ref_x - width, ref_y - height

    def get_axis_label_origin(
        self,
        wcs_offset: Point3D = (0.0, 0.0, 0.0),
        wcs_is_workarea_origin: bool = False,
    ) -> Point3D:
        """Origin offset for axis labels."""
        native = self.space.get_axis_label_origin(
            wcs_offset, wcs_is_workarea_origin
        )
        if self._orientation == PanelOrientation.NATIVE:
            return native
        return (native[1], native[0], native[2])

    # -- Native delegates (no rotation) -------------------------------

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
