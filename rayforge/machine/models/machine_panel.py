"""
Display-facing projection of a machine's coordinate space.

MachineSpace (in rayforge.machine.models.coordspace) describes the
machine in native coordinates -- how the machine "speaks". The
document model lives in WORLD space (canonical, Y-up). MachinePanel
describes the PANEL space: how the bed is *shown* after an optional
90-degree rotation. Keeping these view-facing properties here avoids
mixing presentation concerns into the coordinate model itself.
"""

from enum import Enum
from gettext import gettext as _
from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo import Matrix
from raygeo.geo.types import Point, Point3D, Rect
from raygeo.ops.axis import Axis

from .coordspace import (
    MachineSpace,
    OriginCorner,
)
from .zone import Zone, ZoneShape

if TYPE_CHECKING:
    from .machine import JogDirection, Machine

_DELTA_EPSILON = 1e-9  # mm — filter near-zero jog deltas


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

    @property
    def label(self) -> str:
        """Translated display label for the orientation."""
        if self is PanelOrientation.NATIVE:
            return _("Native")
        if self is PanelOrientation.ROTATED_LEFT:
            return _("Rotate Left")
        return _("Rotate Right")


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
        old_matrix = self._panel_to_world_matrix
        self._orientation = orientation
        new_inverse = np.linalg.inv(self._panel_to_world_matrix)
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
        old_p2w = self._compute_p2w(self._orientation, self._cached_extents)
        new_w2p = np.linalg.inv(self._panel_to_world_matrix)
        self._cached_extents = current
        self._reproject_cameras(old_p2w, new_w2p)

    def _reproject_cameras(
        self,
        old_p2w: np.ndarray,
        new_w2p: np.ndarray,
    ) -> None:
        """Preserve physical camera calibration across orientation changes."""
        for camera in self._machine.cameras:
            if camera.image_to_world is None:
                continue
            image_points, world_points = camera.image_to_world
            reprojected = []
            for wx, wy in world_points:
                world = old_p2w @ np.array([wx, wy, 0.0, 1.0])
                new_panel = new_w2p @ world
                reprojected.append((float(new_panel[0]), float(new_panel[1])))
            alignment_date = camera.alignment_date
            camera.image_to_world = (image_points, reprojected)
            camera.alignment_date = alignment_date

    # -- Rotation matrix ----------------------------------------------

    @staticmethod
    def _compute_p2w(
        orientation: PanelOrientation,
        extents: tuple[float, float],
    ) -> np.ndarray:
        """Rigid 90-degree rotation from the PANEL presentation to WORLD.

        Identity for NATIVE. ROTATED_LEFT maps panel (x, y) to WORLD
        (y, height - x); ROTATED_RIGHT maps it to WORLD (width - y, x).
        Entries are only 0 / +/-1 plus exact translations.
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
    def _panel_to_world_matrix(self) -> np.ndarray:
        """Rigid rotation from PANEL coordinates to canonical WORLD."""
        return self._compute_p2w(self._orientation, self.space.extents)

    @property
    def world_to_panel(self) -> np.ndarray:
        """The 90-degree bed rotation from WORLD geometry to the panel
        presentation.

        Inverse of the panel-to-world rotation. Origin-corner and
        axis-reversal sign flips are not included; the 3D pipeline
        applies those separately via its model matrix and axis flags.
        """
        return np.linalg.inv(self._panel_to_world_matrix)

    def get_world_to_panel_2d(self) -> Matrix:
        """The WORLD→PANEL rotation as a 2D affine matrix.

        Used by the 2D canvas to present document (WORLD) content in
        PANEL space.
        """
        width, height = self.space.extents
        if self._orientation == PanelOrientation.ROTATED_RIGHT:
            return Matrix(
                [[0.0, 1.0, 0.0], [-1.0, 0.0, width], [0.0, 0.0, 1.0]]
            )
        if self._orientation == PanelOrientation.ROTATED_LEFT:
            return Matrix(
                [[0.0, -1.0, height], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]
            )
        return Matrix.identity()

    def _world_point_to_panel(self, x: float, y: float) -> Point:
        """Project a WORLD point into panel coordinates.

        Applies only the presentation rotation; the point is physical
        bed geometry that does not depend on the configured origin or
        axis direction.
        """
        result = self.world_to_panel @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def _world_item_to_panel(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> tuple[Point, tuple[float, float]]:
        """Project an axis-aligned WORLD rectangle into panel
        coordinates."""
        x, y = pos
        width, height = size
        corners = (
            self._world_point_to_panel(x, y),
            self._world_point_to_panel(x + width, y),
            self._world_point_to_panel(x, y + height),
            self._world_point_to_panel(x + width, y + height),
        )
        xs = [point[0] for point in corners]
        ys = [point[1] for point in corners]
        min_x, min_y = min(xs), min(ys)
        return (min_x, min_y), (max(xs) - min_x, max(ys) - min_y)

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

    @property
    def nogo_zones(self) -> dict[str, Zone]:
        """Read-only no-go-zone projections in panel coordinates.

        A detached copy is returned for every orientation, including
        NATIVE, so callers never receive an object whose mutation
        behavior changes when the panel rotates. Edits must go through
        ``machine.nogo_zones`` in WORLD coordinates.
        """
        projected: dict[str, Zone] = {}
        for uid, zone in self._machine.nogo_zones.items():
            panel_zone = Zone.from_dict(zone.to_dict())
            params = panel_zone.params
            x = params.get("x", 0.0)
            y = params.get("y", 0.0)
            if panel_zone.shape == ZoneShape.CYLINDER:
                params["x"], params["y"] = self._world_point_to_panel(x, y)
            else:
                pos, size = self._world_item_to_panel(
                    (x, y),
                    (params.get("w", 10.0), params.get("h", 10.0)),
                )
                params["x"], params["y"] = pos
                params["w"], params["h"] = size
            projected[uid] = panel_zone
        return projected

    # -- Composed transforms ------------------------------------------

    def get_world_to_machine_matrix(self) -> np.ndarray:
        """Full WORLD-to-MACHINE matrix (native, no panel rotation).

        The document model lives in WORLD space, so the encoder uses
        this matrix directly.
        """
        return self.space.get_world_to_machine_matrix()

    def get_machine_to_world_matrix(self) -> np.ndarray:
        """Inverse of get_world_to_machine_matrix()."""
        return self.space.get_machine_to_world_matrix()

    def world_point_to_machine(self, x: float, y: float) -> Point:
        """Transform a point from WORLD space to MACHINE space."""
        return self.space.world_point_to_machine(x, y)

    def machine_point_to_world(self, x: float, y: float) -> Point:
        """Transform a point from MACHINE space to WORLD space."""
        return self.space.machine_point_to_world(x, y)

    def world_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from WORLD space to MACHINE space."""
        return self.space.world_item_to_machine(pos, size)

    def machine_item_to_world(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from MACHINE space to WORLD space."""
        return self.space.machine_item_to_world(pos, size)

    def get_panel_to_machine_matrix(self) -> np.ndarray:
        """Full PANEL-to-MACHINE matrix, including panel rotation."""
        return (
            self.space.get_world_to_machine_matrix()
            @ self._panel_to_world_matrix
        )

    def get_machine_to_panel_matrix(self) -> np.ndarray:
        """Inverse of get_panel_to_machine_matrix()."""
        return np.linalg.inv(self.get_panel_to_machine_matrix())

    def panel_point_to_machine(self, x: float, y: float) -> Point:
        """Transform a point from PANEL space to MACHINE space."""
        matrix = self.get_panel_to_machine_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def machine_point_to_panel(self, x: float, y: float) -> Point:
        """Transform a point from MACHINE space to PANEL space."""
        matrix = self.get_machine_to_panel_matrix()
        result = matrix @ np.array([x, y, 0.0, 1.0])
        return float(result[0]), float(result[1])

    def panel_item_to_machine(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from PANEL space to MACHINE space."""
        px, py = pos
        w, h = size
        corners = (
            self.panel_point_to_machine(px, py),
            self.panel_point_to_machine(px + w, py),
            self.panel_point_to_machine(px, py + h),
            self.panel_point_to_machine(px + w, py + h),
        )
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        mx = max(xs) if self.space.reverse_x else min(xs)
        my = max(ys) if self.space.reverse_y else min(ys)
        return mx, my

    def machine_item_to_panel(
        self,
        pos: Point,
        size: tuple[float, float],
    ) -> Point:
        """Convert item position from MACHINE space to PANEL space."""
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
            self.machine_point_to_panel(x_min, y_min),
            self.machine_point_to_panel(x_max, y_min),
            self.machine_point_to_panel(x_min, y_max),
            self.machine_point_to_panel(x_max, y_max),
        )
        return min(c[0] for c in corners), min(c[1] for c in corners)

    def calculate_jog(
        self, direction: "JogDirection", distance: float
    ) -> dict[Axis, float]:
        """Translate a visual jog direction into native axis deltas.

        Visual directions are expressed in PANEL (presented, rotated)
        coordinates, so the PANEL-to-MACHINE matrix rotates the delta
        vector: on a ROTATED_RIGHT bed a visual east jog drives the
        native Y axis. UP/DOWN are pinned to Axis.Z and delegated to
        the machine, which has no rotation knowledge.
        """
        # Local import: machine.py imports this module before JogDirection
        # is defined, so a module-level import would be circular.
        from .machine import JogDirection

        if direction in (JogDirection.UP, JogDirection.DOWN):
            return {Axis.Z: self._machine.calculate_jog(direction, distance)}
        presented_deltas = {
            JogDirection.EAST: (distance, 0.0),
            JogDirection.WEST: (-distance, 0.0),
            JogDirection.NORTH: (0.0, distance),
            JogDirection.SOUTH: (0.0, -distance),
        }
        dx, dy = presented_deltas[direction]
        machine_delta = self.get_panel_to_machine_matrix() @ np.array(
            [dx, dy, 0.0, 0.0]
        )
        result: dict[Axis, float] = {}
        if abs(machine_delta[0]) > _DELTA_EPSILON:
            result[Axis.X] = float(machine_delta[0])
        if abs(machine_delta[1]) > _DELTA_EPSILON:
            result[Axis.Y] = float(machine_delta[1])
        return result

    # -- Rect / position / label helpers ------------------------------

    def get_workarea_world_rect(self) -> Rect:
        """Work area boundary as a Rect in WORLD space (native)."""
        return self.space.get_workarea_world_rect()

    def get_workarea_panel_rect(self) -> Rect:
        """Work area boundary as a Rect in PANEL space."""
        pos = self.space.get_workarea_origin_in_machine()
        w, h = self.workarea_size
        wx, wy = self.machine_item_to_panel(pos, (w, h))
        return (wx, wy, w, h)

    def world_position_from_origin(
        self,
        ref_x: float,
        ref_y: float,
        size: tuple[float, float],
    ) -> Point:
        """Convert a reference position at the origin corner to WORLD
        coordinates."""
        return self.space.world_position_from_origin(ref_x, ref_y, size)

    def panel_position_from_origin(
        self,
        ref_x: float,
        ref_y: float,
        size: tuple[float, float],
    ) -> Point:
        """Convert a reference position at the origin corner to PANEL
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
        """The reference origin position in WORLD coordinates.

        Combines the machine's reference offset (WCS or workarea origin,
        in machine coordinates) with the native machine→world transform.
        """
        offset_x, offset_y, _ = self._machine.get_reference_offset()
        return self.machine_point_to_world(offset_x, offset_y)

    @property
    def reference_position_panel(self) -> Point:
        """The reference origin position in PANEL coordinates."""
        offset_x, offset_y, _ = self._machine.get_reference_offset()
        return self.machine_point_to_panel(offset_x, offset_y)

    def work_area_center(self) -> Point:
        """Center of the work-area-sized box anchored at the reference
        origin, in WORLD coordinates.

        When the WCS is not the workarea origin, the reference anchor is
        the active WCS position; the returned point is the center of a
        work-area-sized box placed with its origin corner there.
        """
        ref_x, ref_y = self.reference_position_world
        w, h = self.space.workarea_size
        ox, oy = self.world_position_from_origin(ref_x, ref_y, (w, h))
        return (ox + w / 2.0, oy + h / 2.0)

    def work_area_center_panel(self) -> Point:
        """Center of the work-area-sized box anchored at the reference
        origin, in PANEL coordinates."""
        ref_x, ref_y = self.reference_position_panel
        w, h = self.workarea_size
        ox, oy = self.panel_position_from_origin(ref_x, ref_y, (w, h))
        return (ox + w / 2.0, oy + h / 2.0)
