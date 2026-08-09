"""Workspace-facing projection of a machine configuration."""

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import numpy as np
from raygeo.geo.types import Rect
from raygeo.ops.axis import Axis

from ...pipeline.coordspace import MachineSpace
from .zone import Zone, ZoneShape

if TYPE_CHECKING:
    from .machine import Machine


class JogDirection(Enum):
    """Visual direction for jog operations."""

    EAST = "east"
    WEST = "west"
    NORTH = "north"
    SOUTH = "south"
    UP = "up"
    DOWN = "down"


@dataclass(frozen=True)
class MachineView:
    """Read-only workspace presentation of a native machine model.

    The machine remains the source of truth in native bed coordinates. This
    facade projects that state for display and interaction, keeping view-only
    rotation logic out of :class:`Machine` and avoiding mutable projection
    caches that need manual invalidation.
    """

    machine: "Machine"

    @property
    def space(self) -> MachineSpace:
        """Current coordinate-space projection for the machine."""
        return self.machine.get_coordinate_space()

    @property
    def extents(self) -> tuple[float, float]:
        """Dimensions of the machine bed as presented in the editor."""
        return self.space.workspace_extents

    @property
    def margins(self) -> Rect:
        """Native work margins rotated into workspace edge order."""
        return self.space.workspace_margins

    @property
    def work_area(self) -> Rect:
        """Usable work area in workspace coordinates."""
        return self.space.get_workarea_world_rect()

    @property
    def extent_frame(self) -> Rect:
        """Full bed frame relative to the workspace work-area origin."""
        ml, _, _, mb = self.margins
        extent_w, extent_h = self.extents
        return (float(-ml), float(-mb), float(extent_w), float(extent_h))

    @property
    def nogo_zones(self) -> dict[str, Zone]:
        """Return read-only no-go-zone projections in workspace space.

        A detached copy is returned for every orientation, including Native,
        so callers never receive an API whose mutation behavior changes when
        the workspace rotates. Edits must go through ``machine.nogo_zones``
        in native bed coordinates.
        """
        space = self.space
        projected: dict[str, Zone] = {}
        for uid, zone in self.machine.nogo_zones.items():
            workspace_zone = Zone.from_dict(zone.to_dict())
            params = workspace_zone.params
            x = params.get("x", 0.0)
            y = params.get("y", 0.0)
            if workspace_zone.shape == ZoneShape.CYLINDER:
                params["x"], params["y"] = space.native_bed_point_to_workspace(
                    x, y
                )
            else:
                pos, size = space.native_bed_item_to_workspace(
                    (x, y),
                    (params.get("w", 10.0), params.get("h", 10.0)),
                )
                params["x"], params["y"] = pos
                params["w"], params["h"] = size
            projected[uid] = workspace_zone
        return projected

    def calculate_jog(
        self, direction: JogDirection, distance: float
    ) -> dict[Axis, float]:
        """Map a visual jog direction to native controller axis deltas."""
        if direction in (JogDirection.UP, JogDirection.DOWN):
            return {Axis.Z: self.machine.calculate_jog(direction, distance)}

        workspace_deltas = {
            JogDirection.EAST: (distance, 0.0),
            JogDirection.WEST: (-distance, 0.0),
            JogDirection.NORTH: (0.0, distance),
            JogDirection.SOUTH: (0.0, -distance),
        }
        dx, dy = workspace_deltas.get(direction, (0.0, 0.0))
        matrix = self.space.get_world_to_machine_matrix()
        machine_delta = matrix @ np.array([dx, dy, 0.0, 0.0])
        result = {}
        if abs(machine_delta[0]) > 1e-9:
            result[Axis.X] = float(machine_delta[0])
        if abs(machine_delta[1]) > 1e-9:
            result[Axis.Y] = float(machine_delta[1])
        return result
