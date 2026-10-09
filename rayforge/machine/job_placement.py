"""
Resolve a document's job origin ("Start From") against a machine.

The job origin is applied as one rigid XY translation of the whole job
in world space. This module turns the abstract setting into that
translation: it finds the start point (the zero point of the active
WCS, or the reported head position) and the offset that moves the
job's anchor onto it.

The G-code stays absolute in the active WCS; nothing is written to the
controller (no G92 or G10), so an aborted job leaves no hidden offset
behind.
"""

from gettext import gettext as _
from typing import TYPE_CHECKING

from raygeo.geo.types import Point

from ..core.job_origin import BBox, JobOrigin, StartFrom, placement_offset
from .driver.driver import DeviceStatus
from .models.coordspace import MachineSpace

if TYPE_CHECKING:
    from .models.machine import Machine


class JobPlacementError(ValueError):
    """
    Raised when a job cannot be placed as requested, e.g. because the
    head position is unknown. The message is user-facing.
    """


def resolve_start_point(
    machine: "Machine", start_from: StartFrom
) -> Point | None:
    """
    The world-space point the job anchor is placed on, or None for
    absolute coordinates.

    Raises:
        JobPlacementError: If the point cannot be determined.
    """
    if start_from == StartFrom.USER_ORIGIN:
        return _user_origin(machine)
    if start_from == StartFrom.CURRENT_POSITION:
        return _current_position(machine)
    return None


def compute_job_shift(
    machine: "Machine", job_origin: JobOrigin, design_rect: BBox | None
) -> Point | None:
    """
    The world-space (dx, dy) translation that places a job whose
    bounding box is *design_rect* as requested by *job_origin*.

    Returns None when no translation applies (absolute coordinates or
    an empty job).

    Raises:
        JobPlacementError: If the start point cannot be determined.
    """
    if job_origin.is_absolute or design_rect is None:
        return None
    start_point = resolve_start_point(machine, job_origin.start_from)
    if start_point is None:
        return None
    return placement_offset(design_rect, job_origin.anchor, start_point)


def _user_origin(machine: "Machine") -> Point:
    """The zero point of the active WCS in world space."""
    space = MachineSpace.from_machine(machine)
    zero_x, zero_y, _z = space.get_command_offset(
        wcs_offset=machine.get_active_wcs_offset(),
        wcs_is_workarea_origin=machine.wcs_origin_is_workarea_origin,
    )
    return space.machine_point_to_world(zero_x, zero_y)


def _current_position(machine: "Machine") -> Point:
    """
    The reported head position in world space. While pointer alignment
    is on, this is the position of the pointer dot.
    """
    if not machine.is_connected():
        raise JobPlacementError(
            _(
                "Start From “Current Position” needs a connected "
                "machine. Connect it, or use “User Origin” for "
                "files you run elsewhere."
            )
        )
    state = machine.device_state
    if state.status != DeviceStatus.IDLE:
        raise JobPlacementError(
            _(
                "Start From “Current Position” needs an idle "
                "machine; it is {status}."
            ).format(status=state.status.name.lower())
        )
    pos_x, pos_y = state.machine_pos[0], state.machine_pos[1]
    if pos_x is None or pos_y is None:
        raise JobPlacementError(
            _(
                "Start From “Current Position” needs the head "
                "position, but the machine has not reported it yet."
            )
        )
    if machine.pointer_alignment_enabled:
        dx, dy = machine.get_pointer_offset()
        pos_x += dx
        pos_y += dy
    space = MachineSpace.from_machine(machine)
    return space.machine_point_to_world(pos_x, pos_y)
