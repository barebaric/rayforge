"""
The web UI coordinate contract.

Rayforge spaces (see :mod:`rayforge.machine.models.coordspace`):

- ``WORLD``: canonical document space, mm, origin bottom-left, Y-up.
- ``MACHINE``: the bed in native coordinates; origin corner and axis
  directions vary per machine (plus reverse_x/reverse_y sign flips).
- ``WORKAREA``: the usable area inside the bed (margins).
- ``COMMAND``: G-code output coordinates relative to the active WCS.
- ``PIXEL``: raster images, top-left origin, Y-down.
- ``PANEL``: the presented view, i.e. WORLD rotated by the optional
  ±90° ``PanelOrientation`` held by :class:`MachinePanel`.

Transformation chains:

- **Document → Screen**: document items live in WORLD. This module
  projects them into PANEL via the rigid ``world_to_panel`` 2x3 matrix
  (:meth:`MachinePanel.get_world_to_panel_2d`); the frontend maps PANEL
  X/Y onto the three.js ground plane (X, Z) with Y-up.
- **Document → Machine → Gcode**: world ops go through the
  ``MachineTransformSpec`` (raygeo) using the machine's origin corner,
  axis directions, and reverse flags to produce MACHINE coords, then
  through the active WCS offset to COMMAND coords.
- **Machine report → Screen**: driver ``DeviceState`` reports machine
  positions; the beam position (:meth:`Machine.get_beam_position`) is
  projected MACHINE → PANEL via :meth:`MachinePanel.machine_point_to_panel`.

This module is a thin wrapper: it NEVER re-derives conversion math.
Everything delegates to :class:`MachineSpace` / :class:`MachinePanel`
on the active machine.
"""

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raygeo.geo.types import Point, Rect
from raygeo.ops.axis import Axis

from rayforge.machine.models.zone import Zone, ZoneShape

if TYPE_CHECKING:
    from rayforge.context import RayforgeContext
    from rayforge.machine.models.machine import Machine

_DIRECTION_MAP = {
    "+x": "east",
    "-x": "west",
    "+y": "north",
    "-y": "south",
    "+z": "up",
    "-z": "down",
}


@dataclass
class ViewportContract:
    """
    Everything the web frontend needs to present and interact with the
    active machine's bed.

    Coordinates are PANEL (presented) space unless a field name says
    otherwise. The panel is the WORLD document space rotated by the
    panel orientation; the frontend maps panel X/Y onto its ground
    plane (X/Z axes, Y-up).
    """

    orientation: str
    world_to_panel: list[list[float]]
    x_axis_right: bool
    y_axis_down: bool
    x_axis_negative: bool
    y_axis_negative: bool
    extents: list[float]
    margins: list[float]
    workarea: list[float]
    workarea_origin: list[float]
    work_origin: list[float]
    soft_limits: list[float]
    soft_limits_panel: list[float]
    rotary_active: bool
    nogo_zones: list[list[float]]
    wcs: dict[str, Any] | None = None
    beam: list[float] | None = None


def _matrix_to_2x3(matrix) -> list[list[float]]:
    """The rigid 2x3 top rows of a raygeo 2D affine matrix."""
    return [
        [matrix.get(0, 0), matrix.get(0, 1), matrix.get(0, 2)],
        [matrix.get(1, 0), matrix.get(1, 1), matrix.get(1, 2)],
    ]


def _rect_to_list(rect: Rect) -> list[float]:
    x, y, w, h = rect
    return [float(x), float(y), float(w), float(h)]


def _point_to_list(point: Point) -> list[float]:
    return [float(point[0]), float(point[1])]


def _soft_limits_panel(machine: "Machine") -> list[float]:
    """Machine-space soft limits projected to an axis-aligned PANEL
    bounding rect."""
    x_min, y_min, x_max, y_max = machine.get_soft_limits()
    panel = machine.panel
    corners = (
        panel.machine_point_to_panel(x_min, y_min),
        panel.machine_point_to_panel(x_max, y_min),
        panel.machine_point_to_panel(x_min, y_max),
        panel.machine_point_to_panel(x_max, y_max),
    )
    xs = [corner[0] for corner in corners]
    ys = [corner[1] for corner in corners]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def _machine_wcs(machine: "Machine") -> dict[str, Any] | None:
    """The active WCS name and its offset in PANEL coordinates."""
    off_x, off_y, _ = machine.get_active_wcs_offset()
    if machine.wcs_origin_is_workarea_origin:
        return None
    px, py = machine.panel.machine_point_to_panel(off_x, off_y)
    return {
        "name": machine.active_wcs,
        "offset_panel": [px, py],
        "offset_machine": [float(off_x), float(off_y)],
    }


NOGO_CIRCLE_SEGMENTS = 24


def _zone_machine_points(zone: Zone) -> list[tuple[float, float]]:
    """Outline of a zone in MACHINE space: rect corners or a sampled
    circle (closed loop)."""
    if zone.shape is ZoneShape.CYLINDER:
        cx = zone.params.get("x", 0.0)
        cy = zone.params.get("y", 0.0)
        radius = zone.params.get("radius", 5.0)
        points = []
        for i in range(NOGO_CIRCLE_SEGMENTS + 1):
            angle = (i / NOGO_CIRCLE_SEGMENTS) * 2.0 * math.pi
            points.append(
                (cx + radius * math.cos(angle), cy + radius * math.sin(angle))
            )
        return points
    x = zone.params.get("x", 0.0)
    y = zone.params.get("y", 0.0)
    w = zone.params.get("w", 0.0)
    h = zone.params.get("h", 0.0)
    return [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]


def _nogo_zone_polygons(machine: "Machine") -> list[list[float]]:
    """Enabled no-go zones as closed PANEL-space polygons
    ``[x, y, x, y, ...]``, one list per zone."""
    panel = machine.panel
    polygons: list[list[float]] = []
    for zone in machine.nogo_zones.values():
        if not zone.enabled:
            continue
        polygon: list[float] = []
        for mx, my in _zone_machine_points(zone):
            px, py = panel.machine_point_to_panel(mx, my)
            polygon.extend((px, py))
        polygons.append(polygon)
    return polygons


def beam_panel(machine: "Machine") -> list[float] | None:
    """The reported beam position in PANEL coordinates, or None."""
    beam = machine.get_beam_position()
    if beam is None:
        return None
    px, py = machine.panel.machine_point_to_panel(*beam)
    return [px, py]


def work_origin_panel(machine: "Machine") -> list[float]:
    """The active work origin in PANEL coordinates.

    In workarea-origin mode the reference is the workarea origin in
    MACHINE coordinates; otherwise it is the active WCS offset.
    """
    if machine.wcs_origin_is_workarea_origin:
        offset = machine.panel.space.get_workarea_origin_in_machine()
    else:
        offset = machine.get_active_wcs_offset()[:2]
    px, py = machine.panel.machine_point_to_panel(*offset)
    return [px, py]


def _active_machine(context: "RayforgeContext") -> "Machine":
    """The active machine, raising when none is configured."""
    machine = context.machine
    if machine is None:
        raise ValueError("No active machine configured")
    return machine


def viewport_contract(context: "RayforgeContext") -> ViewportContract:
    """
    Build the current viewport contract from the active machine.

    Returns a :class:`ViewportContract`; the HTTP and WebSocket layers
    serialize it to JSON via :func:`serialize_contract`.
    """
    machine = _active_machine(context)
    panel = machine.panel
    space = panel.space
    origin_in_machine = space.get_workarea_origin_in_machine()
    panel_origin = panel.machine_point_to_panel(*origin_in_machine)
    return ViewportContract(
        orientation=panel.orientation.value,
        world_to_panel=_matrix_to_2x3(panel.get_world_to_panel_2d()),
        x_axis_right=panel.x_axis_right,
        y_axis_down=panel.y_axis_down,
        x_axis_negative=panel.x_axis_negative,
        y_axis_negative=panel.y_axis_negative,
        extents=_point_to_list(panel.extents),
        margins=_rect_to_list(panel.margins),
        workarea=_rect_to_list(panel.get_workarea_panel_rect()),
        workarea_origin=_point_to_list(panel_origin),
        work_origin=work_origin_panel(machine),
        soft_limits=_rect_to_list(machine.get_soft_limits()),
        soft_limits_panel=_soft_limits_panel(machine),
        rotary_active=bool(machine.rotary_modules),
        nogo_zones=_nogo_zone_polygons(machine),
        wcs=_machine_wcs(machine),
        beam=beam_panel(machine),
    )


def serialize_contract(contract: ViewportContract) -> dict[str, Any]:
    """JSON-safe representation of a :class:`ViewportContract`."""
    return {
        "orientation": contract.orientation,
        "world_to_panel": contract.world_to_panel,
        "x_axis_right": contract.x_axis_right,
        "y_axis_down": contract.y_axis_down,
        "x_axis_negative": contract.x_axis_negative,
        "y_axis_negative": contract.y_axis_negative,
        "extents": contract.extents,
        "margins": contract.margins,
        "workarea": contract.workarea,
        "workarea_origin": contract.workarea_origin,
        "work_origin": contract.work_origin,
        "soft_limits": contract.soft_limits,
        "soft_limits_panel": contract.soft_limits_panel,
        "rotary_active": contract.rotary_active,
        "nogo_zones": contract.nogo_zones,
        "wcs": contract.wcs,
        "beam": contract.beam,
    }


def viewport_contract_dict(context: "RayforgeContext") -> dict[str, Any]:
    """Convenience: contract for *context* as a JSON dict."""
    return serialize_contract(viewport_contract(context))


def panel_point_to_machine(
    context: "RayforgeContext", x: float, y: float
) -> Point:
    """PANEL (presented) point to MACHINE (native) point, in mm."""
    return _active_machine(context).panel.panel_point_to_machine(x, y)


def world_point_to_panel(
    context: "RayforgeContext", x: float, y: float
) -> Point:
    """WORLD (document) point to PANEL (presented) point, in mm."""
    panel = _active_machine(context).panel
    return panel.get_world_to_panel_2d().transform_point((x, y))


def _jog_direction(direction: str):
    from rayforge.machine.models.machine import JogDirection

    try:
        name = _DIRECTION_MAP[direction.lower()]
    except KeyError:
        raise ValueError(
            f"Invalid direction '{direction}'; expected one of "
            f"{sorted(_DIRECTION_MAP)}"
        ) from None
    return JogDirection(name)


def jog_for_panel_direction(
    context: "RayforgeContext",
    direction: str,
    distance_mm: float,
) -> dict["Axis", float]:
    """
    Translate a PANEL-space direction (``+x``/``-x``/``+y``/``-y`` or
    ``+z``/``-z``) into the native axis deltas the controller expects.

    The mapping is delegated to :meth:`MachinePanel.calculate_jog`,
    which rotates the presented delta through the PANEL-to-MACHINE
    matrix (so a rotated bed drives the correct native axis) and pins
    Z to the machine. Returns e.g. ``{Axis.X: 12.0, Axis.Y: -3.0}``.
    """
    machine = _active_machine(context)
    return machine.panel.calculate_jog(_jog_direction(direction), distance_mm)
