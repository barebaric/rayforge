from __future__ import annotations
import math
import logging
from typing import (
    Callable,
    Iterator,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Generator,
    Dict,
    Any,
)
import numpy as np
import json
from raygeo import Geometry, Point3D, Rect, Polygon, CMD_TYPE_ARC
from raygeo.algo.clipping import (
    clip_line_segment_with_rect,
    subtract_polygons_from_line_segment,
)
from raygeo.path import PyCommand
from raygeo.shape.arc import get_arc_bounds, linearize_arc
from raygeo.shape.bezier import linearize_bezier_segment
from .axis import Axis
from .enums import CommandType, CommandCategory, SectionType
from .state import State
from .timing import estimate_time


logger = logging.getLogger(__name__)


class MachineState(NamedTuple):
    power: float
    air_assist: bool
    cut_speed: Optional[int]
    travel_speed: Optional[int]
    active_laser_uid: Optional[str]
    frequency: Optional[int]
    pulse_width: Optional[float]


class CommandInfo(NamedTuple):
    type: CommandType
    end: Optional[Any] = None
    extra_axes: Optional[Dict[Any, float]] = None
    state: Optional[Any] = None
    center_offset: Optional[Any] = None
    clockwise: Optional[bool] = None
    control1: Optional[Any] = None
    control2: Optional[Any] = None
    control: Optional[Any] = None
    power_values: Optional[bytes] = None
    power: Optional[float] = None
    speed: Optional[int] = None
    frequency: Optional[int] = None
    pulse_width: Optional[float] = None
    laser_uid: Optional[str] = None
    duration_ms: Optional[int] = None
    layer_uid: Optional[str] = None
    workpiece_uid: Optional[str] = None
    section_type: Optional[str] = None


class OpsSection(NamedTuple):
    """A parsed section of Ops commands, bounded by section markers."""

    section_type: Optional[SectionType]
    marker_indices: List[int]
    content_indices: List[int]


class OpsSectionRange(NamedTuple):
    """Like OpsSection but with index ranges instead of command lists."""

    section_type: Optional[SectionType]
    marker_indices: List[int]
    content_indices: List[int]


_COMMAND_TYPE_TO_CATEGORY: Dict[CommandType, CommandCategory] = {
    CommandType.MOVE_TO: CommandCategory.MOVING,
    CommandType.LINE_TO: CommandCategory.MOVING,
    CommandType.ARC_TO: CommandCategory.MOVING,
    CommandType.BEZIER_TO: CommandCategory.MOVING,
    CommandType.QUADRATIC_BEZIER_TO: CommandCategory.MOVING,
    CommandType.SCAN_LINE: CommandCategory.MOVING,
    CommandType.DWELL: CommandCategory.STATE,
    CommandType.SET_POWER: CommandCategory.STATE,
    CommandType.SET_CUT_SPEED: CommandCategory.STATE,
    CommandType.SET_TRAVEL_SPEED: CommandCategory.STATE,
    CommandType.SET_FREQUENCY: CommandCategory.STATE,
    CommandType.SET_PULSE_WIDTH: CommandCategory.STATE,
    CommandType.ENABLE_AIR_ASSIST: CommandCategory.STATE,
    CommandType.DISABLE_AIR_ASSIST: CommandCategory.STATE,
    CommandType.SET_LASER: CommandCategory.STATE,
    CommandType.JOB_START: CommandCategory.MARKER,
    CommandType.JOB_END: CommandCategory.MARKER,
    CommandType.LAYER_START: CommandCategory.MARKER,
    CommandType.LAYER_END: CommandCategory.MARKER,
    CommandType.WORKPIECE_START: CommandCategory.MARKER,
    CommandType.WORKPIECE_END: CommandCategory.MARKER,
    CommandType.OPS_SECTION_START: CommandCategory.MARKER,
    CommandType.OPS_SECTION_END: CommandCategory.MARKER,
}


def _category(ct: CommandType) -> CommandCategory:
    return _COMMAND_TYPE_TO_CATEGORY[ct]


class _SoA:
    """Struct-of-Arrays storage for Ops command data."""

    def __init__(self) -> None:
        self._types: List[int] = []
        self._endpoints: List[Point3D] = []
        self._arc_data: List[Tuple[float, float, bool]] = []
        self._arc_map: List[int] = []
        self._bezier_data: List[Tuple[Point3D, Point3D]] = []
        self._bezier_map: List[int] = []
        self._quad_data: List[Point3D] = []
        self._quad_map: List[int] = []
        self._scanline_data: List[bytearray] = []
        self._scanline_map: List[int] = []
        self._dwell_durations: List[float] = []
        self._dwell_map: List[int] = []
        self._powers: List[float] = []
        self._power_map: List[int] = []
        self._speeds: List[int] = []
        self._speed_map: List[int] = []
        self._frequencies: List[int] = []
        self._frequency_map: List[int] = []
        self._pulse_widths: List[float] = []
        self._pulse_width_map: List[int] = []
        self._laser_uids: List[str] = []
        self._laser_uid_map: List[int] = []
        self._layer_uids: List[str] = []
        self._layer_uid_map: List[int] = []
        self._workpiece_uids: List[str] = []
        self._workpiece_uid_map: List[int] = []
        self._section_types: List[SectionType] = []
        self._section_workpiece_uids: List[Optional[str]] = []
        self._section_map: List[int] = []
        self._extra_axes: List[Dict[Axis, float]] = []
        self._extra_axes_map: List[int] = []
        self._states: List[State] = []
        self._state_map: List[int] = []

    def __len__(self) -> int:
        return len(self._types)

    def command_type(self, idx: int) -> CommandType:
        return CommandType(self._types[idx])

    def category(self, idx: int) -> CommandCategory:
        return _COMMAND_TYPE_TO_CATEGORY[CommandType(self._types[idx])]

    def append(
        self,
        ct: CommandType,
        end: Optional[Point3D] = None,
        arc_params: Optional[Tuple[float, float, bool]] = None,
        bezier_params: Optional[Tuple[Point3D, Point3D]] = None,
        quad_params: Optional[Point3D] = None,
        scanline: Optional[bytearray] = None,
        dwell_duration: Optional[float] = None,
        power: Optional[float] = None,
        speed: Optional[int] = None,
        frequency: Optional[int] = None,
        pulse_width: Optional[float] = None,
        laser_uid: Optional[str] = None,
        layer_uid: Optional[str] = None,
        workpiece_uid: Optional[str] = None,
        section_type: Optional[SectionType] = None,
        section_workpiece_uid: Optional[str] = None,
        extra_axes: Optional[Dict[Axis, float]] = None,
        state: Optional[State] = None,
    ) -> None:
        if ct == CommandType.SCAN_LINE and scanline is None:
            scanline = bytearray([255])
        self._types.append(ct.value)
        self._endpoints.append(end if end is not None else (0, 0, 0))

        if arc_params is not None:
            self._arc_map.append(len(self._arc_data))
            self._arc_data.append(arc_params)
        else:
            self._arc_map.append(-1)

        if bezier_params is not None:
            self._bezier_map.append(len(self._bezier_data))
            self._bezier_data.append(bezier_params)
        else:
            self._bezier_map.append(-1)

        if quad_params is not None:
            self._quad_map.append(len(self._quad_data))
            self._quad_data.append(quad_params)
        else:
            self._quad_map.append(-1)

        if scanline is not None:
            self._scanline_map.append(len(self._scanline_data))
            self._scanline_data.append(scanline)
        else:
            self._scanline_map.append(-1)

        if dwell_duration is not None:
            self._dwell_map.append(len(self._dwell_durations))
            self._dwell_durations.append(dwell_duration)
        else:
            self._dwell_map.append(-1)

        if power is not None:
            self._power_map.append(len(self._powers))
            self._powers.append(power)
        else:
            self._power_map.append(-1)

        if speed is not None:
            self._speed_map.append(len(self._speeds))
            self._speeds.append(speed)
        else:
            self._speed_map.append(-1)

        if frequency is not None:
            self._frequency_map.append(len(self._frequencies))
            self._frequencies.append(frequency)
        else:
            self._frequency_map.append(-1)

        if pulse_width is not None:
            self._pulse_width_map.append(len(self._pulse_widths))
            self._pulse_widths.append(pulse_width)
        else:
            self._pulse_width_map.append(-1)

        if laser_uid is not None:
            self._laser_uid_map.append(len(self._laser_uids))
            self._laser_uids.append(laser_uid)
        else:
            self._laser_uid_map.append(-1)

        if layer_uid is not None:
            self._layer_uid_map.append(len(self._layer_uids))
            self._layer_uids.append(layer_uid)
        else:
            self._layer_uid_map.append(-1)

        if workpiece_uid is not None:
            self._workpiece_uid_map.append(len(self._workpiece_uids))
            self._workpiece_uids.append(workpiece_uid)
        else:
            self._workpiece_uid_map.append(-1)

        if section_type is not None:
            self._section_map.append(len(self._section_types))
            self._section_types.append(section_type)
            self._section_workpiece_uids.append(section_workpiece_uid)
        else:
            self._section_map.append(-1)

        if extra_axes is not None and extra_axes:
            self._extra_axes_map.append(len(self._extra_axes))
            self._extra_axes.append(extra_axes)
        else:
            self._extra_axes_map.append(-1)

        if state is not None:
            self._state_map.append(len(self._states))
            self._states.append(state)
        else:
            self._state_map.append(-1)

    def endpoint(self, idx: int) -> Point3D:
        return self._endpoints[idx]

    def set_endpoint(self, idx: int, end: Point3D) -> None:
        self._endpoints[idx] = end

    def arc_params(self, idx: int) -> Tuple[float, float, bool]:
        return self._arc_data[self._arc_map[idx]]

    def arc_center_offset(self, idx: int) -> Tuple[float, float]:
        ai = self._arc_map[idx]
        return self._arc_data[ai][0], self._arc_data[ai][1]

    def arc_clockwise(self, idx: int) -> bool:
        ai = self._arc_map[idx]
        return self._arc_data[ai][2]

    def set_arc_params(
        self,
        idx: int,
        center_offset: Optional[Tuple[float, float]] = None,
        clockwise: Optional[bool] = None,
    ) -> None:
        ai = self._arc_map[idx]
        old = self._arc_data[ai]
        co = center_offset if center_offset is not None else (old[0], old[1])
        cw = clockwise if clockwise is not None else old[2]
        self._arc_data[ai] = (co[0], co[1], cw)

    def bezier_params(self, idx: int) -> Tuple[Point3D, Point3D]:
        return self._bezier_data[self._bezier_map[idx]]

    def set_bezier_params(self, idx: int, c1: Point3D, c2: Point3D) -> None:
        ai = self._bezier_map[idx]
        self._bezier_data[ai] = (c1, c2)

    def quad_params(self, idx: int) -> Point3D:
        return self._quad_data[self._quad_map[idx]]

    def set_quad_params(self, idx: int, control: Point3D) -> None:
        qi = self._quad_map[idx]
        self._quad_data[qi] = control

    def scanline_data(self, idx: int) -> bytearray:
        si = self._scanline_map[idx]
        if si < 0 or si >= len(self._scanline_data):
            raise IndexError(
                f"scanline_data corruption: idx={idx} "
                f"map={si} data_len={len(self._scanline_data)} "
                f"types_len={len(self._types)}"
            )
        return self._scanline_data[si]

    def set_scanline_data(self, idx: int, pv: bytearray) -> None:
        self._scanline_data[self._scanline_map[idx]] = pv

    def dwell_duration(self, idx: int) -> float:
        return self._dwell_durations[self._dwell_map[idx]]

    def power(self, idx: int) -> float:
        return self._powers[self._power_map[idx]]

    def speed(self, idx: int) -> int:
        return self._speeds[self._speed_map[idx]]

    def frequency(self, idx: int) -> int:
        return self._frequencies[self._frequency_map[idx]]

    def pulse_width(self, idx: int) -> float:
        return self._pulse_widths[self._pulse_width_map[idx]]

    def laser_uid(self, idx: int) -> str:
        return self._laser_uids[self._laser_uid_map[idx]]

    def layer_uid(self, idx: int) -> str:
        return self._layer_uids[self._layer_uid_map[idx]]

    def workpiece_uid(self, idx: int) -> str:
        return self._workpiece_uids[self._workpiece_uid_map[idx]]

    def section_type(self, idx: int) -> SectionType:
        si = self._section_map[idx]
        return self._section_types[si]

    def section_workpiece_uid(self, idx: int) -> Optional[str]:
        si = self._section_map[idx]
        return self._section_workpiece_uids[si]

    def extra_axes(self, idx: int) -> Optional[Dict[Axis, float]]:
        ei = self._extra_axes_map[idx]
        if ei == -1:
            return None
        return self._extra_axes[ei]

    def set_extra_axes(self, idx: int, ea: Dict[Axis, float]) -> None:
        ei = self._extra_axes_map[idx]
        if ei == -1:
            self._extra_axes_map[idx] = len(self._extra_axes)
            self._extra_axes.append(ea)
        else:
            self._extra_axes[ei] = ea

    def state(self, idx: int) -> Optional[State]:
        si = self._state_map[idx]
        if si == -1:
            return None
        return self._states[si]

    def set_state(self, idx: int, st: State) -> None:
        si = self._state_map[idx]
        if si == -1:
            self._state_map[idx] = len(self._states)
            self._states.append(st)
        else:
            self._states[si] = st

    def copy_entry(self, src_idx: int) -> Dict[str, Any]:
        ct = CommandType(self._types[src_idx])
        kwargs: Dict[str, Any] = {"ct": ct}
        cat = _category(ct)
        if cat == CommandCategory.MOVING:
            kwargs["end"] = self._endpoints[src_idx]
        if ct == CommandType.ARC_TO:
            kwargs["arc_params"] = self.arc_params(src_idx)
        if ct == CommandType.BEZIER_TO:
            kwargs["bezier_params"] = self.bezier_params(src_idx)
        if ct == CommandType.QUADRATIC_BEZIER_TO:
            kwargs["quad_params"] = self.quad_params(src_idx)
        if ct == CommandType.SCAN_LINE:
            kwargs["scanline"] = self.scanline_data(src_idx)
        if ct == CommandType.DWELL:
            kwargs["dwell_duration"] = self.dwell_duration(src_idx)
        if ct == CommandType.SET_POWER:
            kwargs["power"] = self.power(src_idx)
        if ct in (CommandType.SET_CUT_SPEED, CommandType.SET_TRAVEL_SPEED):
            kwargs["speed"] = self.speed(src_idx)
        if ct == CommandType.SET_FREQUENCY:
            kwargs["frequency"] = self.frequency(src_idx)
        if ct == CommandType.SET_PULSE_WIDTH:
            kwargs["pulse_width"] = self.pulse_width(src_idx)
        if ct == CommandType.SET_LASER:
            kwargs["laser_uid"] = self.laser_uid(src_idx)
        if ct in (CommandType.LAYER_START, CommandType.LAYER_END):
            kwargs["layer_uid"] = self.layer_uid(src_idx)
        if ct in (CommandType.WORKPIECE_START, CommandType.WORKPIECE_END):
            kwargs["workpiece_uid"] = self.workpiece_uid(src_idx)
        if ct in (CommandType.OPS_SECTION_START, CommandType.OPS_SECTION_END):
            kwargs["section_type"] = self.section_type(src_idx)
            kwargs["section_workpiece_uid"] = self.section_workpiece_uid(
                src_idx
            )
        ea = self.extra_axes(src_idx)
        if ea is not None:
            kwargs["extra_axes"] = dict(ea)
        st = self.state(src_idx)
        if st is not None:
            kwargs["state"] = st.__copy__()
        return kwargs

    def deep_copy_entry(self, src_idx: int) -> Dict[str, Any]:
        kwargs = self.copy_entry(src_idx)
        if "scanline" in kwargs:
            kwargs["scanline"] = bytearray(kwargs["scanline"])
        if "extra_axes" in kwargs:
            kwargs["extra_axes"] = dict(kwargs["extra_axes"])
        if "state" in kwargs:
            kwargs["state"] = kwargs["state"].__copy__()
        return kwargs


class Ops:
    """
    Represents a set of generated path segments and instructions that
    are used for making gcode, but also to generate vector graphics
    for display.
    """

    def __init__(self) -> None:
        self._soa: _SoA = _SoA()
        self.last_move_to: Point3D = (0.0, 0.0, 0.0)
        self._time_dirty: bool = True
        self._cached_time: float = 0.0
        self._time_params: Optional[tuple] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Ops object to a dictionary."""
        commands: List[Dict[str, Any]] = []
        for i in range(len(self._soa)):
            commands.append(self._cmd_to_dict(i))
        return {
            "commands": commands,
            "last_move_to": self.last_move_to,
        }

    def _cmd_to_dict(self, idx: int) -> Dict[str, Any]:
        ct = self._soa.command_type(idx)
        d: Dict[str, Any] = {"type": ct.name}
        cat = _category(ct)
        if cat == CommandCategory.MOVING:
            d["end"] = self._soa.endpoint(idx)
            ea = self._soa.extra_axes(idx)
            if ea:
                d["extra_axes"] = {
                    axis.name: float(value) for axis, value in ea.items()
                }
        if ct == CommandType.ARC_TO:
            ci, cj = self._soa.arc_center_offset(idx)
            d["center_offset"] = (ci, cj)
            d["clockwise"] = self._soa.arc_clockwise(idx)
        if ct == CommandType.BEZIER_TO:
            c1, c2 = self._soa.bezier_params(idx)
            d["control1"] = c1
            d["control2"] = c2
        if ct == CommandType.QUADRATIC_BEZIER_TO:
            c = self._soa.quad_params(idx)
            d["control"] = c
        if ct == CommandType.SCAN_LINE:
            d["power_values"] = list(self._soa.scanline_data(idx))
        if ct == CommandType.DWELL:
            d["duration_ms"] = self._soa.dwell_duration(idx)
        if ct == CommandType.SET_POWER:
            d["power"] = self._soa.power(idx)
        if ct in (CommandType.SET_CUT_SPEED, CommandType.SET_TRAVEL_SPEED):
            d["speed"] = self._soa.speed(idx)
        if ct == CommandType.SET_FREQUENCY:
            d["frequency"] = self._soa.frequency(idx)
        if ct == CommandType.SET_PULSE_WIDTH:
            d["pulse_width"] = self._soa.pulse_width(idx)
        if ct == CommandType.SET_LASER:
            d["laser_uid"] = self._soa.laser_uid(idx)
        if ct in (CommandType.LAYER_START, CommandType.LAYER_END):
            d["layer_uid"] = self._soa.layer_uid(idx)
        if ct in (CommandType.WORKPIECE_START, CommandType.WORKPIECE_END):
            d["workpiece_uid"] = self._soa.workpiece_uid(idx)
        if ct == CommandType.OPS_SECTION_START:
            d["section_type"] = self._soa.section_type(idx).name
            d["workpiece_uid"] = self._soa.section_workpiece_uid(idx)
        if ct == CommandType.OPS_SECTION_END:
            d["section_type"] = self._soa.section_type(idx).name
        return d

    @property
    def scanline_count(self) -> int:
        return sum(
            1
            for i in range(len(self._soa))
            if self._soa.command_type(i) == CommandType.SCAN_LINE
        )

    def _invalidate_time_cache(self) -> None:
        self._time_dirty = True

    _LEGACY_TYPE_NAMES: Dict[str, str] = {
        "MoveToCommand": "MOVE_TO",
        "LineToCommand": "LINE_TO",
        "ArcToCommand": "ARC_TO",
        "BezierToCommand": "BEZIER_TO",
        "QuadraticBezierToCommand": "QUADRATIC_BEZIER_TO",
        "DwellCommand": "DWELL",
        "ScanLinePowerCommand": "SCAN_LINE",
        "SetPowerCommand": "SET_POWER",
        "SetCutSpeedCommand": "SET_CUT_SPEED",
        "SetTravelSpeedCommand": "SET_TRAVEL_SPEED",
        "SetFrequencyCommand": "SET_FREQUENCY",
        "SetPulseWidthCommand": "SET_PULSE_WIDTH",
        "EnableAirAssistCommand": "ENABLE_AIR_ASSIST",
        "DisableAirAssistCommand": "DISABLE_AIR_ASSIST",
        "SetLaserCommand": "SET_LASER",
        "JobStartCommand": "JOB_START",
        "JobEndCommand": "JOB_END",
        "LayerStartCommand": "LAYER_START",
        "LayerEndCommand": "LAYER_END",
        "WorkpieceStartCommand": "WORKPIECE_START",
        "WorkpieceEndCommand": "WORKPIECE_END",
        "OpsSectionStartCommand": "OPS_SECTION_START",
        "OpsSectionEndCommand": "OPS_SECTION_END",
    }

    @classmethod
    def _create_command_from_dict(
        cls,
        cmd_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        cmd_type: str = cmd_data["type"]
        enum_name = cls._LEGACY_TYPE_NAMES.get(cmd_type, cmd_type)
        ct = CommandType[enum_name]
        ea_raw = cmd_data.get("extra_axes")
        extra_axes = None
        if ea_raw:
            extra_axes = {Axis[k]: v for k, v in ea_raw.items()}
        cat = _category(ct)
        if cat == CommandCategory.MOVING:
            kwargs: Dict[str, Any] = {
                "ct": ct,
                "end": tuple(cmd_data["end"]),
                "extra_axes": extra_axes,
            }
            if ct == CommandType.ARC_TO:
                kwargs["arc_params"] = (
                    cmd_data["center_offset"][0],
                    cmd_data["center_offset"][1],
                    cmd_data["clockwise"],
                )
            elif ct == CommandType.BEZIER_TO:
                kwargs["bezier_params"] = (
                    tuple(cmd_data["control1"]),
                    tuple(cmd_data["control2"]),
                )
            elif ct == CommandType.QUADRATIC_BEZIER_TO:
                kwargs["quad_params"] = tuple(cmd_data["control"])
            elif ct == CommandType.SCAN_LINE:
                kwargs["scanline"] = bytearray(cmd_data["power_values"])
        elif ct == CommandType.DWELL:
            kwargs = {
                "ct": ct,
                "dwell_duration": cmd_data["duration_ms"],
            }
        elif ct == CommandType.SET_POWER:
            kwargs = {"ct": ct, "power": cmd_data["power"]}
        elif ct in (
            CommandType.SET_CUT_SPEED,
            CommandType.SET_TRAVEL_SPEED,
        ):
            kwargs = {"ct": ct, "speed": cmd_data["speed"]}
        elif ct == CommandType.SET_FREQUENCY:
            kwargs = {
                "ct": ct,
                "frequency": cmd_data["frequency"],
            }
        elif ct == CommandType.SET_PULSE_WIDTH:
            kwargs = {
                "ct": ct,
                "pulse_width": cmd_data["pulse_width"],
            }
        elif ct == CommandType.SET_LASER:
            kwargs = {
                "ct": ct,
                "laser_uid": cmd_data["laser_uid"],
            }
        elif ct in (
            CommandType.LAYER_START,
            CommandType.LAYER_END,
        ):
            kwargs = {
                "ct": ct,
                "layer_uid": cmd_data["layer_uid"],
            }
        elif ct in (
            CommandType.WORKPIECE_START,
            CommandType.WORKPIECE_END,
        ):
            kwargs = {
                "ct": ct,
                "workpiece_uid": cmd_data["workpiece_uid"],
            }
        elif ct == CommandType.OPS_SECTION_START:
            kwargs = {
                "ct": ct,
                "section_type": SectionType[cmd_data["section_type"]],
                "section_workpiece_uid": cmd_data["workpiece_uid"],
            }
        elif ct == CommandType.OPS_SECTION_END:
            kwargs = {
                "ct": ct,
                "section_type": SectionType[cmd_data["section_type"]],
            }
        else:
            kwargs = {"ct": ct}
        return kwargs

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Ops:
        """Deserializes a dictionary into an Ops instance."""
        new_ops = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_ops.last_move_to = last_move

        for cmd_data in data.get("commands", []):
            try:
                kwargs = cls._create_command_from_dict(cmd_data)
                new_ops._soa.append(**kwargs)
            except TypeError as e:
                logger.warning(
                    f"Skipping unknown command during deserialization: {e}"
                )
        return new_ops

    def to_numpy_arrays(self) -> Dict[str, np.ndarray]:
        """
        Serializes the command list into a dictionary of NumPy arrays
        for efficient storage and transfer. This uses a
        Struct-of-Arrays approach.
        """
        num_cmds = len(self._soa)

        num_arcs = sum(
            1
            for i in range(num_cmds)
            if self._soa.command_type(i) == CommandType.ARC_TO
        )
        num_beziers = sum(
            1
            for i in range(num_cmds)
            if self._soa.command_type(i)
            in (CommandType.BEZIER_TO, CommandType.QUADRATIC_BEZIER_TO)
        )
        scanline_lengths = [
            len(self._soa.scanline_data(i))
            for i in range(num_cmds)
            if self._soa.command_type(i) == CommandType.SCAN_LINE
        ]
        total_scanline_bytes = sum(scanline_lengths)
        num_scanlines = len(scanline_lengths)

        # Array Definitions
        types = np.zeros(num_cmds, dtype=np.int32)
        endpoints = np.zeros((num_cmds, 3), dtype=np.float32)

        arc_data = np.zeros((num_arcs, 3), dtype=np.float32)
        arc_map = np.full(num_cmds, -1, dtype=np.int32)

        bezier_data = np.zeros((num_beziers, 6), dtype=np.float32)
        bezier_map = np.full(num_cmds, -1, dtype=np.int32)

        scanline_data_arr = np.zeros(total_scanline_bytes, dtype=np.uint8)
        scanline_map = np.full(
            num_cmds, -1, dtype=np.int32
        )  # Maps command index to scanline_indices index
        scanline_indices = np.zeros(
            (num_scanlines, 2), dtype=np.int32
        )  # start, end into scanline_data

        # Store non-geometric command data in a dict to be JSON serialized
        state_marker_cmds_data = {}

        # Data Population
        arc_idx, bezier_idx = 0, 0
        scanline_idx, scanline_offset = 0, 0
        extra_axes_map: Dict[str, Any] = {}
        for i in range(num_cmds):
            ct = self._soa.command_type(i)
            cat = _category(ct)
            types[i] = ct.value

            if ct == CommandType.BEZIER_TO:
                c1, c2 = self._soa.bezier_params(i)
                endpoints[i] = self._soa.endpoint(i)
                bezier_data[bezier_idx] = (
                    c1[0],
                    c1[1],
                    c1[2],
                    c2[0],
                    c2[1],
                    c2[2],
                )
                bezier_map[i] = bezier_idx
                bezier_idx += 1
            elif ct == CommandType.QUADRATIC_BEZIER_TO:
                c = self._soa.quad_params(i)
                endpoints[i] = self._soa.endpoint(i)
                bezier_data[bezier_idx] = (
                    c[0],
                    c[1],
                    c[2],
                    0.0,
                    0.0,
                    0.0,
                )
                bezier_map[i] = bezier_idx
                bezier_idx += 1
            elif cat == CommandCategory.MOVING:
                endpoints[i] = self._soa.endpoint(i)
            else:
                state_marker_cmds_data[str(i)] = self._cmd_to_dict(i)

            if ct == CommandType.ARC_TO:
                ci, cj, cw = self._soa.arc_params(i)
                arc_data[arc_idx] = (ci, cj, 1.0 if cw else 0.0)
                arc_map[i] = arc_idx
                arc_idx += 1

            if ct == CommandType.SCAN_LINE:
                pv = self._soa.scanline_data(i)
                length = len(pv)
                scanline_data_arr[
                    scanline_offset : scanline_offset + length
                ] = pv
                scanline_indices[scanline_idx] = (
                    scanline_offset,
                    scanline_offset + length,
                )
                scanline_map[i] = scanline_idx
                scanline_offset += length
                scanline_idx += 1

            if cat == CommandCategory.MOVING:
                ea = self._soa.extra_axes(i)
                if ea:
                    extra_axes_map[str(i)] = {
                        axis.name: float(value) for axis, value in ea.items()
                    }

        json_str = json.dumps(state_marker_cmds_data)
        json_bytes = np.frombuffer(json_str.encode("utf-8"), dtype=np.uint8)

        result = {
            "types": types,
            "endpoints": endpoints,
            "arc_data": arc_data,
            "arc_map": arc_map,
            "bezier_data": bezier_data,
            "bezier_map": bezier_map,
            "scanline_data": scanline_data_arr,
            "scanline_indices": scanline_indices,
            "scanline_map": scanline_map,
            "state_marker_json_bytes": json_bytes,
        }
        if extra_axes_map:
            ea_json_str = json.dumps(extra_axes_map)
            result["extra_axes_json"] = np.frombuffer(
                ea_json_str.encode("utf-8"), dtype=np.uint8
            )
        return result

    @classmethod
    def from_numpy_arrays(cls, arrays: Dict[str, np.ndarray]) -> "Ops":
        """
        Reconstructs an Ops object from a dictionary of NumPy arrays.
        """
        new_ops = cls()

        types = arrays["types"]
        endpoints = arrays["endpoints"]
        arc_data = arrays["arc_data"]
        arc_map = arrays["arc_map"]
        bezier_data = arrays.get(
            "bezier_data", np.zeros((0, 6), dtype=np.float32)
        )
        bezier_map = arrays.get("bezier_map", np.full(0, -1, dtype=np.int32))
        scanline_data = arrays["scanline_data"]
        scanline_indices = arrays["scanline_indices"]
        scanline_map = arrays["scanline_map"]

        json_bytes = arrays.get(
            "state_marker_json_bytes", np.array([], dtype=np.uint8)
        )
        if json_bytes.size > 0:
            json_str = json_bytes.tobytes().decode("utf-8")
            state_marker_cmds_data = json.loads(json_str)
        else:
            state_marker_cmds_data = {}
        extra_axes_data: Dict[str, Any] = {}
        ea_bytes = arrays.get("extra_axes_json")
        if ea_bytes is not None and ea_bytes.size > 0:
            ea_str = ea_bytes.tobytes().decode("utf-8")
            extra_axes_data = json.loads(ea_str)

        for i in range(len(types)):
            # Check if this index corresponds to a state/marker command
            if str(i) in state_marker_cmds_data:
                cmd_data = state_marker_cmds_data[str(i)]
                kwargs = cls._create_command_from_dict(cmd_data)
                new_ops._soa.append(**kwargs)
                continue

            # If not, it must be a geometric command.
            cmd_type_enum = types[i]
            ct = CommandType(cmd_type_enum)
            cat = _COMMAND_TYPE_TO_CATEGORY.get(ct)

            if cat != CommandCategory.MOVING:
                logger.warning(
                    f"Skipping unexpected non-moving command type: {ct}"
                )
                continue

            end_tuple = tuple(endpoints[i])

            if ct in (CommandType.MOVE_TO, CommandType.LINE_TO):
                new_ops._soa.append(ct=ct, end=end_tuple)
            elif ct == CommandType.ARC_TO:
                arc_idx = arc_map[i]
                i_val, j_val, is_cw = arc_data[arc_idx]
                new_ops._soa.append(
                    ct=ct,
                    end=end_tuple,
                    arc_params=(i_val, j_val, bool(is_cw)),
                )
            elif ct == CommandType.BEZIER_TO:
                bez_idx = bezier_map[i]
                c1x, c1y, c1z, c2x, c2y, c2z = bezier_data[bez_idx]
                new_ops._soa.append(
                    ct=ct,
                    end=end_tuple,
                    bezier_params=(
                        (float(c1x), float(c1y), float(c1z)),
                        (float(c2x), float(c2y), float(c2z)),
                    ),
                )
            elif ct == CommandType.QUADRATIC_BEZIER_TO:
                bez_idx = bezier_map[i]
                cx, cy, cz = bezier_data[bez_idx][:3]
                new_ops._soa.append(
                    ct=ct,
                    end=end_tuple,
                    quad_params=(float(cx), float(cy), float(cz)),
                )
            elif ct == CommandType.SCAN_LINE:
                scan_idx = scanline_map[i]
                start, end = scanline_indices[scan_idx]
                power_values = bytearray(scanline_data[start:end])
                new_ops._soa.append(
                    ct=ct, end=end_tuple, scanline=power_values
                )
            else:
                continue

            if str(i) in extra_axes_data:
                ea_raw = extra_axes_data[str(i)]
                ea = {Axis[k]: v for k, v in ea_raw.items()}
                new_ops._soa.set_extra_axes(new_ops._soa.__len__() - 1, ea)

        return new_ops

    @classmethod
    def from_geometry(cls, geometry: "Geometry") -> "Ops":
        """
        Creates an Ops object from a Geometry object, converting its
        path.
        """
        new_ops = cls()
        if geometry.data is None:
            new_ops.last_move_to = geometry.last_move_to
            return new_ops

        last_pos = (0.0, 0.0, 0.0)
        for cmd in geometry.iter_typed_commands():
            end = cmd.end
            if isinstance(cmd, PyCommand.Move):
                new_ops._soa.append(ct=CommandType.MOVE_TO, end=end)
            elif isinstance(cmd, PyCommand.Line):
                new_ops._soa.append(ct=CommandType.LINE_TO, end=end)
            elif isinstance(cmd, PyCommand.Arc):
                center_offset = cmd.center_offset
                clockwise = cmd.clockwise
                new_ops._soa.append(
                    ct=CommandType.ARC_TO,
                    end=end,
                    arc_params=(
                        center_offset[0],
                        center_offset[1],
                        clockwise,
                    ),
                )
            elif isinstance(cmd, PyCommand.Bezier):
                c1 = cmd.control1
                c2 = cmd.control2
                z0, z1 = last_pos[2], end[2]
                c1_3d = (
                    c1[0],
                    c1[1],
                    z0 * (2 / 3) + z1 * (1 / 3),
                )
                c2_3d = (
                    c2[0],
                    c2[1],
                    z0 * (1 / 3) + z1 * (2 / 3),
                )
                new_ops._soa.append(
                    ct=CommandType.BEZIER_TO,
                    end=end,
                    bezier_params=(c1_3d, c2_3d),
                )
            last_pos = end

        new_ops.last_move_to = geometry.last_move_to
        return new_ops

    def to_geometry(self) -> "Geometry":
        """
        Creates a Geometry path from this Ops object, including only
        the geometric commands.
        """
        new_geo = Geometry()
        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            end = self._soa.endpoint(i)
            if ct == CommandType.MOVE_TO:
                new_geo.move_to(*end)
            elif ct == CommandType.LINE_TO:
                new_geo.line_to(*end)
            elif ct == CommandType.ARC_TO:
                ci, cj = self._soa.arc_center_offset(i)
                cw = self._soa.arc_clockwise(i)
                new_geo.arc_to(end[0], end[1], ci, cj, cw, end[2])
            elif ct == CommandType.BEZIER_TO:
                c1, c2 = self._soa.bezier_params(i)
                new_geo.bezier_to(
                    end[0],
                    end[1],
                    c1[0],
                    c1[1],
                    c2[0],
                    c2[1],
                    end[2],
                )
        return new_geo

    def split_into_subpaths(self) -> List["Ops"]:
        """
        Split commands into subpaths at MoveToCommand boundaries.

        Each subpath begins with a MoveToCommand followed by zero or
        more non-MoveTo commands. State commands and markers that
        appear between MoveTo commands are grouped with the subpath
        that precedes them.

        Returns a list of Ops objects, one per subpath.
        """
        subpaths: List[List[int]] = []
        current: List[int] = []
        has_move_to = False
        for i in range(len(self._soa)):
            is_move = self._soa.command_type(i) == CommandType.MOVE_TO
            if is_move and has_move_to:
                subpaths.append(current)
                current = []
            if is_move:
                has_move_to = True
            current.append(i)
        if current:
            subpaths.append(current)
        result: List["Ops"] = []
        for indices in subpaths:
            sub = self.sub_ops(indices)
            result.append(sub)
        return result

    def iter_sections(self) -> Iterator[OpsSection]:
        """
        Iterate over parsed sections of this Ops object.

        Yields OpsSection tuples, each representing either a marked
        section (bounded by OpsSectionStartCommand/OpsSectionEndCommand)
        or a run of commands outside any section.
        """
        active_type: Optional[SectionType] = None
        marker_indices: List[int] = []
        content_indices: List[int] = []

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            if ct == CommandType.OPS_SECTION_START:
                if content_indices or marker_indices:
                    yield OpsSection(
                        active_type, marker_indices, content_indices
                    )
                    marker_indices = []
                    content_indices = []
                active_type = self._soa.section_type(i)
                marker_indices = [i]
            elif ct == CommandType.OPS_SECTION_END:
                marker_indices.append(i)
                yield OpsSection(active_type, marker_indices, content_indices)
                active_type = None
                marker_indices = []
                content_indices = []
            else:
                content_indices.append(i)

        if content_indices or marker_indices:
            yield OpsSection(active_type, marker_indices, content_indices)

    def iter_section_ranges(self) -> Iterator[OpsSectionRange]:
        """
        Like iter_sections() but yields index ranges instead of
        command lists.
        """
        active_type: Optional[SectionType] = None
        marker_indices: List[int] = []
        content_indices: List[int] = []

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            if ct == CommandType.OPS_SECTION_START:
                if content_indices or marker_indices:
                    yield OpsSectionRange(
                        active_type, marker_indices, content_indices
                    )
                    marker_indices = []
                    content_indices = []
                active_type = self._soa.section_type(i)
                marker_indices = [i]
            elif ct == CommandType.OPS_SECTION_END:
                marker_indices.append(i)
                yield OpsSectionRange(
                    active_type, marker_indices, content_indices
                )
                active_type = None
                marker_indices = []
                content_indices = []
            else:
                content_indices.append(i)

        if content_indices or marker_indices:
            yield OpsSectionRange(active_type, marker_indices, content_indices)

    def subpath_indices(self) -> List[List[int]]:
        """
        Like split_into_subpaths() but returns index ranges.
        """
        subpaths: List[List[int]] = []
        current: List[int] = []
        has_move_to = False
        for i in range(len(self._soa)):
            is_move = self._soa.command_type(i) == CommandType.MOVE_TO
            if is_move and has_move_to:
                subpaths.append(current)
                current = []
            if is_move:
                has_move_to = True
            current.append(i)
        if current:
            subpaths.append(current)
        return subpaths

    def sub_ops(self, indices: List[int]) -> Ops:
        """Creates a new Ops with copies of commands at the given
        indices."""
        result = Ops()
        for i in indices:
            kwargs = self._soa.deep_copy_entry(i)
            result._soa.append(**kwargs)
        result._invalidate_time_cache()
        return result

    def flip_ops(self) -> Ops:
        """
        Returns a new Ops with moving commands reversed (flipped).

        The states attached to each point describe the intended
        machine state while traveling TO that point. When flipping,
        states must be shifted to maintain this relationship. Arcs
        must also have their parameters recalculated relative to
        their new start point.
        """
        moving_indices = [
            i
            for i in range(len(self._soa))
            if _category(self._soa.command_type(i)) == CommandCategory.MOVING
        ]
        if len(moving_indices) <= 1:
            result = Ops()
            for i in moving_indices:
                kwargs = self._soa.deep_copy_entry(i)
                result._soa.append(**kwargs)
            result._invalidate_time_cache()
            return result

        # The first command of the new segment is a MoveTo, created
        # from the original segment's first MoveTo command.
        last_moving_end = self._soa.endpoint(moving_indices[-1])
        first_state = self._soa.state(moving_indices[0])
        result = Ops()
        result._soa.append(
            ct=CommandType.MOVE_TO,
            end=last_moving_end,
            state=first_state.__copy__() if first_state else None,
        )

        # Process the rest of the commands in reverse
        for k in range(len(moving_indices) - 2, -1, -1):
            orig_k_idx = moving_indices[k + 1]
            orig_prev_idx = moving_indices[k]
            ct = self._soa.command_type(orig_k_idx)
            new_end = self._soa.endpoint(orig_prev_idx)
            orig_state = self._soa.state(orig_k_idx)
            orig_ea = self._soa.extra_axes(orig_k_idx)

            if ct == CommandType.SCAN_LINE:
                # For a reversed scanline, we only need to reverse
                # the power data.
                pv = self._soa.scanline_data(orig_k_idx)
                result._soa.append(
                    ct=ct,
                    end=new_end,
                    scanline=bytearray(pv[::-1]),
                    extra_axes=dict(orig_ea) if orig_ea else None,
                    state=orig_state.__copy__() if orig_state else None,
                )
            elif ct == CommandType.BEZIER_TO:
                # A reversed cubic bezier P0→C1→C2→P3 becomes
                # P3→C2→C1→P0, so the control points must be swapped.
                c1, c2 = self._soa.bezier_params(orig_k_idx)
                result._soa.append(
                    ct=ct,
                    end=new_end,
                    bezier_params=(c2, c1),
                    extra_axes=dict(orig_ea) if orig_ea else None,
                    state=orig_state.__copy__() if orig_state else None,
                )
            elif ct == CommandType.ARC_TO:
                # The original arc's start point is the endpoint of
                # the command before it in the original segment.
                original_start = self._soa.endpoint(orig_prev_idx)
                original_end = self._soa.endpoint(orig_k_idx)
                ci, cj, cw = self._soa.arc_params(orig_k_idx)
                center_x = original_start[0] + ci
                center_y = original_start[1] + cj
                new_i = center_x - original_end[0]
                new_j = center_y - original_end[1]
                result._soa.append(
                    ct=ct,
                    end=new_end,
                    arc_params=(new_i, new_j, not cw),
                    extra_axes=dict(orig_ea) if orig_ea else None,
                    state=orig_state.__copy__() if orig_state else None,
                )
            elif ct == CommandType.MOVE_TO:
                result._soa.append(
                    ct=ct,
                    end=new_end,
                    extra_axes=dict(orig_ea) if orig_ea else None,
                    state=orig_state.__copy__() if orig_state else None,
                )
            elif ct == CommandType.LINE_TO:
                result._soa.append(
                    ct=ct,
                    end=new_end,
                    extra_axes=dict(orig_ea) if orig_ea else None,
                    state=orig_state.__copy__() if orig_state else None,
                )
            else:
                kwargs = self._soa.deep_copy_entry(orig_k_idx)
                result._soa.append(**kwargs)

        result._invalidate_time_cache()
        return result

    def __add__(self, ops: Ops) -> Ops:
        result = Ops()
        for i in range(len(self._soa)):
            kwargs = self._soa.deep_copy_entry(i)
            result._soa.append(**kwargs)
        for i in range(len(ops._soa)):
            kwargs = ops._soa.deep_copy_entry(i)
            result._soa.append(**kwargs)
        return result

    def __mul__(self, count: int) -> Ops:
        result = Ops()
        for _ in range(count):
            for i in range(len(self._soa)):
                kwargs = self._soa.deep_copy_entry(i)
                result._soa.append(**kwargs)
        return result

    def __len__(self) -> int:
        return len(self._soa)

    def is_empty(self) -> bool:
        """Checks if the Ops object contains any commands."""
        return len(self._soa) == 0

    def len(self) -> int:
        """Returns the number of commands in this Ops object."""
        return len(self._soa)

    def command_type(self, idx: int) -> CommandType:
        """Returns the CommandType of the command at the given index."""
        return self._soa.command_type(idx)

    def category(self, idx: int) -> CommandCategory:
        """Returns the CommandCategory of the command at the given
        index."""
        return self._soa.category(idx)

    def is_travel(self, idx: int) -> bool:
        """Returns True if the command at idx is a travel move."""
        return self._soa.command_type(idx) == CommandType.MOVE_TO

    def is_cutting(self, idx: int) -> bool:
        """Returns True if the command at idx is a cutting move."""
        return (
            self._soa.category(idx) == CommandCategory.MOVING
            and self._soa.command_type(idx) != CommandType.MOVE_TO
        )

    def is_state(self, idx: int) -> bool:
        """Returns True if the command at idx is a state command."""
        return self._soa.category(idx) == CommandCategory.STATE

    def is_marker(self, idx: int) -> bool:
        """Returns True if the command at idx is a marker command."""
        return self._soa.category(idx) == CommandCategory.MARKER

    def is_scanline(self, idx: int) -> bool:
        """Returns True if the command at idx is a
        ScanLinePowerCommand."""
        return self._soa.command_type(idx) == CommandType.SCAN_LINE

    def indices_of(self, ct: CommandType) -> List[int]:
        """Returns all indices whose command_type matches *ct*."""
        return [
            i for i in range(len(self._soa)) if self._soa.command_type(i) == ct
        ]

    def endpoint(self, idx: int) -> Point3D:
        """Returns the endpoint of a MovingCommand at the given
        index."""
        if self._soa.category(idx) != CommandCategory.MOVING:
            raise TypeError(f"Command at index {idx} is not a MovingCommand")
        return self._soa.endpoint(idx)

    def distance_at(self, idx: int, last_point: Optional[Point3D]) -> float:
        """
        Returns the 2D distance covered by the command at the given
        index.
        """
        if self._soa.category(idx) != CommandCategory.MOVING:
            return 0.0
        if last_point is None:
            return 0.0
        end = self._soa.endpoint(idx)
        return math.hypot(end[0] - last_point[0], end[1] - last_point[1])

    def arc_params(self, idx: int) -> Tuple[float, float, bool]:
        """
        Returns the arc parameters (i, j, clockwise) for an
        ArcToCommand at the given index.
        """
        if self._soa.command_type(idx) != CommandType.ARC_TO:
            raise TypeError(f"Command at index {idx} is not an ArcToCommand")
        return self._soa.arc_params(idx)

    def scanline_data(self, idx: int) -> memoryview:
        """
        Returns a memoryview of the power values for a
        ScanLinePowerCommand at the given index.
        """
        if self._soa.command_type(idx) != CommandType.SCAN_LINE:
            raise TypeError(
                f"Command at index {idx} is not a ScanLinePowerCommand"
            )
        return memoryview(self._soa.scanline_data(idx))

    def _apply_state_at(self, idx: int, state: State) -> None:
        ct = self._soa.command_type(idx)
        if ct == CommandType.SET_POWER:
            state.power = self._soa.power(idx)
        elif ct == CommandType.SET_CUT_SPEED:
            state.cut_speed = self._soa.speed(idx)
        elif ct == CommandType.SET_TRAVEL_SPEED:
            state.travel_speed = self._soa.speed(idx)
        elif ct == CommandType.ENABLE_AIR_ASSIST:
            state.air_assist = True
        elif ct == CommandType.DISABLE_AIR_ASSIST:
            state.air_assist = False
        elif ct == CommandType.SET_LASER:
            state.active_laser_uid = self._soa.laser_uid(idx)
        elif ct == CommandType.SET_FREQUENCY:
            state.frequency = self._soa.frequency(idx)
        elif ct == CommandType.SET_PULSE_WIDTH:
            state.pulse_width = self._soa.pulse_width(idx)

    def state_at(self, idx: int) -> MachineState:
        """Returns the effective machine state at the given command
        index."""
        state = State()
        for i in range(idx + 1):
            if self._soa.category(i) == CommandCategory.STATE:
                self._apply_state_at(i, state)
        return MachineState(
            power=state.power,
            air_assist=state.air_assist,
            cut_speed=state.cut_speed,
            travel_speed=state.travel_speed,
            active_laser_uid=state.active_laser_uid,
            frequency=state.frequency,
            pulse_width=state.pulse_width,
        )

    def preloaded_state(self, idx: int) -> MachineState:
        """Returns the preloaded machine state at the given index.
        Requires preload_state() to have been called first."""
        state = self._soa.state(idx)
        if state is None:
            raise ValueError(
                f"No preloaded state at index {idx}. "
                "Call preload_state() first."
            )
        return MachineState(
            power=state.power,
            air_assist=state.air_assist,
            cut_speed=state.cut_speed,
            travel_speed=state.travel_speed,
            active_laser_uid=state.active_laser_uid,
            frequency=state.frequency,
            pulse_width=state.pulse_width,
        )

    def set_state_on_moving(self, state_input: MachineState) -> None:
        """Sets state on all moving commands in this Ops."""
        state = State(
            power=state_input.power,
            air_assist=state_input.air_assist,
            cut_speed=state_input.cut_speed,
            travel_speed=state_input.travel_speed,
            active_laser_uid=state_input.active_laser_uid,
            frequency=state_input.frequency,
            pulse_width=state_input.pulse_width,
        )
        for i in range(len(self._soa)):
            if self._soa.category(i) == CommandCategory.MOVING:
                self._soa.set_state(i, state.__copy__())

    def set_state_at(self, idx: int, state_input: MachineState) -> None:
        """Sets state on the moving command at the given index."""
        state = State(
            power=state_input.power,
            air_assist=state_input.air_assist,
            cut_speed=state_input.cut_speed,
            travel_speed=state_input.travel_speed,
            active_laser_uid=state_input.active_laser_uid,
            frequency=state_input.frequency,
            pulse_width=state_input.pulse_width,
        )
        self._soa.set_state(idx, state)

    def inspect(self, idx: int) -> CommandInfo:
        """Returns a structured representation of the command at the
        given index, for testing assertions."""
        ct = self._soa.command_type(idx)
        cat = _category(ct)
        if cat == CommandCategory.MOVING:
            ea = self._soa.extra_axes(idx)
            result = CommandInfo(
                type=ct,
                end=self._soa.endpoint(idx),
                extra_axes=dict(ea) if ea else None,
                state=self._soa.state(idx),
            )
        else:
            result = CommandInfo(type=ct)
        if ct == CommandType.ARC_TO:
            ci, cj, cw = self._soa.arc_params(idx)
            result = result._replace(
                center_offset=(ci, cj),
                clockwise=cw,
            )
        elif ct == CommandType.BEZIER_TO:
            c1, c2 = self._soa.bezier_params(idx)
            result = result._replace(control1=c1, control2=c2)
        elif ct == CommandType.QUADRATIC_BEZIER_TO:
            c = self._soa.quad_params(idx)
            result = result._replace(control=c)
        elif ct == CommandType.SCAN_LINE:
            result = result._replace(
                power_values=bytes(self._soa.scanline_data(idx))
            )
        elif ct == CommandType.SET_POWER:
            result = result._replace(power=self._soa.power(idx))
        elif ct == CommandType.SET_CUT_SPEED:
            result = result._replace(speed=self._soa.speed(idx))
        elif ct == CommandType.SET_TRAVEL_SPEED:
            result = result._replace(speed=self._soa.speed(idx))
        elif ct == CommandType.SET_FREQUENCY:
            result = result._replace(frequency=self._soa.frequency(idx))
        elif ct == CommandType.SET_PULSE_WIDTH:
            result = result._replace(pulse_width=self._soa.pulse_width(idx))
        elif ct == CommandType.SET_LASER:
            result = result._replace(laser_uid=self._soa.laser_uid(idx))
        elif ct == CommandType.DWELL:
            result = result._replace(duration_ms=self._soa.dwell_duration(idx))
        elif ct in (CommandType.LAYER_START, CommandType.LAYER_END):
            result = result._replace(layer_uid=self._soa.layer_uid(idx))
        elif ct in (
            CommandType.WORKPIECE_START,
            CommandType.WORKPIECE_END,
        ):
            result = result._replace(
                workpiece_uid=self._soa.workpiece_uid(idx)
            )
        elif ct == CommandType.OPS_SECTION_START:
            result = result._replace(
                section_type=self._soa.section_type(idx).name,
                workpiece_uid=self._soa.section_workpiece_uid(idx),
            )
        elif ct == CommandType.OPS_SECTION_END:
            result = result._replace(
                section_type=self._soa.section_type(idx).name
            )
        return result

    def copy_command_from(self, source: "Ops", idx: int) -> None:
        """Deep-copies a command from a source Ops into this Ops
        object."""
        kwargs = source._soa.deep_copy_entry(idx)
        self._soa.append(**kwargs)
        self._invalidate_time_cache()

    def transfer_command_from(self, source: "Ops", idx: int) -> None:
        """Transfers a command from source into this Ops object.
        Use when the source Ops will not outlive this one."""
        kwargs = source._soa.copy_entry(idx)
        self._soa.append(**kwargs)
        self._invalidate_time_cache()

    def bezier_params(self, idx: int) -> Tuple[Point3D, Point3D]:
        """Returns (control1, control2) for a BezierToCommand."""
        if self._soa.command_type(idx) != CommandType.BEZIER_TO:
            raise TypeError(f"Command at index {idx} is not a BezierToCommand")
        return self._soa.bezier_params(idx)

    def quadratic_bezier_params(self, idx: int) -> Tuple[Point3D]:
        """Returns (control,) for a QuadraticBezierToCommand."""
        if self._soa.command_type(idx) != CommandType.QUADRATIC_BEZIER_TO:
            raise TypeError(
                f"Command at index {idx} is not a QuadraticBezierToCommand"
            )
        return (self._soa.quad_params(idx),)

    def dwell_duration(self, idx: int) -> float:
        """Returns the duration in ms for a DwellCommand."""
        if self._soa.command_type(idx) != CommandType.DWELL:
            raise TypeError(f"Command at index {idx} is not a DwellCommand")
        return self._soa.dwell_duration(idx)

    def power(self, idx: int) -> float:
        """Returns the power value for a SetPowerCommand."""
        if self._soa.command_type(idx) != CommandType.SET_POWER:
            raise TypeError(f"Command at index {idx} is not a SetPowerCommand")
        return self._soa.power(idx)

    def speed(self, idx: int) -> float:
        """Returns the speed value for SetCutSpeedCommand or
        SetTravelSpeedCommand."""
        ct = self._soa.command_type(idx)
        if ct not in (
            CommandType.SET_CUT_SPEED,
            CommandType.SET_TRAVEL_SPEED,
        ):
            raise TypeError(f"Command at index {idx} is not a speed command")
        return self._soa.speed(idx)

    def frequency(self, idx: int) -> int:
        """Returns the frequency for a SetFrequencyCommand."""
        if self._soa.command_type(idx) != CommandType.SET_FREQUENCY:
            raise TypeError(
                f"Command at index {idx} is not a SetFrequencyCommand"
            )
        return self._soa.frequency(idx)

    def pulse_width(self, idx: int) -> float:
        """Returns the pulse width for a SetPulseWidthCommand."""
        if self._soa.command_type(idx) != CommandType.SET_PULSE_WIDTH:
            raise TypeError(
                f"Command at index {idx} is not a SetPulseWidthCommand"
            )
        return self._soa.pulse_width(idx)

    def laser_uid(self, idx: int) -> str:
        """Returns the laser UID for a SetLaserCommand."""
        if self._soa.command_type(idx) != CommandType.SET_LASER:
            raise TypeError(f"Command at index {idx} is not a SetLaserCommand")
        return self._soa.laser_uid(idx)

    def layer_uid(self, idx: int) -> str:
        """Returns the layer UID for LayerStart/EndCommand."""
        ct = self._soa.command_type(idx)
        if ct not in (CommandType.LAYER_START, CommandType.LAYER_END):
            raise TypeError(f"Command at index {idx} is not a Layer command")
        return self._soa.layer_uid(idx)

    def workpiece_uid(self, idx: int) -> str:
        """Returns the workpiece UID for WorkpieceStart/EndCommand."""
        ct = self._soa.command_type(idx)
        if ct not in (
            CommandType.WORKPIECE_START,
            CommandType.WORKPIECE_END,
        ):
            raise TypeError(
                f"Command at index {idx} is not a Workpiece command"
            )
        return self._soa.workpiece_uid(idx)

    def section_params(self, idx: int) -> Tuple[SectionType, Optional[str]]:
        """Returns (section_type, workpiece_uid) for
        OpsSectionStartCommand, or (section_type, None) for
        OpsSectionEndCommand."""
        ct = self._soa.command_type(idx)
        if ct == CommandType.OPS_SECTION_START:
            return (
                self._soa.section_type(idx),
                self._soa.section_workpiece_uid(idx),
            )
        elif ct == CommandType.OPS_SECTION_END:
            return (self._soa.section_type(idx), None)
        raise TypeError(f"Command at index {idx} is not an OpsSection command")

    def extra_axes(self, idx: int) -> Optional[Dict[Axis, float]]:
        """Returns the extra axes dict for a MovingCommand, or None."""
        if self._soa.category(idx) != CommandCategory.MOVING:
            raise TypeError(f"Command at index {idx} is not a MovingCommand")
        return self._soa.extra_axes(idx)

    def linearize(self, idx: int, start_point: Point3D) -> "Ops":
        """Linearizes a geometric command into simple LineTo and
        SetPower commands, returning a new Ops object."""
        ct = self._soa.command_type(idx)
        end = self._soa.endpoint(idx)
        ea = self._soa.extra_axes(idx)
        if ct == CommandType.SCAN_LINE:
            result = Ops()
            pv = self._soa.scanline_data(idx)
            num_steps = len(pv)
            if num_steps == 0:
                return result
            p_start = np.array(start_point)
            p_end = np.array(end)
            line_vec = p_end - p_start
            seg_start_power = pv[0]
            result.set_power(seg_start_power / 255.0)
            for i in range(1, num_steps):
                cur_power = pv[i]
                if cur_power != seg_start_power:
                    t_end = i / float(num_steps)
                    seg_end = tuple(p_start + t_end * line_vec)
                    result.line_to(
                        *seg_end,
                        extra=dict(ea) if ea else None,
                    )
                    seg_start_power = cur_power
                    result.set_power(seg_start_power / 255.0)
            result.line_to(*end, extra=dict(ea) if ea else None)
            return result
        elif ct == CommandType.ARC_TO:
            ci, cj, cw = self._soa.arc_params(idx)
            arc_row = [
                CMD_TYPE_ARC,
                end[0],
                end[1],
                end[2],
                ci,
                cj,
                1.0 if cw else 0.0,
                0.0,
            ]
            segments = linearize_arc(arc_row, start_point)
            result = Ops()
            for _, seg_end in segments:
                result.line_to(
                    *seg_end,
                    extra=dict(ea) if ea else None,
                )
            return result
        elif ct == CommandType.BEZIER_TO:
            c1, c2 = self._soa.bezier_params(idx)
            polyline = linearize_bezier_segment(start_point, c1, c2, end)
            result = Ops()
            for pt in polyline[1:]:
                result.line_to(
                    *pt,
                    extra=dict(ea) if ea else None,
                )
            return result
        elif ct in (CommandType.MOVE_TO, CommandType.LINE_TO):
            result = Ops()
            if ct == CommandType.MOVE_TO:
                result.move_to(*end, extra=dict(ea) if ea else None)
            else:
                result.line_to(*end, extra=dict(ea) if ea else None)
            return result
        else:
            raise TypeError(
                f"Cannot linearize command at index {idx}: {ct.name}"
            )

    def copy(self) -> Ops:
        """Creates a copy of the Ops object."""
        new_ops = Ops()
        for i in range(len(self._soa)):
            kwargs = self._soa.deep_copy_entry(i)
            new_ops._soa.append(**kwargs)
        new_ops.last_move_to = self.last_move_to
        new_ops._time_dirty = self._time_dirty
        new_ops._cached_time = self._cached_time
        new_ops._time_params = self._time_params
        return new_ops

    def preload_state(self) -> None:
        """
        Walks through all commands, enriching each by the intended
        state of the machine.
        """
        state = State()
        for i in range(len(self._soa)):
            if self._soa.category(i) == CommandCategory.STATE:
                self._apply_state_at(i, state)
            elif self._soa.category(i) == CommandCategory.MOVING:
                self._soa.set_state(i, state.__copy__())

    def clear(self) -> None:
        self._soa = _SoA()
        self._invalidate_time_cache()

    def replace_all(self, source: "Ops") -> None:
        self._soa = _SoA()
        for i in range(len(source._soa)):
            kw = source._soa.deep_copy_entry(i)
            self._soa.append(**kw)
        self._invalidate_time_cache()

    def replace_with(self, source: "Ops") -> None:
        """Replaces all commands with those from source Ops."""
        self._soa = _SoA()
        for i in range(len(source._soa)):
            kwargs = source._soa.deep_copy_entry(i)
            self._soa.append(**kwargs)
        self.last_move_to = source.last_move_to
        self._invalidate_time_cache()

    def extend(self, other_ops: "Ops") -> None:
        """
        Appends all commands from another Ops object to this one.
        """
        if other_ops and len(other_ops._soa) > 0:
            for i in range(len(other_ops._soa)):
                kwargs = other_ops._soa.deep_copy_entry(i)
                self._soa.append(**kwargs)
            self._invalidate_time_cache()

    def move_to(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        self.last_move_to = (float(x), float(y), float(z))
        self._soa.append(
            ct=CommandType.MOVE_TO,
            end=self.last_move_to,
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def line_to(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        self._soa.append(
            ct=CommandType.LINE_TO,
            end=(float(x), float(y), float(z)),
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def close_path(self) -> None:
        """
        Convenience method that wraps line_to(). Makes a line to
        the last move_to point.
        """
        self.line_to(*self.last_move_to)

    def arc_to(
        self,
        x: float,
        y: float,
        i: float,
        j: float,
        clockwise: bool = True,
        z: float = 0.0,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """
        Adds an arc command with specified endpoint, center offsets,
        and direction (cw/ccw).
        """
        self._soa.append(
            ct=CommandType.ARC_TO,
            end=(float(x), float(y), float(z)),
            arc_params=(float(i), float(j), bool(clockwise)),
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def bezier_to(
        self,
        c1: Point3D,
        c2: Point3D,
        end: Point3D,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """
        Adds a cubic Bézier curve command.
        The curve starts from the current last point in the path.
        """
        if len(self._soa) == 0:
            logger.warning("bezier_to called without a starting point.")
            return
        last_end = None
        for j in range(len(self._soa) - 1, -1, -1):
            if self._soa.category(j) == CommandCategory.MOVING:
                last_end = self._soa.endpoint(j)
                break
        if last_end is None:
            logger.warning("bezier_to called without a starting point.")
            return

        self._soa.append(
            ct=CommandType.BEZIER_TO,
            end=end,
            bezier_params=(c1, c2),
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def quadratic_bezier_to(
        self,
        control: Point3D,
        end: Point3D,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """
        Adds a quadratic Bézier curve command.
        """
        self._soa.append(
            ct=CommandType.QUADRATIC_BEZIER_TO,
            end=end,
            quad_params=control,
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def set_power(self, power: float) -> None:
        """
        Sets the intended laser power for subsequent cutting commands.
        This is a state declaration, not an immediate command to turn
        on the laser.

        Args:
            power: The normalized power level, from 0.0 (off) to
                   1.0 (full power).
        """
        self._soa.append(ct=CommandType.SET_POWER, power=power)

    def set_cut_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent cutting commands.
        This is a state declaration.
        """
        self._soa.append(ct=CommandType.SET_CUT_SPEED, speed=int(speed))
        self._invalidate_time_cache()

    def set_travel_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent travel commands.
        This is a state declaration.
        """
        self._soa.append(ct=CommandType.SET_TRAVEL_SPEED, speed=int(speed))
        self._invalidate_time_cache()

    def dwell(self, duration_ms: float) -> None:
        self._soa.append(ct=CommandType.DWELL, dwell_duration=duration_ms)
        self._invalidate_time_cache()

    def enable_air_assist(self, enable: bool = True) -> None:
        """
        Sets the intended state of the air assist for subsequent
        commands. This is a state declaration.
        """
        if enable:
            self._soa.append(ct=CommandType.ENABLE_AIR_ASSIST)
            self._invalidate_time_cache()
        else:
            self.disable_air_assist()

    def disable_air_assist(self) -> None:
        """
        Sets the intended state of the air assist for subsequent
        commands. This is a state declaration.
        """
        self._soa.append(ct=CommandType.DISABLE_AIR_ASSIST)
        self._invalidate_time_cache()

    def set_laser(self, laser_uid: str) -> None:
        """
        Sets the intended active laser for subsequent commands.
        This is a state declaration.
        """
        self._soa.append(ct=CommandType.SET_LASER, laser_uid=laser_uid)
        self._invalidate_time_cache()

    def set_frequency(self, frequency: int) -> None:
        """
        Sets the laser PWM frequency for subsequent cutting commands.
        This is a state declaration.

        Args:
            frequency: The PWM frequency in Hz.
        """
        self._soa.append(ct=CommandType.SET_FREQUENCY, frequency=frequency)
        self._invalidate_time_cache()

    def set_pulse_width(self, pulse_width: float) -> None:
        """
        Sets the laser pulse width for subsequent cutting commands.
        This is a state declaration.

        Args:
            pulse_width: The pulse width in microseconds.
        """
        self._soa.append(
            ct=CommandType.SET_PULSE_WIDTH, pulse_width=pulse_width
        )
        self._invalidate_time_cache()

    def scan_to(
        self,
        x: float,
        y: float,
        z: float = 0.0,
        power_values: Optional[bytearray] = None,
        extra: Optional[Dict[Axis, float]] = None,
    ) -> None:
        """
        Adds a scan line command with variable power values.
        """
        if power_values is None:
            power_values = bytearray([255])

        self._soa.append(
            ct=CommandType.SCAN_LINE,
            end=(float(x), float(y), float(z)),
            scanline=power_values,
            extra_axes=extra,
        )
        self._invalidate_time_cache()

    def job_start(self) -> None:
        """
        Adds a job start marker command.
        This is a logical marker for the generator.
        """
        self._soa.append(ct=CommandType.JOB_START)
        self._invalidate_time_cache()

    def job_end(self) -> None:
        """
        Adds a job end marker command.
        This is a logical marker for the generator.
        """
        self._soa.append(ct=CommandType.JOB_END)
        self._invalidate_time_cache()

    def layer_start(self, layer_uid: str) -> None:
        """
        Adds a layer start marker command.
        This is a logical marker for the generator.

        Args:
            layer_uid: Unique identifier for the layer.
        """
        self._soa.append(ct=CommandType.LAYER_START, layer_uid=layer_uid)
        self._invalidate_time_cache()

    def layer_end(self, layer_uid: str) -> None:
        """
        Adds a layer end marker command.
        This is a logical marker for the generator.

        Args:
            layer_uid: Unique identifier for the layer.
        """
        self._soa.append(ct=CommandType.LAYER_END, layer_uid=layer_uid)
        self._invalidate_time_cache()

    def workpiece_start(self, workpiece_uid: str) -> None:
        """
        Adds a workpiece start marker command.
        This is a logical marker for the generator.

        Args:
            workpiece_uid: Unique identifier for the workpiece.
        """
        self._soa.append(
            ct=CommandType.WORKPIECE_START,
            workpiece_uid=workpiece_uid,
        )
        self._invalidate_time_cache()

    def workpiece_end(self, workpiece_uid: str) -> None:
        """
        Adds a workpiece end marker command.
        This is a logical marker for the generator.

        Args:
            workpiece_uid: Unique identifier for the workpiece.
        """
        self._soa.append(
            ct=CommandType.WORKPIECE_END,
            workpiece_uid=workpiece_uid,
        )
        self._invalidate_time_cache()

    def ops_section_start(
        self, section_type: SectionType, workpiece_uid: str
    ) -> None:
        """
        Adds an ops section start marker command.
        This marks the beginning of a semantically distinct block.

        Args:
            section_type: The semantic type of the section.
            workpiece_uid: Unique identifier for the workpiece.
        """
        self._soa.append(
            ct=CommandType.OPS_SECTION_START,
            section_type=section_type,
            section_workpiece_uid=workpiece_uid,
        )
        self._invalidate_time_cache()

    def ops_section_end(self, section_type: SectionType) -> None:
        """
        Adds an ops section end marker command.
        This marks the end of a semantically distinct block.

        Args:
            section_type: The semantic type of the section.
        """
        self._soa.append(
            ct=CommandType.OPS_SECTION_END,
            section_type=section_type,
        )
        self._invalidate_time_cache()

    def rect(self, include_travel: bool = False) -> Rect:
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area in the XY plane.

        Args:
            include_travel: If True, travel moves are included in
            the bounds.
        """
        min_x, min_y = float("inf"), float("inf")
        max_x, max_y = float("-inf"), float("-inf")

        curr_x, curr_y = 0.0, 0.0
        if self.last_move_to and include_travel:
            curr_x, curr_y = self.last_move_to[0], self.last_move_to[1]

        # Accumulate points in lists for vectorized min/max at the end.
        points_x: List[float] = []
        points_y: List[float] = []

        # Store arc parameters for post-processing their bulges
        arcs: List[Tuple[float, float, float, float, float, float, bool]] = []

        has_content = False

        for i in range(len(self._soa)):
            if self._soa.category(i) != CommandCategory.MOVING:
                continue

            ct = self._soa.command_type(i)
            end = self._soa.endpoint(i)
            end_x, end_y = end[0], end[1]

            if ct == CommandType.MOVE_TO:
                if include_travel:
                    points_x.append(curr_x)
                    points_y.append(curr_y)
                    points_x.append(end_x)
                    points_y.append(end_y)
                    has_content = True
                curr_x, curr_y = end_x, end_y
                continue

            # Drawing Command (Line, Arc, Scan)
            points_x.append(curr_x)
            points_y.append(curr_y)
            points_x.append(end_x)
            points_y.append(end_y)
            has_content = True

            if ct == CommandType.ARC_TO:
                ci, cj, cw = self._soa.arc_params(i)
                arcs.append((curr_x, curr_y, end_x, end_y, ci, cj, cw))

            curr_x, curr_y = end_x, end_y

        if not has_content:
            return 0.0, 0.0, 0.0, 0.0

        # Vectorized min/max
        if points_x:
            arr_x = np.array(points_x)
            arr_y = np.array(points_y)
            min_x = min(min_x, np.min(arr_x))
            max_x = max(max_x, np.max(arr_x))
            min_y = min(min_y, np.min(arr_y))
            max_y = max(max_y, np.max(arr_y))

        # Process Arcs to adjust bounds for bulges
        for ax, ay, bx, by, i, j, cw in arcs:
            radius = math.hypot(i, j)

            # Explicit full-circle check
            if (
                math.isclose(ax, bx, abs_tol=1e-9)
                and math.isclose(ay, by, abs_tol=1e-9)
                and radius > 1e-9
            ):
                cx, cy = ax + i, ay + j
                min_x = min(min_x, cx - radius)
                max_x = max(max_x, cx + radius)
                min_y = min(min_y, cy - radius)
                max_y = max(max_y, cy + radius)
            else:
                abox = get_arc_bounds((ax, ay), (bx, by), (i, j), cw)
                min_x = min(min_x, abox[0])
                min_y = min(min_y, abox[1])
                max_x = max(max_x, abox[2])
                max_y = max(max_y, abox[3])

        return min_x, min_y, max_x, max_y

    def get_frame(
        self,
        power: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Ops:
        """
        Returns a new Ops object containing four move_to operations
        forming a frame around the occupied area.
        """
        min_x, min_y, max_x, max_y = self.rect()
        if (min_x, min_y, max_x, max_y) == (0.0, 0.0, 0.0, 0.0):
            return Ops()

        frame_ops = Ops()
        if power is not None:
            frame_ops.set_power(power)
        if speed is not None:
            frame_ops.set_cut_speed(speed)
        frame_ops.move_to(min_x, min_y)
        frame_ops.line_to(min_x, max_y)
        frame_ops.line_to(max_x, max_y)
        frame_ops.line_to(max_x, min_y)
        frame_ops.line_to(min_x, min_y)
        return frame_ops

    def distance(self) -> float:
        """
        Calculates the total 2D path length for all moving commands.
        """
        total = 0.0
        last: Optional[Point3D] = None
        for i in range(len(self._soa)):
            if self._soa.category(i) == CommandCategory.MOVING:
                end = self._soa.endpoint(i)
                if last is not None:
                    total += math.hypot(end[0] - last[0], end[1] - last[1])
                last = end
        return total

    def cut_distance(self) -> float:
        """
        Like distance(), but only counts 2D cut distance.
        """
        total = 0.0
        last: Optional[Point3D] = None
        for i in range(len(self._soa)):
            if self._soa.category(i) == CommandCategory.MOVING:
                end = self._soa.endpoint(i)
                if self.is_cutting(i) and last is not None:
                    total += math.hypot(end[0] - last[0], end[1] - last[1])
                last = end
        return total

    def estimate_time(
        self,
        default_cut_speed: float = 1000.0,
        default_travel_speed: float = 3000.0,
        acceleration: float = 1000.0,
    ) -> float:
        """
        Estimates the execution time of the operations in seconds.

        This method calculates the time required to execute all
        commands in the Ops object, taking into account different
        speeds for cutting and travel movements, as well as
        acceleration considerations. Results are cached and only
        recomputed when the command list changes or when different
        machine parameters are used.

        Args:
            default_cut_speed: Default cutting speed in mm/min if
                               not specified by state commands.
            default_travel_speed: Default travel speed in mm/min
                                  if not specified by state commands.
            acceleration: Machine acceleration in mm/s² for more
                         accurate time estimation.

        Returns:
            The estimated execution time in seconds.
        """
        if len(self._soa) == 0:
            return 0.0

        params = (default_cut_speed, default_travel_speed, acceleration)
        if not self._time_dirty and self._time_params == params:
            return self._cached_time

        self._cached_time = estimate_time(
            self,
            default_cut_speed,
            default_travel_speed,
            acceleration,
        )
        self._time_dirty = False
        self._time_params = params
        return self._cached_time

    def segments(self) -> Generator[List[int], None, None]:
        segment: List[int] = []
        for i in range(len(self._soa)):
            if not segment:
                segment.append(i)
                continue

            if self.is_travel(i):
                yield segment
                segment = [i]

            elif self.is_cutting(i):
                segment.append(i)

            elif self.is_state(i) or self.is_marker(i):
                yield segment
                yield [i]
                segment = []

        if segment:
            yield segment

    def segment_indices(self) -> Generator[List[int], None, None]:
        """
        Like segments() but yields lists of command indices instead
        of command objects.
        """
        segment: List[int] = []
        for i in range(len(self._soa)):
            if not segment:
                segment.append(i)
                continue

            if self.is_travel(i):
                yield segment
                segment = [i]
            elif self.is_cutting(i):
                segment.append(i)
            elif self.is_state(i) or self.is_marker(i):
                yield segment
                yield [i]
                segment = []

        if segment:
            yield segment

    def without_state(self) -> Ops:
        """Returns new Ops containing only moving and marker
        commands."""
        result = Ops()
        for i in range(len(self._soa)):
            if self._soa.category(i) != CommandCategory.STATE:
                kwargs = self._soa.deep_copy_entry(i)
                result._soa.append(**kwargs)
        result._invalidate_time_cache()
        return result

    def group_by_state_continuity(self) -> List[Ops]:
        """Splits into segments based on state/marker boundaries."""
        if len(self._soa) == 0:
            return []

        seg_indices: List[List[int]] = []
        current: List[int] = []

        for i in range(len(self._soa)):
            if self.is_marker(i):
                if current:
                    seg_indices.append(current)
                seg_indices.append([i])
                current = []
                continue

            if not current:
                current.append(i)
                continue

            last_state = self._soa.state(current[-1])
            op_state = self._soa.state(i)
            if (
                last_state
                and op_state
                and last_state.air_assist == op_state.air_assist
            ):
                current.append(i)
            else:
                seg_indices.append(current)
                current = [i]

        if current:
            seg_indices.append(current)

        result = []
        for seg in seg_indices:
            ops = Ops()
            for idx in seg:
                kwargs = self._soa.deep_copy_entry(idx)
                ops._soa.append(**kwargs)
            ops._invalidate_time_cache()
            result.append(ops)
        return result

    def transform(self, matrix: "np.ndarray") -> "Ops":
        """
        Applies a transformation matrix to all geometric commands.
        If the transform is non-uniform (contains non-uniform
        scaling or shear), arcs will be linearized to preserve
        their shape.

        Args:
            matrix: A 4x4 NumPy transformation matrix.

        Returns:
            The Ops object itself for chaining.
        """
        # Check for non-uniform scaling or shear by comparing the
        # length of transformed basis vectors.
        v_x = matrix @ np.array([1, 0, 0, 0])
        v_y = matrix @ np.array([0, 1, 0, 0])
        len_x = np.linalg.norm(v_x[:2])
        len_y = np.linalg.norm(v_y[:2])
        is_non_uniform = not np.isclose(len_x, len_y)

        # A reflection (negative determinant) reverses the arc
        # direction.
        det = matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]
        flip_cw = det < 0
        rot_scale_matrix = matrix[:3, :3]

        new_soa = _SoA()
        last_point_untransformed: Optional[Point3D] = None

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            cat = self._soa.category(i)
            original_cmd_end = (
                self._soa.endpoint(i)
                if cat == CommandCategory.MOVING
                else None
            )

            if ct == CommandType.ARC_TO and is_non_uniform:
                start_point = last_point_untransformed or (0.0, 0.0, 0.0)
                end = self._soa.endpoint(i)
                ci, cj, cw = self._soa.arc_params(i)
                arc_row = [
                    CMD_TYPE_ARC,
                    end[0],
                    end[1],
                    end[2],
                    ci,
                    cj,
                    1.0 if cw else 0.0,
                    0.0,
                ]
                segments = linearize_arc(arc_row, start_point)
                st = self._soa.state(i)
                ea = self._soa.extra_axes(i)
                for _, p2 in segments:
                    point_vec = np.array([p2[0], p2[1], p2[2], 1.0])
                    tv = matrix @ point_vec
                    new_soa.append(
                        ct=CommandType.LINE_TO,
                        end=tuple(tv[:3]),
                        extra_axes=dict(ea) if ea else None,
                        state=st.__copy__() if st else None,
                    )

            elif cat == CommandCategory.MOVING:
                end = self._soa.endpoint(i)
                point_vec = np.array([*end, 1.0])
                transformed_vec = matrix @ point_vec
                new_end = tuple(transformed_vec[:3])
                st = self._soa.state(i)
                ea = self._soa.extra_axes(i)

                if ct == CommandType.ARC_TO:
                    ci, cj = self._soa.arc_center_offset(i)
                    cw = self._soa.arc_clockwise(i)
                    offset_vec_3d = np.array([ci, cj, 0])
                    new_offset = rot_scale_matrix @ offset_vec_3d
                    new_cw = (not cw) if flip_cw else cw
                    new_soa.append(
                        ct=ct,
                        end=new_end,
                        arc_params=(new_offset[0], new_offset[1], new_cw),
                        extra_axes=dict(ea) if ea else None,
                        state=st.__copy__() if st else None,
                    )
                elif ct == CommandType.BEZIER_TO:
                    c1, c2 = self._soa.bezier_params(i)
                    c1_vec = np.array([*c1, 1.0])
                    t_c1 = matrix @ c1_vec
                    c2_vec = np.array([*c2, 1.0])
                    t_c2 = matrix @ c2_vec
                    new_soa.append(
                        ct=ct,
                        end=new_end,
                        bezier_params=(tuple(t_c1[:3]), tuple(t_c2[:3])),
                        extra_axes=dict(ea) if ea else None,
                        state=st.__copy__() if st else None,
                    )
                elif ct == CommandType.QUADRATIC_BEZIER_TO:
                    c = self._soa.quad_params(i)
                    c_vec = np.array([*c, 1.0])
                    t_c = matrix @ c_vec
                    new_soa.append(
                        ct=ct,
                        end=new_end,
                        quad_params=tuple(t_c[:3]),
                        extra_axes=dict(ea) if ea else None,
                        state=st.__copy__() if st else None,
                    )
                else:
                    new_soa.append(
                        ct=ct,
                        end=new_end,
                        extra_axes=dict(ea) if ea else None,
                        state=st.__copy__() if st else None,
                    )

            else:
                # Non-moving commands (state, markers) are passed
                # through.
                kw = self._soa.deep_copy_entry(i)
                new_soa.append(**kw)

            # Crucially, update the last_point tracker with the
            # endpoint from BEFORE the transformation for the next
            # iteration.
            if original_cmd_end is not None:
                last_point_untransformed = original_cmd_end

        self._soa = new_soa
        self._invalidate_time_cache()
        last_move_vec = np.array([*self.last_move_to, 1.0])
        transformed_last_move_vec = matrix @ last_move_vec
        self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> Ops:
        """Translate geometric commands."""
        matrix = np.identity(4)
        matrix[0, 3] = dx
        matrix[1, 3] = dy
        matrix[2, 3] = dz
        return self.transform(matrix)

    def scale(self, sx: float, sy: float, sz: float = 1.0) -> Ops:
        """Scales all geometric commands."""
        matrix = np.diag([sx, sy, sz, 1.0])
        return self.transform(matrix)

    def rotate(self, angle_deg: float, cx: float, cy: float) -> Ops:
        """Rotates all points around a center (cx, cy) in the XY
        plane."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Create a 4x4 transformation matrix for rotation around
        # a point: T(-cx, -cy) * R(angle) * T(cx, cy)
        translate_to_origin = np.identity(4)
        translate_to_origin[0, 3] = -cx
        translate_to_origin[1, 3] = -cy

        rotation_matrix = np.identity(4)
        rotation_matrix[0, 0] = cos_a
        rotation_matrix[0, 1] = -sin_a
        rotation_matrix[1, 0] = sin_a
        rotation_matrix[1, 1] = cos_a

        translate_back = np.identity(4)
        translate_back[0, 3] = cx
        translate_back[1, 3] = cy

        matrix = translate_back @ rotation_matrix @ translate_to_origin
        return self.transform(matrix)

    def translate_layers(
        self,
        default_offset: Tuple[float, float, float],
        layer_offsets: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> None:
        """
        Apply per-layer coordinate offsets.

        Commands outside any layer get *default_offset*.
        When a LayerStartCommand is encountered, *layer_offsets* is
        checked for that layer's offset. If the layer is not in the
        dict the default offset is used.

        Each offset is subtracted from every MovingCommand's end
        point (i.e. ``end = end - offset``).
        """
        active_offset = default_offset
        in_named_layer = False

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            if ct == CommandType.LAYER_START:
                if layer_offsets is not None:
                    active_offset = layer_offsets.get(
                        self._soa.layer_uid(i), default_offset
                    )
                else:
                    active_offset = default_offset
                in_named_layer = True
                continue

            if ct == CommandType.LAYER_END:
                active_offset = default_offset
                in_named_layer = False
                continue

            if not in_named_layer:
                active_offset = default_offset

            if self._soa.category(i) == CommandCategory.MOVING:
                end = self._soa.endpoint(i)
                ox, oy, oz = active_offset
                if ox != 0.0 or oy != 0.0 or oz != 0.0:
                    self._soa.set_endpoint(
                        i,
                        (end[0] - ox, end[1] - oy, end[2] - oz),
                    )

        self._invalidate_time_cache()

    def transform_layers(self, callback: Callable[[str, Ops], None]) -> None:
        """
        Call *callback(layer_uid, layer_ops)* for each layer.

        For each ``LayerStartCommand`` in the stream, collects all
        commands from that marker through the matching
        ``LayerEndCommand`` (inclusive) and passes them to
        *callback* as a sub-``Ops``. The callback may modify the
        commands in-place via the ``Ops`` API — changes are
        spliced back automatically.

        Commands that are not inside any layer are never passed to
        the callback.
        """
        i = 0
        while i < len(self._soa):
            if self._soa.command_type(i) != CommandType.LAYER_START:
                i += 1
                continue

            layer_uid = self._soa.layer_uid(i)
            layer_start = i
            collected_indices: List[int] = []
            while i < len(self._soa):
                collected_indices.append(i)
                i += 1
                if self._soa.command_type(i - 1) == CommandType.LAYER_END:
                    break

            layer_end = i
            layer_ops = self.sub_ops(collected_indices)
            callback(layer_uid, layer_ops)

            # Splice the (possibly modified) layer_ops back
            new_soa = _SoA()
            for j in range(layer_start):
                kw = self._soa.deep_copy_entry(j)
                new_soa.append(**kw)
            for j in range(layer_ops.len()):
                kw = layer_ops._soa.deep_copy_entry(j)
                new_soa.append(**kw)
            for j in range(layer_end, len(self._soa)):
                kw = self._soa.deep_copy_entry(j)
                new_soa.append(**kw)
            self._soa = new_soa
            # Adjust i for the new soa length
            i = layer_start + layer_ops.len()

    def transform_moving(
        self,
        on_endpoint: Callable[[List[float], Dict[Axis, float]], None],
        on_aux_point: Optional[Callable[[List[float]], None]] = None,
    ) -> None:
        """
        Transform all moving commands via in-place callbacks.

        Iterates over every ``MovingCommand`` in this ``Ops``
        object. For each one, *on_endpoint* is called with the
        endpoint as a mutable ``List[float]`` and the
        ``extra_axes`` dict. If *on_aux_point* is provided, it is
        called for each auxiliary geometry point:

        * ``ArcToCommand`` — center_offset as ``List[float]``.
        * ``BezierToCommand`` — control1 and control2 each as
          ``List[float]``.
        * ``QuadraticBezierToCommand`` — control as
          ``List[float]``.

        Both callbacks modify their arguments in-place. After the
        callback returns the values are written back. The time
        cache is invalidated automatically.
        """
        for i in range(len(self._soa)):
            if self._soa.category(i) != CommandCategory.MOVING:
                continue

            ct = self._soa.command_type(i)
            end = list(self._soa.endpoint(i))
            ea = self._soa.extra_axes(i) or {}
            on_endpoint(end, ea)
            self._soa.set_endpoint(i, (end[0], end[1], end[2]))
            if ea:
                self._soa.set_extra_axes(i, ea)

            if on_aux_point is not None:
                if ct == CommandType.ARC_TO:
                    ci, cj = self._soa.arc_center_offset(i)
                    off = [ci, cj]
                    on_aux_point(off)
                    self._soa.set_arc_params(i, center_offset=(off[0], off[1]))
                elif ct == CommandType.BEZIER_TO:
                    c1, c2 = self._soa.bezier_params(i)
                    for cp_idx, cp in enumerate([c1, c2]):
                        cp_list = list(cp)
                        on_aux_point(cp_list)
                        if cp_idx == 0:
                            c1 = (
                                cp_list[0],
                                cp_list[1],
                                cp_list[2],
                            )
                        else:
                            c2 = (
                                cp_list[0],
                                cp_list[1],
                                cp_list[2],
                            )
                    self._soa.set_bezier_params(i, c1, c2)
                elif ct == CommandType.QUADRATIC_BEZIER_TO:
                    c = list(self._soa.quad_params(i))
                    on_aux_point(c)
                    self._soa.set_quad_params(i, (c[0], c[1], c[2]))

        self._invalidate_time_cache()

    def linearize_all(self) -> None:
        """
        Replaces all complex commands (e.g., Arcs) with simple
        LineToCommands.
        """
        new_soa = _SoA()
        last_point: Point3D = (0.0, 0.0, 0.0)
        # Find initial position, in case path doesn't start with
        # MoveTo
        for i in range(len(self._soa)):
            if self._soa.command_type(i) == CommandType.MOVE_TO:
                last_point = self._soa.endpoint(i)
                break

        for i in range(len(self._soa)):
            if self._soa.category(i) == CommandCategory.MOVING:
                linearized_ops = self.linearize(i, last_point)
                for j in range(linearized_ops.len()):
                    kw = linearized_ops._soa.deep_copy_entry(j)
                    new_soa.append(**kw)
                    end = linearized_ops._soa.endpoint(j)
                    # Update last_point to the end of the last
                    # generated segment
                    last_point = end
            else:
                # Non-moving commands (state, markers) are passed
                # through.
                kw = self._soa.deep_copy_entry(i)
                new_soa.append(**kw)
        self._soa = new_soa
        self._invalidate_time_cache()

    def linearize_curves(self) -> None:
        """
        Replaces all CurveToCommand instances with their linearized
        LineToCommand output. This is the safety-net fallback used
        before encoding when the machine does not support native
        curves.
        """
        new_soa = _SoA()
        last_point: Point3D = (0.0, 0.0, 0.0)
        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            if ct == CommandType.MOVE_TO:
                last_point = self._soa.endpoint(i)
            if ct in (
                CommandType.BEZIER_TO,
                CommandType.QUADRATIC_BEZIER_TO,
            ):
                linearized_ops = self.linearize(i, last_point)
                for j in range(linearized_ops.len()):
                    kw = linearized_ops._soa.deep_copy_entry(j)
                    new_soa.append(**kw)
                    end = linearized_ops._soa.endpoint(j)
                    last_point = end
            else:
                kw = self._soa.deep_copy_entry(i)
                new_soa.append(**kw)
                if self._soa.category(i) == CommandCategory.MOVING:
                    last_point = self._soa.endpoint(i)
        self._soa = new_soa
        self._invalidate_time_cache()

    def linearize_arcs(self) -> None:
        """
        Replaces all ArcToCommand instances with their linearized
        LineToCommand output. Curve commands are left intact.
        """
        new_soa = _SoA()
        last_point: Point3D = (0.0, 0.0, 0.0)
        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            if ct == CommandType.MOVE_TO:
                last_point = self._soa.endpoint(i)
            if ct == CommandType.ARC_TO:
                linearized_ops = self.linearize(i, last_point)
                for j in range(linearized_ops.len()):
                    kw = linearized_ops._soa.deep_copy_entry(j)
                    new_soa.append(**kw)
                    end = linearized_ops._soa.endpoint(j)
                    last_point = end
            else:
                kw = self._soa.deep_copy_entry(i)
                new_soa.append(**kw)
                if self._soa.category(i) == CommandCategory.MOVING:
                    last_point = self._soa.endpoint(i)
        self._soa = new_soa
        self._invalidate_time_cache()

    def clip_at(self, x: float, y: float, width: float) -> bool:
        """
        Finds the closest point on a continuous path to (x, y) and
        creates a gap of the given width centered at that point,
        measured along the path's length.

        This method works by linearizing the target subpath into a
        polyline, calculating the clip region parametrically along
        the polyline's length, and then reconstructing the polyline
        with the gap. Arcs within the affected subpath will be
        converted to line segments.

        Args:
            x: The x-coordinate of the point to clip near.
            y: The y-coordinate of the point to clip near.
            width: The desired width of the gap in world units.

        Returns:
            True if a clip was successfully performed, False
            otherwise.
        """
        if width <= 1e-6:
            return False

        # 1. Find the closest segment on the entire path
        temp_geo = self.to_geometry()
        closest = temp_geo.find_closest_point(x, y)
        if not closest:
            return False

        segment_index, _, point_on_path = closest
        dist_sq = (x - point_on_path[0]) ** 2 + (y - point_on_path[1]) ** 2
        if dist_sq > (width * 2) ** 2:
            return False

        # 2. Convert geometry index to command index (geometry
        # excludes state commands, so we need to map)
        command_index = 0
        geo_idx = 0
        for cmd_idx in range(len(self._soa)):
            if self._soa.category(cmd_idx) == CommandCategory.MOVING:
                if geo_idx == segment_index:
                    command_index = cmd_idx
                    break
                geo_idx += 1
        else:
            return False

        # 3. Identify the continuous subpath containing the hit
        # segment
        start_idx = 0
        for i in range(command_index, -1, -1):
            if self._soa.command_type(i) == CommandType.MOVE_TO:
                start_idx = i
                break

        end_idx = len(self._soa)
        for i in range(start_idx + 1, len(self._soa)):
            if self._soa.command_type(i) == CommandType.MOVE_TO:
                end_idx = i
                break

        if start_idx >= len(self._soa):
            return False
        if self._soa.category(start_idx) != CommandCategory.MOVING:
            return False

        # 4. Create a temporary, linearized version of the subpath
        subpath_indices = list(range(start_idx, end_idx))
        temp_ops = self.sub_ops(subpath_indices)
        temp_ops.preload_state()
        temp_ops.linearize_all()

        if temp_ops.len() < 2:
            return False

        # 4. Find closest point on the *linearized* path and
        # calculate distance
        linear_geo_cmds = [
            j
            for j in range(temp_ops.len())
            if temp_ops.command_type(j)
            in (CommandType.MOVE_TO, CommandType.LINE_TO)
        ]

        if len(linear_geo_cmds) < 2:
            return False

        linear_temp_geo = temp_ops.to_geometry()
        linear_closest = linear_temp_geo.find_closest_point(x, y)
        if not linear_closest:
            return False

        linear_segment_idx, linear_t, _ = linear_closest

        hit_dist = 0.0
        last_pos = temp_ops.endpoint(linear_geo_cmds[0])

        for idx_i in range(1, linear_segment_idx):
            j = linear_geo_cmds[idx_i]
            end_pt = temp_ops.endpoint(j)
            hit_dist += math.dist(last_pos[:2], end_pt[:2])
            last_pos = end_pt

        hit_segment_j = linear_geo_cmds[linear_segment_idx]
        hit_end = temp_ops.endpoint(hit_segment_j)
        dist = math.dist(last_pos[:2], hit_end[:2])
        hit_dist += linear_t * dist

        # 5. Define gap start and end distances
        gap_start_dist = max(0.0, hit_dist - width / 2.0)
        gap_end_dist = hit_dist + width / 2.0

        # 6. Rebuild the subpath
        def _clip_1d(d1, d2, g1, g2):
            kept = []
            if d1 < g1:
                kept.append((d1, min(d2, g1)))
            if d2 > g2:
                kept.append((max(d1, g2), d2))
            return kept

        new_subpath = Ops()
        # Start with MoveTo from the first command
        new_subpath._soa.append(**temp_ops._soa.deep_copy_entry(0))
        accum_dist = 0.0
        last_pos = temp_ops.endpoint(0)

        for j in range(1, temp_ops.len()):
            ct_j = temp_ops.command_type(j)
            if ct_j == CommandType.LINE_TO:
                p1 = last_pos
                p2 = temp_ops.endpoint(j)
                seg_len = math.dist(p1[:2], p2[:2])

                if seg_len < 1e-9:
                    last_pos = p2
                    continue

                seg_start_dist = accum_dist
                seg_end_dist = accum_dist + seg_len

                kept = _clip_1d(
                    seg_start_dist,
                    seg_end_dist,
                    gap_start_dist,
                    gap_end_dist,
                )

                p1_np, p2_np = np.array(p1), np.array(p2)
                vec = p2_np - p1_np

                for start_d, end_d in kept:
                    t_start = (start_d - seg_start_dist) / seg_len
                    t_end = (end_d - seg_start_dist) / seg_len
                    start_pt = tuple(p1_np + t_start * vec)
                    end_pt = tuple(p1_np + t_end * vec)

                    last_kept_pos: Optional[Tuple[float, ...]] = None
                    # Find the last endpoint in new_subpath
                    for ri in range(new_subpath.len() - 1, -1, -1):
                        if (
                            new_subpath._soa.category(ri)
                            == CommandCategory.MOVING
                        ):
                            last_kept_pos = new_subpath._soa.endpoint(ri)
                            break
                    assert last_kept_pos is not None

                    if math.dist(last_kept_pos, start_pt) > 1e-6:
                        new_subpath.move_to(*start_pt)

                    new_subpath.line_to(*end_pt)

                last_pos = p2
                accum_dist += seg_len
            else:
                # Handle state/marker commands
                if not (gap_start_dist < accum_dist < gap_end_dist):
                    kw = temp_ops._soa.deep_copy_entry(j)
                    new_subpath._soa.append(**kw)

        # Post-process to preserve original endpoint if it was
        # clipped
        original_endpoint = self._soa.endpoint(end_idx - 1)
        new_endpoint = None
        if new_subpath.len() > 0:
            for ri in range(new_subpath.len() - 1, -1, -1):
                if new_subpath._soa.category(ri) == CommandCategory.MOVING:
                    new_endpoint = new_subpath._soa.endpoint(ri)
                    break

        if (
            new_endpoint is None
            or math.dist(original_endpoint, new_endpoint) > 1e-6
        ):
            new_subpath.move_to(*original_endpoint)

        # 7. Replace original subpath
        new_soa = _SoA()
        for j in range(start_idx):
            kw = self._soa.deep_copy_entry(j)
            new_soa.append(**kw)
        for j in range(new_subpath.len()):
            kw = new_subpath._soa.deep_copy_entry(j)
            new_soa.append(**kw)
        for j in range(end_idx, len(self._soa)):
            kw = self._soa.deep_copy_entry(j)
            new_soa.append(**kw)
        self._soa = new_soa
        self._invalidate_time_cache()
        return True

    def _add_clipped_segment_to_ops(
        self,
        segment: Optional[Tuple[Point3D, Point3D]],
        new_ops: Ops,
        current_pen_pos: Optional[Point3D],
    ) -> Optional[Point3D]:
        """
        Processes a single clipped segment, adding MoveTo/LineTo
        commands to the new_ops object as needed.

        Returns the updated pen position.
        """
        if segment:
            p1_clipped, p2_clipped = segment

            # A new move is needed if the pen is up (None) or if
            # there's a gap.
            dist_to_start = (
                math.dist(current_pen_pos, p1_clipped)
                if current_pen_pos
                else float("inf")
            )

            # Use a small tolerance for floating point comparisons
            if dist_to_start > 1e-6:
                new_ops.move_to(*p1_clipped)

            new_ops.line_to(*p2_clipped)
            # The new pen position is the end of the clipped segment
            return p2_clipped
        else:
            # The segment was fully clipped, so the pen is now "up"
            return None

    def clip_rect(self, rect: Rect) -> Ops:
        """
        Clips the Ops to the given rectangle.
        Returns a new, clipped Ops object.
        """
        new_ops = Ops()
        if len(self._soa) == 0:
            return new_ops

        last_point: Point3D = (0.0, 0.0, 0.0)
        # Tracks the last known position of the pen *within the
        # clipped area*. None means the pen is "up" or outside the
        # clip rect.
        clipped_pen_pos: Optional[Point3D] = None

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            cat = self._soa.category(i)

            if cat in (CommandCategory.STATE, CommandCategory.MARKER):
                kw = self._soa.deep_copy_entry(i)
                new_ops._soa.append(**kw)
                continue

            if cat != CommandCategory.MOVING:
                continue

            # Special handling for ScanLinePowerCommand to prevent
            # linearization.
            if ct == CommandType.SCAN_LINE:
                end = self._soa.endpoint(i)
                clipped_segment = clip_line_segment_with_rect(
                    last_point, end, rect
                )
                if clipped_segment:
                    new_start, new_end = clipped_segment

                    # Calculate the start and end `t` values (0-1) of
                    # the clipped segment relative to the original.
                    p_start_orig = np.array(last_point)
                    p_end_orig = np.array(end)
                    vec_orig = p_end_orig - p_start_orig
                    len_sq = np.dot(vec_orig, vec_orig)

                    if len_sq > 1e-9:
                        t_start = (
                            np.dot(
                                np.array(new_start) - p_start_orig,
                                vec_orig,
                            )
                            / len_sq
                        )
                        t_end = (
                            np.dot(
                                np.array(new_end) - p_start_orig,
                                vec_orig,
                            )
                            / len_sq
                        )
                    else:
                        t_start, t_end = 0.0, 1.0

                    t_start = max(0.0, min(1.0, t_start))
                    t_end = max(0.0, min(1.0, t_end))

                    # Slice the power_values array based on the `t`
                    # values.
                    pv = self._soa.scanline_data(i)
                    num_values = len(pv)
                    idx_start = int(num_values * t_start)
                    idx_end = int(num_values * t_end)
                    new_power_values = bytearray(pv[idx_start:idx_end])

                    if new_power_values:
                        # Since scanlines are discrete, we always need
                        # a move
                        if (
                            clipped_pen_pos is None
                            or math.dist(clipped_pen_pos, new_start) > 1e-6
                        ):
                            new_ops.move_to(*new_start)
                        new_ops._soa.append(
                            ct=CommandType.SCAN_LINE,
                            end=new_end,
                            scanline=new_power_values,
                        )
                        clipped_pen_pos = new_end

                last_point = end
                continue

            if ct == CommandType.MOVE_TO:
                end = self._soa.endpoint(i)
                last_point = end
                clipped_pen_pos = None
                continue

            # Linearize the command into a series of simpler commands
            linearized_ops = self.linearize(i, last_point)

            # Process each linearized segment
            p_current_segment_start = last_point
            for j in range(linearized_ops.len()):
                p_current_segment_end = linearized_ops.endpoint(j)

                clipped_segment = clip_line_segment_with_rect(
                    p_current_segment_start,
                    p_current_segment_end,
                    rect,
                )
                clipped_pen_pos = self._add_clipped_segment_to_ops(
                    clipped_segment, new_ops, clipped_pen_pos
                )
                p_current_segment_start = p_current_segment_end

            # The next command starts where the original unclipped
            # command ended
            last_point = self._soa.endpoint(i)

        return new_ops

    def dump(self) -> None:
        for seg_indices in self.segment_indices():
            for idx in seg_indices:
                ct = self._soa.command_type(idx)
                cat = self._soa.category(idx)
                if cat == CommandCategory.MOVING:
                    end = self._soa.endpoint(idx)
                    print(f"  {ct.name}: {end}")
                else:
                    print(f"  {ct.name}")

    def subtract_regions(self, regions: List[Polygon]) -> "Ops":
        """
        Clips the Ops by subtracting a list of polygonal regions.
        This modifies the Ops object in place and returns it.
        """
        if not regions or len(self._soa) == 0:
            return self

        new_ops = Ops()
        last_point: Point3D = (0.0, 0.0, 0.0)
        # Tracks the last known pen position of a *kept* segment
        pen_pos: Optional[Point3D] = None

        # Add any leading state/marker commands before the first move
        first_move_idx = next(
            (
                i
                for i in range(len(self._soa))
                if self._soa.category(i) == CommandCategory.MOVING
            ),
            len(self._soa),
        )
        for i in range(first_move_idx):
            kw = self._soa.deep_copy_entry(i)
            new_ops._soa.append(**kw)

        for i in range(len(self._soa)):
            ct = self._soa.command_type(i)
            cat = self._soa.category(i)

            if cat != CommandCategory.MOVING:
                # State/marker commands are handled as they appear
                # between moves
                if new_ops.len() == 0:
                    kw = self._soa.deep_copy_entry(i)
                    new_ops._soa.append(**kw)
                else:
                    kw = self._soa.deep_copy_entry(i)
                    new_ops._soa.append(**kw)
                continue

            end = self._soa.endpoint(i)

            if ct == CommandType.MOVE_TO:
                last_point = end
                pen_pos = None
                continue

            if ct == CommandType.SCAN_LINE:
                kept_segments = subtract_polygons_from_line_segment(
                    last_point, end, regions
                )
                pv = self._soa.scanline_data(i)
                num_values = len(pv)
                p_start_orig = np.array(last_point)
                p_end_orig = np.array(end)
                vec_orig = p_end_orig - p_start_orig
                len_sq = np.dot(vec_orig, vec_orig)

                for new_start, new_end in kept_segments:
                    if len_sq > 1e-9:
                        t_start = (
                            np.dot(
                                np.array(new_start) - p_start_orig,
                                vec_orig,
                            )
                            / len_sq
                        )
                        t_end = (
                            np.dot(
                                np.array(new_end) - p_start_orig,
                                vec_orig,
                            )
                            / len_sq
                        )
                    else:
                        t_start, t_end = 0.0, 1.0

                    idx_start = int(num_values * t_start)
                    idx_end = int(num_values * t_end)
                    new_power = bytearray(pv[idx_start:idx_end])

                    if new_power:
                        if (
                            pen_pos is None
                            or math.dist(pen_pos, new_start) > 1e-6
                        ):
                            new_ops.move_to(*new_start)
                        new_ops._soa.append(
                            ct=CommandType.SCAN_LINE,
                            end=new_end,
                            scanline=new_power,
                        )
                        pen_pos = new_end
                last_point = end
                continue

            # Linearize cutting command into segments
            linearized_ops = self.linearize(i, last_point)

            p_current_segment_start = last_point
            for j in range(linearized_ops.len()):
                p_current_segment_end = linearized_ops.endpoint(j)

                kept_segments = subtract_polygons_from_line_segment(
                    p_current_segment_start,
                    p_current_segment_end,
                    regions,
                )
                for sub_p1, sub_p2 in kept_segments:
                    if pen_pos is None or math.dist(pen_pos, sub_p1) > 1e-6:
                        new_ops.move_to(*sub_p1)
                    new_ops.line_to(*sub_p2)
                    pen_pos = sub_p2
                p_current_segment_start = p_current_segment_end

            last_point = end

        # Replace self with new_ops
        self._soa = new_ops._soa
        self._invalidate_time_cache()
        # Update last_move_to to a valid point if ops is not empty
        if len(self._soa) > 0:
            for j in range(len(self._soa) - 1, -1, -1):
                if self._soa.command_type(j) == CommandType.MOVE_TO:
                    self.last_move_to = self._soa.endpoint(j)
                    break
        return self

    def clip_to_regions(
        self,
        regions: List[List[Tuple[float, float]]],
        tolerance: float = 0.3,
    ) -> "Ops":
        """
        Clips the Ops to only include parts that lie inside the
        given polygonal regions.

        Arcs that are fully contained within the regions are
        preserved as arc commands. Arcs that intersect region
        boundaries fall back to linearised clipping with arc
        re-fitting.

        This method modifies the Ops object in place and returns
        it.

        Args:
            regions: List of closed polygons defining valid areas.
                     Each polygon is a list of (x, y) tuples.
            tolerance: Tolerance for linearization and polygon
                       operations

        Returns:
            The modified Ops object (self), containing only the
            clipped paths
        """
        from .clipping import clip_ops_to_regions

        if not regions or len(self._soa) == 0:
            return self
        return clip_ops_to_regions(self, regions, tolerance)
