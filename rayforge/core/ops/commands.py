from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
import math
from ..geo import linearize as geo_linearize
from ..geo.bezier import linearize_bezier_segment
from ..geo.types import Point3D
from .axis import Axis


@dataclass
class State:
    power: float = 0.0  # Normalized power from 0.0 to 1.0
    air_assist: bool = False
    cut_speed: Optional[int] = None
    travel_speed: Optional[int] = None
    active_laser_uid: Optional[str] = None

    def allow_rapid_change(self, target_state: State) -> bool:
        """
        Returns True if a change to the target state should be allowed
        in a rapid manner, i.e. for each gcode instruction. For example,
        changing air-assist should not be done too frequently, because
        it could damage the air pump.

        Changing the laser power rapidly is unproblematic.
        """
        return self.air_assist == target_state.air_assist


class Command:
    """
    Note that the state attribute is not set by default. It is later
    filled during the pre-processing stage, where state commands are
    removed.
    """

    def __init__(
        self,
        end: Optional[Point3D] = None,
        state: Optional["State"] = None,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        self.end: Optional[Point3D] = end
        self.state: Optional["State"] = (
            state  # Intended state during execution
        )
        self.extra_axes: Dict[Axis, float] = extra_axes or {}

    def __repr__(self) -> str:
        return f"<{super().__repr__()} {self.__dict__}"

    def apply_to_state(self, state: "State") -> None:
        pass

    def distance(self, last_point: Optional[Point3D]) -> float:
        """Calculates the 2D distance covered by this command."""
        return 0.0

    def is_state_command(self) -> bool:
        """Whether this command modifies the machine state (power, speed)."""
        return False

    def is_cutting_command(self) -> bool:
        """Whether it is a cutting movement."""
        return False

    def is_travel_command(self) -> bool:
        """Whether it is a non-cutting movement."""
        return False

    def is_marker(self) -> bool:
        """Whether this is a logical marker for the generator."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the command to a dictionary."""
        return {"type": self.__class__.__name__}


class MovingCommand(Command, ABC):
    end: Point3D  # type: ignore[reportRedeclaration]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["end"] = self.end
        if self.extra_axes:
            d["extra_axes"] = {
                axis.name: value for axis, value in self.extra_axes.items()
            }
        return d

    @abstractmethod
    def linearize(self, start_point: Point3D) -> List[Command]:
        """
        Returns a list of simpler commands (e.g., LineToCommand) that
        approximate this command. For simple commands, it may return a list
        containing only itself.
        """
        pass

    def distance(self, last_point: Optional[Point3D]) -> float:
        """Calculates the 2D distance of the move."""
        # Use the command's own start_point if it has one (duck typing),
        # otherwise use the endpoint of the last command.
        start = getattr(self, "start_point", last_point)
        if start is None:
            return 0.0
        return math.hypot(self.end[0] - start[0], self.end[1] - start[1])


class MoveToCommand(MovingCommand):
    def is_travel_command(self) -> bool:
        return True

    def linearize(self, start_point: Point3D) -> List[Command]:
        return [self]


class LineToCommand(MovingCommand):
    def is_cutting_command(self) -> bool:
        return True

    def linearize(self, start_point: Point3D) -> List[Command]:
        return [self]


class ArcToCommand(MovingCommand):
    def __init__(
        self,
        end: Point3D,
        center_offset: Tuple[float, float],
        clockwise: bool,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        super().__init__(end, extra_axes=extra_axes)
        self.center_offset = center_offset
        self.clockwise = clockwise

    def is_cutting_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["center_offset"] = self.center_offset
        d["clockwise"] = self.clockwise
        return d

    def linearize(self, start_point: Point3D) -> List[Command]:
        """Approximates the arc with a series of LineToCommands."""
        segments = geo_linearize.linearize_arc(self, start_point)
        new_cmds = []
        for _, end in segments:
            line_cmd = LineToCommand(end, extra_axes=dict(self.extra_axes))
            line_cmd.state = self.state
            new_cmds.append(line_cmd)
        return new_cmds

    def reverse_geometry(
        self,
        original_start: Point3D,
        original_end: Point3D,
    ) -> None:
        """
        Recalculates the center offset and direction for when this arc
        is used in a reversed segment. The command's own `end` property is
        assumed to have already been set to the new endpoint (the original
        start point).
        """
        # Original center is calculated from the original start point
        center_x = original_start[0] + self.center_offset[0]
        center_y = original_start[1] + self.center_offset[1]

        # New offset is from the new start point (original end) to the center
        new_i = center_x - original_end[0]
        new_j = center_y - original_end[1]

        self.center_offset = (new_i, new_j)
        self.clockwise = not self.clockwise


class CurveToCommand(MovingCommand, ABC):
    def is_cutting_command(self) -> bool:
        return True


class BezierToCommand(CurveToCommand):
    def __init__(
        self,
        end: Point3D,
        control1: Point3D,
        control2: Point3D,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        super().__init__(end, extra_axes=extra_axes)
        self.control1 = control1
        self.control2 = control2

    def linearize(self, start_point: Point3D) -> List[Command]:
        polyline = linearize_bezier_segment(
            start_point, self.control1, self.control2, self.end
        )
        new_cmds = []
        for pt in polyline[1:]:
            line_cmd = LineToCommand(pt, extra_axes=dict(self.extra_axes))
            line_cmd.state = self.state
            new_cmds.append(line_cmd)
        return new_cmds

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["control1"] = self.control1
        d["control2"] = self.control2
        return d


class QuadraticBezierToCommand(CurveToCommand):
    def __init__(
        self,
        end: Point3D,
        control: Point3D,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        super().__init__(end, extra_axes=extra_axes)
        self.control = control

    def linearize(self, start_point: Point3D) -> List[Command]:
        p0 = start_point
        p1 = self.end
        c = self.control
        c1 = (
            p0[0] + 2.0 / 3.0 * (c[0] - p0[0]),
            p0[1] + 2.0 / 3.0 * (c[1] - p0[1]),
            p0[2] + 2.0 / 3.0 * (c[2] - p0[2]),
        )
        c2 = (
            p1[0] + 2.0 / 3.0 * (c[0] - p1[0]),
            p1[1] + 2.0 / 3.0 * (c[1] - p1[1]),
            p1[2] + 2.0 / 3.0 * (c[2] - p1[2]),
        )
        polyline = linearize_bezier_segment(p0, c1, c2, p1)
        new_cmds = []
        for pt in polyline[1:]:
            line_cmd = LineToCommand(pt, extra_axes=dict(self.extra_axes))
            line_cmd.state = self.state
            new_cmds.append(line_cmd)
        return new_cmds

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["control"] = self.control
        return d


class SetPowerCommand(Command):
    def __init__(self, power: float) -> None:
        """
        Initializes a command to set the laser power.

        Args:
            power: The normalized power level, from 0.0 (off) to
                   1.0 (full power).

        Raises:
            ValueError: If power is outside the [0.0, 1.0] range.
        """
        super().__init__()
        power = float(power)
        if not (0.0 <= power <= 1.0):
            raise ValueError(
                f"Power must be between 0.0 and 1.0, but got {power}"
            )
        self.power: float = power

    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.power = self.power

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["power"] = self.power
        return d


class SetCutSpeedCommand(Command):
    def __init__(self, speed: int) -> None:
        super().__init__()
        self.speed: int = speed

    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.cut_speed = self.speed

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["speed"] = self.speed
        return d


class SetTravelSpeedCommand(Command):
    def __init__(self, speed: int) -> None:
        super().__init__()
        self.speed: int = speed

    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.travel_speed = self.speed

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["speed"] = self.speed
        return d


class EnableAirAssistCommand(Command):
    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.air_assist = True


class DisableAirAssistCommand(Command):
    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.air_assist = False


class SetLaserCommand(Command):
    def __init__(self, laser_uid: str) -> None:
        super().__init__()
        self.laser_uid = laser_uid

    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: "State") -> None:
        state.active_laser_uid = self.laser_uid

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["laser_uid"] = self.laser_uid
        return d


@dataclass(frozen=True, repr=False)
class JobStartCommand(Command):
    def is_marker(self) -> bool:
        return True


@dataclass(frozen=True, repr=False)
class JobEndCommand(Command):
    def is_marker(self) -> bool:
        return True


@dataclass(frozen=True, repr=False)
class LayerStartCommand(Command):
    layer_uid: str

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["layer_uid"] = self.layer_uid
        return d


@dataclass(frozen=True, repr=False)
class LayerEndCommand(Command):
    layer_uid: str

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["layer_uid"] = self.layer_uid
        return d


@dataclass(frozen=True, repr=False)
class WorkpieceStartCommand(Command):
    workpiece_uid: str

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["workpiece_uid"] = self.workpiece_uid
        return d


@dataclass(frozen=True, repr=False)
class WorkpieceEndCommand(Command):
    workpiece_uid: str

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["workpiece_uid"] = self.workpiece_uid
        return d


class SectionType(Enum):
    """Defines the semantic type of a block of Ops commands."""

    VECTOR_OUTLINE = auto()
    RASTER_FILL = auto()


@dataclass(frozen=True)
class OpsSectionStartCommand(Command):
    """Marks the beginning of a semantically distinct block of Ops."""

    section_type: SectionType
    workpiece_uid: str  # Provides context to downstream transformers

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["section_type"] = self.section_type.name
        d["workpiece_uid"] = self.workpiece_uid
        return d


@dataclass(frozen=True)
class OpsSectionEndCommand(Command):
    """Marks the end of a block."""

    section_type: SectionType

    def is_marker(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["section_type"] = self.section_type.name
        return d


class DwellCommand(Command):
    def __init__(self, duration_ms: float) -> None:
        super().__init__()
        self.duration_ms: float = duration_ms

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["duration_ms"] = self.duration_ms
        return d


class ScanLinePowerCommand(MovingCommand):
    """
    A specialized command for raster engraving that encodes a line segment
    with continuously varying power levels. This is more efficient than a long
    sequence of SetPower and LineTo commands.

    The power is internally stored as a bytearray of values from 0-255,
    where 0 is off and 255 is full power. The length of the array must
    match the number of pixels in the line.
    """

    def __init__(
        self,
        end: Point3D,
        power_values: bytearray,
        extra_axes: Optional[Dict[Axis, float]] = None,
    ) -> None:
        super().__init__(end, extra_axes=extra_axes)
        self.power_values = power_values

    @property
    def normalized_power_values(self) -> np.ndarray:
        """
        Public API to get power values as a normalized float array (0.0-1.0).
        """
        return np.frombuffer(self.power_values, dtype=np.uint8) / 255.0

    def is_cutting_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the command to a dictionary."""
        d = super().to_dict()
        d["power_values"] = list(self.power_values)
        return d

    def linearize(self, start_point: Point3D) -> List[Command]:
        """
        Deconstructs the scan line into an efficient sequence of SetPower and
        LineTo commands by grouping consecutive pixels of the same power.
        """
        commands: List[Command] = []
        num_steps = len(self.power_values)
        if num_steps == 0:
            return []

        p_start_vec = np.array(start_point)
        p_end_vec = np.array(self.end)
        line_vec = p_end_vec - p_start_vec

        # Start the first segment
        segment_start_power = self.power_values[0]
        commands.append(SetPowerCommand(segment_start_power / 255.0))

        for i in range(1, num_steps):
            current_power = self.power_values[i]
            if current_power != segment_start_power:
                # Power has changed. The previous segment ended at pixel i-1.
                # The geometric end point corresponds to t = i / num_steps.
                t_end = i / float(num_steps)
                segment_end_point = p_start_vec + t_end * line_vec
                commands.append(
                    LineToCommand(
                        tuple(segment_end_point),
                        extra_axes=dict(self.extra_axes),
                    )
                )

                # Start the new segment
                segment_start_power = current_power
                commands.append(SetPowerCommand(segment_start_power / 255.0))

        # Add the final LineTo command for the last segment, which always
        # goes to the very end of the scan line.
        commands.append(
            LineToCommand(self.end, extra_axes=dict(self.extra_axes))
        )

        return commands

    def split_by_power(
        self, start_point: Point3D, min_power: int
    ) -> List[Command]:
        """
        Splits the scanline into multiple segments
          (MoveTo, ScanLinePowerCommand)
        based on a power threshold, creating segments for only the "on" parts.
        This method does NOT add overscan; it creates commands with the exact
        geometry of the active segments.

        Args:
            start_point: The (x, y, z) starting point of the full scanline.
            min_power: Power values <= this will be considered "off".

        Returns:
            A list of Command objects representing the active segments.
        """
        if not self.power_values:
            return []

        powers = np.array(self.power_values, dtype=np.uint8)
        is_on = powers > min_power
        if not np.any(is_on):
            return []  # Fully blank line

        # Find contiguous blocks of "on" pixels
        padded = np.concatenate(([False], is_on, [False]))
        diffs = np.diff(padded.astype(int))
        starts = np.where(diffs == 1)[0]
        ends = np.where(diffs == -1)[0]

        p_start_vec = np.array(start_point)
        p_end_vec = np.array(self.end)
        line_vec = p_end_vec - p_start_vec

        result_commands: List[Command] = []
        num_steps = len(self.power_values)

        for start_idx, end_idx in zip(starts, ends):
            # A chunk from start_idx to end_idx-1
            t_start = start_idx / num_steps
            t_end = end_idx / num_steps

            chunk_start_pt = p_start_vec + t_start * line_vec
            chunk_end_pt = p_start_vec + t_end * line_vec

            power_slice = self.power_values[start_idx:end_idx]

            move_cmd = MoveToCommand(tuple(chunk_start_pt))
            scan_cmd = ScanLinePowerCommand(
                tuple(chunk_end_pt), bytearray(power_slice)
            )
            result_commands.extend([move_cmd, scan_cmd])

        return result_commands


COMMAND_TYPE_MAP = {
    MoveToCommand: 1,
    LineToCommand: 2,
    ArcToCommand: 3,
    BezierToCommand: 6,
    QuadraticBezierToCommand: 7,
    ScanLinePowerCommand: 4,
    DwellCommand: 5,
    SetPowerCommand: 10,
    SetCutSpeedCommand: 11,
    SetTravelSpeedCommand: 12,
    EnableAirAssistCommand: 13,
    DisableAirAssistCommand: 14,
    SetLaserCommand: 15,
    JobStartCommand: 100,
    JobEndCommand: 101,
    LayerStartCommand: 102,
    LayerEndCommand: 103,
    WorkpieceStartCommand: 104,
    WorkpieceEndCommand: 105,
    OpsSectionStartCommand: 106,
    OpsSectionEndCommand: 107,
}

# The reverse mapping for deserialization
COMMAND_CLASS_MAP = {v: k for k, v in COMMAND_TYPE_MAP.items()}
