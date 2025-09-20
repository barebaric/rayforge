from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any, List
from abc import ABC, abstractmethod
import numpy as np
import math
from ..geo import linearize as geo_linearize


@dataclass
class State:
    power: int = 0
    air_assist: bool = False
    cut_speed: Optional[int] = None
    travel_speed: Optional[int] = None

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
        end: Optional[Tuple[float, float, float]] = None,
        state: Optional["State"] = None,
    ) -> None:
        # x/y/z of the end position. Is None for state commands
        self.end: Optional[Tuple[float, float, float]] = end
        self.state: Optional["State"] = (
            state  # Intended state during execution
        )
        self._state_ref_for_pyreverse: "State"

    def __repr__(self) -> str:
        return f"<{super().__repr__()} {self.__dict__}"

    def apply_to_state(self, state: "State") -> None:
        pass

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
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

    def is_marker_command(self) -> bool:
        """Whether this is a logical marker for the generator."""
        return False

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the command to a dictionary."""
        return {"type": self.__class__.__name__}


class MovingCommand(Command, ABC):
    end: Tuple[float, float, float]  # type: ignore[reportRedeclaration]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["end"] = self.end
        return d

    @abstractmethod
    def linearize(
        self, start_point: Tuple[float, float, float]
    ) -> List[Command]:
        """
        Returns a list of simpler commands (e.g., LineToCommand) that
        approximate this command. For simple commands, it may return a list
        containing only itself.
        """
        pass

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
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

    def linearize(
        self, start_point: Tuple[float, float, float]
    ) -> List[Command]:
        return [self]


class LineToCommand(MovingCommand):
    def is_cutting_command(self) -> bool:
        return True

    def linearize(
        self, start_point: Tuple[float, float, float]
    ) -> List[Command]:
        return [self]


class ArcToCommand(MovingCommand):
    def __init__(
        self,
        end: Tuple[float, float, float],
        center_offset: Tuple[float, float],
        clockwise: bool,
    ) -> None:
        super().__init__(end)
        self.center_offset = center_offset
        self.clockwise = clockwise

    def is_cutting_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["center_offset"] = self.center_offset
        d["clockwise"] = self.clockwise
        return d

    def linearize(
        self, start_point: Tuple[float, float, float]
    ) -> List[Command]:
        """Approximates the arc with a series of LineToCommands."""
        segments = geo_linearize.linearize_arc(self, start_point)
        return [LineToCommand(end) for start, end in segments]

    def reverse_geometry(
        self,
        original_start: Tuple[float, float, float],
        original_end: Tuple[float, float, float],
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


class SetPowerCommand(Command):
    def __init__(self, power: int) -> None:
        super().__init__()
        self.power: int = power

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


@dataclass(frozen=True, repr=False)
class JobStartCommand(Command):
    def is_marker_command(self) -> bool:
        return True


@dataclass(frozen=True, repr=False)
class JobEndCommand(Command):
    def is_marker_command(self) -> bool:
        return True


@dataclass(frozen=True, repr=False)
class LayerStartCommand(Command):
    layer_uid: str

    def is_marker_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["layer_uid"] = self.layer_uid
        return d


@dataclass(frozen=True, repr=False)
class LayerEndCommand(Command):
    layer_uid: str

    def is_marker_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["layer_uid"] = self.layer_uid
        return d


@dataclass(frozen=True, repr=False)
class WorkpieceStartCommand(Command):
    workpiece_uid: str

    def is_marker_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["workpiece_uid"] = self.workpiece_uid
        return d


@dataclass(frozen=True, repr=False)
class WorkpieceEndCommand(Command):
    workpiece_uid: str

    def is_marker_command(self) -> bool:
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

    def is_marker_command(self) -> bool:
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

    def is_marker_command(self) -> bool:
        return True

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["section_type"] = self.section_type.name
        return d


class ScanLinePowerCommand(MovingCommand):
    """
    A specialized command for raster engraving that encodes a line segment
    with continuously varying power levels. This is more efficient than a long
    sequence of SetPower and LineTo commands.
    """

    def __init__(
        self,
        start_point: Tuple[float, float, float],
        end: Tuple[float, float, float],
        power_values: bytearray,
    ) -> None:
        super().__init__(end)
        self.start_point = start_point
        self.power_values = power_values

    def is_cutting_command(self) -> bool:
        return True

    def linearize(
        self, start_point: Tuple[float, float, float]
    ) -> List[Command]:
        """
        Deconstructs the scan line into a sequence of SetPower and LineTo
        commands. The `start_point` parameter from the interface is ignored
        as this command is self-contained with its own start point.
        """
        commands: List[Command] = []
        num_steps = len(self.power_values)
        if num_steps == 0:
            return []

        p_start_vec = np.array(self.start_point)
        p_end_vec = np.array(self.end)
        line_vec = p_end_vec - p_start_vec

        last_power = -1  # Sentinel value

        for i in range(num_steps):
            t = (i + 1) / num_steps
            current_point = p_start_vec + t * line_vec
            current_power = self.power_values[i]

            if current_power != last_power:
                commands.append(SetPowerCommand(current_power))
                last_power = current_power

            commands.append(LineToCommand(tuple(current_point)))
        return commands

    def reverse_geometry(self) -> None:
        """
        Reverses the command by swapping its start and end points and
        reversing the array of power values.
        """
        self.start_point, self.end = self.end, self.start_point
        self.power_values.reverse()
