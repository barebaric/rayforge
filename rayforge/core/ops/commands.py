from __future__ import annotations
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple, Dict, Any


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


class MovingCommand(Command):
    end: Tuple[float, float, float]  # type: ignore[reportRedeclaration]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["end"] = self.end
        return d


class MoveToCommand(MovingCommand):
    def is_travel_command(self) -> bool:
        return True


class LineToCommand(MovingCommand):
    def is_cutting_command(self) -> bool:
        return True


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
