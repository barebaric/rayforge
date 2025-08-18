from __future__ import annotations
import math
import logging
from copy import copy, deepcopy
from typing import Iterator, List, Optional, Tuple, Generator, Dict, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


# Cohen-Sutherland outcodes
_INSIDE = 0  # 0000
_LEFT = 1  # 0001
_RIGHT = 2  # 0010
_BOTTOM = 4  # 0100
_TOP = 8  # 1000


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
        state: Optional[State] = None,
    ) -> None:
        # x/y/z of the end position. Is None for state commands
        self.end: Optional[Tuple[float, float, float]] = end
        self.state: Optional[State] = state  # Intended state during execution
        self._state_ref_for_pyreverse: State

    def __repr__(self) -> str:
        return f"<{super().__repr__()} {self.__dict__}"

    def apply_to_state(self, state: State) -> None:
        pass

    def is_state_command(self) -> bool:
        return False

    def is_cutting_command(self) -> bool:
        """Whether it is a cutting movement"""
        return False

    def is_travel_command(self) -> bool:
        """Whether it is a non-cutting movement"""
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

    def apply_to_state(self, state: State) -> None:
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

    def apply_to_state(self, state: State) -> None:
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

    def apply_to_state(self, state: State) -> None:
        state.travel_speed = self.speed

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["speed"] = self.speed
        return d


class EnableAirAssistCommand(Command):
    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: State) -> None:
        state.air_assist = True


class DisableAirAssistCommand(Command):
    def is_state_command(self) -> bool:
        return True

    def apply_to_state(self, state: State) -> None:
        state.air_assist = False


class Ops:
    """
    Represents a set of generated path segments and instructions that
    are used for making gcode, but also to generate vector graphics
    for display.
    """

    def __init__(self) -> None:
        self.commands: List[Command] = []
        self._commands_ref_for_pyreverse: Command
        self.last_move_to: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Ops object to a dictionary."""
        return {
            "commands": [cmd.to_dict() for cmd in self.commands],
            "last_move_to": self.last_move_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Ops:
        """Deserializes a dictionary into an Ops instance."""
        new_ops = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_ops.last_move_to = last_move

        for cmd_data in data.get("commands", []):
            cmd_type = cmd_data.get("type")
            if cmd_type == "MoveToCommand":
                new_ops.add(MoveToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "LineToCommand":
                new_ops.add(LineToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "ArcToCommand":
                new_ops.add(
                    ArcToCommand(
                        end=tuple(cmd_data["end"]),
                        center_offset=tuple(cmd_data["center_offset"]),
                        clockwise=cmd_data["clockwise"],
                    )
                )
            elif cmd_type == "SetPowerCommand":
                new_ops.add(SetPowerCommand(power=cmd_data["power"]))
            elif cmd_type == "SetCutSpeedCommand":
                new_ops.add(SetCutSpeedCommand(speed=cmd_data["speed"]))
            elif cmd_type == "SetTravelSpeedCommand":
                new_ops.add(SetTravelSpeedCommand(speed=cmd_data["speed"]))
            elif cmd_type == "EnableAirAssistCommand":
                new_ops.add(EnableAirAssistCommand())
            elif cmd_type == "DisableAirAssistCommand":
                new_ops.add(DisableAirAssistCommand())
            else:
                logger.warning(
                    "Skipping unknown command type during deserialization:"
                    f" {cmd_type}"
                )
        return new_ops

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    def __add__(self, ops: Ops) -> Ops:
        result = Ops()
        result.commands = self.commands + ops.commands
        return result

    def __mul__(self, count: int) -> Ops:
        result = Ops()
        result.commands = count * self.commands
        return result

    def __len__(self) -> int:
        return len(self.commands)

    def is_empty(self) -> bool:
        """Checks if the Ops object contains any commands."""
        return not self.commands

    def copy(self) -> Ops:
        """Creates a deep copy of the Ops object."""
        new_ops = Ops()
        new_ops.commands = deepcopy(self.commands)
        new_ops.last_move_to = self.last_move_to
        return new_ops

    def preload_state(self) -> None:
        """
        Walks through all commands, enriching each by the indended
        state of the machine. The state is useful for some post-processors
        that need to re-order commands without changing the intended
        state during each command.
        """
        state = State()
        for cmd in self.commands:
            if cmd.is_state_command():
                cmd.apply_to_state(state)
            else:
                cmd.state = copy(state)

    def clear(self) -> None:
        self.commands = []

    def add(self, command: Command) -> None:
        self.commands.append(command)

    def move_to(self, x: float, y: float, z: float = 0.0) -> None:
        self.last_move_to = (float(x), float(y), float(z))
        cmd = MoveToCommand(self.last_move_to)
        self.commands.append(cmd)

    def line_to(self, x: float, y: float, z: float = 0.0) -> None:
        cmd = LineToCommand((float(x), float(y), float(z)))
        self.commands.append(cmd)

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
    ) -> None:
        """
        Adds an arc command with specified endpoint, center offsets,
        and direction (cw/ccw).
        """
        self.commands.append(
            ArcToCommand(
                (float(x), float(y), float(z)),
                (float(i), float(j)),
                bool(clockwise),
            )
        )

    def set_power(self, power: float) -> None:
        """
        Sets the intended laser power for subsequent cutting commands.
        This is a state declaration, not an immediate command to turn on
        the laser.
        """
        cmd = SetPowerCommand(int(power))
        self.commands.append(cmd)

    def set_cut_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent cutting commands.
        This is a state declaration.
        """
        cmd = SetCutSpeedCommand(int(speed))
        self.commands.append(cmd)

    def set_travel_speed(self, speed: float) -> None:
        """
        Sets the intended feed rate for subsequent travel commands.
        This is a state declaration.
        """
        cmd = SetTravelSpeedCommand(int(speed))
        self.commands.append(cmd)

    def enable_air_assist(self, enable: bool = True) -> None:
        """
        Sets the intended state of the air assist for subsequent commands.
        This is a state declaration.
        """
        if enable:
            self.commands.append(EnableAirAssistCommand())
        else:
            self.disable_air_assist()

    def disable_air_assist(self) -> None:
        """
        Sets the intended state of the air assist for subsequent commands.
        This is a state declaration.
        """
        self.commands.append(DisableAirAssistCommand())

    def rect(self) -> Tuple[float, float, float, float]:
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area in the XY plane.
        """
        occupied_points: List[Tuple[float, float, float]] = []
        last_point: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if cmd.is_travel_command() and cmd.end:
                last_point = cmd.end
            elif cmd.is_cutting_command() and cmd.end:
                if last_point is not None:
                    occupied_points.append(last_point)
                occupied_points.append(cmd.end)
                last_point = cmd.end

        if not occupied_points:
            return 0.0, 0.0, 0.0, 0.0

        xs = [p[0] for p in occupied_points if p]
        ys = [p[1] for p in occupied_points if p]
        if not xs or not ys:
            return 0.0, 0.0, 0.0, 0.0
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        return min_x, min_y, max_x, max_y

    def get_frame(
        self,
        power: Optional[float] = None,
        speed: Optional[float] = None,
    ) -> Ops:
        """
        Returns a new Ops object containing four move_to operations forming
        a frame around the occupied area of the original Ops. The occupied
        area includes all points from line_to and close_path commands.
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
        Calculates the total 2D distance of all moves in the XY plane.
        Mostly exists to help debug the optimize() method.
        """
        total = 0.0

        last: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if cmd.is_travel_command():
                if last is not None and cmd.end is not None:
                    total += math.hypot(
                        cmd.end[0] - last[0], cmd.end[1] - last[1]
                    )
                last = cmd.end
            elif cmd.is_cutting_command():
                # treating arcs as lines is probably good enough
                if last is not None and cmd.end is not None:
                    total += math.hypot(
                        cmd.end[0] - last[0], cmd.end[1] - last[1]
                    )
                last = cmd.end
        return total

    def cut_distance(self) -> float:
        """
        Like distance(), but only counts 2D cut distance.
        """
        total = 0.0

        last: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if cmd.is_travel_command():
                last = cmd.end
            elif cmd.is_cutting_command():
                # treating arcs as lines is probably good enough
                if last is not None and cmd.end is not None:
                    total += math.hypot(
                        cmd.end[0] - last[0], cmd.end[1] - last[1]
                    )
                last = cmd.end
        return total

    def segments(self) -> Generator[List[Command], None, None]:
        segment: List[Command] = []
        for command in self.commands:
            if not segment:
                segment.append(command)
                continue

            if command.is_travel_command():
                yield segment
                segment = [command]

            elif command.is_cutting_command():
                segment.append(command)

            elif command.is_state_command():
                yield segment
                yield [command]
                segment = []

        if segment:
            yield segment

    def transform(self, matrix: "np.ndarray") -> "Ops":
        """
        Applies a transformation matrix to all geometric commands. If the
        transform is non-uniform (contains non-uniform scaling or shear),
        arcs will be linearized to preserve their shape.

        Args:
            matrix: A 4x4 NumPy transformation matrix.

        Returns:
            The Ops object itself for chaining.
        """
        # Check for non-uniform scaling or shear by comparing the length of
        # transformed basis vectors.
        v_x = matrix @ np.array([1, 0, 0, 0])
        v_y = matrix @ np.array([0, 1, 0, 0])
        len_x = np.linalg.norm(v_x[:2])
        len_y = np.linalg.norm(v_y[:2])
        is_non_uniform = not np.isclose(len_x, len_y)

        transformed_commands: List[Command] = []
        last_point_untransformed: Optional[Tuple[float, float, float]] = None

        for cmd in self.commands:
            original_cmd_end = (
                cmd.end if isinstance(cmd, MovingCommand) else None
            )

            if isinstance(cmd, ArcToCommand) and is_non_uniform:
                # Use the last known untransformed point as the start for
                # linearization
                start_point = last_point_untransformed or (0.0, 0.0, 0.0)
                segments = self._linearize_arc(cmd, start_point)
                for p1, p2 in segments:
                    point_vec = np.array([p2[0], p2[1], p2[2], 1.0])
                    transformed_vec = matrix @ point_vec
                    transformed_commands.append(
                        LineToCommand(tuple(transformed_vec[:3]))
                    )
            elif isinstance(cmd, MovingCommand):
                point_vec = np.array([*cmd.end, 1.0])
                transformed_vec = matrix @ point_vec
                cmd.end = tuple(transformed_vec[:3])

                if isinstance(cmd, ArcToCommand):
                    # For uniform transforms, we transform the center offset
                    # vector by the 3x3 rotation/scaling part of the matrix.
                    offset_vec_3d = np.array(
                        [cmd.center_offset[0], cmd.center_offset[1], 0]
                    )
                    rot_scale_matrix = matrix[:3, :3]
                    new_offset_vec_3d = rot_scale_matrix @ offset_vec_3d
                    cmd.center_offset = (
                        new_offset_vec_3d[0],
                        new_offset_vec_3d[1],
                    )
                transformed_commands.append(cmd)
            else:
                transformed_commands.append(cmd)

            # Crucially, update the last_point tracker with the endpoint
            # from BEFORE the transformation for the next iteration.
            if original_cmd_end is not None:
                last_point_untransformed = original_cmd_end

        self.commands = transformed_commands
        last_move_vec = np.array(
            [
                self.last_move_to[0],
                self.last_move_to[1],
                self.last_move_to[2],
                1.0,
            ]
        )
        transformed_last_move_vec = matrix @ last_move_vec
        self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def translate(self, dx: float, dy: float, dz: float = 0.0) -> Ops:
        """Translate geometric commands while preserving relative offsets"""
        matrix = np.identity(4)
        matrix[0, 3] = dx
        matrix[1, 3] = dy
        matrix[2, 3] = dz
        return self.transform(matrix)

    def scale(self, sx: float, sy: float, sz: float = 1.0) -> Ops:
        """
        Scales both absolute positions and relative offsets by creating a
        scaling matrix and calling transform().
        """
        matrix = np.diag([sx, sy, sz, 1.0])
        return self.transform(matrix)

    def rotate(self, angle_deg: float, cx: float, cy: float) -> Ops:
        """Rotates all points around a center (cx, cy) in the XY plane."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        def _rotate_point(
            p: Tuple[float, float, float],
        ) -> Tuple[float, float, float]:
            x, y, z = p
            # Translate point to origin
            translated_x = x - cx
            translated_y = y - cy
            # Rotate
            rotated_x = translated_x * cos_a - translated_y * sin_a
            rotated_y = translated_x * sin_a + translated_y * cos_a
            # Translate back and return with original Z
            return rotated_x + cx, rotated_y + cy, z

        def _rotate_vector(v: Tuple[float, float]) -> Tuple[float, float]:
            x, y = v
            # Rotate vector (no translation)
            rotated_x = x * cos_a - y * sin_a
            rotated_y = x * sin_a + y * cos_a
            return rotated_x, rotated_y

        for cmd in self.commands:
            if cmd.end is not None:
                cmd.end = _rotate_point(cmd.end)

            if isinstance(cmd, ArcToCommand):
                cmd.center_offset = _rotate_vector(cmd.center_offset)

        self.last_move_to = _rotate_point(self.last_move_to)
        return self

    def _compute_outcode(
        self, x: float, y: float, rect: Tuple[float, float, float, float]
    ) -> int:
        x_min, y_min, x_max, y_max = rect
        code = _INSIDE
        if x < x_min:
            code |= _LEFT
        elif x > x_max:
            code |= _RIGHT
        if y < y_min:
            code |= _BOTTOM
        elif y > y_max:
            code |= _TOP
        return code

    def _clip_segment(
        self,
        p1: Tuple[float, float, float],
        p2: Tuple[float, float, float],
        rect: Tuple[float, float, float, float],
    ) -> Optional[
        Tuple[Tuple[float, float, float], Tuple[float, float, float]]
    ]:
        """
        Clips a line segment to a rectangle using Cohen-Sutherland.
        Returns the clipped segment or None if it's outside. Z is interpolated.
        """
        x_min, y_min, x_max, y_max = rect
        (x1, y1, z1), (x2, y2, z2) = p1, p2
        outcode1 = self._compute_outcode(x1, y1, rect)
        outcode2 = self._compute_outcode(x2, y2, rect)
        dx, dy, dz = x2 - x1, y2 - y1, z2 - z1

        while True:
            if not (outcode1 | outcode2):  # Trivial accept
                return (x1, y1, z1), (x2, y2, z2)
            if outcode1 & outcode2:  # Trivial reject
                return None

            outcode_out = outcode1 if outcode1 else outcode2
            x, y, z = 0.0, 0.0, 0.0

            if outcode_out & _TOP:
                y = y_max
                x = x1 + dx * (y_max - y1) / dy if dy != 0 else x1
                z = z1 + dz * (y_max - y1) / dy if dy != 0 else z1
            elif outcode_out & _BOTTOM:
                y = y_min
                x = x1 + dx * (y_min - y1) / dy if dy != 0 else x1
                z = z1 + dz * (y_min - y1) / dy if dy != 0 else z1
            elif outcode_out & _RIGHT:
                x = x_max
                y = y1 + dy * (x_max - x1) / dx if dx != 0 else y1
                z = z1 + dz * (x_max - x1) / dx if dx != 0 else z1
            elif outcode_out & _LEFT:
                x = x_min
                y = y1 + dy * (x_min - x1) / dx if dx != 0 else y1
                z = z1 + dz * (x_min - x1) / dx if dx != 0 else z1

            if outcode_out == outcode1:
                x1, y1, z1 = x, y, z
                outcode1 = self._compute_outcode(x1, y1, rect)
            else:
                x2, y2, z2 = x, y, z
                outcode2 = self._compute_outcode(x2, y2, rect)

    def _linearize_arc(
        self, arc_cmd: ArcToCommand, start_point: Tuple[float, float, float]
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        """Converts an ArcToCommand into a list of line segments."""
        segments: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = []
        p0 = start_point
        p1 = arc_cmd.end
        if p1 is None:
            return []
        z0, z1 = p0[2], p1[2]

        center = (
            p0[0] + arc_cmd.center_offset[0],
            p0[1] + arc_cmd.center_offset[1],
        )
        radius = math.dist(p0[:2], center)
        if radius == 0:
            # If radius is zero, it's just a line from p0 to p1
            return [(p0, p1)]

        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])

        # Robust angle normalization to handle atan2 wrap-around
        angle_range = end_angle - start_angle
        if arc_cmd.clockwise:
            if angle_range > 0:
                angle_range -= 2 * math.pi
        else:  # Counter-clockwise
            if angle_range < 0:
                angle_range += 2 * math.pi

        arc_len = abs(angle_range * radius)
        # Use ~0.5mm segments for linearization
        num_segments = max(2, int(arc_len / 0.5))

        prev_pt = p0
        for i in range(1, num_segments + 1):
            t = i / num_segments
            angle = start_angle + angle_range * t
            z = z0 + (z1 - z0) * t
            next_pt = (
                center[0] + radius * math.cos(angle),
                center[1] + radius * math.sin(angle),
                z,
            )
            segments.append((prev_pt, next_pt))
            prev_pt = next_pt
        return segments

    def _add_clipped_segment_to_ops(
        self,
        segment: Optional[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ],
        new_ops: Ops,
        current_pen_pos: Optional[Tuple[float, float, float]],
    ) -> Optional[Tuple[float, float, float]]:
        """
        Processes a single clipped segment, adding MoveTo/LineTo commands
        to the new_ops object as needed.

        Returns the updated pen position.
        """
        if segment:
            p1_clipped, p2_clipped = segment

            # A new move is needed if the pen is up (None) or if there's a gap.
            # Using 3D distance for this check is correct.
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

    def clip(self, rect: Tuple[float, float, float, float]) -> Ops:
        """
        Clips the Ops to the given rectangle.
        Returns a new, clipped Ops object.
        """
        new_ops = Ops()
        if not self.commands:
            return new_ops

        last_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Tracks the last known position of the pen *within the clipped area*.
        # None means the pen is "up" or outside the clip rect.
        clipped_pen_pos: Optional[Tuple[float, float, float]] = None

        for cmd in self.commands:
            if cmd.is_state_command():
                new_ops.add(deepcopy(cmd))
                continue

            if cmd.end is None:
                continue

            if cmd.is_travel_command():
                last_point = cmd.end
                clipped_pen_pos = None  # A travel move always lifts the pen
                continue

            # Linearize the command into one or more line segments
            segments_to_clip: List[
                Tuple[Tuple[float, float, float], Tuple[float, float, float]]
            ] = []
            if isinstance(cmd, LineToCommand):
                segments_to_clip.append((last_point, cmd.end))
            elif isinstance(cmd, ArcToCommand):
                segments_to_clip.extend(self._linearize_arc(cmd, last_point))

            # Process each linearized segment
            for p1, p2 in segments_to_clip:
                clipped_segment = self._clip_segment(p1, p2, rect)
                clipped_pen_pos = self._add_clipped_segment_to_ops(
                    clipped_segment, new_ops, clipped_pen_pos
                )

            # The next command starts where the original unclipped command
            # ended
            last_point = cmd.end

        return new_ops

    def dump(self) -> None:
        for segment in self.segments():
            print(segment)
