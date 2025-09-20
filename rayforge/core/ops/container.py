from __future__ import annotations
import math
import logging
from copy import copy, deepcopy
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    Generator,
    Dict,
    Any,
    TYPE_CHECKING,
)
import numpy as np
from ..geo import linearize, query, clipping
from .commands import (
    State,
    Command,
    MovingCommand,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    SetCutSpeedCommand,
    SetTravelSpeedCommand,
    EnableAirAssistCommand,
    DisableAirAssistCommand,
    JobStartCommand,
    JobEndCommand,
    LayerStartCommand,
    LayerEndCommand,
    WorkpieceStartCommand,
    WorkpieceEndCommand,
    SectionType,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
)


if TYPE_CHECKING:
    from ..geo.geometry import Geometry

logger = logging.getLogger(__name__)


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
            elif cmd_type == "JobStartCommand":
                new_ops.add(JobStartCommand())
            elif cmd_type == "JobEndCommand":
                new_ops.add(JobEndCommand())
            elif cmd_type == "LayerStartCommand":
                new_ops.add(LayerStartCommand(layer_uid=cmd_data["layer_uid"]))
            elif cmd_type == "LayerEndCommand":
                new_ops.add(LayerEndCommand(layer_uid=cmd_data["layer_uid"]))
            elif cmd_type == "WorkpieceStartCommand":
                new_ops.add(
                    WorkpieceStartCommand(
                        workpiece_uid=cmd_data["workpiece_uid"]
                    )
                )
            elif cmd_type == "WorkpieceEndCommand":
                new_ops.add(
                    WorkpieceEndCommand(
                        workpiece_uid=cmd_data["workpiece_uid"]
                    )
                )
            elif cmd_type == "OpsSectionStartCommand":
                new_ops.add(
                    OpsSectionStartCommand(
                        section_type=SectionType[cmd_data["section_type"]],
                        workpiece_uid=cmd_data["workpiece_uid"],
                    )
                )
            elif cmd_type == "OpsSectionEndCommand":
                new_ops.add(
                    OpsSectionEndCommand(
                        section_type=SectionType[cmd_data["section_type"]]
                    )
                )
            else:
                logger.warning(
                    "Skipping unknown command type during deserialization:"
                    f" {cmd_type}"
                )
        return new_ops

    @classmethod
    def from_geometry(cls, geometry: Geometry) -> "Ops":
        """
        Creates an Ops object from a Geometry object, converting its path.
        """
        from .. import geo

        new_ops = cls()
        for cmd in geometry.commands:
            # Explicitly convert from geo.Command to ops.Command
            if isinstance(cmd, geo.MoveToCommand):
                new_ops.add(MoveToCommand(cmd.end))
            elif isinstance(cmd, geo.LineToCommand):
                new_ops.add(LineToCommand(cmd.end))
            elif isinstance(cmd, geo.ArcToCommand):
                new_ops.add(
                    ArcToCommand(cmd.end, cmd.center_offset, cmd.clockwise)
                )
        new_ops.last_move_to = geometry.last_move_to
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
            elif not cmd.is_marker_command():
                cmd.state = copy(state)

    def clear(self) -> None:
        self.commands = []

    def add(self, command: Command) -> None:
        self.commands.append(command)

    def extend(self, other_ops: "Ops") -> None:
        """
        Appends all commands from another Ops object to this one.
        """
        if other_ops and other_ops.commands:
            self.commands.extend(other_ops.commands)

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

    def bezier_to(
        self,
        c1: Tuple[float, float, float],
        c2: Tuple[float, float, float],
        end: Tuple[float, float, float],
        num_steps: int = 20,
    ) -> None:
        """
        Adds a cubic Bézier curve approximated by a series of LineToCommands.
        The curve starts from the current last point in the path. This method
        requires full 3D coordinates for all control and end points.
        """
        if not self.commands or self.commands[-1].end is None:
            logger.warning("bezier_to called without a starting point.")
            return

        start_point = self.commands[-1].end
        segments = linearize.linearize_bezier(
            start_point, c1, c2, end, num_steps
        )
        for _, end_point in segments:
            self.line_to(*end_point)

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
        return query.get_bounding_rect(self.commands)

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
        Calculates the total 2D path length for all moving commands.
        """
        return query.get_total_distance(self.commands)

    def cut_distance(self) -> float:
        """
        Like distance(), but only counts 2D cut distance.
        """
        total = 0.0
        last: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if cmd.is_cutting_command():
                total += cmd.distance(last)

            if isinstance(cmd, MovingCommand):
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

            elif command.is_state_command() or command.is_marker_command():
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
                segments = linearize.linearize_arc(cmd, start_point)
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
        """Rotates all points around a center (cx, cy) in the XY plane."""
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)

        # Create a 4x4 transformation matrix for rotation around a point
        # T(-cx, -cy) * R(angle) * T(cx, cy)
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
            if cmd.is_state_command() or cmd.is_marker_command():
                new_ops.add(deepcopy(cmd))
                continue

            if not isinstance(cmd, MovingCommand):
                continue

            if cmd.is_travel_command():
                last_point = cmd.end
                clipped_pen_pos = None  # A travel move always lifts the pen
                continue

            # Linearize the command into a series of simpler commands
            linearized_commands = cmd.linearize(last_point)

            # Process each linearized segment
            p_current_segment_start = last_point
            for l_cmd in linearized_commands:
                if l_cmd.end is None:
                    continue
                p_current_segment_end = l_cmd.end

                clipped_segment = clipping.clip_line_segment(
                    p_current_segment_start, p_current_segment_end, rect
                )
                clipped_pen_pos = self._add_clipped_segment_to_ops(
                    clipped_segment, new_ops, clipped_pen_pos
                )
                p_current_segment_start = p_current_segment_end

            # The next command starts where the original unclipped command
            # ended
            last_point = cmd.end

        return new_ops

    def dump(self) -> None:
        for segment in self.segments():
            print(segment)

    def subtract_regions(
        self, regions: List[List[Tuple[float, float]]]
    ) -> "Ops":
        """
        Clips the Ops by subtracting a list of polygonal regions.
        This modifies the Ops object in place and returns it.
        """
        if not regions or not self.commands:
            return self

        new_ops = Ops()
        last_point: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        # Tracks the last known pen position of a *kept* segment
        pen_pos: Optional[Tuple[float, float, float]] = None

        # Add any leading state/marker commands before the first move
        first_move_idx = next(
            (
                i
                for i, cmd in enumerate(self.commands)
                if isinstance(cmd, MovingCommand)
            ),
            len(self.commands),
        )
        for i in range(first_move_idx):
            new_ops.add(deepcopy(self.commands[i]))

        for cmd in self.commands:
            if not isinstance(cmd, MovingCommand):
                # State/marker commands are handled as they appear
                # between moves
                if not new_ops.commands or new_ops.commands[-1] is not cmd:
                    new_ops.add(deepcopy(cmd))
                continue

            if isinstance(cmd, MoveToCommand):
                last_point = cmd.end
                pen_pos = None  # Pen is up
                continue

            # Linearize cutting command into segments
            linearized_commands = cmd.linearize(last_point)

            p_current_segment_start = last_point
            for l_cmd in linearized_commands:
                if l_cmd.end is None:
                    continue
                p_current_segment_end = l_cmd.end

                kept_segments = clipping.subtract_regions_from_line_segment(
                    p_current_segment_start, p_current_segment_end, regions
                )
                for sub_p1, sub_p2 in kept_segments:
                    if pen_pos is None or math.dist(pen_pos, sub_p1) > 1e-6:
                        new_ops.move_to(*sub_p1)
                    new_ops.line_to(*sub_p2)
                    pen_pos = sub_p2
                p_current_segment_start = p_current_segment_end

            last_point = cmd.end

        self.commands = new_ops.commands
        # Update last_move_to to a valid point if ops is not empty
        if new_ops.commands:
            for cmd in reversed(new_ops.commands):
                if isinstance(cmd, MoveToCommand):
                    self.last_move_to = cmd.end
                    break
        return self
