from __future__ import annotations
import math
import logging
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    Any,
)
from copy import deepcopy
import numpy as np

logger = logging.getLogger(__name__)

T_Geometry = TypeVar("T_Geometry", bound="Geometry")


class Command:
    """Base for all geometric commands."""

    def __init__(
        self, end: Optional[Tuple[float, float, float]] = None
    ) -> None:
        self.end: Optional[Tuple[float, float, float]] = end

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}


class MovingCommand(Command):
    """A geometric command that involves movement."""

    end: Tuple[float, float, float]  # type: ignore[reportRedeclaration]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["end"] = self.end
        return d


class MoveToCommand(MovingCommand):
    """A move-to command."""

    pass


class LineToCommand(MovingCommand):
    """A line-to command."""

    pass


class ArcToCommand(MovingCommand):
    """An arc-to command."""

    def __init__(
        self,
        end: Tuple[float, float, float],
        center_offset: Tuple[float, float],
        clockwise: bool,
    ) -> None:
        super().__init__(end)
        self.center_offset = center_offset
        self.clockwise = clockwise

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["center_offset"] = self.center_offset
        d["clockwise"] = self.clockwise
        return d


class Geometry:
    """
    Represents pure, process-agnostic shape data. It is completely
    self-contained and has no dependency on Ops.
    """

    def __init__(self) -> None:
        self.commands: List[Command] = []
        self.last_move_to: Tuple[float, float, float] = (0.0, 0.0, 0.0)

    def __iter__(self) -> Iterator[Command]:
        return iter(self.commands)

    def __len__(self) -> int:
        return len(self.commands)

    def copy(self: T_Geometry) -> T_Geometry:
        """Creates a deep copy of the Geometry object."""
        new_geo = self.__class__()
        new_geo.commands = deepcopy(self.commands)
        new_geo.last_move_to = self.last_move_to
        return new_geo

    def is_empty(self) -> bool:
        return not self.commands

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
        self.commands.append(
            ArcToCommand(
                (float(x), float(y), float(z)),
                (float(i), float(j)),
                bool(clockwise),
            )
        )

    def rect(self) -> Tuple[float, float, float, float]:
        occupied_points: List[Tuple[float, float, float]] = []
        last_point: Optional[Tuple[float, float, float]] = None
        for cmd in self.commands:
            if isinstance(cmd, MoveToCommand) and cmd.end:
                last_point = cmd.end
            elif isinstance(cmd, (LineToCommand, ArcToCommand)) and cmd.end:
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

    def _linearize_arc(
        self, arc_cmd: ArcToCommand, start_point: Tuple[float, float, float]
    ) -> List[Tuple[Tuple[float, float, float], Tuple[float, float, float]]]:
        segments: List[
            Tuple[Tuple[float, float, float], Tuple[float, float, float]]
        ] = []
        p0 = start_point
        p1 = arc_cmd.end
        z0, z1 = p0[2], p1[2]

        center = (
            p0[0] + arc_cmd.center_offset[0],
            p0[1] + arc_cmd.center_offset[1],
        )

        radius_start = math.dist(p0[:2], center)
        radius_end = math.dist(p1[:2], center)

        # If the start point is the center, it's just a line to the end.
        if radius_start == 0:
            return [(p0, p1)]

        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        angle_range = end_angle - start_angle
        if arc_cmd.clockwise:
            if angle_range > 0:
                angle_range -= 2 * math.pi
        else:
            if angle_range < 0:
                angle_range += 2 * math.pi

        # Use the average radius to get a better estimate for arc length
        avg_radius = (radius_start + radius_end) / 2
        arc_len = abs(angle_range * avg_radius)
        num_segments = max(2, int(arc_len / 0.5))

        prev_pt = p0
        for i in range(1, num_segments + 1):
            t = i / num_segments
            # Interpolate radius and angle to handle imperfectly defined arcs
            radius = radius_start + (radius_end - radius_start) * t
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

    def transform(self: T_Geometry, matrix: "np.ndarray") -> T_Geometry:
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

            if original_cmd_end is not None:
                last_point_untransformed = original_cmd_end

        self.commands = transformed_commands
        last_move_vec = np.array([*self.last_move_to, 1.0])
        transformed_last_move_vec = matrix @ last_move_vec
        self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def _find_closest_on_line_segment(
        self,
        p1: Tuple[float, float],
        p2: Tuple[float, float],
        x: float,
        y: float,
    ) -> Tuple[float, Tuple[float, float], float]:
        """Finds the closest point on a 2D line segment.

        Returns:
            A tuple containing:
            - The parameter `t` (from 0.0 to 1.0) along the segment.
            - A tuple of the (x, y) coordinates of the closest point.
            - The squared distance from the input point to the closest point.
        """
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        len_sq = dx * dx + dy * dy
        if len_sq < 1e-12:  # Treat as a single point
            t = 0.0
        else:
            # Project (x,y) onto the line defined by p1 and p2
            # The parameter t is ((P - A) dot (B - A)) / |B - A|^2
            t = ((x - p1[0]) * dx + (y - p1[1]) * dy) / len_sq
            t = max(0.0, min(1.0, t))  # Clamp to the segment

        closest_x = p1[0] + t * dx
        closest_y = p1[1] + t * dy
        dist_sq = (x - closest_x) ** 2 + (y - closest_y) ** 2
        return t, (closest_x, closest_y), dist_sq

    def _find_closest_on_linearized_arc(
        self,
        arc_cmd: ArcToCommand,
        start_pos: Tuple[float, float, float],
        x: float,
        y: float,
    ) -> Optional[Tuple[float, Tuple[float, float], float]]:
        """Finds the closest point on a linearized arc.

        Returns:
            A tuple of (t_arc, closest_point, distance_squared), or None.
        """
        arc_segments = self._linearize_arc(arc_cmd, start_pos)
        if not arc_segments:
            return None

        min_dist_sq_sub = float("inf")
        best_sub_result = None

        for j, (p1_3d, p2_3d) in enumerate(arc_segments):
            t_sub, pt_sub, dist_sq_sub = self._find_closest_on_line_segment(
                p1_3d[:2], p2_3d[:2], x, y
            )
            if dist_sq_sub < min_dist_sq_sub:
                min_dist_sq_sub = dist_sq_sub
                # Preserve all info to reconstruct the final result
                best_sub_result = (j, t_sub, pt_sub, dist_sq_sub)

        if not best_sub_result:
            return None

        j_best, t_sub_best, pt_best, dist_sq_best = best_sub_result
        # Re-normalize t for the entire arc command
        t_arc = (j_best + t_sub_best) / len(arc_segments)

        return t_arc, pt_best, dist_sq_best

    def find_closest_point(
        self, x: float, y: float
    ) -> Optional[Tuple[int, float, Tuple[float, float]]]:
        """Finds the closest point on the geometry's path to a given 2D point.

        The path is treated as a sequence of 2D line segments. Arcs are
        linearized for this calculation.

        Args:
            x: The x-coordinate of the point to test.
            y: The y-coordinate of the point to test.

        Returns:
            A tuple containing:
            - The index of the command in `self.commands` representing the
              segment.
            - The parameter `t` (from 0.0 to 1.0) along that segment.
            - A tuple of the (x, y) coordinates of the closest point.
            Returns None if the geometry is empty or has no movable segments.
        """
        min_dist_sq = float("inf")
        closest_info: Optional[Tuple[int, float, Tuple[float, float]]] = None

        last_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        for i, cmd in enumerate(self.commands):
            # Handle state changes and non-moving commands
            if isinstance(cmd, MoveToCommand):
                if cmd.end:
                    last_pos_3d = cmd.end
                continue
            if (
                not isinstance(cmd, (LineToCommand, ArcToCommand))
                or not cmd.end
            ):
                continue

            start_pos = last_pos_3d

            # Process the drawing command
            if isinstance(cmd, LineToCommand):
                t, pt, dist_sq = self._find_closest_on_line_segment(
                    start_pos[:2], cmd.end[:2], x, y
                )
                if dist_sq < min_dist_sq:
                    min_dist_sq = dist_sq
                    closest_info = (i, t, pt)

            elif isinstance(cmd, ArcToCommand):
                # For arcs, find the best sub-segment and its properties
                result = self._find_closest_on_linearized_arc(
                    cmd, start_pos, x, y
                )
                if result:
                    t_arc, pt_arc, dist_sq_arc = result
                    if dist_sq_arc < min_dist_sq:
                        min_dist_sq = dist_sq_arc
                        closest_info = (i, t_arc, pt_arc)

            last_pos_3d = cmd.end

        return closest_info

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Geometry object to a dictionary."""
        return {
            "commands": [cmd.to_dict() for cmd in self.commands],
            "last_move_to": self.last_move_to,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Geometry:
        """Deserializes a dictionary into a Geometry instance."""
        new_geo = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_geo.last_move_to = last_move

        for cmd_data in data.get("commands", []):
            cmd_type = cmd_data.get("type")
            if cmd_type == "MoveToCommand":
                new_geo.add(MoveToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "LineToCommand":
                new_geo.add(LineToCommand(end=tuple(cmd_data["end"])))
            elif cmd_type == "ArcToCommand":
                new_geo.add(
                    ArcToCommand(
                        end=tuple(cmd_data["end"]),
                        center_offset=tuple(cmd_data["center_offset"]),
                        clockwise=cmd_data["clockwise"],
                    )
                )
            else:
                logger.warning(
                    "Skipping non-geometric command type during Geometry"
                    f" deserialization: {cmd_type}"
                )
        return new_geo
