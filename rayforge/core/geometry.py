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
    Set,
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
        self._winding_cache: Dict[int, str] = {}

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
        self._winding_cache.clear()

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

    def _find_closest_on_arc(
        self,
        arc_cmd: ArcToCommand,
        start_pos: Tuple[float, float, float],
        x: float,
        y: float,
    ) -> Optional[Tuple[float, Tuple[float, float], float]]:
        """Finds the closest point on an arc, using an analytical method for
        circular arcs and falling back to linearization for spirals."""
        p0 = start_pos[:2]
        p1 = arc_cmd.end[:2]
        center = (
            p0[0] + arc_cmd.center_offset[0],
            p0[1] + arc_cmd.center_offset[1],
        )
        radius_start = math.dist(p0, center)
        radius_end = math.dist(p1, center)

        # If radii differ, it's a spiral. Fall back to linearization.
        if not math.isclose(radius_start, radius_end):
            return self._find_closest_on_linearized_arc(
                arc_cmd, start_pos, x, y
            )

        radius = radius_start
        if radius < 1e-9:  # Arc with zero radius, treat as a point.
            dist_sq = (x - p0[0]) ** 2 + (y - p0[1]) ** 2
            return 0.0, p0, dist_sq

        # 1. Find point on the full circle closest to (x,y)
        vec_to_point = (x - center[0], y - center[1])
        dist_to_center = math.hypot(vec_to_point[0], vec_to_point[1])
        if dist_to_center < 1e-9:
            closest_on_circle = p0
        else:
            closest_on_circle = (
                center[0] + vec_to_point[0] / dist_to_center * radius,
                center[1] + vec_to_point[1] / dist_to_center * radius,
            )

        # 2. Check if this point lies within the arc's angular sweep.
        start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
        end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
        point_angle = math.atan2(
            closest_on_circle[1] - center[1], closest_on_circle[0] - center[0]
        )

        angle_range = end_angle - start_angle
        angle_to_check = point_angle - start_angle

        # Normalize angles to handle wrapping correctly
        if arc_cmd.clockwise:
            if angle_range > 1e-9:
                angle_range -= 2 * math.pi
            if angle_to_check > 1e-9:
                angle_to_check -= 2 * math.pi
        else:  # counter-clockwise
            if angle_range < -1e-9:
                angle_range += 2 * math.pi
            if angle_to_check < -1e-9:
                angle_to_check += 2 * math.pi

        is_on_arc = False
        if arc_cmd.clockwise:
            if angle_to_check >= angle_range - 1e-9 and angle_to_check <= 1e-9:
                is_on_arc = True
        else:  # counter-clockwise
            if (
                angle_to_check <= angle_range + 1e-9
                and angle_to_check >= -1e-9
            ):
                is_on_arc = True

        # 3. Determine the final closest point and its parameter `t`
        if is_on_arc:
            closest_point = closest_on_circle
            t = (
                angle_to_check / angle_range
                if abs(angle_range) > 1e-9
                else 0.0
            )
        else:
            dist_sq_p0 = (x - p0[0]) ** 2 + (y - p0[1]) ** 2
            dist_sq_p1 = (x - p1[0]) ** 2 + (y - p1[1]) ** 2
            if dist_sq_p0 <= dist_sq_p1:
                closest_point, t = p0, 0.0
            else:
                closest_point, t = p1, 1.0

        dist_sq = (x - closest_point[0]) ** 2 + (y - closest_point[1]) ** 2
        t = max(0.0, min(1.0, t))  # Clamp due to floating point math
        return t, closest_point, dist_sq

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
                # Use the optimized arc calculation
                result = self._find_closest_on_arc(cmd, start_pos, x, y)
                if result:
                    t_arc, pt_arc, dist_sq_arc = result
                    if dist_sq_arc < min_dist_sq:
                        min_dist_sq = dist_sq_arc
                        closest_info = (i, t_arc, pt_arc)

            last_pos_3d = cmd.end

        return closest_info

    def find_closest_point_on_segment(
        self, segment_index: int, x: float, y: float
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Finds the closest point on a specific segment to the given coordinates.
        Returns (t, point) or None.
        """
        if segment_index >= len(self.commands):
            return None

        cmd = self.commands[segment_index]
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            return None

        # Find start point
        start_point = None
        for i in range(segment_index - 1, -1, -1):
            prev_cmd = self.commands[i]
            if isinstance(prev_cmd, MovingCommand) and prev_cmd.end:
                start_point = prev_cmd.end
                break

        if not start_point:
            return None

        if isinstance(cmd, LineToCommand):
            t, point, _ = self._find_closest_on_line_segment(
                start_point[:2], cmd.end[:2], x, y
            )
            return (t, point)
        elif isinstance(cmd, ArcToCommand):
            result = self._find_closest_on_arc(cmd, start_point, x, y)
            if result:
                t_arc, pt_arc, _ = result
                return (t_arc, pt_arc)

        return None

    def _get_subpath_vertices(
        self, start_cmd_index: int
    ) -> List[Tuple[float, float]]:
        """
        Extracts all 2D vertices for a single closed subpath starting at a
        given MoveToCommand index, linearizing any arcs.
        """
        vertices: List[Tuple[float, float]] = []
        last_pos_3d = self.commands[start_cmd_index].end or (0.0, 0.0, 0.0)
        vertices.append(last_pos_3d[:2])

        for i in range(start_cmd_index + 1, len(self.commands)):
            cmd = self.commands[i]
            if isinstance(cmd, MoveToCommand):
                # End of the subpath
                break
            if (
                not isinstance(cmd, (LineToCommand, ArcToCommand))
                or not cmd.end
            ):
                continue

            if isinstance(cmd, LineToCommand):
                vertices.append(cmd.end[:2])
            elif isinstance(cmd, ArcToCommand):
                segments = self._linearize_arc(cmd, last_pos_3d)
                for _, p2 in segments:
                    vertices.append(p2[:2])
            last_pos_3d = cmd.end

        return vertices

    def get_winding_order(self, segment_index: int) -> str:
        """
        Determines the winding order ('cw', 'ccw', or 'unknown') for the
        subpath containing the command at `segment_index`.
        """
        # Find the start of the subpath for the given segment
        subpath_start_index = -1
        for i in range(segment_index, -1, -1):
            if isinstance(self.commands[i], MoveToCommand):
                subpath_start_index = i
                break
        if subpath_start_index == -1:
            return "unknown"

        # Check cache first
        if subpath_start_index in self._winding_cache:
            return self._winding_cache[subpath_start_index]

        vertices = self._get_subpath_vertices(subpath_start_index)
        if len(vertices) < 3:
            return "unknown"  # Not a closed polygon

        # Shoelace formula to calculate signed area
        area = 0.0
        for i in range(len(vertices)):
            p1 = vertices[i]
            p2 = vertices[(i + 1) % len(vertices)]
            area += (p1[0] * p2[1]) - (p2[0] * p1[1])

        # Convention: positive area is CCW, negative is CW in a Y-up system
        # A result of 0 means the path is collinear or self-intersecting.
        if abs(area) < 1e-9:
            result = "unknown"
        elif area > 0:
            result = "ccw"
        else:
            result = "cw"

        self._winding_cache[subpath_start_index] = result
        return result

    def get_point_and_tangent_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculates the 2D point and the normalized 2D tangent vector at a
        parameter `t` (0-1) along a given command segment.
        Returns None if the segment is not a moving command.
        """
        cmd = self.commands[segment_index]
        if not isinstance(cmd, (LineToCommand, ArcToCommand)) or not cmd.end:
            return None

        # Find the start point of this segment
        start_pos_3d: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        for i in range(segment_index - 1, -1, -1):
            prev_cmd = self.commands[i]
            if isinstance(prev_cmd, MovingCommand) and prev_cmd.end:
                start_pos_3d = prev_cmd.end
                break

        p0 = start_pos_3d[:2]
        p1 = cmd.end[:2]

        if isinstance(cmd, LineToCommand):
            point = (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]))
            tangent_vec = (p1[0] - p0[0], p1[1] - p0[1])
        elif isinstance(cmd, ArcToCommand):
            center = (
                p0[0] + cmd.center_offset[0],
                p0[1] + cmd.center_offset[1],
            )
            start_angle = math.atan2(p0[1] - center[1], p0[0] - center[0])
            end_angle = math.atan2(p1[1] - center[1], p1[0] - center[0])
            angle_range = end_angle - start_angle

            if cmd.clockwise:
                if angle_range > 0:
                    angle_range -= 2 * math.pi
            else:
                if angle_range < 0:
                    angle_range += 2 * math.pi

            current_angle = start_angle + t * angle_range
            radius_start = math.dist(p0, center)
            radius_end = math.dist(p1, center)
            radius = radius_start + t * (radius_end - radius_start)

            point = (
                center[0] + radius * math.cos(current_angle),
                center[1] + radius * math.sin(current_angle),
            )

            # The radius vector is normal to the circle.
            radius_vec = (point[0] - center[0], point[1] - center[1])
            # The tangent is perpendicular to the radius vector.
            # Direction depends on clockwise flag.
            if cmd.clockwise:
                tangent_vec = (radius_vec[1], -radius_vec[0])
            else:
                tangent_vec = (-radius_vec[1], radius_vec[0])
        else:
            # This path should not be reachable due to the guard clause above,
            # but is included to satisfy static analysis.
            return None

        # Normalize the tangent vector
        norm = math.sqrt(tangent_vec[0] ** 2 + tangent_vec[1] ** 2)
        if norm < 1e-9:
            return point, (1.0, 0.0)  # Fallback for zero-length segments

        normalized_tangent = (tangent_vec[0] / norm, tangent_vec[1] / norm)
        return point, normalized_tangent

    def get_outward_normal_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the outward-pointing, normalized 2D normal vector for a
        point on the geometry path.

        Returns:
            A tuple (nx, ny) representing the normal vector, or None if the
            direction cannot be determined (e.g., for an open path).
        """
        winding = self.get_winding_order(segment_index)
        if winding == "unknown":
            return None

        result = self.get_point_and_tangent_at(segment_index, t)
        if not result:
            return None

        _, tangent = result
        tx, ty = tangent

        # Right-hand rule:
        # For a CCW path, the interior is to the left. The outward normal
        # is a 90-degree clockwise rotation of the tangent: (ty, -tx).
        # For a CW path, the interior is to the right. The outward normal
        # is a 90-degree counter-clockwise rotation: (-ty, tx).
        if winding == "ccw":
            return (ty, -tx)
        else:  # winding == "cw"
            return (-ty, tx)

    def _get_contours(self) -> List[List[Command]]:
        """Splits the command list into a list of contours."""
        if not self.commands:
            return []
        contours = []
        current_contour: List[Command] = []
        for cmd in self.commands:
            if isinstance(cmd, MoveToCommand):
                if current_contour:
                    contours.append(current_contour)
                current_contour = [cmd]
            else:
                if not current_contour:
                    # Geometry starts with a drawing command, treat (0,0) as
                    # start
                    current_contour.append(MoveToCommand((0, 0, 0)))
                current_contour.append(cmd)
        if current_contour:
            contours.append(current_contour)
        return contours

    @staticmethod
    def _is_point_in_polygon(
        point: Tuple[float, float], polygon: List[Tuple[float, float]]
    ) -> bool:
        """
        Checks if a point is inside a polygon using the ray casting algorithm.
        """
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (
                                p2y - p1y
                            ) + p1x
                            if p1x == p2x or x <= xinters:
                                inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def _get_valid_contours_data(
        self, contours: List[List[Command]]
    ) -> List[Dict]:
        """Filters degenerate contours and pre-calculates their data."""
        contour_data = []
        for contour_cmds in contours:
            # A valid contour must have a move and at least one other command
            if len(contour_cmds) < 2:
                continue
            temp_geo = Geometry()
            temp_geo.commands = contour_cmds
            vertices = temp_geo._get_subpath_vertices(0)
            if not vertices:
                continue
            min_x, min_y, max_x, max_y = temp_geo.rect()
            bbox_center = ((min_x + max_x) / 2, (min_y + max_y) / 2)
            contour_data.append(
                {
                    "cmds": contour_cmds,
                    "vertices": vertices,
                    "bbox_center": bbox_center,
                }
            )
        return contour_data

    def _build_adjacency_graph(
        self, contour_data: List[Dict]
    ) -> List[List[int]]:
        """Builds a graph based on contour containment."""
        num_contours = len(contour_data)
        adj: List[List[int]] = [[] for _ in range(num_contours)]
        for i in range(num_contours):
            for j in range(i + 1, num_contours):
                data_i, data_j = contour_data[i], contour_data[j]
                # Check for containment using the center of the bounding box
                center_i_in_j = self._is_point_in_polygon(
                    data_i["bbox_center"], data_j["vertices"]
                )
                center_j_in_i = self._is_point_in_polygon(
                    data_j["bbox_center"], data_i["vertices"]
                )
                if center_i_in_j or center_j_in_i:
                    adj[i].append(j)
                    adj[j].append(i)
        return adj

    def _find_connected_components_bfs(
        self, num_contours: int, adj: List[List[int]]
    ) -> List[List[int]]:
        """Finds connected components in the graph using BFS."""
        visited: Set[int] = set()
        components: List[List[int]] = []
        for i in range(num_contours):
            if i not in visited:
                component = []
                q = [i]
                visited.add(i)
                while q:
                    u = q.pop(0)
                    component.append(u)
                    for v in adj[u]:
                        if v not in visited:
                            visited.add(v)
                            q.append(v)
                components.append(component)
        return components

    def split_into_components(self) -> List["Geometry"]:
        """
        Analyzes the geometry and splits it into a list of separate,
        logically connected shapes (components). For example, a letter 'O'
        with an outer and inner path will be treated as a single component.
        """
        if self.is_empty():
            return []

        contours = self._get_contours()
        contour_data = self._get_valid_contours_data(contours)

        if not contour_data:
            return []
        if len(contour_data) == 1:
            new_geo = Geometry()
            new_geo.commands = contour_data[0]["cmds"]
            return [new_geo]

        adj = self._build_adjacency_graph(contour_data)
        components = self._find_connected_components_bfs(
            len(contour_data), adj
        )

        # Create a new Geometry object for each component
        result_geometries = []
        for component_indices in components:
            component_geo = Geometry()
            for idx in component_indices:
                component_geo.commands.extend(contour_data[idx]["cmds"])
            result_geometries.append(component_geo)

        return result_geometries

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
