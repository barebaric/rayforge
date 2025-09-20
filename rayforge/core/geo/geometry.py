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
from .linearize import linearize_arc
from .analysis import (
    get_path_winding_order,
    get_point_and_tangent_at,
    get_outward_normal_at,
    get_subpath_vertices,
)
from .query import (
    get_bounding_rect,
    find_closest_point_on_path,
    get_total_distance,
)
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
    is_point_in_polygon,
)


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

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
        """Calculates the 2D distance covered by this command."""
        return 0.0


class MovingCommand(Command):
    """A geometric command that involves movement."""

    end: Tuple[float, float, float]  # type: ignore[reportRedeclaration]

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["end"] = self.end
        return d

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
        """
        Calculates the 2D distance of the move (approximating arcs as lines).
        """
        if last_point is None:
            return 0.0
        return math.hypot(
            self.end[0] - last_point[0], self.end[1] - last_point[1]
        )


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
        return get_bounding_rect(self.commands)

    def distance(self) -> float:
        """Calculates the total 2D path length for all moving commands."""
        return get_total_distance(self.commands)

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
                segments = linearize_arc(cmd, start_point)
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

    def find_closest_point(
        self, x: float, y: float
    ) -> Optional[Tuple[int, float, Tuple[float, float]]]:
        """
        Finds the closest point on the geometry's path to a given 2D point.
        """
        return find_closest_point_on_path(self.commands, x, y)

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
            t, point = find_closest_point_on_line_segment(
                start_point[:2], cmd.end[:2], x, y
            )[:2]
            return (t, point)
        elif isinstance(cmd, ArcToCommand):
            result = find_closest_point_on_arc(cmd, start_point, x, y)
            if result:
                t_arc, pt_arc, _ = result
                return (t_arc, pt_arc)

        return None

    def get_winding_order(self, segment_index: int) -> str:
        """
        Determines the winding order ('cw', 'ccw', or 'unknown') for the
        subpath containing the command at `segment_index`.
        """
        # Caching is useful here because winding order is expensive to compute
        # and may be requested multiple times for the same subpath.
        subpath_start_index = -1
        for i in range(segment_index, -1, -1):
            if isinstance(self.commands[i], MoveToCommand):
                subpath_start_index = i
                break
        if subpath_start_index == -1:
            return "unknown"

        if subpath_start_index in self._winding_cache:
            return self._winding_cache[subpath_start_index]

        result = get_path_winding_order(self.commands, segment_index)
        self._winding_cache[subpath_start_index] = result
        return result

    def get_point_and_tangent_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculates the 2D point and the normalized 2D tangent vector at a
        parameter `t` (0-1) along a given command segment.
        """
        return get_point_and_tangent_at(self.commands, segment_index, t)

    def get_outward_normal_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the outward-pointing, normalized 2D normal vector for a
        point on the geometry path.
        """
        return get_outward_normal_at(self.commands, segment_index, t)

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
                    # Geometry starts with a drawing command, treat (0,0)
                    # as start
                    current_contour.append(MoveToCommand((0, 0, 0)))
                current_contour.append(cmd)
        if current_contour:
            contours.append(current_contour)
        return contours

    def _get_valid_contours_data(
        self, contours: List[List[Command]]
    ) -> List[Dict]:
        """
        Filters degenerate contours and pre-calculates their data, including
        whether they are closed.
        """
        contour_data = []
        for i, contour_cmds in enumerate(contours):
            if len(contour_cmds) < 2 or not isinstance(
                contour_cmds[0], MoveToCommand
            ):
                continue

            start_cmd = contour_cmds[0]
            end_cmd = contour_cmds[-1]
            if not isinstance(start_cmd, MovingCommand) or not isinstance(
                end_cmd, MovingCommand
            ):
                continue

            start_point = start_cmd.end
            end_point = end_cmd.end
            if start_point is None or end_point is None:
                continue

            min_x, min_y, max_x, max_y = get_bounding_rect(contour_cmds)
            bbox_area = (max_x - min_x) * (max_y - min_y)

            is_closed = (
                math.isclose(start_point[0], end_point[0])
                and math.isclose(start_point[1], end_point[1])
                and bbox_area > 1e-6
            )

            # Find the index of the start command in the full command list to
            # pass to get_subpath_vertices. This is inefficient but necessary
            # until `split_into_components` is also refactored.
            try:
                start_idx_in_full_list = self.commands.index(start_cmd)
                vertices = get_subpath_vertices(
                    self.commands, start_idx_in_full_list
                )
            except ValueError:
                vertices = []

            if not vertices:
                continue

            contour_data.append(
                {
                    "cmds": contour_cmds,
                    "vertices": vertices,
                    "is_closed": is_closed,
                    "original_index": i,
                }
            )
        return contour_data

    @staticmethod
    def _find_connected_components_bfs(
        num_contours: int, adj: List[List[int]]
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
        logically connected shapes (components).
        """
        logger.debug("Starting to split_into_components")
        if self.is_empty():
            logger.debug("Geometry is empty, returning empty list.")
            return []

        contours = self._get_contours()
        if len(contours) <= 1:
            logger.debug("<= 1 contour, returning a copy of the whole.")
            return [self.copy()]

        all_contour_data = self._get_valid_contours_data(contours)
        if not all_contour_data:
            logger.debug("No valid contours found after filtering.")
            return []

        if not any(c["is_closed"] for c in all_contour_data):
            logger.debug("No closed paths found. Returning single component.")
            return [self.copy()]

        num_contours = len(all_contour_data)
        adj: List[List[int]] = [[] for _ in range(num_contours)]
        for i in range(num_contours):
            if not all_contour_data[i]["is_closed"]:
                continue
            for j in range(num_contours):
                if i == j:
                    continue
                data_i = all_contour_data[i]
                data_j = all_contour_data[j]
                if is_point_in_polygon(
                    data_j["vertices"][0], data_i["vertices"]
                ):
                    adj[i].append(j)
                    adj[j].append(i)

        component_indices_list = self._find_connected_components_bfs(
            num_contours, adj
        )
        logger.debug(f"Found {len(component_indices_list)} raw components.")

        final_geometries: List[Geometry] = []
        stray_open_geo = Geometry()
        for i, indices in enumerate(component_indices_list):
            component_geo = Geometry()
            has_closed_path = False
            for idx in indices:
                contour = all_contour_data[idx]
                component_geo.commands.extend(contour["cmds"])
                if contour["is_closed"]:
                    has_closed_path = True

            if has_closed_path:
                final_geometries.append(component_geo)
            else:
                stray_open_geo.commands.extend(component_geo.commands)

        if not stray_open_geo.is_empty():
            logger.debug(
                "Found stray open paths, creating a final component for them."
            )
            final_geometries.append(stray_open_geo)

        return final_geometries

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
