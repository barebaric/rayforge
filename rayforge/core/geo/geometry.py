from __future__ import annotations
import math
import logging
import cairo
from typing import (
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
    Dict,
    Any,
    Iterable,
    Type,
)
from copy import deepcopy
import numpy as np
from .analysis import (
    is_closed,
    get_path_winding_order,
    get_point_and_tangent_at,
    get_outward_normal_at,
    get_subpath_area,
)
from .query import (
    get_bounding_rect,
    find_closest_point_on_path,
    get_total_distance,
)
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
)


logger = logging.getLogger(__name__)

T_Geometry = TypeVar("T_Geometry", bound="Geometry")


class Command:
    """Base for all geometric commands."""

    __slots__ = ("end",)

    def __init__(
        self, end: Optional[Tuple[float, float, float]] = None
    ) -> None:
        self.end: Optional[Tuple[float, float, float]] = end

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Command):
            return NotImplemented
        return self.end == other.end

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.__class__.__name__}

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
        """Calculates the 2D distance covered by this command."""
        return 0.0


class MovingCommand(Command):
    """A geometric command that involves movement."""

    __slots__ = ()

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

    __slots__ = ()


class LineToCommand(MovingCommand):
    """A line-to command."""

    __slots__ = ()


class ArcToCommand(MovingCommand):
    """An arc-to command."""

    __slots__ = ("center_offset", "clockwise")

    def __init__(
        self,
        end: Tuple[float, float, float],
        center_offset: Tuple[float, float],
        clockwise: bool,
    ) -> None:
        super().__init__(end)
        self.center_offset = center_offset
        self.clockwise = clockwise

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ArcToCommand):
            return NotImplemented
        return (
            self.end == other.end
            and self.center_offset == other.center_offset
            and self.clockwise == other.clockwise
        )

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d["center_offset"] = self.center_offset
        d["clockwise"] = self.clockwise
        return d

    def distance(
        self, last_point: Optional[Tuple[float, float, float]]
    ) -> float:
        """
        Calculates the true 2D length of the arc path.
        """
        if not last_point or not self.end:
            return 0.0

        # The center of the arc's circle in the XY plane
        center_x = last_point[0] + self.center_offset[0]
        center_y = last_point[1] + self.center_offset[1]

        # The radius is the distance from the center to the start point
        radius = math.hypot(self.center_offset[0], self.center_offset[1])

        if radius < 1e-9:
            # If the radius is zero, the arc is just a point.
            return 0.0

        # Calculate the start and end angles relative to the center
        start_angle = math.atan2(
            last_point[1] - center_y, last_point[0] - center_x
        )
        end_angle = math.atan2(self.end[1] - center_y, self.end[0] - center_x)

        # Calculate the sweep of the angle
        angle_span = end_angle - start_angle

        # Adjust the angle span based on direction (clockwise/ccw) and wrapping
        if self.clockwise:
            if angle_span > 1e-9:  # Ensure we subtract to go negative
                angle_span -= 2 * math.pi
        else:  # Counter-clockwise
            if angle_span < -1e-9:  # Ensure we add to go positive
                angle_span += 2 * math.pi

        # Arc length is radius times the absolute angle span in radians
        return abs(angle_span * radius)


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

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented
        if len(self.commands) != len(other.commands):
            return False
        for command, other_command in zip(self.commands, other.commands):
            if command != other_command:
                return False
        return True

    def __hash__(self):
        # A simple hash based on the commands. This allows Geometry objects
        # to be used in sets and as dictionary keys.
        return hash(tuple(str(cmd.to_dict()) for cmd in self.commands))

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

    def close_gaps(self: T_Geometry, tolerance: float = 1e-6) -> T_Geometry:
        """
        Closes small gaps between endpoints in the geometry to form clean,
        connected paths. This method operates in-place.

        This is a convenience wrapper around the `close_geometry_gaps`
        function in the `contours` module.

        Args:
            tolerance: The maximum distance between two points to be
                       considered "the same".

        Returns:
            The modified Geometry object (self).
        """
        from . import contours  # Local import to prevent circular dependency

        # The function returns a new object; we update self with its data.
        new_geo = contours.close_geometry_gaps(self, tolerance=tolerance)
        self.commands = new_geo.commands
        self._winding_cache.clear()  # Winding order might have changed
        return self

    def rect(self) -> Tuple[float, float, float, float]:
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area in the XY plane.
        """
        return get_bounding_rect(self.commands)

    def distance(self) -> float:
        """Calculates the total 2D path length for all moving commands."""
        return get_total_distance(self.commands)

    def area(self) -> float:
        """
        Calculates the total area of all closed subpaths in the geometry.

        This method correctly handles complex shapes with holes by summing the
        signed areas of each subpath (contour). An outer, counter-clockwise
        path will have a positive area, while an inner, clockwise path (a hole)
        will have a negative area. The absolute value of the final sum is
        returned.
        """
        total_signed_area = 0.0
        for i, cmd in enumerate(self.commands):
            if isinstance(cmd, MoveToCommand):
                total_signed_area += get_subpath_area(self.commands, i)
        return abs(total_signed_area)

    def segments(self) -> List[List[Tuple[float, float, float]]]:
        """
        Returns a list of segments, where each segment is a list of points
        defining a continuous subpath.

        A new segment is started by a MoveToCommand. No linearization of
        arcs is performed; only the end points of commands are used.

        Returns:
            A list of lists, where each inner list contains the (x, y, z)
            points of a subpath.
        """
        if not self.commands:
            return []

        all_segments: List[List[Tuple[float, float, float]]] = []
        current_segment_points: List[Tuple[float, float, float]] = []

        for cmd in self.commands:
            if isinstance(cmd, MoveToCommand):
                if current_segment_points:
                    all_segments.append(current_segment_points)
                # Start a new segment with the move_to point
                current_segment_points = [cmd.end]
            elif isinstance(cmd, MovingCommand):
                if not current_segment_points:
                    # Geometry starts with a drawing command, assume (0,0,0)
                    # start
                    current_segment_points.append((0.0, 0.0, 0.0))
                current_segment_points.append(cmd.end)

        # Add the last segment if it exists
        if current_segment_points:
            all_segments.append(current_segment_points)

        return all_segments

    def transform(self: T_Geometry, matrix: "np.ndarray") -> T_Geometry:
        from . import transform  # Local import to prevent circular dependency

        self.commands = transform.apply_affine_transform(self.commands, matrix)

        last_move_vec = np.array([*self.last_move_to, 1.0])
        transformed_last_move_vec = matrix @ last_move_vec
        self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def grow(self: T_Geometry, amount: float) -> T_Geometry:
        """
        Offsets the contours of any closed shape in the geometry by a
        given amount.

        This method grows (positive offset) or shrinks (negative offset) the
        area enclosed by closed paths. Arcs are linearized into polylines for
        the offsetting process. Open paths are ignored and not included in
        the returned geometry.

        Args:
            amount: The distance to offset the geometry. Positive values
                    expand the shape, negative values contract it.

        Returns:
            A new Geometry object containing the offset shape(s).
        """
        from . import transform  # Local import to prevent circular dependency

        return transform.grow_geometry(self, offset=amount)

    def split_inner_and_outer_contours(
        self,
    ) -> Tuple[List["Geometry"], List["Geometry"]]:
        """
        Splits the geometry's contours into two distinct lists: internal
        contours (holes) and external contours (solids).

        This is a convenience wrapper around the
        `split_inner_and_outer_contours`
        function in the `contours` module.

        Returns:
            A tuple containing two lists of Geometry objects:
            (internal_contours, external_contours).
        """
        from . import contours as contours_module
        from . import split as split_module

        # 1. Split self into individual contours
        contour_list = split_module.split_into_contours(self)
        if not contour_list:
            return [], []

        # 2. Split the list of contours into inner and outer
        return contours_module.split_inner_and_outer_contours(contour_list)

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

    def is_closed(self, tolerance: float = 1e-6) -> bool:
        """
        Checks if the geometry's path is closed.

        This method assumes the Geometry object represents a single contour.
        It checks if the start point (from the first MoveTo) and the end
        point (from the last moving command) are within the given tolerance.

        For geometries with multiple contours, use `split_into_contours()`
        and call this method on each resulting part.

        Args:
            tolerance: The maximum distance to consider start and end points
                       equal.

        Returns:
            True if the path is closed, False otherwise.
        """
        return is_closed(self.commands, tolerance=tolerance)

    def _get_valid_contours_data(
        self, contour_geometries: List["Geometry"]
    ) -> List[Dict]:
        """
        Filters degenerate contours and pre-calculates their data, including
        whether they are closed.
        """
        contour_data = []
        for i, contour_geo in enumerate(contour_geometries):
            if len(contour_geo.commands) < 2 or not isinstance(
                contour_geo.commands[0], MoveToCommand
            ):
                continue

            start_cmd = contour_geo.commands[0]
            end_cmd = contour_geo.commands[-1]
            if not isinstance(start_cmd, MovingCommand) or not isinstance(
                end_cmd, MovingCommand
            ):
                continue

            start_point = start_cmd.end
            end_point = end_cmd.end
            if start_point is None or end_point is None:
                continue

            min_x, min_y, max_x, max_y = contour_geo.rect()
            bbox_area = (max_x - min_x) * (max_y - min_y)

            # A contour is valid and closed if its path is closed AND it's
            # not degenerate (has some area).
            is_closed = contour_geo.is_closed() and bbox_area > 1e-9

            # A single-contour geometry by definition has only one segment list
            segments = contour_geo.segments()
            if not segments:
                continue
            vertices_3d = segments[0]
            vertices_2d = [p[:2] for p in vertices_3d]

            contour_data.append(
                {
                    "geo": contour_geo,
                    "vertices": vertices_2d,
                    "is_closed": is_closed,
                    "original_index": i,
                }
            )
        return contour_data

    def remove_inner_edges(self) -> "Geometry":
        """
        Filters the geometry, keeping all open paths and only the external-most
        closed paths (contours).

        This is a convenience wrapper around the `remove_inner_edges` function
        in the `contours` module. It effectively removes any "holes" from
        closed shapes while preserving any open lines or arcs.

        Returns:
            A new Geometry object containing the filtered paths.
        """
        from . import contours  # Local import to prevent circular dependency

        return contours.remove_inner_edges(self)

    def split_into_components(self) -> List["Geometry"]:
        """
        Analyzes the geometry and splits it into a list of separate,
        logically connected shapes (components).
        """
        from . import split as split_module

        return split_module.split_into_components(self)

    def split_into_contours(self) -> List["Geometry"]:
        """
        Splits the geometry into a list of separate, single-contour
        Geometry objects.
        """
        from . import split as split_module

        return split_module.split_into_contours(self)

    def has_self_intersections(self, fail_on_t_junction: bool = False) -> bool:
        """
        Checks if any subpath within the geometry intersects with itself.
        Adjacent segments meeting at a vertex are not considered intersections.

        Args:
            fail_on_t_junction: If False (default), T-junctions where a vertex
                                lies on another segment are not considered
                                intersections. If True, they are flagged.
        """
        from .intersect import check_self_intersection  # Local import

        return check_self_intersection(
            self.commands, fail_on_t_junction=fail_on_t_junction
        )

    def intersects_with(self, other: "Geometry") -> bool:
        """
        Checks if this geometry's path intersects with another geometry's path.
        """
        from .intersect import check_intersection  # Local import

        # When checking two different geometries, T-junctions are always
        # intersections.
        return check_intersection(self.commands, other.commands)

    def encloses(self, other: "Geometry") -> bool:
        """
        Checks if this geometry fully encloses another geometry.

        This method performs a series of checks to determine containment.
        The 'other' geometry must be fully inside this geometry's boundary,
        not intersecting it, and not located within any of this geometry's
        holes.

        Args:
            other: The Geometry object to check for containment.

        Returns:
            True if this geometry encloses the other, False otherwise.
        """
        from . import analysis  # Local import to prevent circular dependency

        return analysis.encloses(self, other)

    @classmethod
    def from_cairo_path(
        cls: Type[T_Geometry], path_data: cairo.Path
    ) -> T_Geometry:
        """
        Creates a Geometry instance from a flattened Cairo path data structure.

        Args:
            path_data: An iterable of (path_type, points) tuples, as returned
                       by `cairo.Context.copy_path_flat()`.

        Returns:
            A new Geometry instance.
        """
        new_geo = cls()
        for path_type, points in path_data:  # type: ignore
            if path_type == cairo.PATH_MOVE_TO:
                new_geo.move_to(points[0], points[1])
            elif path_type == cairo.PATH_LINE_TO:
                new_geo.line_to(points[0], points[1])
            elif path_type == cairo.PATH_CLOSE_PATH:
                new_geo.close_path()
        return new_geo

    @classmethod
    def from_points(
        cls: Type[T_Geometry],
        points: Iterable[Tuple[float, ...]],
        close: bool = True,
    ) -> T_Geometry:
        """
        Creates a Geometry path from a list of points.

        Args:
            points: An iterable of points, where each point is a tuple of
                    (x, y) or (x, y, z).
            close: If True (default), a final segment will be added to close
                   the path, forming a polygon. If False, an open polyline
                   is created.

        Returns:
            A new Geometry instance representing the polygon or polyline.
        """
        new_geo = cls()
        point_iterator = iter(points)

        try:
            first_point = next(point_iterator)
        except StopIteration:
            return new_geo  # Return empty geometry for empty list

        new_geo.move_to(*first_point)

        has_segments = False
        for point in point_iterator:
            new_geo.line_to(*point)
            has_segments = True

        # Only close the path if requested and it's a valid path
        if close and has_segments:
            new_geo.close_path()

        return new_geo

    def dump(self) -> Dict[str, Any]:
        """
        Returns a space-efficient, serializable representation of the Geometry.

        This is a more compact alternative to to_dict().

        Returns:
            A dictionary with a compact representation of the geometry data.
        """
        compact_cmds = []
        for cmd in self.commands:
            if isinstance(cmd, MoveToCommand):
                compact_cmds.append(["M", *cmd.end])
            elif isinstance(cmd, LineToCommand):
                compact_cmds.append(["L", *cmd.end])
            elif isinstance(cmd, ArcToCommand):
                compact_cmds.append(
                    [
                        "A",
                        *cmd.end,
                        *cmd.center_offset,
                        1 if cmd.clockwise else 0,
                    ]
                )
            # Non-geometric commands are skipped
        return {
            "last_move_to": list(self.last_move_to),
            "commands": compact_cmds,
        }

    @classmethod
    def load(cls: Type[T_Geometry], data: Dict[str, Any]) -> T_Geometry:
        """
        Creates a Geometry instance from its space-efficient representation
        generated by dump().

        Args:
            data: The dictionary created by the dump() method.

        Returns:
            A new Geometry instance.
        """
        new_geo = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_geo.last_move_to = last_move

        for cmd_data in data.get("commands", []):
            cmd_type = cmd_data[0]
            if cmd_type == "M":
                new_geo.add(MoveToCommand(end=tuple(cmd_data[1:4])))
            elif cmd_type == "L":
                new_geo.add(LineToCommand(end=tuple(cmd_data[1:4])))
            elif cmd_type == "A":
                new_geo.add(
                    ArcToCommand(
                        end=tuple(cmd_data[1:4]),
                        center_offset=tuple(cmd_data[4:6]),
                        clockwise=bool(cmd_data[6]),
                    )
                )
            else:
                logger.warning(
                    "Skipping unknown command type during Geometry.load():"
                    f" {cmd_type}"
                )
        return new_geo

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
