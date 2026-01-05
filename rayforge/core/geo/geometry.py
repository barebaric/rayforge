from __future__ import annotations
import logging
import cairo
from typing import (
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
import math
import numpy as np
from .analysis import (
    is_closed,
    get_path_winding_order_from_array,
    get_point_and_tangent_at_from_array,
    get_outward_normal_at_from_array,
    get_area_from_array,
    get_subpath_vertices_from_array,
)
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
)
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
)
from .query import (
    get_bounding_rect_from_array,
    find_closest_point_on_path_from_array,
    get_total_distance_from_array,
)
from .simplify import simplify_geometry_from_array


logger = logging.getLogger(__name__)

T_Geometry = TypeVar("T_Geometry", bound="Geometry")


class Geometry:
    """
    Represents pure, process-agnostic shape data, stored internally as a
    NumPy array for performance.
    """

    def __init__(self) -> None:
        self.last_move_to: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._winding_cache: Dict[int, str] = {}
        self._pending_data: List[List[float]] = []
        self._data: Optional[np.ndarray] = None

    @property
    def data(self) -> Optional[np.ndarray]:
        """
        Provides read-only access to the internal NumPy data array.
        Ensures any pending data is synchronized before access.
        """
        self._sync_to_numpy()
        return self._data

    def _sync_to_numpy(self) -> None:
        """
        Consolidates pending data into the main NumPy array.
        """
        if not self._pending_data:
            return

        new_block = np.array(self._pending_data, dtype=np.float64)
        if self._data is None or len(self._data) == 0:
            self._data = new_block
        else:
            self._data = np.vstack((self._data, new_block))

        self._pending_data = []

    def __len__(self) -> int:
        data_len = 0 if self._data is None else len(self._data)
        pending_len = len(self._pending_data)
        return data_len + pending_len

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Geometry):
            return NotImplemented

        # Accessing .data property on both handles the sync
        if (self.data is None or len(self.data) == 0) and (
            other.data is None or len(other.data) == 0
        ):
            return True
        if self.data is None or other.data is None:
            return False

        return np.array_equal(self.data, other.data)

    def __hash__(self):
        """
        Calculates a hash based on the binary representation of the geometry
        data.
        """
        if self.data is None:
            return 0
        return hash(self.data.tobytes())

    def copy(self: T_Geometry) -> T_Geometry:
        """Creates a deep copy of the Geometry object."""
        new_geo = self.__class__()
        new_geo.last_move_to = self.last_move_to

        # Manually sync before copying internal state to avoid double-sync
        self._sync_to_numpy()
        new_geo._pending_data = []  # Copied data is already synced
        if self._data is not None:
            new_geo._data = self._data.copy()

        return new_geo

    def is_empty(self) -> bool:
        data_is_empty = self._data is None or len(self._data) == 0
        pending_is_empty = not self._pending_data
        return data_is_empty and pending_is_empty

    def clear(self) -> None:
        self._winding_cache.clear()
        self._pending_data = []
        self._data = None

    def extend(self, other: "Geometry") -> None:
        """Extends this geometry with commands from another."""
        # Accessing other.data ensures it's synced
        if other.data is not None and len(other.data) > 0:
            # Fast path: append numpy data directly
            self._sync_to_numpy()  # sync self first
            if self._data is None:
                self._data = other.data.copy()
            else:
                self._data = np.vstack((self._data, other.data))
        elif other._pending_data:
            # If other only has pending data, we can just extend our list
            self._pending_data.extend(deepcopy(other._pending_data))

    def move_to(self, x: float, y: float, z: float = 0.0) -> None:
        self.last_move_to = (float(x), float(y), float(z))
        self._pending_data.append(
            [
                CMD_TYPE_MOVE,
                self.last_move_to[0],
                self.last_move_to[1],
                self.last_move_to[2],
                0.0,
                0.0,
                0.0,
            ]
        )

    def line_to(self, x: float, y: float, z: float = 0.0) -> None:
        self._pending_data.append(
            [
                CMD_TYPE_LINE,
                float(x),
                float(y),
                float(z),
                0.0,
                0.0,
                0.0,
            ]
        )

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
        self._pending_data.append(
            [
                CMD_TYPE_ARC,
                float(x),
                float(y),
                float(z),
                float(i),
                float(j),
                1.0 if bool(clockwise) else 0.0,
            ]
        )

    def append_numpy_data(self, new_data: np.ndarray) -> None:
        """
        Directly appends a block of command data (N, 7) to the internal
        storage. This bypasses the overhead of Python list construction
        for bulk operations.
        """
        if new_data is None or len(new_data) == 0:
            return

        self._sync_to_numpy()

        if self._data is None:
            self._data = new_data.copy()
        else:
            self._data = np.vstack((self._data, new_data))

    def simplify(self: T_Geometry, tolerance: float = 0.01) -> T_Geometry:
        """
        Reduces the number of segments in the geometry using the
        Ramer-Douglas-Peucker algorithm. This preserves the overall shape
        while removing redundant collinear or near-collinear points.

        Args:
            tolerance: The maximum perpendicular distance deviation (mm).

        Returns:
            The modified Geometry object (self).
        """
        if self.is_empty() or self.data is None:
            return self

        self._data = simplify_geometry_from_array(self.data, tolerance)
        self._winding_cache.clear()
        return self

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
        from . import contours

        if self.is_empty() or self.data is None:
            return self

        new_geo = contours.close_geometry_gaps(
            self.copy(), tolerance=tolerance
        )

        self.clear()
        self.extend(new_geo)
        self._winding_cache.clear()

        return self

    def rect(self) -> Tuple[float, float, float, float]:
        """
        Returns a rectangle (x1, y1, x2, y2) that encloses the
        occupied area in the XY plane.
        """
        if self.data is not None and len(self.data) > 0:
            return get_bounding_rect_from_array(self.data)
        return 0.0, 0.0, 0.0, 0.0

    def distance(self) -> float:
        """Calculates the total 2D path length for all moving commands."""
        if self.data is None:
            return 0.0
        return get_total_distance_from_array(self.data)

    def area(self) -> float:
        """
        Calculates the total area of all closed subpaths in the geometry.

        This method correctly handles complex shapes with holes by summing the
        signed areas of each subpath (contour). An outer, counter-clockwise
        path will have a positive area, while an inner, clockwise path (a hole)
        will have a negative area. The absolute value of the final sum is
        returned.
        """
        if self.data is None:
            return 0.0
        return get_area_from_array(self.data)

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
        if self.data is None or len(self.data) == 0:
            return []

        all_segments: List[List[Tuple[float, float, float]]] = []
        current_segment_points: List[Tuple[float, float, float]] = []

        # Find the first real command to establish a start point if needed
        implicit_start: Tuple[float, float, float] = (0.0, 0.0, 0.0)

        for i in range(self.data.shape[0]):
            row = self.data[i]
            cmd_type = row[COL_TYPE]
            end_point = (row[COL_X], row[COL_Y], row[COL_Z])

            if cmd_type == CMD_TYPE_MOVE:
                if current_segment_points:
                    all_segments.append(current_segment_points)
                current_segment_points = [end_point]
            else:  # Line, Arc, etc.
                if not current_segment_points:
                    current_segment_points.append(implicit_start)
                current_segment_points.append(end_point)

        if current_segment_points:
            all_segments.append(current_segment_points)

        return all_segments

    def transform(self: T_Geometry, matrix: "np.ndarray") -> T_Geometry:
        from . import (
            transform as tr,
        )  # Local import to prevent circular dependency

        if self.data is not None and len(self.data) > 0:
            self._data = tr.apply_affine_transform_to_array(self.data, matrix)
            # Update last_move_to by transforming it
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
        if self.data is None:
            return None
        return find_closest_point_on_path_from_array(self.data, x, y)

    def find_closest_point_on_segment(
        self, segment_index: int, x: float, y: float
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Finds the closest point on a specific segment to the given coordinates.
        Returns (t, point) or None.
        """
        if self.data is None or segment_index >= len(self.data):
            return None

        row = self.data[segment_index]
        cmd_type = row[COL_TYPE]
        end_point_3d = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type not in (CMD_TYPE_LINE, CMD_TYPE_ARC):
            return None

        # Find start point
        if segment_index > 0:
            start_point = tuple(
                self.data[segment_index - 1, COL_X : COL_Z + 1]
            )
        else:
            start_point = (0.0, 0.0, 0.0)

        if cmd_type == CMD_TYPE_LINE:
            t, point, _ = find_closest_point_on_line_segment(
                start_point[:2], end_point_3d[:2], x, y
            )
            return (t, point)
        elif cmd_type == CMD_TYPE_ARC:
            result = find_closest_point_on_arc(row, start_point, x, y)
            if result:
                t_arc, pt_arc, _ = result
                return (t_arc, pt_arc)
        return None

    def get_winding_order(self, segment_index: int) -> str:
        """
        Determines the winding order ('cw', 'ccw', or 'unknown') for the
        subpath containing the command at `segment_index`.
        """
        if self.data is None:
            return "unknown"
        # Caching is useful here because winding order is expensive to compute
        # and may be requested multiple times for the same subpath.
        subpath_start_index = -1
        for i in range(segment_index, -1, -1):
            if self.data[i, COL_TYPE] == CMD_TYPE_MOVE:
                subpath_start_index = i
                break
        if subpath_start_index == -1:
            subpath_start_index = 0

        if subpath_start_index in self._winding_cache:
            return self._winding_cache[subpath_start_index]

        result = get_path_winding_order_from_array(
            self.data, subpath_start_index
        )
        self._winding_cache[subpath_start_index] = result
        return result

    def get_point_and_tangent_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """
        Calculates the 2D point and the normalized 2D tangent vector at a
        parameter `t` (0-1) along a given command segment.
        """
        if self.data is None:
            return None
        return get_point_and_tangent_at_from_array(self.data, segment_index, t)

    def get_outward_normal_at(
        self, segment_index: int, t: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calculates the outward-pointing, normalized 2D normal vector for a
        point on the geometry path.
        """
        if self.data is None:
            return None
        return get_outward_normal_at_from_array(self.data, segment_index, t)

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
        if self.data is None:
            return False
        return is_closed(self.data, tolerance=tolerance)

    def _get_valid_contours_data(
        self, contour_geometries: List["Geometry"]
    ) -> List[Dict]:
        """
        Filters degenerate contours and pre-calculates their data, including
        whether they are closed.
        """
        contour_data = []
        for i, contour_geo in enumerate(contour_geometries):
            if contour_geo.is_empty():
                continue

            # Access .data to trigger sync
            data = contour_geo.data
            if (
                data is None
                or data.shape[0] < 2
                or data[0, COL_TYPE] != CMD_TYPE_MOVE
            ):
                continue

            min_x, min_y, max_x, max_y = contour_geo.rect()
            bbox_area = (max_x - min_x) * (max_y - min_y)
            is_closed_flag = is_closed(data) and bbox_area > 1e-9

            if not is_closed_flag:
                continue

            # A single contour geometry has one "move" at the start (index 0).
            vertices_2d = get_subpath_vertices_from_array(data, 0)

            contour_data.append(
                {
                    "geo": contour_geo,
                    "vertices": vertices_2d,
                    "is_closed": is_closed_flag,
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
        from .intersect import (
            check_self_intersection_from_array,
        )  # Local import

        if self.data is None:
            return False
        return check_self_intersection_from_array(
            self.data, fail_on_t_junction=fail_on_t_junction
        )

    def intersects_with(self, other: "Geometry") -> bool:
        """
        Checks if this geometry's path intersects with another geometry's path.
        """
        from .intersect import check_intersection_from_array  # Local import

        if self.data is None or other.data is None:
            return False
        return check_intersection_from_array(self.data, other.data)

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

    def to_cairo(self, ctx: cairo.Context) -> None:
        """
        Draws this geometry's path to a Cairo context.

        This method iterates through the geometry's commands and translates
        them into the corresponding Cairo drawing operations.

        Args:
            ctx: The Cairo context to draw on.
        """
        last_point = (0.0, 0.0)
        data = self.data
        if data is None:
            return

        for i in range(len(data)):
            row = data[i]
            cmd_type = row[COL_TYPE]
            end = (row[COL_X], row[COL_Y])

            if cmd_type == CMD_TYPE_MOVE:
                ctx.move_to(end[0], end[1])
            elif cmd_type == CMD_TYPE_LINE:
                ctx.line_to(end[0], end[1])
            elif cmd_type == CMD_TYPE_ARC:
                center_x = last_point[0] + row[COL_I]
                center_y = last_point[1] + row[COL_J]
                radius = math.hypot(row[COL_I], row[COL_J])

                start_angle = math.atan2(-row[COL_J], -row[COL_I])
                end_angle = math.atan2(end[1] - center_y, end[0] - center_x)

                clockwise = bool(row[COL_CW])
                if clockwise:
                    ctx.arc_negative(
                        center_x, center_y, radius, start_angle, end_angle
                    )
                else:
                    ctx.arc(center_x, center_y, radius, start_angle, end_angle)

            last_point = end

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
        if self.data is not None:
            for row in self.data:
                cmd_type = row[COL_TYPE]
                if cmd_type == CMD_TYPE_MOVE:
                    compact_cmds.append(
                        ["M", row[COL_X], row[COL_Y], row[COL_Z]]
                    )
                elif cmd_type == CMD_TYPE_LINE:
                    compact_cmds.append(
                        ["L", row[COL_X], row[COL_Y], row[COL_Z]]
                    )
                elif cmd_type == CMD_TYPE_ARC:
                    compact_cmds.append(
                        [
                            "A",
                            row[COL_X],
                            row[COL_Y],
                            row[COL_Z],
                            row[COL_I],
                            row[COL_J],
                            int(row[COL_CW]),
                        ]
                    )
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
                new_geo.move_to(cmd_data[1], cmd_data[2], cmd_data[3])
            elif cmd_type == "L":
                new_geo.line_to(cmd_data[1], cmd_data[2], cmd_data[3])
            elif cmd_type == "A":
                new_geo.arc_to(
                    cmd_data[1],
                    cmd_data[2],
                    i=cmd_data[4],
                    j=cmd_data[5],
                    clockwise=bool(cmd_data[6]),
                    z=cmd_data[3],
                )
            else:
                logger.warning(
                    "Skipping unknown command type during Geometry.load():"
                    f" {cmd_type}"
                )
        return new_geo

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Geometry object to a dictionary."""
        return self.dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Geometry":
        """Deserializes a dictionary into a Geometry instance."""
        new_geo = cls()
        last_move = tuple(data.get("last_move_to", (0.0, 0.0, 0.0)))
        assert len(last_move) == 3, "last_move_to must be a 3-tuple"
        new_geo.last_move_to = last_move

        commands = data.get("commands", [])
        if not commands:
            return new_geo

        # Check format: verbose dicts or compact lists
        first_cmd = commands[0]
        is_compact_format = isinstance(first_cmd, list)

        if is_compact_format:
            return cls.load(data)
        else:
            # Handle verbose format
            for cmd_data in commands:
                cmd_type = cmd_data.get("type")
                if cmd_type == "MoveToCommand":
                    end = tuple(cmd_data["end"])
                    new_geo.move_to(end[0], end[1], end[2])
                elif cmd_type == "LineToCommand":
                    end = tuple(cmd_data["end"])
                    new_geo.line_to(end[0], end[1], end[2])
                elif cmd_type == "ArcToCommand":
                    end = tuple(cmd_data["end"])
                    offset = tuple(cmd_data["center_offset"])
                    new_geo.arc_to(
                        end[0],
                        end[1],
                        offset[0],
                        offset[1],
                        cmd_data["clockwise"],
                        end[2],
                    )
                else:
                    # Silently ignore non-geometric commands (e.g., from Ops)
                    pass
        return new_geo

    def iter_commands(
        self,
    ) -> Iterable[Tuple[int, float, float, float, float, float, float]]:
        """
        Yields command data tuples for each command in the geometry.

        Each yielded tuple contains:
        (cmd_type, x, y, z, i, j, cw)

        This method ensures data is synced before iteration and provides
        a clean interface without exposing the raw NumPy array.

        Yields:
            Tuples of (cmd_type, x, y, z, i, j, cw) for each command.
        """
        if self.data is None:
            return

        for row in self.data:
            yield (
                int(row[COL_TYPE]),
                float(row[COL_X]),
                float(row[COL_Y]),
                float(row[COL_Z]),
                float(row[COL_I]),
                float(row[COL_J]),
                float(row[COL_CW]),
            )

    def get_command_at(
        self, index: int
    ) -> Optional[Tuple[int, float, float, float, float, float, float]]:
        """
        Returns command data tuple at the specified index.

        Args:
            index: The index of the command to retrieve.

        Returns:
            A tuple (cmd_type, x, y, z, i, j, cw) or None if
            the index is out of bounds or data is None.
        """
        if self.data is None or index < 0 or index >= len(self.data):
            return None

        row = self.data[index]
        return (
            int(row[COL_TYPE]),
            float(row[COL_X]),
            float(row[COL_Y]),
            float(row[COL_Z]),
            float(row[COL_I]),
            float(row[COL_J]),
            float(row[COL_CW]),
        )
