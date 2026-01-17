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
    Callable,
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
)
from .constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    CMD_TYPE_BEZIER,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_Z,
    COL_I,
    COL_J,
    COL_CW,
    COL_C1X,
    COL_C1Y,
    COL_C2X,
    COL_C2Y,
)
from .fitting import (
    convert_arc_to_beziers_from_array,
    fit_arcs,
    optimize_path_from_array,
)
from .linearize import linearize_geometry
from .primitives import (
    find_closest_point_on_line_segment,
    find_closest_point_on_arc,
    find_closest_point_on_bezier,
)
from .query import (
    get_bounding_rect_from_array,
    find_closest_point_on_path_from_array,
    get_total_distance_from_array,
)


logger = logging.getLogger(__name__)

T_Geometry = TypeVar("T_Geometry", bound="Geometry")


class Geometry:
    """
    Represents pure, process-agnostic shape data, stored internally as a
    NumPy array for performance.

    The geometry tracks whether it contains circular arcs, which cannot be
    non-uniformly scaled. Use `arc_to_as_bezier()` to add arcs as Bézier curves
    if non-uniform scaling is needed.
    """

    def __init__(self) -> None:
        """Initializes a new, empty Geometry object."""
        self.last_move_to: Tuple[float, float, float] = (0.0, 0.0, 0.0)
        self._uniform_scalable: bool = True
        self._winding_cache: Dict[int, str] = {}
        self._pending_data: List[List[float]] = []
        self._data: Optional[np.ndarray] = None

    @property
    def uniform_scalable(self) -> bool:
        return self._uniform_scalable

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

    def _get_last_point(self) -> Tuple[float, float, float]:
        """
        Retrieves the end point of the last command in the geometry.
        Returns (0,0,0) if empty.
        """
        if self._pending_data:
            last_row = self._pending_data[-1]
            return (last_row[COL_X], last_row[COL_Y], last_row[COL_Z])
        if self._data is not None and len(self._data) > 0:
            last_row = self._data[-1]
            return (last_row[COL_X], last_row[COL_Y], last_row[COL_Z])
        return (0.0, 0.0, 0.0)

    def __len__(self) -> int:
        """Returns the total number of commands in the geometry."""
        data_len = 0 if self._data is None else len(self._data)
        pending_len = len(self._pending_data)
        return data_len + pending_len

    def __eq__(self, other: object) -> bool:
        """Checks equality between two Geometry objects."""
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
        """
        Creates a deep copy of the Geometry object.

        Returns:
            A new Geometry object with the same data and state.
        """
        new_geo = self.__class__()
        new_geo.last_move_to = self.last_move_to
        new_geo._uniform_scalable = self._uniform_scalable

        # Manually sync before copying internal state to avoid double-sync
        self._sync_to_numpy()
        new_geo._pending_data = []  # Copied data is already synced
        if self._data is not None:
            new_geo._data = self._data.copy()

        return new_geo

    def is_empty(self) -> bool:
        """
        Checks if the geometry contains any commands.

        Returns:
            True if the geometry is empty, False otherwise.
        """
        data_is_empty = self._data is None or len(self._data) == 0
        pending_is_empty = not self._pending_data
        return data_is_empty and pending_is_empty

    def clear(self) -> None:
        """
        Clears all commands from the geometry and resets internal state.
        """
        self._winding_cache.clear()
        self._pending_data = []
        self._data = None
        self._uniform_scalable = True

    def extend(self, other: "Geometry") -> None:
        """
        Extends this geometry with commands from another.

        Args:
            other: The Geometry object to extend from.
        """
        if other.data is not None and len(other.data) > 0:
            self._sync_to_numpy()  # sync self first
            if self._data is None:
                self._data = other.data.copy()
            else:
                self._data = np.vstack((self._data, other.data))
        elif other._pending_data:
            self._pending_data.extend(deepcopy(other._pending_data))

        # Update uniform_scalable flag based on the other geometry
        self._uniform_scalable = (
            self._uniform_scalable and other._uniform_scalable
        )

    def move_to(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Adds a move command to the geometry, starting a new subpath.

        Args:
            x: The x-coordinate of the new position.
            y: The y-coordinate of the new position.
            z: The z-coordinate of the new position.
        """
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
                0.0,
            ]
        )

    def line_to(self, x: float, y: float, z: float = 0.0) -> None:
        """
        Adds a line segment to the geometry.

        Args:
            x: The x-coordinate of the line's endpoint.
            y: The y-coordinate of the line's endpoint.
            z: The z-coordinate of the line's endpoint.
        """
        self._pending_data.append(
            [
                CMD_TYPE_LINE,
                float(x),
                float(y),
                float(z),
                0.0,
                0.0,
                0.0,
                0.0,
            ]
        )

    def close_path(self) -> None:
        """
        Closes the current subpath by adding a line to the last move position.
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
        Adds a circular arc segment to the geometry.

        Args:
            x: The x-coordinate of the arc's endpoint.
            y: The y-coordinate of the arc's endpoint.
            i: The x-offset of the arc's center from the start point.
            j: The y-offset of the arc's center from the start point.
            clockwise: If True, the arc is drawn clockwise.
            z: The z-coordinate of the arc's endpoint.

        Note:
            Adding an arc marks the geometry as non-uniformly scalable.
            Use `arc_to_as_bezier()` if you need non-uniform scaling.
        """
        self._uniform_scalable = False
        self._pending_data.append(
            [
                CMD_TYPE_ARC,
                float(x),
                float(y),
                float(z),
                float(i),
                float(j),
                1.0 if bool(clockwise) else 0.0,
                0.0,
            ]
        )

    def arc_to_as_bezier(
        self,
        x: float,
        y: float,
        i: float,
        j: float,
        clockwise: bool = True,
        z: float = 0.0,
    ) -> None:
        """
        Adds a circular arc segment to the geometry, converted to Bézier
        curves.

        This method converts the arc to one or more cubic Bézier curves,
        allowing the geometry to be non-uniformly scaled.

        Args:
            x: The x-coordinate of the arc's endpoint.
            y: The y-coordinate of the arc's endpoint.
            i: The x-offset of the arc's center from the start point.
            j: The y-offset of the arc's center from the start point.
            clockwise: If True, the arc is drawn clockwise.
            z: The z-coordinate of the arc's endpoint.
        """
        start_point = self._get_last_point()
        end_point = (float(x), float(y), float(z))
        center_offset = (float(i), float(j))

        bezier_rows = convert_arc_to_beziers_from_array(
            start_point, end_point, center_offset, clockwise
        )

        for row in bezier_rows:
            # row is a numpy array, pending_data expects list
            self._pending_data.append(row.tolist())

    def bezier_to(
        self,
        x: float,
        y: float,
        c1x: float,
        c1y: float,
        c2x: float,
        c2y: float,
        z: float = 0.0,
    ) -> None:
        """
        Adds a cubic Bézier curve segment to the geometry.

        Args:
            x: The x-coordinate of the curve's endpoint.
            y: The y-coordinate of the curve's endpoint.
            c1x: The x-coordinate of the first control point.
            c1y: The y-coordinate of the first control point.
            c2x: The x-coordinate of the second control point.
            c2y: The y-coordinate of the second control point.
            z: The z-coordinate of the curve's endpoint.
        """
        self._pending_data.append(
            [
                CMD_TYPE_BEZIER,
                float(x),
                float(y),
                float(z),
                float(c1x),
                float(c1y),
                float(c2x),
                float(c2y),
            ]
        )

    def append_numpy_data(self, new_data: np.ndarray) -> None:
        """
        Directly appends a block of command data (N, 8) to the internal
        storage. This bypasses the overhead of Python list construction
        for bulk operations.

        Args:
            new_data: A NumPy array of shape (N, 8) containing command data.

        Note:
            This is a low-level method that assumes the input data has already
            been processed. It does NOT automatically convert arcs.
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
        Reduces the number of segments in any linear chains using the
        Ramer-Douglas-Peucker algorithm. Arcs and Beziers are preserved.

        Args:
            tolerance: The maximum perpendicular distance deviation (mm).

        Returns:
            The modified Geometry object (self).
        """
        if self.is_empty() or self.data is None:
            return self

        self._data = optimize_path_from_array(
            self.data, tolerance, fit_arcs=False
        )
        self._winding_cache.clear()
        return self

    def linearize(self: T_Geometry, tolerance: float) -> T_Geometry:
        """
        Converts the geometry to a polyline approximation (Lines only),
        reducing vertex count using the Ramer-Douglas-Peucker algorithm.

        Args:
            tolerance: The maximum allowable deviation.

        Returns:
            The modified Geometry object (self).
        """
        if self.is_empty() or self.data is None:
            return self

        self._sync_to_numpy()
        self._data = linearize_geometry(self._data, tolerance)

        # Update last_move_to from the last move command
        if self._data is not None and len(self._data) > 0:
            for r in reversed(self._data):
                if r[COL_TYPE] == CMD_TYPE_MOVE:
                    self.last_move_to = (r[COL_X], r[COL_Y], r[COL_Z])
                    break

        self._winding_cache.clear()
        self._uniform_scalable = True  # Lines are scalable
        return self

    def fit_arcs(
        self: T_Geometry,
        tolerance: float,
        on_progress: Optional[Callable[[float], None]] = None,
    ) -> T_Geometry:
        """
        Reconstructs the geometry using an optimal set of Line and Arc
        commands. This method is optimized to handle both polylines and
        existing curves (like Beziers) efficiently, ensuring the output
        contains only Lines and Arcs.

        Args:
            tolerance: The maximum allowable deviation.
            on_progress: An optional callback function that receives progress
                         updates from 0.0 to 1.0.

        Returns:
            The modified Geometry object (self).
        """
        if self.is_empty() or self.data is None:
            return self

        self._sync_to_numpy()
        new_data = fit_arcs(self._data, tolerance, on_progress)

        if new_data is None or len(new_data) == 0:
            self._data = None
        else:
            self._data = new_data
            self.last_move_to = (0.0, 0.0, 0.0)
            for r in reversed(new_data):
                if r[COL_TYPE] == CMD_TYPE_MOVE:
                    self.last_move_to = (r[COL_X], r[COL_Y], r[COL_Z])
                    break

        self._winding_cache.clear()
        self._uniform_scalable = False
        return self

    def upgrade_to_scalable(self: T_Geometry) -> T_Geometry:
        """
        Converts all circular arcs in the geometry to Bézier curves, making
        the geometry fully non-uniformly scalable.

        This method operates in-place. If the geometry is already scalable
        (i.e., contains no arcs), this method does nothing.

        Returns:
            The modified Geometry object (self).
        """
        if self._uniform_scalable or self.is_empty() or self.data is None:
            return self

        new_rows = []
        last_point = (0.0, 0.0, 0.0)

        for row in self.data:
            cmd_type = row[COL_TYPE]
            end_point = (row[COL_X], row[COL_Y], row[COL_Z])

            if cmd_type == CMD_TYPE_ARC:
                start_point = last_point
                center_offset = (row[COL_I], row[COL_J])
                clockwise = bool(row[COL_CW])

                bezier_rows = convert_arc_to_beziers_from_array(
                    start_point, end_point, center_offset, clockwise
                )
                new_rows.extend(bezier_rows)
            else:
                new_rows.append(row)

            # The start point for the next command is the end point of the
            # original command, regardless of linearization.
            last_point = end_point

        if not new_rows:
            self._data = None
        else:
            self._data = np.vstack(new_rows)

        self._uniform_scalable = True
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

        Returns:
            A tuple containing (min_x, min_y, max_x, max_y) coordinates.
        """
        if self.data is not None and len(self.data) > 0:
            return get_bounding_rect_from_array(self.data)
        return 0.0, 0.0, 0.0, 0.0

    def distance(self) -> float:
        """
        Calculates the total 2D path length for all moving commands.

        Returns:
            The total path length in the XY plane.
        """
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
        """
        Applies an affine transformation matrix to the geometry.

        Args:
            matrix: A 4x4 affine transformation matrix.

        Returns:
            The modified Geometry object (self).

        Raises:
            TypeError: If the geometry contains circular arcs and the
                transformation matrix represents non-uniform scaling.

        Note:
            Non-uniform scaling (different X and Y scale factors) is not
            supported for geometries containing circular arcs. Use
            `arc_to_as_bezier()` instead of `arc_to()` to add arcs if
            non-uniform scaling is needed.
        """
        from . import (
            transform as tr,
        )  # Local import to prevent circular dependency

        if self.data is not None and len(self.data) > 0:
            # Check for non-uniform scaling if geometry is not uniform scalable
            if not self._uniform_scalable:
                # Extract scale factors from the matrix
                sx = math.sqrt(matrix[0, 0] ** 2 + matrix[0, 1] ** 2)
                sy = math.sqrt(matrix[1, 0] ** 2 + matrix[1, 1] ** 2)
                if not math.isclose(sx, sy, rel_tol=1e-9):
                    raise TypeError(
                        "Non-uniform scaling is not supported for "
                        "geometries containing circular arcs. "
                        "Use arc_to_as_bezier() instead of arc_to()."
                    )

            self._data = tr.apply_affine_transform_to_array(self.data, matrix)
            # Update last_move_to by transforming it
            last_move_vec = np.array([*self.last_move_to, 1.0])
            transformed_last_move_vec = matrix @ last_move_vec
            self.last_move_to = tuple(transformed_last_move_vec[:3])
        return self

    def flip_x(self: T_Geometry) -> T_Geometry:
        """
        Flips the geometry across the Y-axis (inverts X coordinates).

        Returns:
            The modified Geometry object (self).
        """
        flip_matrix = np.array(
            [
                [-1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return self.transform(flip_matrix)

    def flip_y(self: T_Geometry) -> T_Geometry:
        """
        Flips the geometry across the X-axis (inverts Y coordinates).

        Returns:
            The modified Geometry object (self).
        """
        flip_matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, -1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return self.transform(flip_matrix)

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

    def map_to_frame(
        self: T_Geometry,
        origin: Tuple[float, float],
        p_width: Tuple[float, float],
        p_height: Tuple[float, float],
    ) -> T_Geometry:
        """
        Transforms the geometry to fit into an affine frame defined by three
        points.

        This is a convenience wrapper around the `map_geometry_to_frame`
        function in the `transform` module. This method returns a new,
        transformed Geometry object, leaving the original unchanged.

        Args:
            origin: The (x, y) coordinate for the bottom-left corner of the
                    target frame.
            p_width: The (x, y) coordinate for the bottom-right corner of the
                     target frame.
            p_height: The (x, y) coordinate for the top-left corner of the
                      target frame.

        Returns:
            A new, transformed Geometry object.
        """
        from .transform import map_geometry_to_frame

        return map_geometry_to_frame(self, origin, p_width, p_height)

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

        Args:
            x: The x-coordinate of the query point.
            y: The y-coordinate of the query point.

        Returns:
            A tuple (segment_index, t, point) where segment_index is the
            index of the closest command segment, t is the parameter along
            that segment (0-1), and point is the (x, y) coordinates of the
            closest point. Returns None if the geometry is empty.
        """
        if self.data is None:
            return None
        return find_closest_point_on_path_from_array(self.data, x, y)

    def find_closest_point_on_segment(
        self, segment_index: int, x: float, y: float
    ) -> Optional[Tuple[float, Tuple[float, float]]]:
        """
        Finds the closest point on a specific segment to the given coordinates.

        Args:
            segment_index: The index of the command segment.
            x: The x-coordinate of the query point.
            y: The y-coordinate of the query point.

        Returns:
            A tuple (t, point) where t is the parameter along the segment
            (0-1) and point is the (x, y) coordinates of the closest point.
            Returns None if the segment is not a valid path command.
        """
        if self.data is None or segment_index >= len(self.data):
            return None

        row = self.data[segment_index]
        cmd_type = row[COL_TYPE]
        end_point_3d = (row[COL_X], row[COL_Y], row[COL_Z])

        if cmd_type not in (CMD_TYPE_LINE, CMD_TYPE_ARC, CMD_TYPE_BEZIER):
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
        elif cmd_type == CMD_TYPE_BEZIER:
            result = find_closest_point_on_bezier(row, start_point, x, y)
            if result:
                t_bezier, pt_bezier, _ = result
                return (t_bezier, pt_bezier)
        return None

    def get_winding_order(self, segment_index: int) -> str:
        """
        Determines the winding order ('cw', 'ccw', or 'unknown') for the
        subpath containing the command at `segment_index`.

        Args:
            segment_index: The index of a command within the subpath.

        Returns:
            'cw' for clockwise, 'ccw' for counter-clockwise, or 'unknown'.
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

        Args:
            segment_index: The index of the command segment.
            t: The parameter along the segment (0 to 1).

        Returns:
            A tuple ((x, y), (tx, ty)) where (x, y) is the point and
            (tx, ty) is the normalized tangent vector. Returns None if the
            geometry is empty or the segment index is invalid.
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

        Args:
            segment_index: The index of the command segment.
            t: The parameter along the segment (0 to 1).

        Returns:
            A tuple (nx, ny) representing the normalized outward normal vector.
            Returns None if the geometry is empty or the segment index is
            invalid.
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

        Returns:
            A list of Geometry objects, each representing a distinct
            component.
        """
        from . import split as split_module

        return split_module.split_into_components(self)

    def split_into_contours(self) -> List["Geometry"]:
        """
        Splits the geometry into a list of separate, single-contour
        Geometry objects.

        Returns:
            A list of Geometry objects, each containing a single contour.
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

        Args:
            other: The Geometry object to check for intersection.

        Returns:
            True if the paths intersect, False otherwise.
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
            elif cmd_type == CMD_TYPE_BEZIER:
                c1 = (row[COL_C1X], row[COL_C1Y])
                c2 = (row[COL_C2X], row[COL_C2Y])
                ctx.curve_to(c1[0], c1[1], c2[0], c2[1], end[0], end[1])

            last_point = end

    @classmethod
    def from_cairo_path(
        cls: Type[T_Geometry],
        path_data: cairo.Path,
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

    @classmethod
    def from_text(
        cls: Type[T_Geometry],
        text: str,
        font_family: str = "sans-serif",
        font_size: float = 10.0,
        is_bold: bool = False,
        is_italic: bool = False,
    ) -> T_Geometry:
        """
        Creates a Geometry instance from a string of text. Text is inserted
        in Y-down!

        This is a convenience wrapper around the `text_to_geometry` function.
        The resulting geometry is generated at the origin with natural font
        dimensions.

        Args:
            text: The string content to render.
            font_family: The font family name.
            font_size: The font size in geometry units.
            is_bold: Whether to use a bold weight.
            is_italic: Whether to use an italic slant.

        Returns:
            A new Geometry instance representing the text path.
        """
        from .text import text_to_geometry

        base_geo = text_to_geometry(
            text,
            font_family=font_family,
            font_size=font_size,
            is_bold=is_bold,
            is_italic=is_italic,
        )
        new_geo = cls()
        new_geo.extend(base_geo)
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
                elif cmd_type == CMD_TYPE_BEZIER:
                    compact_cmds.append(
                        [
                            "B",
                            row[COL_X],
                            row[COL_Y],
                            row[COL_Z],
                            row[COL_C1X],
                            row[COL_C1Y],
                            row[COL_C2X],
                            row[COL_C2Y],
                        ]
                    )
        return {
            "last_move_to": list(self.last_move_to),
            "commands": compact_cmds,
        }

    @classmethod
    def load(
        cls: Type[T_Geometry],
        data: Dict[str, Any],
    ) -> T_Geometry:
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
            elif cmd_type == "B":
                new_geo.bezier_to(
                    x=cmd_data[1],
                    y=cmd_data[2],
                    z=cmd_data[3],
                    c1x=cmd_data[4],
                    c1y=cmd_data[5],
                    c2x=cmd_data[6],
                    c2y=cmd_data[7],
                )
            else:
                logger.warning(
                    "Skipping unknown command type during Geometry.load():"
                    f" {cmd_type}"
                )
        return new_geo

    def to_dict(self) -> Dict[str, Any]:
        """
        Serializes the Geometry object to a dictionary.

        Returns:
            A dictionary representation of the geometry.
        """
        return self.dump()

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Geometry":
        """
        Deserializes a dictionary into a Geometry instance.

        Args:
            data: The dictionary representation.

        Returns:
            A new Geometry instance.
        """
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
    ) -> Iterable[Tuple[int, float, float, float, float, float, float, float]]:
        """
        Yields command data tuples for each command in the geometry.

        Each yielded tuple contains:
        (cmd_type, x, y, z, p1, p2, p3, p4)

        This method ensures data is synced before iteration and provides
        a clean interface without exposing the raw NumPy array.

        Yields:
            Tuples of (cmd_type, x, y, z, p1, p2, p3, p4) for each command.
        """
        if self.data is None:
            return

        for row in self.data:
            yield (
                int(row[COL_TYPE]),
                float(row[COL_X]),
                float(row[COL_Y]),
                float(row[COL_Z]),
                float(row[4]),
                float(row[5]),
                float(row[6]),
                float(row[7]),
            )

    def get_command_at(
        self, index: int
    ) -> Optional[Tuple[int, float, float, float, float, float, float, float]]:
        """
        Returns command data tuple at the specified index.

        Args:
            index: The index of the command to retrieve.

        Returns:
            A tuple (cmd_type, x, y, z, p1, p2, p3, p4) or None if
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
            float(row[4]),
            float(row[5]),
            float(row[6]),
            float(row[7]),
        )
