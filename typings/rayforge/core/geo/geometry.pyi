"""The ``Geometry`` class — a vector path composed of move, line, arc and
cubic Bezier commands.

Internally the path is stored as a flat NumPy array with 8 columns (see
:mod:`rayforge.core.geo` column-index constants).  Most mutating methods
automatically synchronise the pending command buffer to the NumPy backing
store.
"""

from typing import Optional, Tuple, List, Union, TypeAlias

from numpy.typing import NDArray
import numpy as np

from rayforge.core.geo.path import PyCommand

Point: TypeAlias = Tuple[float, float]
Point3D: TypeAlias = Tuple[float, float, float]
Polygon: TypeAlias = List[Tuple[float, float]]
_Point2DOr3D: TypeAlias = Tuple[float, ...]

_CommandRow = Tuple[float, float, float, float, float, float, float, float]
"""A single command row: ``(type, x, y, z, aux1, aux2, aux3, aux4)``."""


class Geometry:
    """A mutable vector geometry path backed by a NumPy data array.

    Use the ``*_to`` family of methods to append drawing commands and
    the analysis methods to query the path.

    Example::

        g = Geometry()
        g.move_to(0, 0)
        g.line_to(10, 0)
        g.line_to(10, 10)
        g.close_path()
        print(g.area())
    """

    COL_TYPE: int
    COL_X: int
    COL_Y: int
    COL_Z: int
    COL_I: int
    COL_J: int
    COL_CW: int
    COL_C1X: int
    COL_C1Y: int
    COL_C2X: int
    COL_C2Y: int
    CMD_TYPE_MOVE: float
    CMD_TYPE_LINE: float
    CMD_TYPE_ARC: float
    CMD_TYPE_BEZIER: float

    def __init__(self) -> None:
        """Create an empty geometry."""
        ...

    # -- drawing commands ---------------------------------------------------

    def move_to(self, x: float, y: float, z: float = ...) -> None:
        """Move the pen to an absolute position without drawing.

        Args:
            x: Target x coordinate.
            y: Target y coordinate.
            z: Target z coordinate.  Defaults to ``0.0``.
        """
        ...

    def line_to(self, x: float, y: float, z: float = ...) -> None:
        """Draw a straight line to an absolute position.

        Args:
            x: Target x coordinate.
            y: Target y coordinate.
            z: Target z coordinate.  Defaults to ``0.0``.
        """
        ...

    def close_path(self) -> None:
        """Close the current sub-path by drawing a line back to the
        last :meth:`move_to` position."""
        ...

    def arc_to(
        self,
        x: float,
        y: float,
        i: float = ...,
        j: float = ...,
        clockwise: bool = ...,
        z: float = ...,
    ) -> None:
        """Draw a circular arc to an absolute position.

        The arc center is specified as an offset ``(i, j)`` relative to
        the current pen position.

        Args:
            x: End point x coordinate.
            y: End point y coordinate.
            i: X offset from current position to arc center.
                Defaults to ``0.0``.
            j: Y offset from current position to arc center.
                Defaults to ``0.0``.
            clockwise: ``True`` for a clockwise arc.
                Defaults to ``True``.
            z: End point z coordinate.  Defaults to ``0.0``.
        """
        ...

    def bezier_to(
        self,
        x: float,
        y: float,
        c1x: float,
        c1y: float,
        c2x: float,
        c2y: float,
        z: float = ...,
    ) -> None:
        """Draw a cubic Bezier curve to an absolute position.

        Args:
            x: End point x coordinate.
            y: End point y coordinate.
            c1x: First control point x.
            c1y: First control point y.
            c2x: Second control point x.
            c2y: Second control point y.
            z: End point z coordinate.  Defaults to ``0.0``.
        """
        ...

    def arc_to_as_bezier(
        self,
        x: float,
        y: float,
        i: float,
        j: float,
        clockwise: bool = ...,
        z: float = ...,
    ) -> None:
        """Draw an arc by converting it to one or more cubic Bezier
        segments.

        The resulting geometry is marked as uniformly scalable (safe for
        non-uniform scaling transforms).

        Args:
            x: End point x coordinate.
            y: End point y coordinate.
            i: X offset from current position to arc center.
            j: Y offset from current position to arc center.
            clockwise: ``True`` for a clockwise arc.
                Defaults to ``True``.
            z: End point z coordinate.  Defaults to ``0.0``.
        """
        ...

    # -- sync / low-level ---------------------------------------------------

    def sync_to_data(self) -> None:
        """Flush pending commands to the NumPy backing store."""
        ...

    def _sync_to_numpy(self) -> None:
        """Alias for :meth:`sync_to_data`."""
        ...

    # -- dunder methods -----------------------------------------------------

    def __len__(self) -> int:
        """Return the number of commands in the geometry.

        Implicitly calls :meth:`sync_to_data` first.

        Returns:
            Command count as a non-negative integer.
        """
        ...

    def __hash__(self) -> int:
        """Compute a hash from the raw data rows.

        Returns:
            A 64-bit unsigned integer hash.
        """
        ...

    def __eq__(self, other: object) -> bool:
        """Check exact equality with another :class:`Geometry`.

        Args:
            other: The object to compare against.

        Returns:
            ``True`` if both geometries contain identical command data.
        """
        ...

    def __repr__(self) -> str:
        """Return a human-readable representation.

        Returns:
            A string like ``<Geometry commands=5 closed=True>``.
        """
        ...

    # -- properties ---------------------------------------------------------

    @property
    def data(self) -> Optional[NDArray[np.float64]]:
        """The command data as an ``(N, 8)`` NumPy float64 array, or
        ``None`` when the geometry is empty.

        Implicitly calls :meth:`sync_to_data`.

        Setting this property replaces the entire command buffer.
        Pass ``None`` to clear.
        """
        ...

    @data.setter
    def data(self, value: Optional[NDArray[np.float64]]) -> None: ...

    @property
    def last_move_to(self) -> Point3D:
        """The last position set by :meth:`move_to` as ``(x, y, z)``."""
        ...

    @last_move_to.setter
    def last_move_to(self, value: Point3D) -> None:
        """Set the last move-to position.

        Args:
            value: A ``(x, y, z)`` tuple.
        """
        ...

    @property
    def uniform_scalable(self) -> bool:
        """Whether the geometry can be non-uniformly scaled without
        distortion.

        Becomes ``True`` after :meth:`upgrade_to_scalable` converts arcs
        to Bezier curves.
        """
        ...

    @uniform_scalable.setter
    def uniform_scalable(self, value: bool) -> None:
        """Set the uniform-scalable flag.

        Args:
            value: ``True`` to mark the geometry as safe for non-uniform
                scaling.
        """
        ...

    @property
    def _pending_data(self) -> List[List[float]]:
        """Commands that have not yet been flushed to :attr:`data`.

        Returns:
            A list of 8-element float lists.
        """
        ...

    def _get_last_point(self) -> Point3D:
        """Return the last data row's ``(x, y, z)`` coordinates.

        Returns ``(0.0, 0.0, 0.0)`` if the geometry is empty.

        Returns:
            ``(x, y, z)`` of the last command row.
        """
        ...

    # -- query methods ------------------------------------------------------

    def is_empty(self) -> bool:
        """Check whether the geometry contains no commands.

        Returns:
            ``True`` if there are zero commands.
        """
        ...

    def rect(self) -> Tuple[float, float, float, float]:
        """Compute the 2D axis-aligned bounding box.

        Implicitly calls :meth:`sync_to_data`.

        Returns:
            ``(x_min, y_min, x_max, y_max)``.
        """
        ...

    def distance(self) -> float:
        """Compute the total path length (2D).

        Implicitly calls :meth:`sync_to_data`.

        Returns:
            Sum of all segment lengths.
        """
        ...

    def area(self) -> float:
        """Compute the signed enclosed area of the path.

        Implicitly calls :meth:`sync_to_data`.

        A positive value indicates counter-clockwise winding;
        negative indicates clockwise.

        Returns:
            The signed area in square coordinate units.
        """
        ...

    def is_closed(self, tolerance: float = ...) -> bool:
        """Check whether the path returns to its start point.

        Args:
            tolerance: Maximum allowed gap between start and end points.
                Defaults to ``1e-6``.

        Returns:
            ``True`` if the path is closed within *tolerance*.
        """
        ...

    def segments(self) -> List[List[Point3D]]:
        """Split the geometry into sub-paths (segments).

        Each sub-path starts at a move-to command and contains all
        subsequent commands up to the next move-to.

        Implicitly calls :meth:`sync_to_data`.

        Returns:
            A list of sub-paths, each a list of ``(x, y, z)`` vertices.
        """
        ...

    def get_command_at(
        self, index: int,
    ) -> Optional[_CommandRow]:
        """Retrieve a single command by index.

        Implicitly calls :meth:`sync_to_data`.

        Args:
            index: Zero-based command index.

        Returns:
            An 8-element tuple ``(type, x, y, z, aux1, aux2, aux3, aux4)``
            or ``None`` if *index* is out of range.
        """
        ...

    def iter_commands(self) -> List[_CommandRow]:
        """Return all commands as a flat list.

        Implicitly calls :meth:`sync_to_data`.

        Returns:
            A list of 8-element tuples.
        """
        ...

    def iter_typed_commands(self) -> List["PyCommand"]:
        """Yield typed :class:`PyCommand` objects.

        Backward-compatible; existing :meth:`iter_commands` still
        returns raw tuples.

        Implicitly calls :meth:`sync_to_data`.

        Returns:
            A list of :class:`PyCommand` instances.

        Raises:
            ValueError: If a row contains an unknown command type.
        """
        ...

    def get_typed_command_at(
        self, index: int,
    ) -> Optional["PyCommand"]:
        """Retrieve a typed command by index.

        Implicitly calls :meth:`sync_to_data`.

        Args:
            index: Zero-based command index.

        Returns:
            A :class:`PyCommand` instance, or ``None`` if *index*
            is out of range or negative.

        Raises:
            ValueError: If the row contains an unknown command type.
        """
        ...

    def find_closest_point(
        self, x: float, y: float,
    ) -> Optional[Tuple[int, float, Point]]:
        """Find the closest point on the path to a query position.

        Args:
            x: Query x coordinate.
            y: Query y coordinate.

        Returns:
            A tuple ``(segment_index, distance, (px, py))`` or ``None``
            if the geometry is empty.
        """
        ...

    def get_point_and_tangent_at(
        self, segment_index: int, t: float,
    ) -> Optional[Tuple[Point, Point]]:
        """Get position and tangent at parametric value *t* on a segment.

        Args:
            segment_index: Zero-based segment index.
            t: Parameter in ``[0, 1]``.

        Returns:
            ``((px, py), (tx, ty))`` — position and unit tangent — or
            ``None`` if the geometry is empty.
        """
        ...

    def get_outward_normal_at(
        self, segment_index: int, t: float,
    ) -> Optional[Point]:
        """Get the outward-facing normal at parametric value *t*.

        Args:
            segment_index: Zero-based segment index.
            t: Parameter in ``[0, 1]``.

        Returns:
            ``(nx, ny)`` — the outward unit normal — or ``None`` if the
            geometry is empty.
        """
        ...

    def has_self_intersections(
        self, fail_on_t_junction: bool = ...,
    ) -> bool:
        """Check whether the path self-intersects.

        Args:
            fail_on_t_junction: If ``True``, T-junctions are treated as
                self-intersections.  Defaults to ``False``.

        Returns:
            ``True`` if any self-intersection is detected.
        """
        ...

    def intersects_with(self, other: Geometry) -> bool:
        """Check whether this geometry intersects another.

        Args:
            other: The geometry to test against.

        Returns:
            ``True`` if the two geometries share at least one point.
        """
        ...

    def encloses(self, other: Geometry) -> bool:
        """Check whether this geometry fully encloses another.

        Args:
            other: The geometry to test.

        Returns:
            ``True`` if every point of *other* is inside this geometry.

        Raises:
            RuntimeError: If the underlying computation fails.
        """
        ...

    # -- mutation / transformation ------------------------------------------

    def clear(self) -> None:
        """Remove all commands."""
        ...

    def copy(self) -> "Geometry":
        """Return a deep copy.

        Returns:
            A new :class:`Geometry` with identical data.
        """
        ...

    def transform(self, matrix: Union[List[List[float]], NDArray[np.float64]]) -> "Geometry":
        """Apply a 4×4 affine transformation matrix in-place.

        Args:
            matrix: A ``4×4`` list-of-lists of floats representing the
                transformation.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def extend(self, other: "Geometry") -> None:
        """Append all commands from *other* to this geometry.

        Args:
            other: Geometry whose commands to append.
        """
        ...

    def simplify(self, tolerance: float) -> "Geometry":
        """Simplify the path by removing redundant points.

        Args:
            tolerance: Maximum allowed deviation from the original path.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def linearize(self, tolerance: float) -> "Geometry":
        """Convert arcs and Beziers to line segments.

        Args:
            tolerance: Maximum allowed deviation from the curves.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def fit_curves(
        self,
        tolerance: float,
        beziers: bool,
        arcs: bool,
        on_progress: Optional[object] = ...,
    ) -> "Geometry":
        """Fit curves to the linear path data.

        Args:
            tolerance: Maximum allowed deviation from the original data.
            beziers: Whether to fit cubic Bezier curves.
            arcs: Whether to fit circular arcs.
            on_progress: Optional progress callback (ignored).

        Returns:
            ``self`` (for chaining).
        """
        ...

    def fit_arcs(self, tolerance: float) -> "Geometry":
        """Shortcut for :meth:`fit_curves` with only arcs enabled.

        Args:
            tolerance: Maximum allowed deviation.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def upgrade_to_scalable(self) -> "Geometry":
        """Convert all arcs to Bezier curves so the geometry can be
        non-uniformly scaled.

        Sets :attr:`uniform_scalable` to ``True``.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def close_gaps(self, tolerance: float) -> "Geometry":
        """Close small gaps between adjacent segments.

        Args:
            tolerance: Maximum gap size to close.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def cleanup(self, tolerance: float) -> "Geometry":
        """Remove duplicate segments from the path.

        Args:
            tolerance: Maximum distance for considering two segments
                identical.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def append_data(
        self, rows: Optional[NDArray[np.float64]] = ...,
    ) -> None:
        """Append raw command rows from a NumPy array.

        Args:
            rows: An ``(N, 8)`` float64 array.  ``None`` is a no-op.
        """
        ...

    def flip_x(self) -> "Geometry":
        """Negate all x coordinates in-place.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def flip_y(self) -> "Geometry":
        """Negate all y coordinates in-place.

        Returns:
            ``self`` (for chaining).
        """
        ...

    def grow(self, amount: float) -> "Geometry":
        """Offset the geometry outward (positive) or inward (negative).

        Args:
            amount: Offset distance.  Positive grows outward.

        Returns:
            A **new** :class:`Geometry` with the offset result.
        """
        ...

    def remove_inner_edges(self) -> "Geometry":
        """Remove edges that are shared between two sub-paths.

        Returns:
            A new :class:`Geometry` with only outer edges.

        Raises:
            RuntimeError: If the underlying computation fails.
        """
        ...

    def split_inner_and_outer_contours(
        self,
    ) -> Tuple[List["Geometry"], List["Geometry"]]:
        """Partition sub-paths into inner (holes) and outer contours.

        Returns:
            A tuple ``(inner, outer)`` of :class:`Geometry` lists.

        Raises:
            RuntimeError: If the underlying computation fails.
        """
        ...

    def map_to_frame(
        self,
        origin: Point,
        p_width: Tuple[float, float],
        p_height: Tuple[float, float],
        anchor_y: Optional[float] = ...,
        stable_src_height: Optional[float] = ...,
        anchor_x: Optional[float] = ...,
        stable_src_width: Optional[float] = ...,
    ) -> "Geometry":
        """Map this geometry into a rectangular frame with anchoring.

        Typically used for positioning a workpiece on a machine bed.

        Args:
            origin: The ``(x, y)`` origin of the frame.
            p_width: ``(source_width, target_width)`` pair.
            p_height: ``(source_height, target_height)`` pair.
            anchor_y: Optional y anchor position.
            stable_src_height: Source height that should remain stable.
            anchor_x: Optional x anchor position.
            stable_src_width: Source width that should remain stable.

        Returns:
            A new :class:`Geometry` transformed to the frame.
        """
        ...

    def split_into_contours(self) -> List["Geometry"]:
        """Split into individual closed contours.

        Each contour starts with a move-to and ends where it began.

        Returns:
            A list of :class:`Geometry` objects, one per contour.
        """
        ...

    def split_into_components(self) -> List["Geometry"]:
        """Split into disconnected components.

        Components are separated by move-to commands that jump to a new
        location without drawing.

        Returns:
            A list of :class:`Geometry` objects, one per component.
        """
        ...

    def to_polygons(self, tolerance: float = ...) -> List[Polygon]:
        """Convert the geometry to a list of simple polygons.

        Curves are first linearised within *tolerance*.

        Args:
            tolerance: Linearisation tolerance.  Defaults to ``0.01``.

        Returns:
            A list of polygons (each a list of ``(x, y)`` tuples).
        """
        ...

    # -- serialisation ------------------------------------------------------

    def dump(self) -> dict:
        """Serialise the geometry to a plain dict.

        The dict contains ``"last_move_to"``, ``"uniform_scalable"``,
        and ``"commands"`` keys suitable for JSON storage.

        Returns:
            A dictionary representation of the geometry.
        """
        ...

    @classmethod
    def load(cls, data: dict) -> "Geometry":
        """Deserialise a geometry from a dict produced by :meth:`dump`.

        Args:
            data: A dict with ``"last_move_to"``, ``"uniform_scalable"``
                and ``"commands"`` keys.

        Returns:
            The reconstructed :class:`Geometry`.

        Raises:
            TypeError: If *data* is not a dict.
            ValueError: If the dict contents cannot be parsed.
        """
        ...

    @classmethod
    def from_dict(cls, data: dict) -> "Geometry":
        """Alias for :meth:`load`.

        Args:
            data: A dict with geometry data.

        Returns:
            The reconstructed :class:`Geometry`.
        """
        ...

    @classmethod
    def from_points(
        cls,
        points: List[_Point2DOr3D],
        close: bool = ...,
    ) -> "Geometry":
        """Create a geometry from a list of 2D or 3D points connected
        by straight lines.

        The first point becomes a move-to; subsequent points are
        line-to commands.  If *close* is ``True`` and there are enough
        points, the path is closed.

        Args:
            points: Ordered vertices.  Each element is either
                ``(x, y)`` or ``(x, y, z)``.
            close: Whether to close the path.  Defaults to ``True``.

        Returns:
            A new :class:`Geometry`.

        Raises:
            ValueError: If a point is not a 2- or 3-tuple of floats.
        """
        ...
