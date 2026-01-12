# Geo Module API Reference

The `geo` module is a high-performance 2D/3D geometry library backed by
NumPy. It creates, analyzes, and transforms geometric paths defined by
lines, arcs, and Bézier curves.

---

## 1. Data Structure & Constants

The core of the library relies on a stateless `(N, 8)` NumPy float64 array.
While the `Geometry` class wraps this, understanding the layout is
crucial for low-level manipulation.

**Command Types (`COL_TYPE`)**

- `CMD_TYPE_MOVE` (1.0): Moves the pen to a new location. Starts a new
  contour.
- `CMD_TYPE_LINE` (2.0): Draws a straight line to `(x, y, z)`.
- `CMD_TYPE_ARC` (3.0): Draws a circular arc to `(x, y, z)`.
- `CMD_TYPE_BEZIER` (4.0): Draws a cubic Bézier curve to `(x, y, z)`.

**Array Layout (Columns)**

The geometry array shape is `(N, 8)`.

| Index | Constant             | Description                                                                    |
| :---- | :------------------- | :----------------------------------------------------------------------------- |
| 0     | `COL_TYPE`           | The command type ID (1.0 - 4.0).                                               |
| 1     | `COL_X`              | End X coordinate.                                                              |
| 2     | `COL_Y`              | End Y coordinate.                                                              |
| 3     | `COL_Z`              | End Z coordinate.                                                              |
| 4     | `COL_I` / `COL_C1X`  | **Arc:** Center offset X from start. **Bezier:** Control Point 1 X (absolute). |
| 5     | `COL_J` / `COL_C1Y`  | **Arc:** Center offset Y from start. **Bezier:** Control Point 1 Y (absolute). |
| 6     | `COL_CW` / `COL_C2X` | **Arc:** 1.0 = Clockwise, 0.0 = CCW. **Bezier:** Control Point 2 X (absolute). |
| 7     | `COL_C2Y`            | **Bezier:** Control Point 2 Y (absolute). Unused for Arcs.                     |

---

## 2. Class: `geometry.Geometry`

The main entry point for the API. This class is mutable and manages the
internal NumPy array.

### Properties

#### `data`

- **Type:** `Optional[np.ndarray]`
- **Description:** Returns the raw `(N, 8)` float64 array representing the
  path. If new commands have been added via `line_to`, etc., they are
  synchronized from a pending list into the NumPy array before returning.
  Returns `None` if the geometry is completely empty.

#### `uniform_scalable`

- **Type:** `bool`
- **Description:** `True` if the geometry contains only Lines or Béziers.
  `False` if it contains Circular Arcs.
- **Implication:** If `False`, applying a non-uniform scale transform
  (e.g., `scale(1, 0.5)`) will raise a `TypeError` because circular
  arcs cannot become elliptical in this engine. Use `upgrade_to_scalable()` or
  `arc_to_as_bezier()` to fix this.

### Construction Methods

#### `move_to(x: float, y: float, z: float = 0.0) -> None`

Starts a new subpath. If the previous path was not closed, it remains
open.

#### `line_to(x: float, y: float, z: float = 0.0) -> None`

Adds a straight line segment from the current position to `(x, y, z)`.

#### `close_path() -> None`

Adds a `line_to` command connecting the current position back to the
coordinates of the last `move_to`.

#### `arc_to(x: float, y: float, i: float, j: float, clockwise: bool = True, z: float = 0.0) -> None`

Adds a circular arc.

- **x, y, z**: The end point of the arc.
- **i, j**: The offset vector from the _start_ point to the _center_ of
  the circle.
- **clockwise**: Direction of the arc.
- _Note:_ Sets `uniform_scalable = False`.

#### `arc_to_as_bezier(...) -> None`

Same signature as `arc_to`. Approximates the circular arc using one or
more cubic Bézier curves (splitting every 90 degrees). Use this if you
intend to apply non-uniform scaling later.

#### `bezier_to(x: float, y: float, c1x: float, c1y: float, c2x: float, c2y: float, z: float = 0.0) -> None`

Adds a cubic Bézier curve.

- **x, y, z**: The end point.
- **c1x, c1y**: Absolute coordinates of the first control point.
- **c2x, c2y**: Absolute coordinates of the second control point.

#### `extend(other: Geometry) -> None`

Appends all commands from the `other` geometry object to the end of this
one. Efficiently stacks NumPy arrays.

#### `clear() -> None`

Resets the object to an empty state, clearing internal arrays and cache.

### Modification Methods

#### `transform(matrix: np.ndarray) -> self`

Applies a 4x4 affine transformation matrix to the geometry in-place.

- **matrix**: A 4x4 numpy array.
- **Exception**: Raises `TypeError` if the matrix implies non-uniform scaling
  and the geometry contains Arcs.

#### `simplify(tolerance: float = 0.01) -> self`

Reduces the number of vertices in linear segments using the
**Ramer-Douglas-Peucker** algorithm.

- **tolerance**: Max perpendicular distance (deviation) allowed.
- _Note:_ Does not affect Arcs or Béziers.

#### `linearize(tolerance: float) -> self`

Converts **all** Arcs and Béziers into sequences of straight `LINE`
commands.

- **tolerance**: Max deviation allowed between the original curve and the
  linear approximation.
- _Result:_ The geometry will contain only `MOVE` and `LINE` commands.
  `uniform_scalable` becomes `True`.

#### `fit_arcs(tolerance: float) -> self`

Reconstructs the path by attempting to fit circular Arcs and straight
Lines to existing points.

- **Algorithm**: Linearizes the path first, simplifies points, then uses
  recursive least-squares fitting to detect lines and circles.
- **Use Case**: Converting dense polylines (e.g., from a scan) into clean
  CAD geometry.

#### `grow(amount: float) -> Geometry`

Performs a polygon offset (buffering).

- **amount**: Distance to offset. Positive expands, negative shrinks.
- **Algorithm**: Uses `pyclipper` (Vatti clipping algorithm).
- **Returns**: A _new_ Geometry object.

#### `close_gaps(tolerance: float = 1e-6) -> self`

Snaps end-points of subpaths to start-points of subsequent subpaths if
they are within `tolerance`. Use this to fix imported CAD files that
look closed but have microscopic gaps.

#### `upgrade_to_scalable() -> self`

Converts all internal Arc commands to Bézier commands in-place. Enables
non-uniform scaling.

#### `remove_inner_edges() -> Geometry`

Filters the geometry. Keeps all open paths. For closed paths, it removes
any that are considered "holes" (internal contours), returning only the
outline.

### Analysis & Query Methods

#### `area() -> float`

Calculates the total signed area.

- **Behavior**: Positive for CCW, Negative for CW. Returns the absolute
  value of the sum. Correctly handles shapes with holes if winding is
  consistent.

#### `distance() -> float`

Returns the total length of the path (sum of all segments).

#### `rect() -> Tuple[float, float, float, float]`

Returns the exact bounding box `(min_x, min_y, max_x, max_y)`.

- _Note:_ Accurately calculates the extents of Arcs (bulges) and
  Béziers (convex hull property), not just endpoints.

#### `is_closed(tolerance: float = 1e-6) -> bool`

Checks if the _first_ contour in the geometry is closed (start point
equals end point). For multi-contour geometries, check
`split_into_contours()` first.

#### `encloses(other: Geometry) -> bool`

Determines if this geometry fully contains `other`.

- **Checks**: Bounding box $\to$ Intersection $\to$ Point-in-Polygon
  (Winding Number).

#### `intersects_with(other: Geometry) -> bool`

Checks if the _lines/curves_ of this geometry intersect with `other`.

#### `find_closest_point(x: float, y: float) -> Optional[Tuple[int, float, Tuple[float, float]]]`

Finds the location on the path closest to the query coordinates.

- **Returns**: `(segment_index, t, (pt_x, pt_y))`.
  - `segment_index`: Index of the command in the data array.
  - `t`: Parameter (0.0 to 1.0) along that segment.

### Topology Methods

#### `split_into_contours() -> List[Geometry]`

Splits the geometry at every `MOVE` command. Returns a list of
geometries, where each contains exactly one continuous path.

#### `split_into_components() -> List[Geometry]`

Logically groups contours.

- **Algorithm**: If a contour A fully encloses contour B, B is considered a
  hole of A. They are returned together as one "Component" Geometry.
  Disjoint shapes are returned as separate Geometries.

#### `split_inner_and_outer_contours() -> Tuple[List[Geometry], List[Geometry]]`

Returns `(holes, solids)`. Uses the Even-Odd rule logic to determine
which contours are internal voids and which are external shapes.

---

## 3. Submodule: `analysis`

Low-level analysis algorithms operating directly on NumPy arrays.

- **`is_closed(commands: np.ndarray, tolerance: float) -> bool`**
  Checks distance between the first and last point of the array.
- **`get_subpath_area_from_array(data: np.ndarray, start_cmd_index: int) -> float`**
  Calculates signed area of a specific subpath using the Shoelace
  formula.
- **`get_path_winding_order_from_array(...) -> str`**
  Returns `'cw'`, `'ccw'`, or `'unknown'` (if degenerate/flat).
- **`get_point_and_tangent_at_from_array(data, index, t) -> Optional[Tuple[Point, Vector]]`**
  Calculates exact point and normalized tangent vector for Line, Arc, or
  Bezier at parameter `t`.

---

## 4. Submodule: `contours`

Algorithms for cleaning and organizing path topology.

- **`close_geometry_gaps(geometry: Geometry, tolerance: float) -> Geometry`**
  The implementation behind `Geometry.close_gaps`. Performs intra-contour
  snapping (closing loops) and inter-contour snapping (connecting lines).
- **`filter_to_external_contours(contours: List[Geometry]) -> List[Geometry]`**
  Accepts a list of single-contour Geometries. Returns a list containing
  only the "Solid" shapes. Normalizes winding orders internally before
  filtering.
- **`normalize_winding_orders(contours: List[Geometry]) -> List[Geometry]`**
  Analyzes nesting. Enforces:
  - **Solids (Even nesting):** CCW
  - **Holes (Odd nesting):** CW
  - _Returns:_ A new list of Geometries with corrected directions.
- **`reverse_contour(contour: Geometry) -> Geometry`**
  Mathematically reverses the path. Handles Arc CW/CCW flipping and
  Bezier control point swapping.

---

## 5. Submodule: `fitting`

Algorithms for converting point clouds to clean CAD geometry.

- **`fit_points_to_primitives(points: List[Point], tolerance: float) -> List[np.ndarray]`**
  Recursive fitting algorithm.
  1. Tries to fit a single Line.
  2. If error > tolerance, tries to fit a single Arc (Least Squares
     circle fit).
  3. If error > tolerance, splits points in half and recurses.
- **`fit_arcs(data: np.ndarray, tolerance: float, ...) -> np.ndarray`**
  Reconstructs an existing path. First linearizes it into points,
  simplifies those points (RDP), then runs `fit_points_to_primitives`.

---

## 6. Submodule: `intersect`

- **`check_intersection_from_array(data1, data2, fail_on_t_junction=False) -> bool`**
  Brute-force intersection check ($O(N*M)$).
  - `fail_on_t_junction`: If `True`, T-junctions (vertex of one segment
    lying on another segment) count as intersections.
- **`check_self_intersection_from_array(data, fail_on_t_junction=False) -> bool`**
  Checks if a path intersects itself. Ignores adjacent segment connections
  (vertices).

---

## 7. Submodule: `linearize`

- **`linearize_geometry(data: np.ndarray, tolerance: float) -> np.ndarray`**
  Converts data to `MOVE` and `LINE` only. Uses `tolerance` to calculate
  an adaptive resolution for curves, then runs simplification.
- **`flatten_to_points(data, resolution) -> List[List[Point]]`**
  Converts geometry into raw lists of 3D coordinates.
- **`linearize_arc(...)`** and **`linearize_bezier(...)`**
  Helpers that return lists of line segments approximating the specific
  primitive.

---

## 8. Submodule: `primitives`

Fundamental geometric tests.

- **`is_point_in_polygon(point, polygon) -> bool`**
  Robust test. Uses AABB fast-fail, then Boundary check, then Ray
  Casting.
- **`clip_line_segment(p1, p2, rect)`** (from `clipping.py`)
  Clips a 3D line to a 2D rectangle using the **Cohen-Sutherland**
  algorithm. Interpolates Z.
- **`find_closest_point_on_arc(...)`**
  Analytic solution for closest point on circle, clamped to arc angles.
- **`get_arc_bounding_box(...)`**
  Calculates tight bounds, accounting for whether the arc sweeps across
  cardinal quadrants (0, 90, 180, 270 degrees).

---

## 9. Submodule: `transform`

- **`grow_geometry(geometry, offset) -> Geometry`**
  Wraps the **PyClipper** library.
  1. Scales floats to integers (PyClipper requirement).
  2. Performs the offset (Miter join, Closed Polygon).
  3. Scales back to floats and reconstructs Geometry.
- **`apply_affine_transform_to_array(data, matrix) -> np.ndarray`**
  - **Uniform Scale/Rotate/Translate**: Transforms endpoints, centers, and
    control points directly.
  - **Non-Uniform Scale**: Detects if scale X != scale Y. If so,
    linearizes Arcs into Lines before transforming, as Arcs cannot be
    non-uniformly scaled. Beziers are transformed mathematically without
    linearization.
