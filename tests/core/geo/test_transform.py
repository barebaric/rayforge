import pytest
import math
import numpy as np
from rayforge.core.geo.constants import (
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
from rayforge.core.geo import (
    Geometry,
)
from rayforge.core.geo.text import text_to_geometry
from rayforge.core.geo.transform import (
    apply_affine_transform_to_array,
    grow_geometry,
    map_geometry_to_frame,
)


def _create_translate_matrix(x, y, z):
    """Creates a NumPy translation matrix."""
    return np.array(
        [
            [1, 0, 0, x],
            [0, 1, 0, y],
            [0, 0, 1, z],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_scale_matrix(sx, sy, sz):
    """Creates a NumPy scaling matrix."""
    return np.array(
        [
            [sx, 0, 0, 0],
            [0, sy, 0, 0],
            [0, 0, sz, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


def _create_z_rotate_matrix(angle_rad):
    """Creates a NumPy Z-axis rotation matrix."""
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return np.array(
        [
            [c, -s, 0, 0],
            [s, c, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=float,
    )


# --- Affine Transform Tests ---


def test_transform_identity():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)
    original_geo = geo.copy()

    identity_matrix = np.identity(4, dtype=float)
    geo.transform(identity_matrix)

    assert geo == original_geo


def test_transform_translate():
    geo = Geometry()
    geo.move_to(10, 20, 30)
    geo.arc_to(50, 60, i=5, j=7, z=40)
    geo.bezier_to(70, 80, c1x=55, c1y=65, c2x=65, c2y=75, z=50)

    translate_matrix = _create_translate_matrix(10, -5, 15)
    geo.transform(translate_matrix)
    assert geo.data is not None

    # Check move
    assert np.allclose(geo.data[0, 1:4], (20, 15, 45))
    # Check arc
    assert np.allclose(geo.data[1, 1:4], (60, 55, 55))
    # Translation should NOT affect arc center offsets (vectors)
    assert np.allclose(geo.data[1, 4:6], (5, 7))
    # Check bezier
    assert np.allclose(geo.data[2, 1:4], (80, 75, 65))
    # Translation SHOULD affect bezier control points (absolute coords)
    assert np.allclose(geo.data[2, COL_C1X : COL_C1Y + 1], (65, 60))
    assert np.allclose(geo.data[2, COL_C2X : COL_C2Y + 1], (75, 70))


def test_transform_scale_non_uniform_preserves_beziers():
    geo = Geometry()
    geo.move_to(10, 20, 5)
    # Arc converted to bezier using arc_to_as_bezier(). This may create
    # one or more bezier segments.
    geo.arc_to_as_bezier(22, 22, i=5, j=7, z=-10)
    # This is the last command added.
    geo.bezier_to(30, 30, c1x=24, c1y=24, c2x=28, c2y=28, z=-20)
    scale_matrix = _create_scale_matrix(2, 3, 4)

    geo.transform(scale_matrix)
    assert geo.data is not None

    # 1. Check Move
    assert np.allclose(geo.data[0, 1:4], (20, 60, 20))

    # 2. Check that all subsequent commands are still Beziers
    assert np.all(geo.data[1:, COL_TYPE] == CMD_TYPE_BEZIER)

    # 3. Check the final state of the explicit bezier_to command.
    # It's the last row.
    final_bezier_row = geo.data[-1]

    # Check end point: (30*2, 30*3, -20*4) -> (60, 90, -80)
    final_point = final_bezier_row[COL_X : COL_Z + 1]
    assert np.allclose(final_point, (60.0, 90.0, -80.0))

    # Check C1: (24*2, 24*3) -> (48, 72)
    final_c1 = final_bezier_row[COL_C1X : COL_C1Y + 1]
    assert np.allclose(final_c1, (48.0, 72.0))

    # Check C2: (28*2, 28*3) -> (56, 84)
    final_c2 = final_bezier_row[COL_C2X : COL_C2Y + 1]
    assert np.allclose(final_c2, (56.0, 84.0))

    # 4. Check the final state of the arc_to_as_bezier command.
    # Its last segment is the second-to-last row.
    arc_end_row = geo.data[-2]

    # Check end point: (22*2, 22*3, -10*4) -> (44, 66, -40)
    arc_final_point = arc_end_row[COL_X : COL_Z + 1]
    assert np.allclose(arc_final_point, (44.0, 66.0, -40.0))


def test_transform_rotate_preserves_z():
    geo = Geometry()
    geo.move_to(10, 10, -5)
    rotate_matrix = _create_z_rotate_matrix(math.radians(90))

    geo.transform(rotate_matrix)
    assert geo.data is not None

    x, y, z = geo.data[0, 1:4]
    assert z == -5
    assert x == pytest.approx(-10)
    assert y == pytest.approx(10)


def test_transform_uniform_scale_preserves_curves():
    geo = Geometry()
    geo.move_to(0, 0, 0)
    # Arc from (0,0) to (10,0) with center at (5,0) -> radius 5
    geo.arc_to(10, 0, i=5, j=0, clockwise=True)
    # Bezier from (10,0) to (20,0)
    geo.bezier_to(20, 0, c1x=12, c1y=2, c2x=18, c2y=-2)

    # Uniform scale by 2
    scale_matrix = _create_scale_matrix(2, 2, 2)
    geo.transform(scale_matrix)
    assert geo.data is not None

    # Check arc
    arc_row = geo.data[1]
    assert arc_row[COL_TYPE] == CMD_TYPE_ARC
    assert np.allclose(arc_row[1:4], (20, 0, 0))
    # Offset should also scale
    assert np.allclose(arc_row[4:6], (10, 0))

    # Check bezier
    bezier_row = geo.data[2]
    assert bezier_row[COL_TYPE] == CMD_TYPE_BEZIER
    assert np.allclose(bezier_row[1:4], (40, 0, 0))
    # Control points should also scale
    assert np.allclose(bezier_row[COL_C1X : COL_C1Y + 1], (24, 4))
    assert np.allclose(bezier_row[COL_C2X : COL_C2Y + 1], (36, -4))


# --- Raw Array Transform Tests (formerly test_transform_numpy.py) ---


def test_transform_array_uniform_translation():
    data = np.array(
        [
            [CMD_TYPE_MOVE, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Manual translation matrix
    matrix = np.eye(4)
    matrix[0, 3] = 5.0
    matrix[1, 3] = 5.0

    transformed = apply_affine_transform_to_array(data, matrix)

    # Check move
    assert transformed[0, COL_X] == 15.0
    assert transformed[0, COL_Y] == 25.0

    # Check line
    assert transformed[1, COL_X] == 35.0
    assert transformed[1, COL_Y] == 45.0


def test_transform_array_uniform_rotation_arc():
    # Arc from (10,0) to (0,10), center at (0,0) -> Offset from start (-10, 0)
    data = np.array(
        [[CMD_TYPE_ARC, 0.0, 10.0, 0.0, -10.0, 0.0, 0.0, 0.0]]  # CCW 0.0
    )

    # Manual 90 deg rotation matrix (CCW)
    matrix = np.eye(4)
    matrix[0, 0] = 0.0
    matrix[0, 1] = -1.0
    matrix[1, 0] = 1.0
    matrix[1, 1] = 0.0

    transformed = apply_affine_transform_to_array(data, matrix)

    # Check end point (0,10) -> (-10, 0)
    assert np.isclose(transformed[0, COL_X], -10.0)
    assert np.isclose(transformed[0, COL_Y], 0.0)

    # Check offset (-10, 0) -> (0, -10)
    # Rotation of vector (-10, 0) by 90 deg is (0, -10)
    assert np.isclose(transformed[0, COL_I], 0.0)
    assert np.isclose(transformed[0, COL_J], -10.0)


def test_transform_array_flip():
    # Arc
    data = np.array(
        [[CMD_TYPE_ARC, 10.0, 10.0, 0.0, 5.0, 0.0, 0.0, 0.0]]  # CW=0 (CCW)
    )

    # Flip X: Scale(-1, 1)
    matrix = np.eye(4)
    matrix[0, 0] = -1.0

    transformed = apply_affine_transform_to_array(data, matrix)

    assert transformed[0, COL_X] == -10.0
    assert transformed[0, COL_I] == -5.0
    assert transformed[0, COL_CW] == 1.0  # Flipped to CW


def test_transform_array_non_uniform():
    # Input: Move -> Arc -> Bezier
    # Arc: Start (10,0), End (-10,0), Center (0,0) -> Offset (-10,0).
    # Bezier: Start (-10,0), End (0,10), C1(-10, 5), C2(-5, 10)
    data = np.array(
        [
            [CMD_TYPE_MOVE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, -10.0, 0.0, 0.0, -10.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_BEZIER, 0.0, 10.0, 0.0, -10.0, 5.0, -5.0, 10.0],
        ]
    )

    # Scale Y by 0.5 -> Ellipse.
    matrix = np.eye(4)
    matrix[1, 1] = 0.5

    transformed = apply_affine_transform_to_array(data, matrix)

    # 1. Check Move
    assert transformed[0, COL_TYPE] == CMD_TYPE_MOVE
    assert transformed[0, COL_X] == 10.0
    assert transformed[0, COL_Y] == 0.0

    # 2. Check Arc -> Lines
    # The middle section should be lines
    assert transformed[1, COL_TYPE] == CMD_TYPE_LINE

    # 3. Check Bezier -> Bezier (Last element)
    last_row = transformed[-1]
    assert last_row[COL_TYPE] == CMD_TYPE_BEZIER

    # Check Bezier Points transformation
    # End: (0, 10) -> (0, 5)
    assert np.isclose(last_row[COL_X], 0.0)
    assert np.isclose(last_row[COL_Y], 5.0)
    # C1: (-10, 5) -> (-10, 2.5)
    assert np.isclose(last_row[COL_C1X], -10.0)
    assert np.isclose(last_row[COL_C1Y], 2.5)
    # C2: (-5, 10) -> (-5, 5)
    assert np.isclose(last_row[COL_C2X], -5.0)
    assert np.isclose(last_row[COL_C2Y], 5.0)


# --- Grow/Offset Tests ---


def test_grow_simple_square():
    """Tests growing and shrinking a simple CCW square."""
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Grow the square
    grown_square = grow_geometry(square, 1.0)
    assert grown_square.area() == pytest.approx(144.0)  # (10+2)^2
    # Check one of the new vertices
    grown_points = grown_square.segments()[0]
    # Use pytest.approx for floating point comparisons of coordinates
    assert any(np.allclose(p, (-1.0, -1.0, 0.0)) for p in grown_points), (
        "Expected grown vertex not found"
    )

    # Shrink the square
    shrunk_square = grow_geometry(square, -1.0)
    assert shrunk_square.area() == pytest.approx(64.0)  # (10-2)^2
    shrunk_points = shrunk_square.segments()[0]
    assert any(np.allclose(p, (1.0, 1.0, 0.0)) for p in shrunk_points), (
        "Expected shrunk vertex not found"
    )


def test_grow_clockwise_square():
    """Tests that offset direction is consistent for a CW shape."""
    # A clockwise square
    square_cw = Geometry.from_points([(0, 0), (0, 10), (10, 10), (10, 0)])

    # A positive offset on any shape should grow it
    grown_square = grow_geometry(square_cw, 1.0)
    assert grown_square.area() == pytest.approx(144.0)

    # A negative offset on any shape should shrink it
    shrunk_square = grow_geometry(square_cw, -1.0)
    assert shrunk_square.area() == pytest.approx(64.0)


def test_grow_shape_with_hole():
    """Tests offsetting a shape containing a hole."""
    # Outer CCW square (0,0) -> (20,20), Area = 400
    outer = Geometry.from_points([(0, 0), (20, 0), (20, 20), (0, 20)])
    # Inner CW square (hole) (5,5) -> (15,15), Area = -100
    inner = Geometry.from_points([(5, 5), (5, 15), (15, 15), (15, 5)])
    shape_with_hole = outer.copy()
    shape_with_hole.extend(inner)
    assert shape_with_hole.area() == pytest.approx(300.0)

    # Grow by 1. Outer becomes 22x22, inner becomes 8x8.
    # New area = 22*22 - 8*8 = 484 - 64 = 420.
    grown_shape = grow_geometry(shape_with_hole, 1.0)
    assert grown_shape.area() == pytest.approx(420.0)

    # Shrink by 1. Outer becomes 18x18, inner becomes 12x12.
    # New area = 18*18 - 12*12 = 324 - 144 = 180.
    shrunk_shape = grow_geometry(shape_with_hole, -1.0)
    assert shrunk_shape.area() == pytest.approx(180.0)


def test_grow_open_path_is_ignored():
    """Tests that open paths result in an empty geometry."""
    open_path = Geometry.from_points([(0, 0), (10, 10), (20, 0)], close=False)
    result = grow_geometry(open_path, 1.0)
    assert result.is_empty()


def test_grow_circle():
    """Tests offsetting a shape with arcs by checking the resulting area."""
    radius = 10.0
    # Create a polygonal approximation of a circle using from_points. This
    # avoids issues with how area() handles ArcTo and ensures a valid, simple
    # polygon for testing the offset logic itself.
    num_points = 100
    angles = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    points = [(radius * np.cos(a), radius * np.sin(a)) for a in angles]
    circle = Geometry.from_points(points)

    original_area = math.pi * radius**2
    assert circle.area() == pytest.approx(original_area, rel=1e-3)

    # Grow the circle
    offset = 2.0
    grown_circle = grow_geometry(circle, offset)
    expected_grown_area = math.pi * (radius + offset) ** 2
    assert grown_circle.area() == pytest.approx(expected_grown_area, rel=1e-2)

    # Shrink the circle
    offset = -2.0
    shrunk_circle = grow_geometry(circle, offset)
    expected_shrunk_area = math.pi * (radius + offset) ** 2
    assert shrunk_circle.area() == pytest.approx(
        expected_shrunk_area, rel=1e-2
    )


def test_shrink_to_nothing():
    """Tests that shrinking a shape by its half-width or more is handled."""
    square = Geometry.from_points([(0, 0), (10, 0), (10, 10), (0, 10)])

    # Shrinking by half the width should result in a zero-area shape
    shrunk_to_point = grow_geometry(square, -5.0)
    assert shrunk_to_point.area() == pytest.approx(0.0)

    # Shrinking by more than the half-width should also result in zero area
    shrunk_past_zero = grow_geometry(square, -6.0)
    # The algorithm might produce a small self-intersecting shape with non-zero
    # area in this case, but it should be very small. A robust offset algorithm
    # would clean this up, but for now we check that it's close to zero.
    assert shrunk_past_zero.area() == pytest.approx(0.0, abs=1.0)


# --- Map to Frame Tests ---


def test_map_geometry_to_frame_identity():
    """Tests mapping a geometry to a frame matching its own bounding box."""
    geo = Geometry.from_points([(10, 20), (30, 20), (30, 50), (10, 50)])
    original_geo = geo.copy()

    # Define a frame that is identical to the geometry's bounding box
    origin = (10, 20)
    p_width = (30, 20)
    p_height = (10, 50)

    mapped_geo = map_geometry_to_frame(geo, origin, p_width, p_height)

    # The result should be identical to the original
    assert mapped_geo == original_geo


def test_map_geometry_to_frame_translate_scale():
    """Tests mapping a unit square to a larger, translated rectangle."""
    # Source is a 1x1 square at the origin
    unit_square = Geometry.from_points([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Target is a 50x20 rectangle at (100, 200)
    origin = (100, 200)
    p_width = (150, 200)  # 50 units wide
    p_height = (100, 220)  # 20 units high

    mapped_geo = map_geometry_to_frame(unit_square, origin, p_width, p_height)

    # Check the bounding box of the result
    min_x, min_y, max_x, max_y = mapped_geo.rect()
    assert min_x == pytest.approx(100)
    assert min_y == pytest.approx(200)
    assert max_x == pytest.approx(150)
    assert max_y == pytest.approx(220)


def test_map_geometry_to_frame_non_uniform_scale():
    """Tests mapping (stretching) a geometry non-uniformly."""
    # Source is text "I", which has a non-square aspect ratio
    text_geo = text_to_geometry("I", font_size=10)
    min_x_s, min_y_s, max_x_s, max_y_s = text_geo.rect()
    # The text is not at (0,0), so this is a good test case

    # Target is a 50x100 rectangle at (0,0)
    origin = (0, 0)
    p_width = (50, 0)
    p_height = (0, 100)

    mapped_geo = map_geometry_to_frame(text_geo, origin, p_width, p_height)

    # The result should be stretched to fill the 50x100 box exactly
    min_x, min_y, max_x, max_y = mapped_geo.rect()
    assert min_x == pytest.approx(0)
    assert min_y == pytest.approx(0)
    assert max_x == pytest.approx(50)
    assert max_y == pytest.approx(100)


def test_map_geometry_to_frame_rotate_and_shear():
    """Tests mapping to a rotated and sheared parallelogram frame."""
    unit_square = Geometry.from_points([(0, 0), (1, 0), (1, 1), (0, 1)])

    # Target is a parallelogram
    origin = (10, 10)
    p_width = (20, 15)  # Vector U = (10, 5)
    p_height = (5, 20)  # Vector V = (-5, 10)

    mapped_geo = map_geometry_to_frame(unit_square, origin, p_width, p_height)

    # The four corners of the unit square should map to the four corners of
    # the parallelogram: P0, P_width, P_height, P_width+P_height-P0
    segments = mapped_geo.segments()[0]
    expected_corners = [
        (10, 10, 0),  # origin
        (20, 15, 0),  # p_width
        (15, 25, 0),  # implicit 4th point: origin + U + V
        (5, 20, 0),  # p_height
    ]

    # Check if all expected corners are present in the transformed geometry's
    # vertices. The order might change due to how from_points works.
    assert len(segments) == 5  # 4 points + closing point
    for expected_corner in expected_corners:
        found = any(np.allclose(p, expected_corner) for p in segments)
        assert found, f"Corner {expected_corner} not found in transformed geo"


def test_map_geometry_to_frame_empty_geometry():
    """Tests that mapping an empty geometry results in an empty geometry."""
    empty_geo = Geometry()
    mapped_geo = map_geometry_to_frame(empty_geo, (0, 0), (10, 0), (0, 10))
    assert mapped_geo.is_empty()


def test_map_geometry_to_frame_degenerate_source():
    """
    Tests that mapping a geometry with zero width or height is handled.
    """
    # A single line has zero width/height depending on orientation
    line_geo = Geometry.from_points([(0, 0), (10, 0)], close=False)
    assert line_geo.rect()[3] - line_geo.rect()[1] == 0  # Zero height

    mapped_geo = map_geometry_to_frame(line_geo, (0, 0), (10, 0), (0, 10))
    # Should return an empty geometry as the scaling would be infinite
    assert mapped_geo.is_empty()
