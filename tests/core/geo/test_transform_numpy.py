import numpy as np
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
    COL_TYPE,
    COL_X,
    COL_Y,
    COL_I,
    COL_J,
    COL_CW,
)
from rayforge.core.geo.transform import apply_affine_transform_to_array


def test_transform_array_uniform_translation():
    data = np.array(
        [
            [CMD_TYPE_MOVE, 10.0, 20.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 30.0, 40.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Manual translation matrix
    matrix = np.eye(4)
    matrix[3, 0] = 5.0
    matrix[3, 1] = 5.0

    transformed = apply_affine_transform_to_array(data, matrix.T)

    # Check move
    assert transformed[0, COL_X] == 15.0
    assert transformed[0, COL_Y] == 25.0

    # Check line
    assert transformed[1, COL_X] == 35.0
    assert transformed[1, COL_Y] == 45.0


def test_transform_array_uniform_rotation_arc():
    # Arc from (10,0) to (0,10), center at (0,0) -> Offset from start (-10, 0)
    data = np.array(
        [
            [CMD_TYPE_ARC, 0.0, 10.0, 0.0, -10.0, 0.0, 0.0]  # CCW 0.0
        ]
    )

    # Manual 90 deg rotation matrix (CCW)
    # [ 0 -1  0  0]
    # [ 1  0  0  0]
    # ...
    matrix = np.eye(4)
    matrix[0, 0] = 0.0
    matrix[0, 1] = 1.0
    matrix[1, 0] = -1.0
    matrix[1, 1] = 0.0

    transformed = apply_affine_transform_to_array(data, matrix.T)

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
        [
            [CMD_TYPE_ARC, 10.0, 10.0, 0.0, 5.0, 0.0, 0.0]  # CW=0 (CCW)
        ]
    )

    # Flip X: Scale(-1, 1)
    matrix = np.eye(4)
    matrix[0, 0] = -1.0

    transformed = apply_affine_transform_to_array(data, matrix.T)

    assert transformed[0, COL_X] == -10.0
    assert transformed[0, COL_I] == -5.0
    assert transformed[0, COL_CW] == 1.0  # Flipped to CW


def test_transform_array_non_uniform():
    # Arc circle
    # Start (10,0), End (-10,0), Center (0,0) -> Offset (-10,0). Half circle.
    data = np.array(
        [
            [CMD_TYPE_MOVE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_ARC, -10.0, 0.0, 0.0, -10.0, 0.0, 0.0],
        ]
    )

    # Scale Y by 0.5 -> Ellipse
    matrix = np.eye(4)
    matrix[1, 1] = 0.5

    transformed = apply_affine_transform_to_array(data, matrix.T)

    # First row is still the transformed Move
    assert transformed[0, COL_TYPE] == CMD_TYPE_MOVE
    # Subsequent rows should all be LINE commands
    assert transformed.shape[0] > 2
    assert np.all(transformed[1:, COL_TYPE] == CMD_TYPE_LINE)

    # End point should be at (-10, 0) still
    last_row = transformed[-1]
    assert np.isclose(last_row[COL_X], -10.0)
    assert np.isclose(last_row[COL_Y], 0.0)
