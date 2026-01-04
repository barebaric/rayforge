import numpy as np
from rayforge.core.geo.constants import (
    CMD_TYPE_MOVE,
    CMD_TYPE_LINE,
    CMD_TYPE_ARC,
)
from rayforge.core.geo.query import get_bounding_rect_from_array


def test_rect_empty():
    data = np.array([])
    res = get_bounding_rect_from_array(data)
    assert res == (0.0, 0.0, 0.0, 0.0)


def test_rect_lines():
    # Square 0,0 to 10,10
    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 10.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            [CMD_TYPE_LINE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    min_x, min_y, max_x, max_y = get_bounding_rect_from_array(data)
    assert min_x == 0.0
    assert min_y == 0.0
    assert max_x == 10.0
    assert max_y == 10.0


def test_rect_arc_bulge():
    # Semi-circle from (0,0) to (10,0).
    # Start: (0,0). End: (10,0). Center (5,0).
    # Radius = 5.
    # Start Angle (from center): atan2(0, -5) = 180 deg (PI)
    # End Angle (from center): atan2(0, 5) = 0 deg

    # CASE 1: Clockwise (180 -> 90 -> 0)
    # Sweeps through 90 deg (+Y).
    # Expect Max Y = 5. Min Y = 0.

    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # End(10,0), Offset(5,0), CW=1
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 1.0],
        ]
    )

    min_x, min_y, max_x, max_y = get_bounding_rect_from_array(data)

    assert min_x == 0.0
    assert max_x == 10.0
    assert min_y == 0.0
    assert max_y == 5.0


def test_rect_arc_bulge_negative():
    # Semi-circle from (0,0) to (10,0).
    # Start: (0,0). End: (10,0). Center (5,0).

    # CASE 2: Counter-Clockwise (180 -> 270 -> 360/0)
    # Sweeps through 270 deg (-90) (-Y).
    # Expect Min Y = -5. Max Y = 0.

    data = np.array(
        [
            [CMD_TYPE_MOVE, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            # End(10,0), Offset(5,0), CW=0
            [CMD_TYPE_ARC, 10.0, 0.0, 0.0, 5.0, 0.0, 0.0],
        ]
    )

    min_x, min_y, max_x, max_y = get_bounding_rect_from_array(data)

    assert min_x == 0.0
    assert max_x == 10.0
    assert min_y == -5.0
    assert max_y == 0.0
