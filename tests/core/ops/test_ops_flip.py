import pytest
from typing import cast, List
from rayforge.core.ops import (
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    MovingCommand,
    ScanLinePowerCommand,
)
from rayforge.core.ops.flip import flip_segment


def test_flip_segment_empty_or_short():
    assert flip_segment([]) == []
    segment: List[MovingCommand] = [MoveToCommand((1, 2, 3))]
    assert flip_segment(segment) == segment


def test_flip_segment_lines_only():
    segment: List[MovingCommand] = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((10, 0, 1)),
        LineToCommand((10, 10, 2)),
    ]
    # Assign states for a more robust test
    for i, cmd in enumerate(segment):
        cmd.state = f"state_{i}"  # type: ignore

    flipped = flip_segment(segment)

    assert len(flipped) == 3
    # Check points are reversed
    assert flipped[0].end == (10, 10, 2)
    assert flipped[1].end == (10, 0, 1)
    assert flipped[2].end == (0, 0, 0)

    # Check that states are correctly shifted
    assert flipped[0].state == "state_0"
    assert flipped[1].state == "state_2"
    assert flipped[2].state == "state_1"


def test_flip_segment_with_arc():
    # Arc from (10,0) to (0,0) with center at (5,0) [CCW]
    # i, j is offset from start: i= -5, j=0
    segment: List[MovingCommand] = [
        MoveToCommand((0, 10, 0)),
        LineToCommand((10, 0, 0)),
        ArcToCommand((0, 0, 0), (-5, 0), False),
    ]

    flipped = flip_segment(segment)
    assert len(flipped) == 3

    # Check point reversal
    assert flipped[0].end == (0, 0, 0)
    assert flipped[1].end == (10, 0, 0)
    assert flipped[2].end == (0, 10, 0)

    # Validate the flipped arc command
    flipped_arc = cast(ArcToCommand, flipped[1])
    assert isinstance(flipped_arc, ArcToCommand)
    # New endpoint is the start of the original LineTo
    assert flipped_arc.end == (10, 0, 0)
    # Direction should be inverted
    assert flipped_arc.clockwise is True
    # Center is (5,0). New start is (0,0). New end is (10,0).
    # New offset i,j is from the new start (0,0) to center (5,0)
    assert flipped_arc.center_offset == pytest.approx((5, 0))


def test_flip_segment_with_scanline():
    """Tests that flipping a segment with a ScanLinePowerCommand works."""
    powers = bytearray([10, 20, 30])
    segment: List[MovingCommand] = [
        MoveToCommand((0, 0, 0)),
        ScanLinePowerCommand(end=(10, 10, 10), power_values=powers),
    ]

    flipped = flip_segment(segment)
    assert len(flipped) == 2
    assert isinstance(flipped[0], MoveToCommand)
    assert isinstance(flipped[1], ScanLinePowerCommand)

    # Check geometry
    assert flipped[0].end == (10, 10, 10)
    assert flipped[1].end == (0, 0, 0)

    # Check power values
    flipped_scan = cast(ScanLinePowerCommand, flipped[1])
    assert flipped_scan.power_values == bytearray([30, 20, 10])
