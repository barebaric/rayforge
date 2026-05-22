import pytest
from rayforge.core.ops import Ops
from rayforge.core.ops.enums import CommandType, CommandCategory
from post_processors.transformers import (
    Optimize,
    greedy_order_segments,
    two_opt,
    kdtree_order_segments,
)


def _make_seg(start: tuple, end: tuple) -> Ops:
    """Create a 2-point segment: move_to then line_to."""
    ops = Ops()
    ops.move_to(*start)
    ops.line_to(*end)
    return ops


@pytest.fixture
def ctx(mock_progress_context) -> object:
    """Provides a dummy execution context for functions that require it."""
    return mock_progress_context


def test_greedy_order_segments(mock_progress_context):
    """Test the greedy algorithm for initial segment ordering."""
    # Seg1: (0,0) -> (10,0) - should be chosen first
    # Seg2: (100,100) -> (110,100)
    # Seg3: (10,0) -> (10,10) - should be chosen second
    s1 = _make_seg((0, 0, 0), (10, 0, 0))
    s2 = _make_seg((100, 100, 0), (110, 100, 0))
    s3 = _make_seg((10, 0, 0), (10, 10, 0))
    segments = [s1, s2, s3]

    ordered = greedy_order_segments(mock_progress_context, segments)
    assert len(ordered) == 3
    # Expected order: s1, s3, s2
    assert ordered[0] is s1
    assert ordered[1] is s3
    assert ordered[2] is s2


def test_greedy_order_with_flip(mock_progress_context):
    """Test greedy ordering when flipping a segment is optimal."""
    # Seg1: (0,0) -> (10,0)
    # Seg2: (100,100) -> (110,100)
    # Seg3: (10,10) -> (10,0) <-- start is far, end is near
    s1 = _make_seg((0, 0, 0), (10, 0, 0))
    s2 = _make_seg((100, 100, 0), (110, 100, 0))
    s3 = _make_seg((10, 10, 0), (10, 0, 0))
    segments = [s1, s2, s3]

    ordered = greedy_order_segments(mock_progress_context, segments)

    # Expected: s1, flipped(s3), s2
    assert ordered[0] is s1
    assert ordered[1] is not s3  # Should be a new, flipped list
    assert ordered[1].endpoint(0) == (10, 0, 0)  # Start of flipped s3
    last = ordered[1].len() - 1
    assert ordered[1].endpoint(last) == (10, 10, 0)  # End of flipped s3
    assert ordered[2] is s2


def test_kdtree_order_segments(mock_progress_context):
    """
    Test the k-d tree algorithm for initial segment ordering and flipping.
    """
    # A(0,0 -> 10,0), B(100,0 -> 110,0), C(10,10 -> 10,0), D(110,0 -> 110,10)
    # Optimal path should be A, C(flipped), B, D
    sA = _make_seg((0, 0, 0), (10, 0, 0))
    sB = _make_seg((100, 0, 0), (110, 0, 0))
    sC = _make_seg((10, 10, 0), (10, 0, 0))  # Reversed
    sD = _make_seg((110, 0, 0), (110, 10, 0))
    segments = [sA, sB, sC, sD]

    ordered = kdtree_order_segments(mock_progress_context, segments)

    assert len(ordered) == 4
    # Expected order: A, flipped(C), B, D
    # 1. Start with A, ends at (10,0).
    # 2. Closest point is end of C (10,0). C is chosen and flipped.
    #    Path is now at original start of C (10,10).
    # 3. From (10,10), closest is start of B (100,0). B is chosen.
    #    Path is now at end of B (110,0).
    # 4. From (110,0), closest is start of D (110,0). D is chosen.
    assert ordered[0] is sA
    assert ordered[1].endpoint(0) == (10, 0, 0)  # start of flipped sC
    last = ordered[1].len() - 1
    assert ordered[1].endpoint(last) == (10, 10, 0)  # end of flipped sC
    assert ordered[2] is sB
    assert ordered[3] is sD


def test_two_opt(mock_progress_context):
    """Test the 2-opt algorithm for un-crossing paths."""
    # A(0,0->1,0), B(10,10->11,10), C(2,0->1,0), D(11,10->12,10)
    # Order A, B, C, D is crossed. Optimal is A, C, B, D.
    # sC is reversed to make simple greedy fail.
    sA = _make_seg((0, 0, 0), (1, 0, 0))
    sB = _make_seg((10, 10, 0), (11, 10, 0))
    sC = _make_seg((2, 0, 0), (1, 0, 0))
    sD = _make_seg((11, 10, 0), (12, 10, 0))

    ordered = [sA, sB, sC, sD]

    optimized = two_opt(mock_progress_context, ordered, 10)

    # 2-opt should reverse [sB, sC] to [sC, sB] and flip each segment.
    # Expected final sequence: [sA, flipped(sC), flipped(sB), sD]
    assert optimized[0] is sA
    assert optimized[1].endpoint(0) == (1, 0, 0)  # start of flipped sC
    assert optimized[2].endpoint(0) == (11, 10, 0)  # start of flipped sB
    assert optimized[3] is sD


def _calculate_travel_distance(ops: Ops) -> float:
    """Helper to calculate only the travel distance."""
    return ops.distance() - ops.cut_distance()


def test_run_optimization(mock_progress_context):
    """Test the full optimization process on a sample Ops object."""
    # Create an inefficient path
    # It draws two separate squares, but jumps between them for each segment
    ops = Ops()
    ops.set_power(1.0)

    # Square 1 (at 0,0)
    ops.move_to(0, 0)
    ops.line_to(10, 0)  # Seg 1
    # Square 2 (at 100,100)
    ops.move_to(100, 100)
    ops.line_to(110, 100)  # Seg 2
    # Square 1
    ops.move_to(10, 0)
    ops.line_to(10, 10)  # Seg 3
    # Square 2
    ops.move_to(110, 100)
    ops.line_to(110, 110)  # Seg 4

    # Calculate travel distance before optimization
    ops_copy = ops.copy()
    ops_copy.preload_state()
    travel_before = _calculate_travel_distance(ops_copy)

    # Run the optimizer
    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    # Calculate travel distance after optimization
    ops.preload_state()
    travel_after = _calculate_travel_distance(ops)

    # The optimizer should significantly reduce travel distance
    assert travel_before > 250, "Initial travel should be large"
    assert travel_after < travel_before, "Optimized travel should be smaller"
    assert travel_after < 150, "Optimized travel should be just one jump"

    # Check that the number of cutting commands is the same
    cuts_after = sum(1 for i in range(ops.len()) if ops.is_cutting(i))
    assert cuts_after == 4


def test_run_with_air_assist_change(mock_progress_context):
    """
    Verify that segments with different air assist states are not reordered.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Part 1: Air Assist OFF - Inefficient path
    ops.move_to(0, 0)
    ops.line_to(10, 0)  # Seg A1
    ops.move_to(0, 10)
    ops.line_to(10, 10)  # Seg A2

    ops.enable_air_assist(True)

    # Part 2: Air Assist ON - Inefficient path
    ops.move_to(100, 100)
    ops.line_to(110, 100)  # Seg B1
    ops.move_to(100, 110)
    ops.line_to(110, 110)  # Seg B2

    # Run optimizer
    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    ops.preload_state()

    # After optimization, find the first command with air assist ON.
    air_on_idx = -1
    for i in range(ops.len()):
        if ops.category(i) == CommandCategory.MOVING:
            state = ops.preloaded_state(i)
            if state.air_assist:
                air_on_idx = i
                break

    assert air_on_idx != -1, "A segment with air assist ON should exist"

    # All points before this index should be from Part 1
    for i in range(air_on_idx):
        if ops.category(i) == CommandCategory.MOVING:
            assert ops.endpoint(i)[0] < 50, (
                "Points from Part 1 should be in first half"
            )
            state = ops.preloaded_state(i)
            assert not state.air_assist, "State should be air OFF"

    # All points from this index on should be from Part 2
    for i in range(air_on_idx, ops.len()):
        if ops.category(i) == CommandCategory.MOVING:
            assert ops.endpoint(i)[0] > 50, (
                "Points from Part 2 should be second half"
            )
            state = ops.preloaded_state(i)
            assert state.air_assist, "State should be air ON"


def test_run_preserves_markers(mock_progress_context):
    """Verify that marker commands act as optimization boundaries."""
    ops = Ops()
    ops.set_power(1.0)

    # Inefficient path with a marker in the middle
    ops.move_to(0, 0)
    ops.line_to(10, 0)  # Seg 1
    ops.move_to(100, 100)
    ops.line_to(110, 100)  # Seg 2
    ops.job_start()  # Marker
    ops.move_to(10, 0)
    ops.line_to(10, 10)  # Seg 3
    ops.move_to(110, 100)
    ops.line_to(110, 110)  # Seg 4

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    # Find the marker
    marker_idx = -1
    for i in range(ops.len()):
        if ops.command_type(i) == CommandType.JOB_START:
            marker_idx = i
            break

    assert marker_idx != -1, "Marker command should be preserved"

    # Check that segments before the marker were optimized together
    moving_before = [
        ops.endpoint(i)
        for i in range(marker_idx)
        if ops.category(i) == CommandCategory.MOVING
    ]
    assert len(moving_before) == 4
    starts_before = {
        ops.endpoint(i) for i in range(marker_idx) if ops.is_travel(i)
    }
    # After optimization, there will be one travel to the start of the first
    # segment, and one travel between segments. The exact points depend on
    # the optimizer's choice, so we check that the original start points exist.
    assert (0, 0, 0) in starts_before or (100, 100, 0) in starts_before

    # Check that segments after the marker were optimized together
    moving_after = [
        ops.endpoint(i)
        for i in range(marker_idx + 1, ops.len())
        if ops.category(i) == CommandCategory.MOVING
    ]
    assert len(moving_after) == 4
    starts_after = {
        ops.endpoint(i)
        for i in range(marker_idx + 1, ops.len())
        if ops.is_travel(i)
    }
    assert (10, 0, 0) in starts_after or (110, 100, 0) in starts_after


def test_run_optimization_with_unsplit_scanline(mock_progress_context):
    """
    Verify the optimizer can flip a fully "on" ScanLinePowerCommand
    without splitting it.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Path 1: A simple vector line from (0,0) to (10,0)
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)

    # Path 2: A raster line that is fully "on". It should be flipped.
    ops.move_to(20, 0, 0)
    ops.scan_to(10, 0, 0, power_values=bytearray([10, 20, 30]))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()
    travel_after = _calculate_travel_distance(ops)

    # Travel should be zero after flipping.
    assert travel_after == pytest.approx(0.0)

    moving_indices = [
        i
        for i in range(ops.len())
        if ops.category(i) == CommandCategory.MOVING
    ]

    # Original unoptimized: M, L, M, S (4)
    # Optimized: M, L, M, S_flipped (4)
    assert len(moving_indices) == 4

    # Check the final flipped segment
    flipped_move_idx = moving_indices[2]
    flipped_scan_idx = moving_indices[3]
    assert ops.command_type(flipped_move_idx) == CommandType.MOVE_TO
    assert ops.command_type(flipped_scan_idx) == CommandType.SCAN_LINE

    # The new segment should start where the old one ended
    assert ops.endpoint(flipped_move_idx) == pytest.approx((10.0, 0.0, 0.0))
    # The scan command's geometry should reflect the flipped segment
    assert ops.endpoint(flipped_scan_idx) == pytest.approx((20.0, 0.0, 0.0))
    # Power values should be reversed
    assert bytearray(ops.scanline_data(flipped_scan_idx)) == bytearray(
        [30, 20, 10]
    )


def test_run_optimization_with_split_scanline(mock_progress_context):
    """
    Verify the optimizer splits a ScanLine with blank areas and optimizes
    the resulting segment. This version uses geometry that forces a reorder.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Path A: A vector line that ends at x=108.
    ops.move_to(0, 5, 0)
    ops.line_to(108, 5, 0)

    # Path B+C: A raster line from (100, 5) to (110, 5) with a blank middle.
    # This gets split into two segments:
    # Path B: from x=100 to x=102.857
    # Path C: from x=107.142 to x=110
    ops.move_to(100, 5, 0)
    ops.scan_to(110, 5, 0, power_values=bytearray([50, 50, 0, 0, 0, 60, 60]))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    moving_indices = [
        i
        for i in range(ops.len())
        if ops.category(i) == CommandCategory.MOVING
    ]

    # Original: M, L, M, S (4 commands total)
    # The ScanLine is split into two segments: [M, S] and [M, S].
    # The original M is removed, so we add 2*2=4 new commands.
    # Total expanded: M, L, M_B, S_B, M_C, S_C -> 6 commands
    assert len(moving_indices) == 6

    # The greedy k-d tree algorithm will produce the path [A, C, flipped(B)].
    # 1. Start with A, ending at x=108.
    # 2. The closest unvisited point is the start of C (x=107.142).
    # 3. From the end of C (x=110), the closest unvisited point is the
    #    end of B (x=102.857), so B is flipped.

    # Check segment C (part 2), which should be next.
    move_cmd_1_idx = moving_indices[2]
    scan_cmd_1_idx = moving_indices[3]
    assert ops.command_type(move_cmd_1_idx) == CommandType.MOVE_TO
    assert ops.command_type(scan_cmd_1_idx) == CommandType.SCAN_LINE

    # It should connect to the start of C, its original start.
    assert ops.endpoint(move_cmd_1_idx) == pytest.approx(
        (107.142, 5.0, 0.0), abs=1e-3
    )
    # The scan should proceed to the original end of C.
    assert ops.endpoint(scan_cmd_1_idx) == pytest.approx((110.0, 5.0, 0.0))
    assert bytearray(ops.scanline_data(scan_cmd_1_idx)) == bytearray([60, 60])

    # Check segment flipped(B) (part 1), which should be last.
    move_cmd_2_idx = moving_indices[4]
    scan_cmd_2_idx = moving_indices[5]
    assert ops.command_type(move_cmd_2_idx) == CommandType.MOVE_TO
    assert ops.command_type(scan_cmd_2_idx) == CommandType.SCAN_LINE
    # It should connect to the start of flipped(B), which is B's original end.
    assert ops.endpoint(move_cmd_2_idx) == pytest.approx(
        (102.857, 5.0, 0.0), abs=1e-3
    )
    # The scan should proceed to the end of flipped(B), B's original start.
    assert ops.endpoint(scan_cmd_2_idx) == pytest.approx((100.0, 5.0, 0.0))
    assert (
        bytearray(ops.scanline_data(scan_cmd_2_idx))
        == bytearray([50, 50])[::-1]
    )


def test_optimizer_does_not_split_overscanned_scanline(
    mock_progress_context,
):
    """
    Tests that the optimizer does not split a ScanLinePowerCommand that has
    been padded with zero-power values by the OverscanTransformer.

    The optimizer's splitting logic is designed to break up scanlines with
    large empty areas to improve travel paths. However, an overscanned line
    intentionally has zero-power lead-in/outs. The optimizer must treat
    this entire overscanned line as a single, unbreakable segment.
    """
    # Arrange: Create an Ops object that simulates the output of an
    # OverscanTransformer. This is a single scanline with zero-power padding.
    ops = Ops()
    ops.set_power(1.0)

    # This represents a 10mm content line (15-5) with 5mm overscan on each side
    start_pt = (0.0, 10.0, 0.0)
    end_pt = (20.0, 10.0, 0.0)
    # Padded power values: 2 bytes for lead-in, 3 for content, 2 for lead-out
    power_values = bytearray([0, 0] + [50, 100, 150] + [0, 0])

    ops.move_to(*start_pt)
    ops.scan_to(*end_pt, power_values=power_values)

    # Act: Run the optimizer
    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    # Assert: The optimizer should NOT have split the scanline.
    scan_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SCAN_LINE
    ]
    move_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.MOVE_TO
    ]

    # 1. There should still be exactly one ScanLinePowerCommand
    assert len(scan_indices) == 1
    final_scan_idx = scan_indices[0]

    # 2. The move command preceding it should still start at the overscan point
    assert len(move_indices) == 1
    assert ops.endpoint(move_indices[0]) == pytest.approx(start_pt)

    # 3. The scanline's geometry should be unchanged. If it were split, the
    #    endpoint would be shortened to the end of the content area.
    assert ops.endpoint(final_scan_idx) == pytest.approx(end_pt)

    # 4. The power values should still contain the zero-power padding.
    assert bytearray(ops.scanline_data(final_scan_idx)) == power_values


def test_run_optimization_scanline_flip_preserves_state(
    mock_progress_context,
):
    """
    Verify that when a ScanLine segment is flipped, the new commands
    (MoveTo, ScanLinePowerCommand) correctly inherit the state.
    """
    ops = Ops()
    ops.set_power(0.85)
    ops.set_cut_speed(1234)
    ops.enable_air_assist(True)

    # Path 1: A vector line from (0,0) to (10,0)
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    # Path 2: A raster line that should be flipped to minimize travel
    ops.move_to(20, 0)
    ops.scan_to(10, 0, power_values=bytearray([10, 20, 30]))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    scan_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SCAN_LINE
    ]
    assert len(scan_indices) == 1
    scan_idx = scan_indices[0]

    move_idx = scan_idx - 1
    assert ops.command_type(move_idx) == CommandType.MOVE_TO

    # Check state on the new MoveTo for the flipped segment
    move_state = ops.preloaded_state(move_idx)
    assert move_state.power == pytest.approx(0.85)
    assert move_state.cut_speed == pytest.approx(1234)
    assert move_state.air_assist is True

    # Check state on the flipped ScanLinePowerCommand
    scan_state = ops.preloaded_state(scan_idx)
    assert scan_state.power == pytest.approx(0.85)
    assert scan_state.cut_speed == pytest.approx(1234)
    assert scan_state.air_assist is True


def test_run_optimization_scanline_split_preserves_state(
    mock_progress_context,
):
    """
    Verify that when a ScanLine is split, all new sub-segments correctly
    inherit the original state.
    """
    ops = Ops()
    ops.set_power(0.77)
    ops.set_travel_speed(5678)
    ops.enable_air_assist(False)

    # A raster line that will be split into two segments
    ops.move_to(0, 0)
    ops.scan_to(10, 0, power_values=bytearray([50, 50, 0, 0, 60, 60]))
    # A far away vector line to ensure no reordering happens
    ops.move_to(100, 100)
    ops.line_to(101, 101)

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    # The original ScanLine should be replaced by two new ones
    scan_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SCAN_LINE
    ]
    assert len(scan_indices) == 2

    for scan_idx in scan_indices:
        move_idx = scan_idx - 1
        assert ops.command_type(move_idx) == CommandType.MOVE_TO

        # Verify state on the new MoveTo for the sub-segment
        move_state = ops.preloaded_state(move_idx)
        assert move_state.power == pytest.approx(0.77)
        assert move_state.travel_speed == pytest.approx(5678)
        assert move_state.air_assist is False

        # Verify state on the new ScanLinePowerCommand for the sub-segment
        scan_state = ops.preloaded_state(scan_idx)
        assert scan_state.power == pytest.approx(0.77)
        assert scan_state.travel_speed == pytest.approx(5678)
        assert scan_state.air_assist is False


def test_run_with_state_change_and_scanlines(mock_progress_context):
    """
    Verify that ScanLine segments with different states are not reordered
    across state boundaries, and that each optimized block has the correct
    state.
    """
    ops = Ops()

    # Part 1: Power 0.4 - Inefficient path with scanlines
    ops.set_power(0.4)
    ops.move_to(0, 0)
    ops.scan_to(10, 0, power_values=bytearray([10]))
    ops.move_to(0, 10)
    ops.scan_to(10, 10, power_values=bytearray([20]))

    # State change acts as an optimization boundary
    ops.set_power(0.9)

    # Part 2: Power 0.9 - Inefficient path with scanlines
    ops.move_to(100, 100)
    ops.scan_to(110, 100, power_values=bytearray([30]))
    ops.move_to(100, 110)
    ops.scan_to(110, 110, power_values=bytearray([40]))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    # Find the index where the state changes
    power_change_idx = -1
    for i in range(ops.len()):
        if ops.category(i) == CommandCategory.MOVING:
            state = ops.preloaded_state(i)
            if state.power == pytest.approx(0.9):
                power_change_idx = i
                break

    assert power_change_idx != -1, "A segment with power 0.9 should exist"

    # Check all moving commands before the state change
    for i in range(power_change_idx):
        if ops.category(i) == CommandCategory.MOVING:
            state = ops.preloaded_state(i)
            assert ops.endpoint(i)[0] < 50, (
                "Points from Part 1 should be in first half"
            )
            assert state.power == pytest.approx(0.4), (
                "State should be power 0.4"
            )

    # Check all moving commands at and after the state change
    for i in range(power_change_idx, ops.len()):
        if ops.category(i) == CommandCategory.MOVING:
            state = ops.preloaded_state(i)
            assert ops.endpoint(i)[0] > 50, (
                "Points from Part 2 should be in second half"
            )
            assert state.power == pytest.approx(0.9), (
                "State should be power 0.9"
            )

    # Also check that optimization occurred within the first block
    # Initial travel: move(0,0)->scan(10,0) -> move(0,10)->scan(10,10)
    # Travel is from (10,0) to (0,10) = sqrt(10^2 + 10^2) ~= 14.14
    # Optimized travel: move(0,0)->scan(10,0) -> move(10,10)->scan(0,10)
    # Travel is from (10,0) to (10,10) = 10.
    # We can verify this by checking the order of the Y coordinates.
    y_coords = [
        ops.endpoint(i)[1]
        for i in range(power_change_idx)
        if ops.command_type(i) == CommandType.SCAN_LINE
    ]
    assert y_coords == [0, 10] or y_coords == [10, 0], (
        "Optimization should order by y-coord"
    )


def test_run_optimization_with_overscan_and_flip_preserves_state(
    mock_progress_context,
):
    """
    Tests that an overscanned ScanLine that gets flipped by the optimizer
    correctly preserves its state. This simulates the real-world scenario
    where the error was observed.
    """
    ops = Ops()
    ops.set_power(0.66)
    ops.set_cut_speed(2000)

    # Path 1: A vector line from (0,0) to (10,10)
    ops.move_to(0, 0)
    ops.line_to(10, 10)

    # Path 2: An overscanned raster line that should be flipped.
    # The content is from (30,10) to (20,10), but overscan extends it.
    start_pt_overscan = (35.0, 10.0, 0.0)
    end_pt_overscan = (15.0, 10.0, 0.0)
    power_values = bytearray([0, 0] + [100, 120, 140] + [0, 0])
    ops.move_to(*start_pt_overscan)
    ops.scan_to(*end_pt_overscan, power_values=power_values)

    # The optimizer should connect Path 1's end (10,10) to the nearest
    # point on Path 2, which is its end (15,10), causing a flip.

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    # Find the scan command after optimization
    scan_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.SCAN_LINE
    ]
    assert len(scan_indices) == 1
    flipped_scan_idx = scan_indices[0]

    # Find its preceding MoveTo command
    move_idx = flipped_scan_idx - 1
    assert ops.command_type(move_idx) == CommandType.MOVE_TO

    # Check state on the new MoveTo for the flipped segment
    move_state = ops.preloaded_state(move_idx)
    assert move_state.power == pytest.approx(0.66)
    assert move_state.cut_speed == pytest.approx(2000)

    # Check state on the flipped ScanLinePowerCommand
    scan_state = ops.preloaded_state(flipped_scan_idx)
    assert scan_state.power == pytest.approx(0.66)
    assert scan_state.cut_speed == pytest.approx(2000)

    # Verify the geometry and power values were flipped correctly
    assert ops.endpoint(move_idx) == pytest.approx(end_pt_overscan)
    assert ops.endpoint(flipped_scan_idx) == pytest.approx(start_pt_overscan)
    assert bytearray(ops.scanline_data(flipped_scan_idx)) == power_values[::-1]


def test_workpiece_level_optimization(mock_progress_context):
    """
    Test that workpiece-level optimization reorders workpieces to minimize
    travel when run at per-step level (workpiece=None).
    """
    ops = Ops()
    ops.set_power(1.0)

    # Workpiece A at (0,0)
    ops.workpiece_start("wp-a")
    ops.move_to(0, 0)
    ops.line_to(10, 0)
    ops.workpiece_end("wp-a")

    # Workpiece C at (200, 200) - far away
    ops.workpiece_start("wp-c")
    ops.move_to(200, 200)
    ops.line_to(210, 200)
    ops.workpiece_end("wp-c")

    # Workpiece B at (10, 0) - close to A
    ops.workpiece_start("wp-b")
    ops.move_to(10, 0)
    ops.line_to(10, 10)
    ops.workpiece_end("wp-b")

    # Calculate travel before optimization
    ops_copy = ops.copy()
    ops_copy.preload_state()
    travel_before = ops_copy.distance() - ops_copy.cut_distance()

    # Run optimizer at per-step level (workpiece=None)
    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    ops.preload_state()
    travel_after = ops.distance() - ops.cut_distance()

    # Travel should be reduced
    assert travel_after < travel_before, (
        f"Travel should be reduced: {travel_before} -> {travel_after}"
    )

    # Extract workpiece order after optimization
    wp_order = []
    for i in range(ops.len()):
        if ops.command_type(i) == CommandType.WORKPIECE_START:
            wp_order.append(ops.workpiece_uid(i))

    # Should be reordered: A, B, C (not A, C, B)
    assert wp_order == ["wp-a", "wp-b", "wp-c"], (
        f"Workpieces should be reordered to A, B, C, got {wp_order}"
    )


def test_bezier_passes_through_optimizer(mock_progress_context):
    """
    Verify that the optimizer correctly handles segments containing
    BezierToCommand and does not corrupt them.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Path 1: vector line
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    # Path 2: bezier curve
    ops.move_to(100, 0)
    ops.bezier_to((110, 10, 0), (120, 10, 0), (130, 0, 0))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    bezier_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.BEZIER_TO
    ]
    assert len(bezier_indices) == 1
    c1, c2 = ops.bezier_params(bezier_indices[0])
    assert c1 == (110, 10, 0)
    assert c2 == (120, 10, 0)
    assert ops.endpoint(bezier_indices[0]) == (130, 0, 0)

    line_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.LINE_TO
    ]
    assert len(line_indices) == 1

    moving_count = sum(
        1
        for i in range(ops.len())
        if ops.category(i) == CommandCategory.MOVING
    )
    assert moving_count == 4


def test_bezier_segment_flip(mock_progress_context):
    """
    Verify that when the optimizer flips a segment containing a
    BezierToCommand, the control points are correctly swapped.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Path 1: ends at (10, 0)
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    # Path 2: bezier that ends near (10, 0) — should be flipped
    # to connect (10, 0) → (30, 0) via the bezier
    ops.move_to(30, 0)
    ops.bezier_to((25, 5, 0), (15, 5, 0), (10, 0, 0))

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)
    ops.preload_state()

    bezier_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.BEZIER_TO
    ]
    assert len(bezier_indices) == 1
    idx = bezier_indices[0]

    # The bezier should be flipped:
    # - end changes from original (10, 0) to (30, 0)
    # - control1/control2 are swapped
    assert ops.endpoint(idx) == pytest.approx((30, 0, 0))
    c1, c2 = ops.bezier_params(idx)
    assert c1 == pytest.approx((15, 5, 0))
    assert c2 == pytest.approx((25, 5, 0))


def test_mixed_lines_and_bezier(mock_progress_context):
    """
    Verify the optimizer handles a mix of line and bezier segments,
    correctly reordering them by proximity.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Line segment at origin
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    # Bezier segment far away
    ops.move_to(100, 100)
    ops.bezier_to((105, 105, 0), (115, 105, 0), (120, 100, 0))

    # Line segment close to the first one
    ops.move_to(10, 0)
    ops.line_to(10, 10)

    optimizer = Optimize()
    optimizer.run(ops, context=mock_progress_context)

    ops.preload_state()
    travel_after = ops.distance() - ops.cut_distance()

    ops_unoptimized = Ops()
    ops_unoptimized.set_power(1.0)
    ops_unoptimized.move_to(0, 0)
    ops_unoptimized.line_to(10, 0)
    ops_unoptimized.move_to(100, 100)
    ops_unoptimized.bezier_to((105, 105, 0), (115, 105, 0), (120, 100, 0))
    ops_unoptimized.move_to(10, 0)
    ops_unoptimized.line_to(10, 10)
    ops_unoptimized.preload_state()
    travel_before = ops_unoptimized.distance() - ops_unoptimized.cut_distance()

    assert travel_after < travel_before

    bezier_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.BEZIER_TO
    ]
    assert len(bezier_indices) == 1
    line_indices = [
        i
        for i in range(ops.len())
        if ops.command_type(i) == CommandType.LINE_TO
    ]
    assert len(line_indices) == 2
