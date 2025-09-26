import pytest
from rayforge.core.ops import (
    Ops,
    MoveToCommand,
    LineToCommand,
    JobStartCommand,
    MovingCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.transformer.optimize import (
    Optimize,
    greedy_order_segments,
    flip_segments,
    two_opt,
    _dist_2d,
    farthest_insertion_order_segments,
)
from rayforge.shared.tasker.context import (
    BaseExecutionContext,
    ExecutionContext,
)


@pytest.fixture
def ctx() -> BaseExecutionContext:
    """Provides a dummy execution context for functions that require it."""
    return ExecutionContext()


def test_greedy_order_segments(ctx):
    """Test the greedy algorithm for initial segment ordering."""
    # Seg1: (0,0) -> (10,0) - should be chosen first
    # Seg2: (100,100) -> (110,100)
    # Seg3: (10,0) -> (10,10) - should be chosen second
    s1 = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
    s2 = [MoveToCommand((100, 100, 0)), LineToCommand((110, 100, 0))]
    s3 = [MoveToCommand((10, 0, 0)), LineToCommand((10, 10, 0))]
    segments = [s1, s2, s3]

    ordered = greedy_order_segments(ctx, segments)
    assert len(ordered) == 3
    # Expected order: s1, s3, s2
    assert ordered[0] is s1
    assert ordered[1] is s3
    assert ordered[2] is s2


def test_greedy_order_with_flip(ctx):
    """Test greedy ordering when flipping a segment is optimal."""
    # Seg1: (0,0) -> (10,0)
    # Seg2: (100,100) -> (110,100)
    # Seg3: (10,10) -> (10,0) <-- start is far, end is near
    s1 = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
    s2 = [MoveToCommand((100, 100, 0)), LineToCommand((110, 100, 0))]
    s3 = [MoveToCommand((10, 10, 0)), LineToCommand((10, 0, 0))]
    segments = [s1, s2, s3]

    ordered = greedy_order_segments(ctx, segments)

    # Expected: s1, flipped(s3), s2
    assert ordered[0] is s1
    assert ordered[1] is not s3  # Should be a new, flipped list
    assert ordered[1][0].end == (10, 0, 0)  # Start of flipped s3
    assert ordered[1][-1].end == (10, 10, 0)  # End of flipped s3
    assert ordered[2] is s2


def test_flip_segments_local_optimization(ctx):
    """Test the iterative flipping of segments to reduce travel."""
    # Seg1: (0,0) -> (10,0)
    # Seg2: (20,10) -> (10,10) <-- Flipped is better
    # Seg3: (20,0) -> (30,0)
    s1 = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
    s2 = [MoveToCommand((20, 10, 0)), LineToCommand((10, 10, 0))]
    s3 = [MoveToCommand((20, 0, 0)), LineToCommand((30, 0, 0))]

    ordered_segments = [s1, s2, s3]

    # Travel cost before: dist((10,0), (20,10)) + dist((10,10), (20,0))
    # ~ 14.14 + 14.14 = 28.28
    travel_before = _dist_2d((10, 0), (20, 10)) + _dist_2d((10, 10), (20, 0))

    improved = flip_segments(ctx, ordered_segments)

    # Flipped s2 starts at (10,10) and ends at (20,10)
    # Travel cost after: dist((10,0),(10,10)) + dist((20,10),(20,0)) = 20
    p1_end = improved[0][-1].end
    p2_start, p2_end = improved[1][0].end, improved[1][-1].end
    p3_start = improved[2][0].end
    travel_after = _dist_2d(p1_end, p2_start) + _dist_2d(p2_end, p3_start)

    assert travel_after < travel_before
    assert improved[1][0].end == (10, 10, 0)  # Check if s2 was flipped


def test_two_opt(ctx):
    """Test the 2-opt algorithm for un-crossing paths."""
    # A(0,0->1,0), B(10,10->11,10), C(2,0->1,0), D(11,10->12,10)
    # Order A, B, C, D is crossed. Optimal is A, C, B, D.
    # sC is reversed to make simple greedy fail.
    sA = [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]
    sB = [MoveToCommand((10, 10, 0)), LineToCommand((11, 10, 0))]
    sC = [MoveToCommand((2, 0, 0)), LineToCommand((1, 0, 0))]
    sD = [MoveToCommand((11, 10, 0)), LineToCommand((12, 10, 0))]

    ordered = [sA, sB, sC, sD]

    optimized = two_opt(ctx, ordered, 10)

    # 2-opt should reverse [sB, sC] to [sC, sB] and flip each segment.
    # Expected final sequence: [sA, flipped(sC), flipped(sB), sD]
    assert optimized[0] is sA
    assert optimized[1][0].end == (1, 0, 0)  # start of flipped sC
    assert optimized[2][0].end == (11, 10, 0)  # start of flipped sB
    assert optimized[3] is sD


def test_farthest_insertion_order_segments(ctx: BaseExecutionContext):
    """
    Test the Farthest Insertion heuristic for tour construction and flipping.
    """
    # Setup: Define four short, disconnected segments at the corners of a
    # square.
    # Segment C is defined backwards to test the final flipping pass.
    seg_a = [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]
    seg_b = [MoveToCommand((100, 0, 0)), LineToCommand((101, 0, 0))]
    seg_c = [MoveToCommand((101, 100, 0)), LineToCommand((100, 100, 0))]
    seg_d = [MoveToCommand((0, 100, 0)), LineToCommand((1, 100, 0))]

    # Jumble the order to ensure the algorithm sorts them correctly.
    segments = [seg_d, seg_a, seg_c, seg_b]

    ordered_segments = farthest_insertion_order_segments(ctx, segments)
    assert len(ordered_segments) == 4

    # Test 1: Verify that a valid tour was created.
    # The final set of segments should be the original four, just reordered
    # and possibly replaced with their flipped versions.
    original_endpoints = {
        tuple(s[0].end) for s in [seg_a, seg_b, seg_c, seg_d]
    } | {tuple(s[-1].end) for s in [seg_a, seg_b, seg_c, seg_d]}
    final_startpoints = {tuple(s[0].end) for s in ordered_segments}

    assert len(final_startpoints) == 4
    assert final_startpoints.issubset(original_endpoints)

    # Test 2: Verify the reversed segment was flipped correctly.
    # The optimal tour is A -> B -> flipped(C) -> D.
    # Find the segment that corresponds to the original seg_c.
    seg_c_output = None
    for seg in ordered_segments:
        # A segment is identified by its set of endpoints, regardless of flip.
        endpoints = {tuple(seg[0].end), tuple(seg[-1].end)}
        if endpoints == {(101, 100, 0), (100, 100, 0)}:
            seg_c_output = seg
            break
    assert seg_c_output is not None, "Segment C not found in output"

    # In the optimal tour, seg_c will follow seg_b (which ends near (101,0)).
    # The closest point on seg_c is (100,100), not (101,100).
    # Therefore, the final pass MUST have flipped seg_c.
    assert seg_c_output[0].end == pytest.approx((100, 100, 0))


def _calculate_travel_distance(ops: Ops) -> float:
    """Helper to calculate only the travel distance."""
    return ops.distance() - ops.cut_distance()


def test_run_optimization():
    """Test the full optimization process on a sample Ops object."""
    # Create an inefficient path
    # It draws two separate squares, but jumps between them for each segment
    ops = Ops()
    ops.set_power(100)

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
    optimizer.run(ops)

    # Calculate travel distance after optimization
    ops.preload_state()
    travel_after = _calculate_travel_distance(ops)

    # The optimizer should significantly reduce travel distance
    assert travel_before > 250, "Initial travel should be large"
    assert travel_after < travel_before, "Optimized travel should be smaller"
    assert travel_after < 150, "Optimized travel should be just one jump"

    # Check that the number of cutting commands is the same
    cuts_after = sum(1 for c in ops.commands if c.is_cutting_command())
    assert cuts_after == 4


def test_run_with_air_assist_change():
    """
    Verify that segments with different air assist states are not reordered.
    """
    ops = Ops()
    ops.set_power(100)

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
    optimizer.run(ops)

    ops.preload_state()

    # After optimization, find the first command with air assist ON.
    air_on_idx = -1
    for i, cmd in enumerate(ops.commands):
        if cmd.state and cmd.state.air_assist:
            air_on_idx = i
            break

    assert air_on_idx != -1, "A segment with air assist ON should exist"

    # All points before this index should be from Part 1
    for i in range(air_on_idx):
        cmd = ops.commands[i]
        if cmd.end and cmd.state:  # Check only moving commands
            assert cmd.end[0] < 50, (
                "Points from Part 1 should be in first half"
            )
            assert not cmd.state.air_assist, "State should be air OFF"

    # All points from this index on should be from Part 2
    for i in range(air_on_idx, len(ops.commands)):
        cmd = ops.commands[i]
        if cmd.end and cmd.state:  # Check only moving commands
            assert cmd.end[0] > 50, "Points from Part 2 should be second half"
            assert cmd.state.air_assist, "State should be air ON"


def test_run_preserves_markers():
    """Verify that marker commands act as optimization boundaries."""
    ops = Ops()
    ops.set_power(100)

    # Inefficient path with a marker in the middle
    ops.move_to(0, 0)
    ops.line_to(10, 0)  # Seg 1
    ops.move_to(100, 100)
    ops.line_to(110, 100)  # Seg 2
    ops.add(JobStartCommand())  # Marker
    ops.move_to(10, 0)
    ops.line_to(10, 10)  # Seg 3
    ops.move_to(110, 100)
    ops.line_to(110, 110)  # Seg 4

    optimizer = Optimize()
    optimizer.run(ops)

    # Find the marker
    marker_idx = -1
    for i, cmd in enumerate(ops.commands):
        if isinstance(cmd, JobStartCommand):
            marker_idx = i
            break

    assert marker_idx != -1, "Marker command should be preserved"

    # Check that segments before the marker were optimized together
    moving_cmds_before = [
        c for c in ops.commands[:marker_idx] if isinstance(c, MovingCommand)
    ]
    assert len(moving_cmds_before) == 4
    starts_before = {
        c.end for c in moving_cmds_before if c.is_travel_command()
    }
    assert (0, 0, 0) in starts_before
    assert (100, 100, 0) in starts_before

    # Check that segments after the marker were optimized together
    moving_cmds_after = [
        c
        for c in ops.commands[marker_idx + 1 :]
        if isinstance(c, MovingCommand)
    ]
    assert len(moving_cmds_after) == 4
    starts_after = {c.end for c in moving_cmds_after if c.is_travel_command()}
    assert (10, 0, 0) in starts_after
    assert (110, 100, 0) in starts_after


def test_run_optimization_with_unsplit_scanline():
    """
    Verify the optimizer can flip a fully "on" ScanLinePowerCommand
    without splitting it.
    """
    ops = Ops()
    ops.set_power(100)

    # Path 1: A simple vector line from (0,0) to (10,0)
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)

    # Path 2: A raster line that is fully "on". It should be flipped.
    ops.move_to(20, 0, 0)
    ops.add(
        ScanLinePowerCommand(
            end=(10, 0, 0),
            power_values=bytearray([10, 20, 30]),
        )
    )

    optimizer = Optimize()
    optimizer.run(ops)
    ops.preload_state()
    travel_after = _calculate_travel_distance(ops)

    # Travel should be zero after flipping.
    assert travel_after == pytest.approx(0.0)

    moving_cmds = [c for c in ops.commands if isinstance(c, MovingCommand)]

    # Original: M, L, M, S (4)
    # Expanded: M, L, M(new), S(new) (4). The original M before S is removed.
    assert len(moving_cmds) == 4

    # Check the final flipped segment
    flipped_move_cmd = moving_cmds[2]
    flipped_scan_cmd = moving_cmds[3]
    assert isinstance(flipped_move_cmd, MoveToCommand)
    assert isinstance(flipped_scan_cmd, ScanLinePowerCommand)

    # The new segment should start where the old one ended
    assert flipped_move_cmd.end == pytest.approx((10.0, 0.0, 0.0))
    # The scan command's geometry should reflect the flipped segment
    assert flipped_scan_cmd.end == pytest.approx((20.0, 0.0, 0.0))
    # Power values should be reversed
    assert flipped_scan_cmd.power_values == bytearray([30, 20, 10])


def test_run_optimization_with_split_scanline():
    """
    Verify the optimizer splits a ScanLine with blank areas and optimizes
    the resulting segment. This version uses geometry that forces a reorder.
    """
    ops = Ops()
    ops.set_power(100)

    # Path 1: A vector line that ends close to the SECOND part of the raster.
    ops.move_to(0, 5, 0)
    ops.line_to(108, 5, 0)

    # Path 2: A raster line from (100, 5) to (110, 5) with a blank middle.
    ops.move_to(100, 5, 0)
    ops.add(
        ScanLinePowerCommand(
            end=(110, 5, 0),
            power_values=bytearray([50, 50, 0, 0, 0, 60, 60]),
        )
    )

    optimizer = Optimize()
    optimizer.run(ops)
    ops.preload_state()

    moving_cmds = [c for c in ops.commands if isinstance(c, MovingCommand)]

    # Original: M, L, M, S (4 commands total)
    # The ScanLine is split into two segments: [M, S] and [M, S].
    # The original M is removed, so we add 2*2=4 new commands.
    # Total expanded: M, L, M_part1, S_part1, M_part2, S_part2 -> 6 commands
    assert len(moving_cmds) == 6

    # After optimization, the order should be:
    # 1. Original M, L (cmds 0, 1) ending at x=108
    # 2. The *second part* of the raster line, which starts near x=107.142
    # 3. The *first part* of the raster line, which starts at x=100

    # The truly optimal path found by the algorithm is actually
    # [A, flipped(C), flipped(B)], where C is part 2 and B is part 1.

    # Check segment that should be first after the vector line (flipped part 2)
    move_cmd_1 = moving_cmds[2]
    scan_cmd_1 = moving_cmds[3]
    assert isinstance(move_cmd_1, MoveToCommand)
    assert isinstance(scan_cmd_1, ScanLinePowerCommand)

    # Start of flipped C is its original end
    assert move_cmd_1.end == pytest.approx((110.0, 5.0, 0.0))
    # End of flipped C is its original start
    assert scan_cmd_1.end == pytest.approx((107.142, 5.0, 0.0), abs=1e-3)
    assert scan_cmd_1.power_values == bytearray([60, 60])[::-1]

    # Check segment that should be last (flipped part 1)
    move_cmd_2 = moving_cmds[4]
    scan_cmd_2 = moving_cmds[5]
    assert isinstance(move_cmd_2, MoveToCommand)
    assert isinstance(scan_cmd_2, ScanLinePowerCommand)
    # Start of flipped B is its original end
    assert move_cmd_2.end == pytest.approx((102.857, 5.0, 0.0), abs=1e-3)
    # End of flipped B is its original start
    assert scan_cmd_2.end == pytest.approx((100.0, 5.0, 0.0))
    assert scan_cmd_2.power_values == bytearray([50, 50])[::-1]
