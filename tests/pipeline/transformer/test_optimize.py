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
    two_opt,
    kdtree_order_segments,
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


def test_kdtree_order_segments(ctx):
    """
    Test the k-d tree algorithm for initial segment ordering and flipping.
    """
    # A(0,0 -> 10,0), B(100,0 -> 110,0), C(10,10 -> 10,0), D(110,0 -> 110,10)
    # Optimal path should be A, C(flipped), B, D
    sA = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
    sB = [MoveToCommand((100, 0, 0)), LineToCommand((110, 0, 0))]
    sC = [MoveToCommand((10, 10, 0)), LineToCommand((10, 0, 0))]  # Reversed
    sD = [MoveToCommand((110, 0, 0)), LineToCommand((110, 10, 0))]
    segments = [sA, sB, sC, sD]

    ordered = kdtree_order_segments(ctx, segments)

    assert len(ordered) == 4
    # Expected order: A, flipped(C), B, D
    # 1. Start with A, ends at (10,0).
    # 2. Closest point is end of C (10,0). C is chosen and flipped.
    #    Path is now at original start of C (10,10).
    # 3. From (10,10), closest is start of B (100,0). B is chosen.
    #    Path is now at end of B (110,0).
    # 4. From (110,0), closest is start of D (110,0). D is chosen.
    assert ordered[0] is sA
    assert ordered[1][0].end == (10, 0, 0)  # start of flipped sC
    assert ordered[1][-1].end == (10, 10, 0)  # end of flipped sC
    assert ordered[2] is sB
    assert ordered[3] is sD


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
    # After optimization, there will be one travel to the start of the first
    # segment, and one travel between segments. The exact points depend on
    # the optimizer's choice, so we check that the original start points exist.
    assert (0, 0, 0) in starts_before or (100, 100, 0) in starts_before

    # Check that segments after the marker were optimized together
    moving_cmds_after = [
        c
        for c in ops.commands[marker_idx + 1 :]
        if isinstance(c, MovingCommand)
    ]
    assert len(moving_cmds_after) == 4
    starts_after = {c.end for c in moving_cmds_after if c.is_travel_command()}
    assert (10, 0, 0) in starts_after or (110, 100, 0) in starts_after


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

    # Original unoptimized: M, L, M, S (4)
    # Optimized: M, L, M, S_flipped (4)
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

    # Path A: A vector line that ends at x=108.
    ops.move_to(0, 5, 0)
    ops.line_to(108, 5, 0)

    # Path B+C: A raster line from (100, 5) to (110, 5) with a blank middle.
    # This gets split into two segments:
    # Path B: from x=100 to x=102.857
    # Path C: from x=107.142 to x=110
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
    # Total expanded: M, L, M_B, S_B, M_C, S_C -> 6 commands
    assert len(moving_cmds) == 6

    # The greedy k-d tree algorithm will produce the path [A, C, flipped(B)].
    # 1. Start with A, ending at x=108.
    # 2. The closest unvisited point is the start of C (x=107.142).
    # 3. From the end of C (x=110), the closest unvisited point is the
    #    end of B (x=102.857), so B is flipped.

    # Check segment C (part 2), which should be next.
    move_cmd_1 = moving_cmds[2]
    scan_cmd_1 = moving_cmds[3]
    assert isinstance(move_cmd_1, MoveToCommand)
    assert isinstance(scan_cmd_1, ScanLinePowerCommand)

    # It should connect to the start of C, its original start.
    assert move_cmd_1.end == pytest.approx((107.142, 5.0, 0.0), abs=1e-3)
    # The scan should proceed to the original end of C.
    assert scan_cmd_1.end == pytest.approx((110.0, 5.0, 0.0))
    assert scan_cmd_1.power_values == bytearray([60, 60])

    # Check segment flipped(B) (part 1), which should be last.
    move_cmd_2 = moving_cmds[4]
    scan_cmd_2 = moving_cmds[5]
    assert isinstance(move_cmd_2, MoveToCommand)
    assert isinstance(scan_cmd_2, ScanLinePowerCommand)
    # It should connect to the start of flipped(B), which is B's original end.
    assert move_cmd_2.end == pytest.approx((102.857, 5.0, 0.0), abs=1e-3)
    # The scan should proceed to the end of flipped(B), B's original start.
    assert scan_cmd_2.end == pytest.approx((100.0, 5.0, 0.0))
    assert scan_cmd_2.power_values == bytearray([50, 50])[::-1]


def test_optimizer_does_not_split_overscanned_scanline():
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
    ops.set_power(100)

    # This represents a 10mm content line (15-5) with 5mm overscan on each side
    start_pt = (0.0, 10.0, 0.0)
    end_pt = (20.0, 10.0, 0.0)
    # Padded power values: 2 bytes for lead-in, 3 for content, 2 for lead-out
    power_values = bytearray([0, 0] + [50, 100, 150] + [0, 0])

    ops.move_to(*start_pt)
    ops.add(ScanLinePowerCommand(end=end_pt, power_values=power_values))

    # Act: Run the optimizer
    optimizer = Optimize()
    optimizer.run(ops)

    # Assert: The optimizer should NOT have split the scanline.
    scan_cmds = [
        c for c in ops.commands if isinstance(c, ScanLinePowerCommand)
    ]
    move_cmds = [c for c in ops.commands if isinstance(c, MoveToCommand)]

    # 1. There should still be exactly one ScanLinePowerCommand
    assert len(scan_cmds) == 1
    final_scan_cmd = scan_cmds[0]

    # 2. The move command preceding it should still start at the overscan point
    assert len(move_cmds) == 1
    assert move_cmds[0].end == pytest.approx(start_pt)

    # 3. The scanline's geometry should be unchanged. If it were split, the
    #    endpoint would be shortened to the end of the content area.
    assert final_scan_cmd.end == pytest.approx(end_pt)

    # 4. The power values should still contain the zero-power padding.
    assert final_scan_cmd.power_values == power_values
