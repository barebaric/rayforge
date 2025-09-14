import unittest
from typing import List, cast
from rayforge.core.ops import (
    Ops,
    State,
    Command,
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    JobStartCommand,
    MovingCommand,
)
from rayforge.pipeline.transformer.optimize import (
    Optimize,
    split_long_segments,
    split_segments,
    flip_segment,
    greedy_order_segments,
    flip_segments,
    two_opt,
    _dist_2d,
)
from rayforge.shared.tasker.context import (
    BaseExecutionContext,
    ExecutionContext,
)


class TestOptimizerHelpers(unittest.TestCase):
    """Tests for the helper functions used by the Optimize transformer."""

    def setUp(self):
        """Set up a dummy execution context for functions that require it."""
        self.ctx: BaseExecutionContext = ExecutionContext()

    def _create_commands_with_states(
        self, states_config: List[bool]
    ) -> List[Command]:
        """Helper to create commands with specified air_assist states."""
        commands = []
        for i, air_on in enumerate(states_config):
            state = State(power=100, air_assist=air_on)
            # We need MovingCommand for split_segments to work
            cmd = LineToCommand((float(i), float(i), 0.0))
            cmd.state = state
            commands.append(cmd)
        return commands

    def test_split_long_segments(self):
        """Test splitting commands by non-reorderable state changes."""
        # All same state -> 1 segment
        cmds1 = self._create_commands_with_states([True, True, True])
        self.assertEqual(len(split_long_segments(cmds1)), 1)
        self.assertEqual(len(split_long_segments(cmds1)[0]), 3)

        # State change -> 2 segments
        cmds2 = self._create_commands_with_states([True, True, False])
        self.assertEqual(len(split_long_segments(cmds2)), 2)
        self.assertEqual(len(split_long_segments(cmds2)[0]), 2)
        self.assertEqual(len(split_long_segments(cmds2)[1]), 1)

        # Multiple state changes
        cmds3 = self._create_commands_with_states(
            [False, True, True, False, False, True]
        )
        segments = split_long_segments(cmds3)
        self.assertEqual(len(segments), 4)
        self.assertEqual([len(s) for s in segments], [1, 2, 2, 1])

        # Empty and single command lists
        self.assertEqual(split_long_segments([]), [], "Handles empty list")
        cmds4 = self._create_commands_with_states([True])
        self.assertEqual(
            len(split_long_segments(cmds4)), 1, "Handles single command"
        )

        # Test with marker commands
        cmds_with_marker = self._create_commands_with_states([True, True])
        cmds_with_marker.insert(1, JobStartCommand())  # [cmd_T, marker, cmd_T]
        segments_with_marker = split_long_segments(cmds_with_marker)
        self.assertEqual(len(segments_with_marker), 3)
        self.assertEqual([len(s) for s in segments_with_marker], [1, 1, 1])
        self.assertIsInstance(segments_with_marker[1][0], JobStartCommand)

    def test_split_segments(self):
        """Test splitting a list of commands into re-orderable paths."""
        # Type as List[Command] to satisfy the invariant type checker,
        # since MovingCommand is a subtype of Command.
        cmds: List[Command] = [
            MoveToCommand((0, 0, 0)),
            LineToCommand((10, 0, 0)),
            LineToCommand((10, 10, 0)),
            MoveToCommand((100, 100, 0)),
            LineToCommand((110, 100, 0)),
        ]
        segments = split_segments(cmds)
        self.assertEqual(len(segments), 2)
        self.assertEqual(len(segments[0]), 3)
        self.assertIsInstance(segments[0][0], MoveToCommand)
        self.assertEqual(len(segments[1]), 2)
        self.assertIsInstance(segments[1][0], MoveToCommand)

        # Test with a travel command at the end
        cmds.append(MoveToCommand((0, 0, 0)))
        segments = split_segments(cmds)
        self.assertEqual(len(segments), 3)
        self.assertEqual(len(segments[2]), 1)

        # Test error on non-moving command (this is an invalid input)
        with self.assertRaises(ValueError):
            # A state command, which should have been filtered out
            split_segments([Command()])

    def test_flip_segment(self):
        """Test the reversal of a single cutting path segment."""
        state_a = State(power=100)
        state_b = State(power=120)
        state_c = State(power=150)

        p1, p2, p3 = (0, 0, 0), (10, 0, 0), (10, 10, 0)

        # Original commands and their states
        # The state is for the move TO the point.
        cmd1 = MoveToCommand(p1)
        cmd1.state = state_a
        cmd2 = LineToCommand(p2)
        cmd2.state = state_b
        cmd3 = LineToCommand(p3)
        cmd3.state = state_c
        segment = [cmd1, cmd2, cmd3]

        flipped = flip_segment(segment)

        # Check length
        self.assertEqual(len(flipped), 3)

        # Check endpoints: p3 -> p2 -> p1
        self.assertEqual(flipped[0].end, p3)
        self.assertEqual(flipped[1].end, p2)
        self.assertEqual(flipped[2].end, p1)

        # Check command types and shifted states
        # Expected:
        # MoveTo(p3, state_a), LineTo(p2, state_c), LineTo(p1, state_b)
        self.assertIsInstance(flipped[0], MoveToCommand)
        self.assertEqual(flipped[0].state, state_a)

        self.assertIsInstance(flipped[1], LineToCommand)
        self.assertEqual(flipped[1].state, state_c)

        self.assertIsInstance(flipped[2], LineToCommand)
        self.assertEqual(flipped[2].state, state_b)

    def test_flip_segment_with_arc(self):
        """Test flipping a segment that contains an ArcToCommand."""
        p_start = (0, 0, 0)
        p_end = (10, 10, 0)
        center_offset = (10, 0)  # Center is at (10, 0) relative to start

        state1 = State(power=100)
        state2 = State(power=120)

        cmd1 = MoveToCommand(p_start)
        cmd1.state = state1
        cmd2 = ArcToCommand(p_end, center_offset, clockwise=True)
        cmd2.state = state2
        segment = [cmd1, cmd2]

        flipped = flip_segment(segment)
        self.assertEqual(len(flipped), 2)

        # Check basic flip properties
        self.assertIsInstance(flipped[0], MoveToCommand)
        self.assertEqual(flipped[0].end, p_end)
        self.assertEqual(flipped[0].state, state1)

        # Check flipped arc. Assert type then cast for full type safety.
        self.assertIsInstance(flipped[1], ArcToCommand)
        flipped_arc = cast(ArcToCommand, flipped[1])

        self.assertEqual(
            flipped_arc.end,
            p_start,
            "Flipped arc should end at original start",
        )
        self.assertFalse(
            flipped_arc.clockwise, "Arc direction should be inverted"
        )

        # Verify center point remains the same
        # Original arc start is p_start, so center is (0+10, 0+0) = (10,0)
        original_center = (
            p_start[0] + center_offset[0],
            p_start[1] + center_offset[1],
        )
        # Flipped arc starts at p_end=(10,10), its offset should bring it
        # to the same center
        new_center = (
            p_end[0] + flipped_arc.center_offset[0],
            p_end[1] + flipped_arc.center_offset[1],
        )

        self.assertAlmostEqual(original_center[0], new_center[0])
        self.assertAlmostEqual(original_center[1], new_center[1])

    def test_greedy_order_segments(self):
        """Test the greedy algorithm for initial segment ordering."""
        # Seg1: (0,0) -> (10,0) - should be chosen first
        # Seg2: (100,100) -> (110,100)
        # Seg3: (10,0) -> (10,10) - should be chosen second
        s1 = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
        s2 = [MoveToCommand((100, 100, 0)), LineToCommand((110, 100, 0))]
        s3 = [MoveToCommand((10, 0, 0)), LineToCommand((10, 10, 0))]
        segments = [s1, s2, s3]

        ordered = greedy_order_segments(self.ctx, segments)
        self.assertEqual(len(ordered), 3)
        # Expected order: s1, s3, s2
        self.assertIs(ordered[0], s1)
        self.assertIs(ordered[1], s3)
        self.assertIs(ordered[2], s2)

    def test_greedy_order_with_flip(self):
        """Test greedy ordering when flipping a segment is optimal."""
        # Seg1: (0,0) -> (10,0)
        # Seg2: (100,100) -> (110,100)
        # Seg3: (10,10) -> (10,0) <-- start is far, end is near
        s1 = [MoveToCommand((0, 0, 0)), LineToCommand((10, 0, 0))]
        s2 = [MoveToCommand((100, 100, 0)), LineToCommand((110, 100, 0))]
        s3 = [MoveToCommand((10, 10, 0)), LineToCommand((10, 0, 0))]
        segments = [s1, s2, s3]

        ordered = greedy_order_segments(self.ctx, segments)

        # Expected: s1, flipped(s3), s2
        self.assertIs(ordered[0], s1)
        self.assertIsNot(ordered[1], s3)  # Should be a new, flipped list
        self.assertEqual(ordered[1][0].end, (10, 0, 0))  # Start of flipped s3
        self.assertEqual(ordered[1][-1].end, (10, 10, 0))  # End of flipped s3
        self.assertIs(ordered[2], s2)

    def test_flip_segments_local_optimization(self):
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
        travel_before = _dist_2d((10, 0), (20, 10)) + _dist_2d(
            (10, 10), (20, 0)
        )

        improved = flip_segments(self.ctx, ordered_segments)

        # Flipped s2 starts at (10,10) and ends at (20,10)
        # Travel cost after: dist((10,0),(10,10)) + dist((20,10),(20,0)) = 20
        p1_end = improved[0][-1].end
        p2_start, p2_end = improved[1][0].end, improved[1][-1].end
        p3_start = improved[2][0].end
        travel_after = _dist_2d(p1_end, p2_start) + _dist_2d(p2_end, p3_start)

        self.assertLess(travel_after, travel_before)
        self.assertEqual(
            improved[1][0].end, (10, 10, 0)
        )  # Check if s2 was flipped

    def test_two_opt(self):
        """Test the 2-opt algorithm for un-crossing paths."""
        # A(0,0->1,0), B(10,10->11,10), C(2,0->1,0), D(11,10->12,10)
        # Order A, B, C, D is crossed. Optimal is A, C, B, D.
        # sC is reversed to make simple greedy fail.
        sA = [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]
        sB = [MoveToCommand((10, 10, 0)), LineToCommand((11, 10, 0))]
        sC = [MoveToCommand((2, 0, 0)), LineToCommand((1, 0, 0))]
        sD = [MoveToCommand((11, 10, 0)), LineToCommand((12, 10, 0))]

        ordered = [sA, sB, sC, sD]

        optimized = two_opt(self.ctx, ordered, 10)

        # 2-opt should reverse [sB, sC] to [sC, sB] and flip each segment.
        # Expected final sequence: [sA, flipped(sC), flipped(sB), sD]
        self.assertIs(optimized[0], sA)
        self.assertEqual(optimized[1][0].end, (1, 0, 0))  # start of flipped sC
        self.assertEqual(
            optimized[2][0].end, (11, 10, 0)
        )  # start of flipped sB
        self.assertIs(optimized[3], sD)


class TestOptimizerIntegration(unittest.TestCase):
    """Integration test for the main Optimize.run() method."""

    def _calculate_travel_distance(self, ops: Ops) -> float:
        """Helper to calculate only the travel distance."""
        return ops.distance() - ops.cut_distance()

    def test_run_optimization(self):
        """Test the full optimization process on a sample Ops object."""
        # Create an inefficient path
        # It draws two separate squares, but jumps between them for each
        # segment
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
        travel_before = self._calculate_travel_distance(ops_copy)

        # Run the optimizer
        optimizer = Optimize()
        optimizer.run(ops)

        # Calculate travel distance after optimization
        ops.preload_state()
        travel_after = self._calculate_travel_distance(ops)

        # The optimizer should significantly reduce travel distance
        self.assertGreater(
            travel_before, 250, "Initial travel should be large"
        )
        self.assertLess(
            travel_after,
            travel_before,
            "Optimized travel distance should be smaller",
        )
        self.assertLess(
            travel_after, 150, "Optimized travel should be just one jump"
        )

        # Check that the number of cutting commands is the same
        cuts_after = sum(1 for c in ops.commands if c.is_cutting_command())
        self.assertEqual(cuts_after, 4)

    def test_run_with_air_assist_change(self):
        """
        Verify that segments with different air assist states are not reordered
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

        # After optimization, find the first command with air assist ON.
        air_on_idx = -1
        for i, cmd in enumerate(ops.commands):
            if cmd.state and cmd.state.air_assist:
                air_on_idx = i
                break

        self.assertNotEqual(
            air_on_idx, -1, "A segment with air assist ON should exist"
        )

        # All points before this index should be from Part 1
        for i in range(air_on_idx):
            cmd = ops.commands[i]
            if cmd.end:  # Check only moving commands
                self.assertIsNotNone(cmd.state)
                self.assertLess(
                    cmd.end[0],
                    50,
                    "Points from Part 1 should be in first half",
                )
                assert cmd.state
                self.assertFalse(
                    cmd.state.air_assist, "State should be air OFF"
                )

        # All points from this index on should be from Part 2
        for i in range(air_on_idx, len(ops.commands)):
            cmd = ops.commands[i]
            if cmd.end:  # Check only moving commands
                self.assertIsNotNone(cmd.state)
                self.assertGreater(
                    cmd.end[0],
                    50,
                    "Points from Part 2 should be in second half",
                )
                assert cmd.state
                self.assertTrue(cmd.state.air_assist, "State should be air ON")

    def test_run_preserves_markers(self):
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

        self.assertNotEqual(
            marker_idx, -1, "Marker command should be preserved"
        )

        # Check that segments before the marker were optimized together
        moving_cmds_before = [
            c
            for c in ops.commands[:marker_idx]
            if isinstance(c, MovingCommand)
        ]
        self.assertEqual(len(moving_cmds_before), 4)
        self.assertEqual(moving_cmds_before[0].end, (0, 0, 0))  # Start S1
        self.assertEqual(moving_cmds_before[1].end, (10, 0, 0))  # End S1
        self.assertEqual(moving_cmds_before[2].end, (100, 100, 0))  # Start S2
        self.assertEqual(moving_cmds_before[3].end, (110, 100, 0))  # End S2

        # Check that segments after the marker were optimized together
        moving_cmds_after = [
            c
            for c in ops.commands[marker_idx + 1 :]
            if isinstance(c, MovingCommand)
        ]
        self.assertEqual(len(moving_cmds_after), 4)
        self.assertEqual(moving_cmds_after[0].end, (10, 0, 0))  # Start S3
        self.assertEqual(moving_cmds_after[1].end, (10, 10, 0))  # End S3
        self.assertEqual(moving_cmds_after[2].end, (110, 100, 0))  # Start S4
        self.assertEqual(moving_cmds_after[3].end, (110, 110, 0))  # End S4


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
