import math
import unittest
from unittest.mock import Mock
from rayforge.core.ops import ArcToCommand, Ops
from rayforge.pipeline.transformer.smooth import Smooth
from rayforge.shared.tasker.proxy import BaseExecutionContext


class FakeExecutionContext(BaseExecutionContext):
    """
    A fake execution context for testing.
    Implements all abstract methods from the base class.
    """

    def __init__(self):
        self._cancelled = False
        self.progress = 0.0

    def is_cancelled(self) -> bool:
        return self._cancelled

    def set_progress(self, progress: float):
        self.progress = progress

    def cancel(self):
        self._cancelled = True

    def _create_sub_context(self, *args, **kwargs) -> "BaseExecutionContext":
        return self

    def _report_normalized_progress(self, *args, **kwargs):
        pass

    def flush(self, *args, **kwargs):
        pass

    def set_message(self, *args, **kwargs):
        pass


class TestSmooth(unittest.TestCase):
    """Test suite for the Smooth path transformer."""

    def assertPointsAlmostEqual(
        self, p1: tuple, p2: tuple, places=5, msg=None
    ):
        """Asserts that two 3D points are almost equal."""
        self.assertAlmostEqual(
            p1[0], p2[0], places=places, msg=f"{msg} (x-coord)"
        )
        self.assertAlmostEqual(
            p1[1], p2[1], places=places, msg=f"{msg} (y-coord)"
        )
        self.assertAlmostEqual(
            p1[2], p2[2], places=places, msg=f"{msg} (z-coord)"
        )

    def _distance_2d(self, p1: tuple, p2: tuple) -> float:
        """Helper to calculate 2D distance."""
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

    def test_initialization_and_properties(self):
        """Tests constructor and property setters trigger signals."""
        smoother = Smooth(enabled=True, amount=50, corner_angle_threshold=60)
        smoother.changed = Mock()  # Mock the instance attribute

        self.assertTrue(smoother.enabled)
        smoother.amount = 120
        self.assertEqual(smoother.amount, 100)
        smoother.corner_angle_threshold = 90
        self.assertAlmostEqual(smoother.corner_angle_threshold, 90)
        smoother.changed.send.assert_called()

    def test_run_with_zero_amount(self):
        """Tests that run() is a no-op if amount is zero."""
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        original_ops = ops.copy()
        smoother = Smooth(amount=0)
        smoother.run(ops)
        self.assertEqual(len(ops.commands), len(original_ops.commands))

    def test_non_line_only_segment_is_unmodified(self):
        """Tests that segments with arcs are unmodified."""
        ops = Ops()
        ops.move_to(0, 0)
        ops.add(ArcToCommand((10, 10, 0), (5, 0), True))
        original_ops = ops.copy()
        smoother = Smooth(amount=50)
        smoother.run(ops)
        self.assertEqual(len(ops.commands), len(original_ops.commands))
        self.assertIsInstance(ops.commands[1], ArcToCommand)

    def test_smooth_open_path(self):
        """Tests smoothing a simple open line segment."""
        ops = Ops()
        ops.move_to(0, 0, 5)
        ops.line_to(50, 0, 5)  # The corner to be smoothed
        ops.line_to(100, 50, 5)

        smoother = Smooth(amount=50)
        smoother.run(ops)

        self.assertGreater(len(ops.commands), 3, "Path should be subdivided")

        # Endpoints must be preserved
        output_points = [
            cmd.end for cmd in ops.commands if cmd.end is not None
        ]
        self.assertPointsAlmostEqual(output_points[0], (0, 0, 5))
        self.assertPointsAlmostEqual(output_points[-1], (100, 50, 5))

        # Find the output point that is spatially closest to the original
        # corner
        original_corner = 50, 0, 5
        closest_point = min(
            output_points, key=lambda p: math.dist(p, original_corner)
        )

        # The corner should have been "pulled" inwards
        self.assertGreater(closest_point[1], 1e-9)

    def test_corner_preservation(self):
        """
        Tests that sharp corners are preserved while dull ones are smoothed.
        """
        ops = Ops()
        ops.move_to(0, 50)
        ops.line_to(50, 0)  # Sharp corner (~90 deg)
        ops.line_to(100, 50)  # Dull internal corner (~135 deg)
        ops.line_to(150, 50)  # Creates a straight line to the end

        smoother = Smooth(amount=40, corner_angle_threshold=95)
        smoother.run(ops)

        output_points = [
            cmd.end for cmd in ops.commands if cmd.end is not None
        ]

        # Check that the sharp corner is still present.
        found_sharp = any(
            self._distance_2d(p, (50, 0, 0)) < 1e-5 for p in output_points
        )
        self.assertTrue(found_sharp, "Sharp corner was not preserved")

        # Check that the internal dull corner was smoothed away.
        found_dull = any(
            self._distance_2d(p, (100, 50, 0)) < 1e-5 for p in output_points
        )
        self.assertFalse(found_dull, "Dull corner was not smoothed")

    def test_context_cancellation_and_progress(self):
        """
        Tests that the process can be cancelled and that progress is reported.
        """
        ops = Ops()
        for i in range(10):  # 10 segments
            ops.move_to(i * 10, 0)
            ops.line_to(i * 10 + 5, 5)

        context = FakeExecutionContext()
        smoother = Smooth(amount=50)

        original_smooth_segment = smoother._smooth_segment
        call_count = [0]

        def cancelling_smooth_segment(points):
            call_count[0] += 1
            if call_count[0] >= 5:
                context.cancel()
            return original_smooth_segment(points)

        smoother._smooth_segment = cancelling_smooth_segment
        smoother.run(ops, context)

        # The loop runs 5 times before the cancellation is detected on
        # the 6th try
        self.assertEqual(len(list(ops.segments())), 5)
        self.assertAlmostEqual(context.progress, 0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
