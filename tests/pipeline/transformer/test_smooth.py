import math
from unittest.mock import Mock
from rayforge.core.ops import ArcToCommand, Ops
from rayforge.pipeline.transformer.smooth import Smooth
from conftest import MockProgressContext


def assert_points_almost_equal(p1: tuple, p2: tuple, places=5, msg=None):
    """Asserts that two 3D points are almost equal."""
    assert abs(p1[0] - p2[0]) < 10 ** (-places), (
        f"{msg} (x-coord): {p1[0]} != {p2[0]}"
    )
    assert abs(p1[1] - p2[1]) < 10 ** (-places), (
        f"{msg} (y-coord): {p1[1]} != {p2[1]}"
    )
    assert abs(p1[2] - p2[2]) < 10 ** (-places), (
        f"{msg} (z-coord): {p1[2]} != {p2[2]}"
    )


def distance_2d(p1: tuple, p2: tuple) -> float:
    """Helper to calculate 2D distance."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def test_initialization_and_properties():
    """Tests constructor and property setters trigger signals."""
    smoother = Smooth(enabled=True, amount=50, corner_angle_threshold=60)
    smoother.changed = Mock()

    assert smoother.enabled
    smoother.amount = 120
    assert smoother.amount == 100
    smoother.corner_angle_threshold = 90
    assert abs(smoother.corner_angle_threshold - 90) < 1e-9
    smoother.changed.send.assert_called()


def test_run_with_zero_amount():
    """Tests that run() is a no-op if amount is zero."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 0)
    original_ops = ops.copy()
    smoother = Smooth(amount=0)
    smoother.run(ops)
    assert len(ops.commands) == len(original_ops.commands)


def test_non_line_only_segment_is_unmodified():
    """Tests that segments with arcs are unmodified."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.add(ArcToCommand((10, 10, 0), (5, 0), True))
    original_ops = ops.copy()
    smoother = Smooth(amount=50)
    smoother.run(ops)
    assert len(ops.commands) == len(original_ops.commands)
    assert isinstance(ops.commands[1], ArcToCommand)


def test_smooth_open_path():
    """Tests smoothing a simple open line segment."""
    ops = Ops()
    ops.move_to(0, 0, 5)
    ops.line_to(50, 0, 5)
    ops.line_to(100, 50, 5)

    smoother = Smooth(amount=50)
    smoother.run(ops)

    assert len(ops.commands) > 3, "Path should be subdivided"

    output_points = [cmd.end for cmd in ops.commands if cmd.end is not None]

    assert_points_almost_equal(output_points[0], (0, 0, 5))
    assert_points_almost_equal(output_points[-1], (100, 50, 5))

    original_corner = 50, 0, 5
    closest_point = min(
        output_points, key=lambda p: math.dist(p, original_corner)
    )

    assert closest_point[1] > 1e-9


def test_corner_preservation():
    """
    Tests that sharp corners are preserved while dull ones are smoothed.
    """
    ops = Ops()
    ops.move_to(0, 50)
    ops.line_to(50, 0)
    ops.line_to(100, 50)
    ops.line_to(150, 50)

    smoother = Smooth(amount=40, corner_angle_threshold=95)
    smoother.run(ops)

    output_points = [cmd.end for cmd in ops.commands if cmd.end is not None]

    found_sharp = any(distance_2d(p, (50, 0, 0)) < 1e-5 for p in output_points)
    assert found_sharp, "Sharp corner was not preserved"

    found_dull = any(
        distance_2d(p, (100, 50, 0)) < 1e-5 for p in output_points
    )
    assert not found_dull, "Dull corner was not smoothed"


def test_context_cancellation_and_progress():
    """
    Tests that process can be cancelled and that progress is reported.
    """
    ops = Ops()
    for i in range(10):
        ops.move_to(i * 10, 0)
        ops.line_to(i * 10 + 5, 5)

    context = MockProgressContext()
    context._inner._progress_context.set_wrapper(context._inner)
    smoother = Smooth(amount=50)

    original_smooth_segment = smoother._smooth_segment
    call_count = [0]

    def cancelling_smooth_segment(points):
        call_count[0] += 1
        if call_count[0] >= 5:
            context.set_cancelled(True)
        return original_smooth_segment(points)

    smoother._smooth_segment = cancelling_smooth_segment
    smoother.run(ops, context=context)

    assert len(list(ops.segments())) == 5
    assert abs(context.progress_calls[-1] - 0.5) < 1e-9
