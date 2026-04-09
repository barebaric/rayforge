import math
from unittest.mock import Mock, patch
from rayforge.core.ops import (
    ArcToCommand,
    LineToCommand,
    MoveToCommand,
    Ops,
)
from rayforge.core.geo.smooth import smooth_polyline
from rayforge.core.ops.commands import BezierToCommand
from tests.conftest import MockProgressContext
from post_processors.transformers import Smooth


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


def test_arcs_are_linearized_and_smoothed():
    """Tests that segments with arcs are linearized and smoothed."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.add(ArcToCommand((10, 10, 0), (5, 0), True))
    smoother = Smooth(amount=50)
    smoother.run(ops)
    assert len(ops.commands) > 2
    assert isinstance(ops.commands[0], MoveToCommand)
    assert all(isinstance(c, LineToCommand) for c in ops.commands[1:])


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

    call_count = [0]
    original_smooth_polyline = smooth_polyline

    def cancelling_smooth_polyline(
        points, amount, corner_angle, is_closed=None
    ):
        call_count[0] += 1
        if call_count[0] >= 5:
            context.set_cancelled(True)
        return original_smooth_polyline(
            points, amount, corner_angle, is_closed
        )

    with patch(
        "post_processors.transformers.smooth_transformer.smooth_polyline",
        cancelling_smooth_polyline,
    ):
        smoother.run(ops, context=context)

    assert len(list(ops.segments())) == 5
    assert abs(context.progress_calls[-1] - 0.5) < 1e-9


def test_bezier_passes_through_unchanged():
    """
    Bezier commands should pass through the smooth transformer
    without being modified — they're already smooth curves.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to((10, 20, 0), (30, 20, 0), (40, 0, 0))

    smoother = Smooth(amount=50)
    smoother.run(ops)

    bezier_cmds = [c for c in ops.commands if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) == 1
    cmd = bezier_cmds[0]
    assert cmd.control1 == (10, 20, 0)
    assert cmd.control2 == (30, 20, 0)
    assert cmd.end == (40, 0, 0)


def test_mixed_lines_and_bezier():
    """
    A segment with both lines and bezier should pass through unchanged.
    A line-only segment in the same ops should still be smoothed.
    """
    ops = Ops()
    ops.set_power(1.0)

    # Segment 1: line-only (should be smoothed)
    ops.move_to(0, 0, 0)
    ops.line_to(50, 0, 0)
    ops.line_to(100, 50, 0)

    # Segment 2: contains a bezier (should pass through)
    ops.move_to(0, 0, 0)
    ops.line_to(10, 0, 0)
    ops.bezier_to((20, 10, 0), (30, 10, 0), (40, 0, 0))

    smoother = Smooth(amount=50)
    smoother.run(ops)

    bezier_cmds = [c for c in ops.commands if isinstance(c, BezierToCommand)]
    assert len(bezier_cmds) == 1
    assert bezier_cmds[0].control1 == (20, 10, 0)
    assert bezier_cmds[0].control2 == (30, 10, 0)
    assert bezier_cmds[0].end == (40, 0, 0)

    line_cmds = [c for c in ops.commands if isinstance(c, LineToCommand)]
    # Segment 1 was smoothed (subdivided into more lines)
    # Segment 2's line is preserved but not smoothed
    assert len(line_cmds) > 2
