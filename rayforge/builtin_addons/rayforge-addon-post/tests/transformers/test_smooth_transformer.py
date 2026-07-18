import math
from unittest.mock import Mock

import pytest
from post_processors.transformers import Smooth
from raygeo.ops import Ops
from raygeo.ops.types import CommandType

from rayforge.pipeline.transformer.specs import (
    apply_transformer_specs,
    build_transformer_specs,
)
from tests.conftest import MockProgressContext


def _apply(transformer, ops, context=None):
    """Run a transformer through the Rust spec dispatch."""
    specs = build_transformer_specs([transformer], None, None, None)
    apply_transformer_specs(ops, specs, context)


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
    """Tests that the dispatch is a no-op if amount is zero."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 0)
    original_ops = ops.copy()
    smoother = Smooth(amount=0)
    _apply(smoother, ops)
    assert ops.len() == original_ops.len()


def test_arcs_are_linearized_and_smoothed():
    """Tests that segments with arcs are linearized and smoothed."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.arc_to(10, 10, 5, 0, True)
    smoother = Smooth(amount=50)
    _apply(smoother, ops)
    assert ops.len() > 2
    assert ops.command_type(0) == CommandType.MOVE_TO
    assert all(
        ops.command_type(i) == CommandType.LINE_TO for i in range(1, ops.len())
    )


def test_smooth_open_path():
    """Tests smoothing a simple open line segment."""
    ops = Ops()
    ops.move_to(0, 0, 5)
    ops.line_to(50, 0, 5)
    ops.line_to(100, 50, 5)

    smoother = Smooth(amount=50)
    _apply(smoother, ops)

    assert ops.len() > 3, "Path should be subdivided"

    output_points = [ops.endpoint(i) for i in range(ops.len())]

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
    _apply(smoother, ops)

    output_points = [ops.endpoint(i) for i in range(ops.len())]

    found_sharp = any(distance_2d(p, (50, 0, 0)) < 1e-5 for p in output_points)
    assert found_sharp, "Sharp corner was not preserved"

    found_dull = any(
        distance_2d(p, (100, 50, 0)) < 1e-5 for p in output_points
    )
    assert not found_dull, "Dull corner was not smoothed"


def test_context_cancellation_and_progress():
    """
    Tests that progress is reported during dispatch.
    """
    ops = Ops()
    for i in range(10):
        ops.move_to(i * 10, 0)
        ops.line_to(i * 10 + 5, 5)

    context = MockProgressContext()
    context._inner._progress_context.set_wrapper(context._inner)
    smoother = Smooth(amount=50)

    _apply(smoother, ops, context=context)

    assert len(context.progress_calls) > 0


def test_context_cancellation_skips():
    """Tests that a cancelled context aborts the dispatch."""
    ops = Ops()
    ops.move_to(0, 0)
    ops.line_to(10, 0)

    context = MockProgressContext()
    context.set_cancelled(True)
    smoother = Smooth(amount=50)
    with pytest.raises(RuntimeError, match="cancelled"):
        _apply(smoother, ops, context=context)


def test_bezier_passes_through_unchanged():
    """
    Bezier commands should pass through the smooth transformer
    without being modified — they're already smooth curves.
    """
    ops = Ops()
    ops.move_to(0, 0)
    ops.bezier_to((10, 20, 0), (30, 20, 0), (40, 0, 0))

    smoother = Smooth(amount=50)
    _apply(smoother, ops)

    bezier_indices = ops.indices_of(CommandType.BEZIER_TO)
    assert len(bezier_indices) == 1
    bezier_idx = bezier_indices[0]
    c1, c2 = ops.bezier_params(bezier_idx)
    assert c1 == (10, 20, 0)
    assert c2 == (30, 20, 0)
    assert ops.endpoint(bezier_idx) == (40, 0, 0)


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
    _apply(smoother, ops)

    bezier_indices = ops.indices_of(CommandType.BEZIER_TO)
    assert len(bezier_indices) == 1
    bezier_idx = bezier_indices[0]
    c1, c2 = ops.bezier_params(bezier_idx)
    assert c1 == (20, 10, 0)
    assert c2 == (30, 10, 0)
    assert ops.endpoint(bezier_idx) == (40, 0, 0)

    line_count = len(ops.indices_of(CommandType.LINE_TO))
    # Segment 1 was smoothed (subdivided into more lines)
    # Segment 2's line is preserved but not smoothed
    assert line_count > 2
