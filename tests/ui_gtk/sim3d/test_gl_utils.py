"""Tests for the shared sim3d GL helpers in gl_utils."""

import numpy as np
import pytest

from rayforge.ui_gtk.sim3d.gl_utils import (
    LINE_DEPTH_BIAS_MM,
    line_depth_bias,
)


def _perspective_proj(near=0.1, far=10000.0, f=1.0):
    """Builds a perspective projection matrix like Camera's."""
    return np.array(
        [
            [f, 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [
                0.0,
                0.0,
                (far + near) / (near - far),
                (2 * far * near) / (near - far),
            ],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    )


def _ortho_proj(near=0.1, far=10000.0):
    """Builds an orthographic projection matrix like Camera's."""
    return np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -2.0 / (far - near), -(far + near) / (far - near)],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


@pytest.mark.ui
def test_line_depth_bias_is_negative_for_perspective():
    """A perspective projection maps view z to clip z with a ~-1 slope,
    so the bias must be negative (moves fragments toward the camera)."""
    bias = line_depth_bias(_perspective_proj())
    assert bias < 0.0
    assert bias == pytest.approx(-LINE_DEPTH_BIAS_MM, rel=1e-3)


@pytest.mark.ui
def test_line_depth_bias_scales_with_view_depth_slope():
    """The bias must equal proj[2][2] * LINE_DEPTH_BIAS_MM so the
    resulting clip-space offset is a constant view-space shift for both
    projection types."""
    proj = _perspective_proj()
    expected = proj[2, 2] * LINE_DEPTH_BIAS_MM
    assert line_depth_bias(proj) == pytest.approx(float(expected))


@pytest.mark.ui
def test_line_depth_bias_is_negative_for_ortho():
    """Orthographic projections also map farther points to larger clip
    z, so the bias stays negative."""
    bias = line_depth_bias(_ortho_proj())
    assert bias < 0.0
    assert bias == pytest.approx(
        float(_ortho_proj()[2, 2]) * LINE_DEPTH_BIAS_MM
    )
