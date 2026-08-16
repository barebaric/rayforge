"""Tests for the shared sim3d GL helpers in gl_utils."""

import pytest

from rayforge.ui_gtk.sim3d.gl_utils import LINE_DEPTH_WINDOW_BIAS

# One LSB of a 24-bit fixed-point depth buffer (NDC range spans 2).
_LSB_24BIT = 2.0 / ((1 << 24) - 1)


@pytest.mark.ui
def test_line_depth_range_bias_beats_coplanar_geometry():
    """The bias must exceed a couple of depth LSBs so lines win depth
    ties against the coplanar raster texture and the stock fill (which
    glPolygonOffset pushes away by one unit ~= one LSB)."""
    assert LINE_DEPTH_WINDOW_BIAS > 2.0 * _LSB_24BIT


@pytest.mark.ui
def test_line_depth_range_bias_stays_below_laser_head_separation():
    """The bias must stay far below the depth separation of the laser
    head model hovering above the workpiece, so the head still
    occludes the biased lines.  At typical simulator zoom levels the
    head spans well over a hundred LSBs of window depth."""
    assert LINE_DEPTH_WINDOW_BIAS < 10.0 * _LSB_24BIT
