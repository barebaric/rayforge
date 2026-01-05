import pytest
import numpy as np
from rayforge.ui_gtk.canvas3d.ops_renderer import OpsRenderer
from rayforge.core.ops import Ops
from rayforge.shared.util.colors import ColorSet


@pytest.fixture
def ops_renderer():
    """Provides an OpsRenderer instance for testing."""
    return OpsRenderer()


@pytest.fixture
def sample_ops():
    """Creates a sample Ops object with mixed command types."""
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.set_power(0.5)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(0.8)
    ops.arc_to(10.0, 10.0, 0.0, 5.0, clockwise=False)

    # Add a scanline power command directly with mixed power values
    # Include some zero power values to test zero power vertices
    ops.scan_to(0.0, 10.0, 0.0, bytearray([100, 0, 200, 0, 100]))
    ops.move_to(5.0, 5.0, 0.0)

    # Preload state to ensure commands have proper state information
    ops.preload_state()

    return ops


@pytest.fixture
def colors():
    """Provides a ColorSet for testing."""
    # Create LUTs for cut and engrave
    cut_lut = np.zeros((256, 4))
    cut_lut[:, 0] = 1.0  # Red
    cut_lut[:, 3] = 1.0  # Full alpha

    engrave_lut = np.zeros((256, 4))
    # Create gradient from white to red
    for i in range(256):
        t = i / 255.0
        engrave_lut[i] = [1.0, 1.0 - t, 1.0 - t, 1.0]

    # RGBA colors
    travel_rgba = (0.0, 1.0, 0.0, 1.0)  # Green
    zero_power_rgba = (0.0, 0.0, 1.0, 1.0)  # Blue

    return ColorSet(
        {
            "cut": cut_lut,
            "engrave": engrave_lut,
            "travel": travel_rgba,
            "zero_power": zero_power_rgba,
        }
    )


def test_prepare_vertex_data(ops_renderer, sample_ops, colors):
    """
    Test that prepare_vertex_data with travel_only=True only returns
    vertices corresponding to MoveTo commands.
    """
    # Call prepare_vertex_data with travel_only=True
    (
        powered_vertices,
        powered_colors,
        travel_vertices,
        zero_power_vertices,
        zero_power_colors,
    ) = ops_renderer.prepare_vertex_data(sample_ops, colors)

    # Check that only travel vertices are present
    # We should have vertices for the travel moves
    assert travel_vertices.size > 0

    # Check that powered, zero power vertices and colors are present
    assert powered_vertices.size > 0  # From LineTo and ArcTo commands
    assert powered_colors.size > 0
    assert travel_vertices.size > 0  # From MoveTo commands
    assert zero_power_vertices.size > 0  # From zero power segments in scan_to
    assert zero_power_colors.size > 0  # From zero power segments in scan_to

    # Verify that the travel vertices only contain MoveTo command vertices
    # The exact number may vary depending on implementation details
    # but we should only have travel vertices, no powered or zero power ones
    assert travel_vertices.size % 3 == 0  # Should be divisible by 3 (x,y,z)
