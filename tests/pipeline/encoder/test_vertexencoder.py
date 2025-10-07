import pytest
import numpy as np
from rayforge.core.ops import Ops
from rayforge.pipeline.encoder.vertexencoder import VertexEncoder


class TestVertexEncoder:
    """Test suite for VertexEncoder class."""

    @pytest.fixture
    def encoder(self) -> VertexEncoder:
        """Provides a VertexEncoder instance for testing."""
        return VertexEncoder()

    def test_encode_empty_ops(self, encoder: VertexEncoder):
        """Encoding an empty Ops object should return empty arrays."""
        ops = Ops()
        result = encoder.encode(ops)

        assert result.powered_vertices.shape == (0, 3)
        assert result.powered_colors.shape == (0, 4)
        assert result.travel_vertices.shape == (0, 3)
        assert result.zero_power_vertices.shape == (0, 3)

    def test_encode_simple_cut_and_travel(self, encoder: VertexEncoder):
        """Test encoding a simple cut with travel move."""
        ops = Ops()
        # Travel move
        ops.move_to(0.0, 0.0, 0.0)
        ops.move_to(10.0, 0.0, 0.0)
        # Cut move
        ops.set_power(1.0)
        ops.line_to(10.0, 10.0, 0.0)

        result = encoder.encode(ops)

        # Check travel vertices (2 MoveTo commands = 2 segments = 4 vertices)
        assert result.travel_vertices.shape == (4, 3)
        travel_coords = result.travel_vertices
        np.testing.assert_array_equal(travel_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(travel_coords[1], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(travel_coords[2], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(travel_coords[3], [10.0, 0.0, 0.0])

        # Check powered vertices (1 LineTo command = 1 segment = 2 vertices)
        assert result.powered_vertices.shape == (2, 3)
        assert result.powered_colors.shape == (2, 4)
        powered_coords = result.powered_vertices
        np.testing.assert_array_equal(powered_coords[0], [10.0, 0.0, 0.0])
        np.testing.assert_array_equal(powered_coords[1], [10.0, 10.0, 0.0])

        # Check colors (should be white for power 1.0)
        powered_colors = result.powered_colors
        expected_color = [1.0, 1.0, 1.0, 1.0]  # White RGBA
        np.testing.assert_array_equal(powered_colors[0], expected_color)
        np.testing.assert_array_equal(powered_colors[1], expected_color)

    def test_encode_zero_power_move(self, encoder: VertexEncoder):
        """Test encoding zero-power moves."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(0.0)  # Zero power
        ops.line_to(5.0, 5.0, 0.0)

        result = encoder.encode(ops)

        # Should have zero-power vertices, not travel vertices
        assert result.zero_power_vertices.shape == (2, 3)
        assert result.powered_vertices.shape == (0, 3)
        assert result.travel_vertices.shape == (
            2,
            3,
        )  # From the initial MoveTo

        # Check the coordinates of the zero-power move
        zero_power_coords = result.zero_power_vertices
        np.testing.assert_array_equal(zero_power_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(zero_power_coords[1], [5.0, 5.0, 0.0])

    def test_encode_arc_linearization(self, encoder: VertexEncoder):
        """Test that arcs are properly linearized into line segments."""
        ops = Ops()
        ops.set_power(0.5)
        ops.move_to(0.0, 10.0, 0.0)
        # Create a 90-degree arc
        ops.arc_to(10.0, 0.0, 0.0, -10.0, True)

        result = encoder.encode(ops)

        # Arc should be linearized into multiple segments
        assert result.powered_vertices.shape[0] >= 4  # At least 2 segments
        assert result.powered_vertices.shape[0] % 2 == 0  # Even vertices
        assert (
            result.powered_colors.shape[0] == result.powered_vertices.shape[0]
        )

        # Check that colors correspond to power 0.5 (mid-gray)
        powered_colors = result.powered_colors
        expected_color = [0.49803922, 0.49803922, 0.49803922, 1.0]  # 127/255
        np.testing.assert_array_almost_equal(powered_colors[0], expected_color)

        # Check start and end points
        powered_coords = result.powered_vertices
        np.testing.assert_array_almost_equal(
            powered_coords[0], [0.0, 10.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            powered_coords[-1], [10.0, 0.0, 0.0]
        )

    def test_encode_scanline_power_command(self, encoder: VertexEncoder):
        """
        Test that ScanLinePowerCommand only produces zero-power vertices for
        the overscan/unpowered portions.
        """
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        # Create a scanline with mixed power values
        # Off, half, full, off, full
        power_values = bytearray([0, 128, 255, 0, 255])
        ops.scan_to(5.0, 0.0, 0.0, power_values)

        result = encoder.encode(ops)

        # Powered segments should be empty, as the texture handles them.
        assert result.powered_vertices.shape == (0, 3)
        assert result.powered_colors.shape == (0, 4)

        # Zero-power segments: two 1-unit chunks. 2 segments = 4 vertices
        assert result.zero_power_vertices.shape == (4, 3)

        # Check coordinates of the zero-power segments
        zero_v = result.zero_power_vertices

        # First zero-power chunk (0-1mm)
        np.testing.assert_array_almost_equal(zero_v[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(zero_v[1], [1.0, 0.0, 0.0])
        # Second zero-power chunk (3-4mm)
        np.testing.assert_array_almost_equal(zero_v[2], [3.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(zero_v[3], [4.0, 0.0, 0.0])

    def test_encode_complex_path(self, encoder: VertexEncoder):
        """Test encoding a more complex path with multiple move types."""
        ops = Ops()

        # Travel to start
        ops.move_to(0.0, 0.0, 0.0)
        ops.move_to(10.0, 0.0, 0.0)

        # Cut at full power
        ops.set_power(1.0)
        ops.line_to(20.0, 0.0, 0.0)

        # Cut at half power
        ops.set_power(0.5)
        ops.line_to(20.0, 10.0, 0.0)

        # Zero power move
        ops.set_power(0.0)
        ops.line_to(10.0, 10.0, 0.0)

        # Travel back
        ops.move_to(0.0, 10.0, 0.0)

        result = encoder.encode(ops)

        # Check travel vertices (3 MoveTo commands = 6 vertices)
        assert result.travel_vertices.shape == (6, 3)

        # Check powered vertices (2 cut moves = 4 vertices)
        assert result.powered_vertices.shape == (4, 3)
        assert result.powered_colors.shape == (4, 4)

        # Check zero-power vertices (1 zero-power LineTo = 2 vertices)
        assert result.zero_power_vertices.shape == (2, 3)

        # Verify colors match power levels
        powered_colors = result.powered_colors
        # First cut at full power
        np.testing.assert_array_equal(powered_colors[0], [1.0, 1.0, 1.0, 1.0])
        # Second cut at half power
        np.testing.assert_array_almost_equal(
            powered_colors[2], [0.49803922, 0.49803922, 0.49803922, 1.0]
        )

    def test_grayscale_lut_creation(self, encoder: VertexEncoder):
        """Test that the grayscale lookup table is created correctly."""
        lut = encoder._grayscale_lut

        assert lut.shape == (256, 4)
        assert lut.dtype == np.float32

        # Check endpoints
        np.testing.assert_array_equal(lut[0], [0.0, 0.0, 0.0, 1.0])  # Black
        np.testing.assert_array_equal(lut[255], [1.0, 1.0, 1.0, 1.0])  # White

        # Check middle value
        np.testing.assert_array_almost_equal(
            lut[128], [0.5019608, 0.5019608, 0.5019608, 1.0]
        )

    def test_encode_3d_coordinates(self, encoder: VertexEncoder):
        """Test that Z coordinates are properly handled."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 5.0)  # Start at Z=5
        ops.set_power(1.0)
        ops.line_to(10.0, 0.0, 5.0)  # End at Z=5
        ops.line_to(10.0, 10.0, 0.0)  # End at Z=0

        result = encoder.encode(ops)

        powered_coords = result.powered_vertices
        assert powered_coords.shape == (4, 3)

        # Check Z coordinates are preserved
        assert powered_coords[0][2] == 5.0  # First vertex Z=5
        assert powered_coords[1][2] == 5.0  # Second vertex Z=5
        assert powered_coords[2][2] == 5.0  # Third vertex Z=5
        assert powered_coords[3][2] == 0.0  # Fourth vertex Z=0
