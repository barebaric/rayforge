import math
import pytest
import numpy as np
from rayforge.core.ops import Ops, SectionType
from rayforge.core.ops.commands import BezierToCommand
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

        # Check travel vertices (2 MoveTo, first skipped = 1 segment = 2 verts)
        assert result.travel_vertices.shape == (2, 3)
        travel_coords = result.travel_vertices
        np.testing.assert_array_equal(travel_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(travel_coords[1], [10.0, 0.0, 0.0])

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
        # First MoveTo is skipped, so no travel vertices
        assert result.travel_vertices.shape == (0, 3)

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

        # Check travel vertices (3 MoveTo, first skipped = 4 vertices)
        assert result.travel_vertices.shape == (4, 3)

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

    def test_encode_multi_workpiece_contour(self, encoder: VertexEncoder):
        """
        Simulates the exact command stream produced by compute_step_artifacts
        for a contour step with multiple workpieces, each wrapped in
        WorkpieceStart/End and OpsSection markers.
        """
        combined_ops = Ops()

        num_contours = 5
        points_per_contour = 20

        for wp_idx in range(3):
            wp_uid = f"workpiece-{wp_idx}"

            combined_ops.workpiece_start(wp_uid)

            combined_ops.set_power(1.0)
            combined_ops.set_cut_speed(1000)
            combined_ops.set_travel_speed(3000)
            combined_ops.enable_air_assist(True)
            combined_ops.set_laser("laser-1")

            combined_ops.ops_section_start(SectionType.VECTOR_OUTLINE, wp_uid)
            for c_idx in range(num_contours):
                cx = wp_idx * 100.0 + c_idx * 10.0
                cy = 0.0
                combined_ops.move_to(cx, cy, 0.0)
                for p_idx in range(points_per_contour):
                    angle = 2.0 * math.pi * p_idx / points_per_contour
                    nx = cx + 5.0 * math.cos(angle)
                    ny = cy + 5.0 * math.sin(angle)
                    combined_ops.line_to(nx, ny, 0.0)
                combined_ops.line_to(cx, cy, 0.0)

            combined_ops.ops_section_end(SectionType.VECTOR_OUTLINE)
            combined_ops.disable_air_assist()

            combined_ops.workpiece_end(wp_uid)

        result = encoder.encode(combined_ops)

        total_contours = 3 * num_contours
        points_per_contour_with_close = points_per_contour + 1
        expected_powered_verts = (
            total_contours * points_per_contour_with_close * 2
        )
        expected_travel_verts = (total_contours - 1) * 2

        actual_powered = result.powered_vertices.shape[0]
        actual_travel = result.travel_vertices.shape[0]

        assert actual_powered == expected_powered_verts, (
            f"Expected {expected_powered_verts} powered vertices, "
            f"got {actual_powered}"
        )
        assert actual_travel == expected_travel_verts, (
            f"Expected {expected_travel_verts} travel vertices, "
            f"got {actual_travel}"
        )

        last_powered = result.powered_vertices[-1]
        assert last_powered[0] == pytest.approx(2 * 100.0 + 4 * 10.0)
        assert last_powered[1] == pytest.approx(0.0)
        assert last_powered[2] == pytest.approx(0.0)

    def test_encode_bezier_powered(self, encoder: VertexEncoder):
        """Test that BezierToCommand is linearized into powered vertices."""
        ops = Ops()
        ops.set_power(0.5)
        ops.move_to(0.0, 0.0, 0.0)
        ops.commands.append(
            BezierToCommand(
                end=(10.0, 0.0, 0.0),
                control1=(3.0, 5.0, 0.0),
                control2=(7.0, 5.0, 0.0),
            )
        )

        result = encoder.encode(ops)

        assert result.powered_vertices.shape[0] >= 4
        assert result.powered_vertices.shape[0] % 2 == 0
        assert (
            result.powered_colors.shape[0] == result.powered_vertices.shape[0]
        )

        powered_coords = result.powered_vertices
        np.testing.assert_array_almost_equal(
            powered_coords[0], [0.0, 0.0, 0.0]
        )
        np.testing.assert_array_almost_equal(
            powered_coords[-1], [10.0, 0.0, 0.0]
        )

        expected_color = [0.49803922, 0.49803922, 0.49803922, 1.0]
        np.testing.assert_array_almost_equal(
            result.powered_colors[0], expected_color
        )

    def test_encode_bezier_zero_power(self, encoder: VertexEncoder):
        """Test that zero-power bezier goes to zero_power_vertices."""
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(0.0)
        ops.commands.append(
            BezierToCommand(
                end=(10.0, 0.0, 0.0),
                control1=(3.0, 5.0, 0.0),
                control2=(7.0, 5.0, 0.0),
            )
        )

        result = encoder.encode(ops)

        assert result.powered_vertices.shape == (0, 3)
        assert result.zero_power_vertices.shape[0] >= 4
        assert result.zero_power_vertices.shape[0] % 2 == 0

        zp_coords = result.zero_power_vertices
        np.testing.assert_array_almost_equal(zp_coords[0], [0.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(zp_coords[-1], [10.0, 0.0, 0.0])


class TestTransformToCylinder:
    """Tests for transform_to_cylinder with Z-depth awareness."""

    def test_surface_z_zero(self):
        """At Z=0 (surface) vertices sit on the cylinder surface."""
        from rayforge.pipeline.encoder.vertexencoder import (
            transform_to_cylinder,
        )

        diameter = 10.0
        radius = diameter / 2.0
        verts = np.array(
            [[0, 0, 0], [10, 0, 0], [20, 0, 0], [30, 0, 0]],
            dtype=np.float32,
        )
        result, _, _ = transform_to_cylinder(verts, diameter)
        assert result.shape == (4, 3)
        np.testing.assert_almost_equal(
            result[:, 1], radius * np.sin(0.0), decimal=3
        )
        np.testing.assert_almost_equal(
            result[:, 2], radius * np.cos(0.0), decimal=3
        )

    def test_negative_z_smaller_radius(self):
        """Negative Z (step-down) produces smaller effective radius."""
        from rayforge.pipeline.encoder.vertexencoder import (
            transform_to_cylinder,
        )

        diameter = 10.0
        z = -1.0
        verts = np.array(
            [[0, 0, z], [10, 0, z]],
            dtype=np.float32,
        )
        result, _, _ = transform_to_cylinder(verts, diameter)
        expected_radius = diameter / 2.0 + z
        assert result.shape == (2, 3)
        r_actual = float(np.sqrt(result[0, 1] ** 2 + result[0, 2] ** 2))
        assert r_actual == pytest.approx(expected_radius, abs=0.01)
        r_actual_end = float(np.sqrt(result[1, 1] ** 2 + result[1, 2] ** 2))
        assert r_actual_end == pytest.approx(expected_radius, abs=0.01)

    def test_mixed_z_different_radii(self):
        """Vertices with different Z depths produce different radii."""
        from rayforge.pipeline.encoder.vertexencoder import (
            transform_to_cylinder,
        )

        diameter = 20.0
        z1, z2 = 0.0, -1.0
        verts = np.array(
            [[5, 0, z1], [15, 0, z2]],
            dtype=np.float32,
        )
        result, _, _ = transform_to_cylinder(verts, diameter)
        r1 = float(np.sqrt(result[0, 1] ** 2 + result[0, 2] ** 2))
        r2 = float(np.sqrt(result[1, 1] ** 2 + result[1, 2] ** 2))
        assert r1 == pytest.approx(diameter / 2.0 + z1, abs=0.01)
        assert r2 == pytest.approx(diameter / 2.0 + z2, abs=0.01)
