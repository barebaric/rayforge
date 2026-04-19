import pytest
import numpy as np

from rayforge.core.ops import Ops
from rayforge.core.ops.commands import MoveToCommand
from rayforge.pipeline.encoder.textureencoder import TextureEncoder


class TestTextureEncoder:
    """Test suite for TextureEncoder class."""

    @pytest.fixture
    def encoder(self) -> TextureEncoder:
        """Provides a TextureEncoder instance for testing."""
        return TextureEncoder()

    @pytest.fixture
    def default_width_px(self) -> int:
        """Default width for test textures."""
        return 100

    @pytest.fixture
    def default_height_px(self) -> int:
        """Default height for test textures."""
        return 100

    @pytest.fixture
    def default_px_per_mm(self) -> tuple[float, float]:
        """Default resolution for test textures."""
        return (10.0, 10.0)

    @pytest.fixture
    def safe_y_mm(self) -> float:
        """A safe Y coordinate that won't map out of bounds."""
        return 1.0

    def test_encode_empty_ops(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Encoding an empty Ops object should return a zero-filled buffer."""
        ops = Ops()
        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8
        expected = np.zeros(
            (default_height_px, default_width_px), dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_encode_zero_width(
        self,
        encoder: TextureEncoder,
        default_px_per_mm: tuple[float, float],
    ):
        """Encoding with zero width should return an empty array."""
        ops = Ops()
        result = encoder.encode(ops, 0, 100, default_px_per_mm)

        assert result.shape == (0,)
        assert result.dtype == np.uint8

    def test_encode_zero_height(
        self,
        encoder: TextureEncoder,
        default_px_per_mm: tuple[float, float],
    ):
        """Encoding with zero height should return an empty array."""
        ops = Ops()
        result = encoder.encode(ops, 100, 0, default_px_per_mm)

        assert result.shape == (0,)
        assert result.dtype == np.uint8

    def test_encode_negative_dimensions(
        self,
        encoder: TextureEncoder,
        default_px_per_mm: tuple[float, float],
    ):
        """Encoding with negative dimensions should return an empty array."""
        ops = Ops()
        result = encoder.encode(ops, -10, -10, default_px_per_mm)

        assert result.shape == (0,)
        assert result.dtype == np.uint8

    def test_encode_single_scan_line_power_command(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test encoding a single ScanLinePowerCommand."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(10.0, safe_y_mm, 0.0, bytearray([255, 128, 0]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # Y=1mm at 10 px/mm = 10 pixels from bottom
        # In Y-down: height_px - 10 = 90
        expected_y = default_height_px - 10
        # X linspace from 0 to 100 with 3 steps: [0, 50, 100]
        # But X=100 is filtered out (out of bounds)
        assert result[expected_y, 0] == 255
        assert result[expected_y, 50] == 128

    def test_encode_scan_line_with_move_to(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test encoding with a MoveToCommand before ScanLinePowerCommand."""
        ops = Ops()
        ops.move_to(5.0, 5.0, 0.0)
        ops.scan_to(10.0, 5.0, 0.0, bytearray([200, 150, 100]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # Y=5mm at 10 px/mm = 50 pixels from bottom
        # In Y-down: height_px - 50 = 50
        expected_y = default_height_px - 50
        # X linspace from 50 to 100 with 3 steps: [50, 75, 100]
        # X=100 is filtered out
        assert result[expected_y, 50] == 200
        assert result[expected_y, 75] == 150

    def test_encode_scan_line_empty_power_values(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test ScanLinePowerCommand with empty power_values."""
        ops = Ops()
        ops.scan_to(10.0, 0.0, 0.0, bytearray())

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8
        expected = np.zeros(
            (default_height_px, default_width_px), dtype=np.uint8
        )
        np.testing.assert_array_equal(result, expected)

    def test_encode_multiple_scan_lines(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test encoding multiple ScanLinePowerCommands."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(10.0, safe_y_mm, 0.0, bytearray([255]))
        ops.move_to(0.0, 3.0, 0.0)
        ops.scan_to(10.0, 3.0, 0.0, bytearray([128]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # First line at Y=1mm = 10 pixels from bottom
        assert result[default_height_px - 10, 0] == 255
        # Second line at Y=3mm = 30 pixels from bottom
        assert result[default_height_px - 30, 0] == 128

    def test_coordinate_conversion_y_up_to_y_down(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test Y coordinates are correctly converted from Y-up to Y-down."""
        ops = Ops()
        ops.move_to(0.0, 5.0, 0.0)
        ops.scan_to(5.0, 5.0, 0.0, bytearray([200]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        # Y=5mm at 10 px/mm = 50 pixels from origin
        # In Y-down: height_px - 50 = 50
        expected_y = default_height_px - 50
        assert result[expected_y, 0] == 200

    def test_boundary_filtering_out_of_bounds(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test that coordinates outside buffer are filtered out."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(15.0, safe_y_mm, 0.0, bytearray([255] * 150))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # Check that some pixels were set but not all
        expected_y = default_height_px - 10
        assert np.any(result[expected_y, :] > 0)

    def test_boundary_filtering_negative_coordinates(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test that negative coordinates are filtered out."""
        ops = Ops()
        ops.move_to(-5.0, safe_y_mm, 0.0)
        ops.scan_to(5.0, safe_y_mm, 0.0, bytearray([255] * 100))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # Check that some pixels were set
        expected_y = default_height_px - 10
        assert np.any(result[expected_y, :] > 0)

    def test_different_px_per_mm(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
    ):
        """Test encoding with different pixels per millimeter values."""
        ops = Ops()
        ops.move_to(0.0, 4.0, 0.0)
        ops.scan_to(4.0, 4.0, 0.0, bytearray([200]))

        px_per_mm = (20.0, 20.0)
        result = encoder.encode(
            ops, default_width_px, default_height_px, px_per_mm
        )

        # Check that some pixels were set with the expected power value
        assert np.any(result == 200)

    def test_non_square_px_per_mm(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
    ):
        """Test encoding with different X and Y resolutions."""
        ops = Ops()
        ops.move_to(0.0, 5.0, 0.0)
        ops.scan_to(5.0, 5.0, 0.0, bytearray([150]))

        px_per_mm = (10.0, 5.0)
        result = encoder.encode(
            ops, default_width_px, default_height_px, px_per_mm
        )

        # Check that some pixels were set with the expected power value
        assert np.any(result == 150)

    def test_current_position_updates(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test that current position is updated correctly after commands."""
        ops = Ops()
        ops.move_to(0.0, 5.0, 0.0)
        ops.scan_to(5.0, 5.0, 0.0, bytearray([100]))
        ops.scan_to(10.0, 5.0, 0.0, bytearray([200]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 50
        # First scan line starts at origin (0, 5)
        assert result[expected_y, 0] == 100
        # Second scan line starts at (5, 5) - end of first
        assert result[expected_y, 50] == 200

    def test_move_to_updates_current_position(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test that MoveToCommand updates current position."""
        ops = Ops()
        ops.move_to(9.0, 9.0, 0.0)
        ops.scan_to(10.0, 9.0, 0.0, bytearray([180]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 90
        # Scan line starts at (9, 9) = (90, 90) px
        assert result[expected_y, 90] == 180

    def test_power_values_bytearray_handling(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test that power_values bytearray is correctly converted."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(5.0, safe_y_mm, 0.0, bytearray([0, 64, 128, 192, 255]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 10
        # X linspace from 0 to 50 with 5 steps: [0, 12, 25, 37, 50]
        assert result[expected_y, 0] == 0
        assert result[expected_y, 12] == 64
        assert result[expected_y, 25] == 128
        assert result[expected_y, 37] == 192
        assert result[expected_y, 50] == 255

    def test_diagonal_scan_line(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test encoding a diagonal scan line."""
        ops = Ops()
        ops.move_to(0.0, 5.0, 0.0)
        ops.scan_to(5.0, 5.0, 0.0, bytearray([100, 150, 200]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

        # Check that some pixels were set
        assert np.any(result > 0)

    def test_move_to_without_end(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test MoveToCommand without end attribute."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        move_cmd = MoveToCommand(end=None)
        ops.commands.append(move_cmd)
        ops.scan_to(5.0, safe_y_mm, 0.0, bytearray([100]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        # MoveTo with None doesn't update position, so scan line
        # starts from the previous position (0, safe_y_mm)
        expected_y = default_height_px - 10
        assert result[expected_y, 0] == 100

    def test_z_coordinate_ignored(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test that Z coordinate is ignored in encoding."""
        ops = Ops()
        ops.move_to(0.0, 5.0, 0.0)
        ops.scan_to(5.0, 5.0, 10.0, bytearray([120]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 50
        assert result[expected_y, 0] == 120

    def test_large_power_values_array(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
    ):
        """Test encoding with a large power values array."""
        ops = Ops()
        power_values = bytearray([i % 256 for i in range(1000)])
        ops.scan_to(100.0, 0.0, 0.0, power_values)

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        assert result.shape == (default_height_px, default_width_px)
        assert result.dtype == np.uint8

    def test_overlapping_scan_lines(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test that overlapping scan lines write to same pixels."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(5.0, safe_y_mm, 0.0, bytearray([100]))
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(5.0, safe_y_mm, 0.0, bytearray([200]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        # Last write should win
        expected_y = default_height_px - 10
        assert result[expected_y, 0] == 200

    def test_all_zero_power_values(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test encoding with all zero power values."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(10.0, safe_y_mm, 0.0, bytearray([0, 0, 0]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 10
        # X linspace from 0 to 100 with 3 steps: [0, 50, 100]
        # X=100 is filtered out
        assert result[expected_y, 0] == 0
        assert result[expected_y, 50] == 0

    def test_all_max_power_values(
        self,
        encoder: TextureEncoder,
        default_width_px: int,
        default_height_px: int,
        default_px_per_mm: tuple[float, float],
        safe_y_mm: float,
    ):
        """Test encoding with all maximum power values."""
        ops = Ops()
        ops.move_to(0.0, safe_y_mm, 0.0)
        ops.scan_to(10.0, safe_y_mm, 0.0, bytearray([255, 255, 255]))

        result = encoder.encode(
            ops, default_width_px, default_height_px, default_px_per_mm
        )

        expected_y = default_height_px - 10
        # X linspace from 0 to 100 with 3 steps: [0, 50, 100]
        # X=100 is filtered out
        assert result[expected_y, 0] == 255
        assert result[expected_y, 50] == 255
