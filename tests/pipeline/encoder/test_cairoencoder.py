import pytest
import cairo
import numpy as np
from typing import Tuple

from rayforge.core.ops import (
    Ops,
    SetPowerCommand,
    ArcToCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.encoder.cairoencoder import CairoEncoder

# --- Test Constants ---
WIDTH, HEIGHT = 100, 100
BLACK = (0, 0, 0, 255)
CUT_COLOR_RGB = (1, 0, 1)
TRAVEL_COLOR_RGB = (1.0, 0.4, 0.0)
ZERO_POWER_COLOR_RGB = (0.0, 0.2, 0.9)

# Convert RGB (0-1) to RGBA (0-255) for pixel checking
CUT_COLOR = (255, 0, 255, 255)
TRAVEL_COLOR = (255, int(0.4 * 255), 0, 255)
ZERO_POWER_COLOR = (0, int(0.2 * 255), int(0.9 * 255), 255)


# --- Helper Fixtures and Functions ---
@pytest.fixture
def encoder() -> CairoEncoder:
    """Provides a default CairoEncoder instance."""
    return CairoEncoder()


@pytest.fixture
def surface_and_ctx() -> Tuple[cairo.ImageSurface, cairo.Context]:
    """Provides a standard 100x100 black testing surface and its context."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, WIDTH, HEIGHT)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black background
    ctx.paint()
    return surface, ctx


def get_pixel_data(surface: cairo.ImageSurface) -> np.ndarray:
    """Extracts pixel data from a Cairo surface into a NumPy array."""
    buf = surface.get_data()
    # Cairo uses ARGB32, which is BGRA in memory order for
    # little-endian systems.
    # We reshape and then slice to get a standard RGBA array.
    data = np.ndarray(shape=(HEIGHT, WIDTH, 4), dtype=np.uint8, buffer=buf)
    return data[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA


def assert_pixel_color_approx(
    data: np.ndarray, x: int, y: int, expected_color: Tuple, tolerance=10
):
    """Asserts the color of a single pixel within a tolerance."""
    actual_color = tuple(data[y, x])
    for i in range(len(expected_color)):
        assert abs(actual_color[i] - expected_color[i]) <= tolerance, (
            f"Pixel at ({x}, {y}) color channel {i} was {actual_color[i]}, "
            f"expected {expected_color[i]} (tolerance {tolerance})"
        )


# --- Test Suite ---
class TestCairoEncoder:
    def test_encode_empty_ops(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """Encoding an empty Ops object should not draw anything."""
        surface, ctx = surface_and_ctx
        ops = Ops()
        encoder.encode(ops, ctx, scale=(1.0, 1.0))
        data = get_pixel_data(surface)
        # Verify a corner is still black
        assert_pixel_color_approx(data, 10, 10, BLACK)

    def test_encode_simple_line_with_y_inversion(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """
        Tests that a simple LineTo is drawn with the correct color and
        Y-axis inversion.
        """
        surface, ctx = surface_and_ctx
        ops = Ops()
        ops.set_power(100)
        ops.move_to(10, 10)  # Bottom-left in user space
        ops.line_to(90, 90)  # Top-right in user space

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            drawable_height=HEIGHT,
            use_antialias=False,
        )
        data = get_pixel_data(surface)

        # Y is inverted: (10, 10) in ops space -> (10, 90) in pixel space
        assert_pixel_color_approx(data, 10, HEIGHT - 1 - 10, CUT_COLOR)
        # (90, 90) in ops space -> (90, 10) in pixel space
        assert_pixel_color_approx(data, 90, HEIGHT - 1 - 90, CUT_COLOR)
        # Check that an unrelated pixel is still black
        assert_pixel_color_approx(data, 50, 10, BLACK)

    def test_show_and_hide_travel_moves(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """Tests the `show_travel_moves` flag."""
        # Case 1: Show travel moves
        ops_show = Ops()
        ops_show.move_to(25, 25)
        ops_show.move_to(50, 50)  # A second travel move

        encoder.encode(
            ops_show,
            surface_and_ctx[1],
            scale=(1.0, 1.0),
            travel_color=TRAVEL_COLOR_RGB,
            show_travel_moves=True,
            drawable_height=HEIGHT,
            use_antialias=False,
        )
        data = get_pixel_data(surface_and_ctx[0])
        # The first move from origin (0,0) is skipped, so check the second one.
        assert_pixel_color_approx(data, 37, HEIGHT - 1 - 37, TRAVEL_COLOR)

        # Case 2: Hide travel moves
        surface_hide, ctx_hide = surface_and_ctx
        ops_hide = Ops()
        ops_hide.move_to(50, 50)
        encoder.encode(
            ops_hide,
            ctx_hide,
            scale=(1.0, 1.0),
            travel_color=TRAVEL_COLOR_RGB,
            show_travel_moves=False,
            drawable_height=HEIGHT,
        )
        data_hide = get_pixel_data(surface_hide)
        assert_pixel_color_approx(data_hide, 37, HEIGHT - 1 - 37, TRAVEL_COLOR)

    def test_state_tracking_for_zero_power_lines(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """
        Crucial test for the overscan bug: ensures SetPower(0) correctly
        changes the color for the subsequent LineTo.
        """
        surface, ctx = surface_and_ctx
        ops = Ops()
        # A simulated overscan sequence
        ops.add(SetPowerCommand(100))
        ops.move_to(10, 50)
        ops.line_to(40, 50)  # This should be the cut color
        ops.add(SetPowerCommand(0))
        ops.line_to(
            50, 50
        )  # This is the overscan move, should be zero power color

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            zero_power_color=ZERO_POWER_COLOR_RGB,
            show_travel_moves=True,
            use_antialias=False,
        )
        data = get_pixel_data(surface)

        # The content line
        assert_pixel_color_approx(data, 25, HEIGHT - 1 - 50, CUT_COLOR)
        # The zero-power overscan line
        assert_pixel_color_approx(data, 45, HEIGHT - 1 - 50, ZERO_POWER_COLOR)

    def test_arc_to_draws_an_arc(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """Tests that ArcToCommand is rendered correctly."""
        surface, ctx = surface_and_ctx
        ops = Ops()
        ops.set_power(100)
        ops.move_to(20, 50)
        # 90-degree clockwise arc from (20,50) to (50,20), center (20,20)
        ops.add(
            ArcToCommand(
                end=(50, 20, 0), center_offset=(0, -30), clockwise=True
            )
        )

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            drawable_height=HEIGHT,
            use_antialias=False,
        )
        data = get_pixel_data(surface)

        # Check endpoints
        assert_pixel_color_approx(data, 20, HEIGHT - 1 - 50, CUT_COLOR)
        assert_pixel_color_approx(data, 50, HEIGHT - 1 - 20, CUT_COLOR)
        # Check a point on the arc (45 degrees from start) -> (41.2, 41.2)
        assert_pixel_color_approx(data, 41, HEIGHT - 1 - 41, CUT_COLOR)
        # Check a point inside the arc is empty
        assert_pixel_color_approx(data, 30, HEIGHT - 1 - 40, BLACK)

    def test_scanline_with_mixed_power(
        self,
        encoder: CairoEncoder,
        surface_and_ctx: Tuple[cairo.ImageSurface, cairo.Context],
    ):
        """Tests a ScanLinePowerCommand with on/off segments."""
        surface, ctx = surface_and_ctx
        ops = Ops()
        ops.move_to(10, 50)
        ops.add(
            ScanLinePowerCommand(
                end=(90, 50, 0),
                power_values=bytearray([0, 0, 100, 100, 100, 0]),
            )
        )
        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            zero_power_color=ZERO_POWER_COLOR_RGB,
            show_travel_moves=True,
            use_antialias=False,
        )
        data = get_pixel_data(surface)

        y = HEIGHT - 1 - 50
        # Check first zero-power segment (1/3 of the line)
        assert_pixel_color_approx(data, 20, y, ZERO_POWER_COLOR)
        # Check cutting segment (middle half of the line)
        # It's a gradient from white (power=100) to white, so non-black
        # In RGBA this is (0,0,0,255) as power=100 -> color=0.0
        assert_pixel_color_approx(data, 50, y, (0, 0, 0, 255))
        # Check last zero-power segment
        assert_pixel_color_approx(data, 85, y, ZERO_POWER_COLOR)
