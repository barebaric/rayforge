import pytest
import cairo
import numpy as np
from typing import Tuple, Dict, cast

from rayforge.core.ops import (
    Ops,
    ArcToCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.encoder.cairoencoder import CairoEncoder

# --- Test Constants for Matrix-based Testing ---
# Simplified color palette for matrix readability
CUT_COLOR_RGB = (1, 0, 0)  # Red
TRAVEL_COLOR_RGB = (0, 1, 0)  # Green
ZERO_POWER_COLOR_RGB = (0, 0, 1)  # Blue

# RGBA versions for pixel checking
RED = (255, 0, 0, 255)
GREEN = (0, 255, 0, 255)
BLUE = (0, 0, 255, 255)
BLACK = (0, 0, 0, 255)

# Map characters to colors for matrix assertions
COLOR_MAP: Dict[str, Tuple[int, int, int, int]] = {
    "r": RED,
    "g": GREEN,
    "b": BLUE,
    "k": BLACK,
    " ": BLACK,  # Treat spaces as background for easier matrix writing
}

# Unambiguous reverse map for generating actual matrix strings
CHAR_MAP: Dict[Tuple[int, int, int, int], str] = {
    RED: "r",
    GREEN: "g",
    BLUE: "b",
    BLACK: "k",
}

TEST_PALETTE = list(CHAR_MAP.keys())


# --- Helper Fixtures and Functions ---
@pytest.fixture
def encoder() -> CairoEncoder:
    """Provides a default CairoEncoder instance."""
    return CairoEncoder()


def get_pixel_data(surface: cairo.ImageSurface) -> np.ndarray:
    """Extracts pixel data from a Cairo surface into a NumPy array."""
    buf = surface.get_data()
    height, width = surface.get_height(), surface.get_width()
    # Cairo uses ARGB32, which is BGRA in memory order for little-endian
    # systems.
    data = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=buf)
    return data[:, :, [2, 1, 0, 3]]  # BGRA -> RGBA


def create_surface(
    width: int, height: int
) -> Tuple[cairo.ImageSurface, cairo.Context]:
    """Creates a black testing surface of a given size."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)  # Black background
    ctx.paint()
    return surface, ctx


def find_closest_color_char(pixel_rgba: Tuple[int, int, int, int]) -> str:
    """
    Finds the character for the color in the palette closest to the
    pixel color.
    """
    # If the pixel is very dark, just classify it as black.
    if sum(pixel_rgba[:3]) < 128:
        return "k"

    min_dist_sq = float("inf")
    closest_color_tuple = None

    pr, pg, pb, _ = map(int, pixel_rgba)  # Cast to avoid uint8 overflow

    for color_tuple in TEST_PALETTE:
        cr, cg, cb, _ = color_tuple
        dist_sq = (pr - cr) ** 2 + (pg - cg) ** 2 + (pb - cb) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_color_tuple = color_tuple

    return (
        CHAR_MAP.get(closest_color_tuple, "?") if closest_color_tuple else "?"
    )


def assert_matrix_equals_tolerant(
    surface: cairo.ImageSurface, expected_matrix: str
):
    """
    Asserts that the surface's pixel data matches the expected character
    matrix, tolerating 1-pixel offsets caused by anti-aliasing.
    """
    data = get_pixel_data(surface)
    h, w, _ = data.shape

    expected_lines = [
        line.strip() for line in expected_matrix.strip().split("\n")
    ]
    expected_h = len(expected_lines)
    expected_w = len(expected_lines[0]) if expected_h > 0 else 0

    assert h == expected_h, (
        f"Matrix height mismatch: expected {expected_h}, got {h}"
    )
    assert w == expected_w, (
        f"Matrix width mismatch: expected {expected_w}, got {w}"
    )

    mismatches = []
    actual_matrix_lines = []

    for y in range(h):
        actual_line = ""
        for x in range(w):
            actual_color_tuple = cast(
                Tuple[int, int, int, int], tuple(data[y, x])
            )
            actual_line += find_closest_color_char(actual_color_tuple)
        actual_matrix_lines.append(actual_line)

    for y in range(expected_h):
        for x in range(expected_w):
            expected_char = expected_lines[y][x]
            actual_char = actual_matrix_lines[y][x]

            if expected_char == actual_char:
                continue

            # Check if the mismatch is an acceptable anti-aliasing artifact
            is_artifact = False

            # Case 1: Expected a color, but got background. Check if the color
            # is nearby.
            if expected_char != "k" and actual_char == "k":
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and actual_matrix_lines[ny][nx] == expected_char
                        ):
                            is_artifact = True
                            break
                    if is_artifact:
                        break

            # Case 2: Expected background, but got a color. Check if a nearby
            # expected pixel has that color.
            elif expected_char == "k" and actual_char != "k":
                for dy in range(-1, 2):
                    for dx in range(-1, 2):
                        ny, nx = y + dy, x + dx
                        if (
                            0 <= ny < h
                            and 0 <= nx < w
                            and expected_lines[ny][nx] == actual_char
                        ):
                            is_artifact = True
                            break
                    if is_artifact:
                        break

            if not is_artifact:
                mismatches.append(
                    f"  - At ({x},{y}): expected '{expected_char}', "
                    f"got '{actual_char}'"
                )

    if mismatches:
        header = "Pixel matrix did not match expected matrix."

        # Limit the number of mismatches printed for readability
        max_mismatches = 10
        if len(mismatches) > max_mismatches:
            diff_msg = "\n".join(mismatches[:max_mismatches])
            diff_msg += f"\n  ...and {len(mismatches) - max_mismatches}"
            diff_msg += "more mismatches."
        else:
            diff_msg = "\n".join(mismatches)

        expected_formatted = "\n".join(expected_lines)
        actual_formatted = "\n".join(actual_matrix_lines)

        full_message = (
            f"{header}\n\n"
            f"Mismatches found:\n{diff_msg}\n\n"
            f"Expected Matrix ({expected_w}x{expected_h}):\n"
            f"---\n{expected_formatted}\n---\n"
            f"Actual (Closest Color) Matrix ({w}x{h}):\n"
            f"---\n{actual_formatted}\n---"
        )
        pytest.fail(full_message)


# --- Test Suite ---
class TestCairoEncoder:
    def test_encode_empty_ops(self, encoder: CairoEncoder):
        """Encoding an empty Ops object should result in a black canvas."""
        W, H = 5, 5
        surface, ctx = create_surface(W, H)
        ops = Ops()
        encoder.encode(ops, ctx, scale=(1.0, 1.0))

        expected_matrix = """
        kkkkk
        kkkkk
        kkkkk
        kkkkk
        kkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)

    def test_encode_simple_line_with_y_inversion(self, encoder: CairoEncoder):
        """A diagonal line should be drawn correctly, inverting the Y-axis."""
        W, H = 10, 10
        surface, ctx = create_surface(W, H)
        ops = Ops()
        ops.set_power(100)
        ops.move_to(1, 1)  # User space bottom-left
        ops.line_to(8, 8)  # User space top-right

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            show_cut_moves=True,
            drawable_height=H,
        )

        expected_matrix = """
        kkkkkkkkkk
        kkkkkkkkrk
        kkkkkkkrkk
        kkkkkkrkkk
        kkkkkrkkkk
        kkkkrkkkkk
        kkkrkkkkkk
        kkrkkkkkkk
        krkkkkkkkk
        kkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)

    def test_show_and_hide_travel_moves(self, encoder: CairoEncoder):
        """
        Tests that `show_travel_moves` correctly draws or hides travel
        lines.
        """
        W, H = 10, 10
        ops = Ops()
        ops.move_to(2, 2)
        ops.move_to(7, 7)

        # Case 1: Show travel moves
        surface_show, ctx_show = create_surface(W, H)
        encoder.encode(
            ops,
            ctx_show,
            scale=(1.0, 1.0),
            travel_color=TRAVEL_COLOR_RGB,
            show_travel_moves=True,
            drawable_height=H,
        )

        expected_matrix_show = """
        kkkkkkkkkk
        kkkkkkkkkk
        kkkkkkkgkk
        kkkkkkgkkk
        kkkkkgkkkk
        kkkkgkkkkk
        kkkgkkkkkk
        kkgkkkkkkk
        kkkkkkkkkk
        kkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface_show, expected_matrix_show)

        # Case 2: Hide travel moves
        surface_hide, ctx_hide = create_surface(W, H)
        encoder.encode(
            ops,
            ctx_hide,
            scale=(1.0, 1.0),
            travel_color=TRAVEL_COLOR_RGB,
            show_travel_moves=False,
            drawable_height=H,
        )
        expected_matrix_hide = "k" * W + ("\n" + "k" * W) * (H - 1)
        assert_matrix_equals_tolerant(surface_hide, expected_matrix_hide)

    def test_state_tracking_for_zero_power_lines(self, encoder: CairoEncoder):
        """
        Tests that SetPower(0) correctly changes the color for the next
        line.
        """
        W, H = 10, 5
        surface, ctx = create_surface(W, H)
        ops = Ops()
        ops.set_power(100)
        ops.move_to(1, 2)
        ops.line_to(5, 2)
        ops.set_power(0)
        ops.line_to(9, 2)

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            zero_power_color=ZERO_POWER_COLOR_RGB,
            show_cut_moves=True,
            show_zero_power_moves=True,
            drawable_height=H,
        )

        expected_matrix = """
        kkkkkkkkkk
        kkkkkkkkkk
        krrrrbbbbk
        kkkkkkkkkk
        kkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)

    def test_arc_to_draws_an_arc(self, encoder: CairoEncoder):
        """Tests that ArcToCommand is rendered correctly."""
        W, H = 10, 10
        surface, ctx = create_surface(W, H)
        ops = Ops()
        ops.set_power(100)
        ops.move_to(2, 8)
        ops.add(
            ArcToCommand(end=(8, 2, 0), center_offset=(0, -6), clockwise=True)
        )

        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            cut_color=CUT_COLOR_RGB,
            show_cut_moves=True,
            drawable_height=H,
        )

        expected_matrix = """
        kkkkkkkkkk
        kkrrrkkkkk
        kkkkkrrkkk
        kkkkkkrrkk
        kkkkkkkrkk
        kkkkkkkkrk
        kkkkkkkkrk
        kkkkkkkkrk
        kkkkkkkkkk
        kkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)

    def test_scanline_with_mixed_power(self, encoder: CairoEncoder):
        """Tests a ScanLinePowerCommand with on/off segments."""
        W, H = 12, 5
        surface, ctx = create_surface(W, H)
        ops = Ops()
        ops.move_to(1, 2)
        ops.add(
            ScanLinePowerCommand(
                end=(10, 2, 0),
                power_values=bytearray([0, 255, 0]),
            )
        )
        encoder.encode(
            ops,
            ctx,
            scale=(1.0, 1.0),
            zero_power_color=ZERO_POWER_COLOR_RGB,
            engrave_color=CUT_COLOR_RGB,
            show_engrave_moves=True,
            show_zero_power_moves=True,
            drawable_height=H,
        )

        expected_matrix = """
        kkkkkkkkkkkk
        kkkkkkkkkkkk
        kbbbrrrbbbkk
        kkkkkkkkkkkk
        kkkkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)

    @pytest.mark.parametrize(
        "move_type, show_flag, should_draw",
        [
            ("cut", "show_cut_moves", True),
            ("cut", "show_cut_moves", False),
            ("zero_power", "show_zero_power_moves", True),
            ("zero_power", "show_zero_power_moves", False),
            ("engrave", "show_engrave_moves", True),
            ("engrave", "show_engrave_moves", False),
        ],
    )
    def test_visibility_toggles(
        self, encoder, move_type, show_flag, should_draw
    ):
        """
        Tests that each visibility flag correctly shows or hides its
        corresponding move type.
        """
        W, H = 10, 5
        surface, ctx = create_surface(W, H)
        ops = Ops()

        encode_kwargs = {
            "scale": (1.0, 1.0),
            "cut_color": CUT_COLOR_RGB,
            "engrave_color": CUT_COLOR_RGB,
            "zero_power_color": ZERO_POWER_COLOR_RGB,
            "drawable_height": H,
            show_flag: should_draw,
        }

        expected_char = "k"

        if move_type == "cut":
            ops.set_power(100)
            ops.move_to(1, 2)
            ops.line_to(8, 2)
            if should_draw:
                expected_char = "r"
        elif move_type == "zero_power":
            ops.set_power(0)
            ops.move_to(1, 2)
            ops.line_to(8, 2)
            if should_draw:
                expected_char = "b"
        elif move_type == "engrave":
            ops.move_to(1, 2)
            ops.add(
                ScanLinePowerCommand(
                    end=(8, 2, 0), power_values=bytearray([255] * 7)
                )
            )
            if should_draw:
                expected_char = "r"

        encoder.encode(ops, ctx, **encode_kwargs)

        line = expected_char * 7
        expected_matrix = f"""
        kkkkkkkkkk
        kkkkkkkkkk
        k{line}kk
        kkkkkkkkkk
        kkkkkkkkkk
        """
        assert_matrix_equals_tolerant(surface, expected_matrix)
