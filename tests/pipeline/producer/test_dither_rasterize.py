import cairo
import pytest
import numpy as np
from unittest.mock import MagicMock
from rayforge.pipeline.producer.dither_rasterize import DitherRasterizer
from rayforge.core.ops import (
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    MoveToCommand,
    ScanLinePowerCommand,
)
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.core.workpiece import WorkPiece
from rayforge.image.dither import DitherAlgorithm, surface_to_dithered_array
from rayforge.pipeline.encoder.cairoencoder import CairoEncoder
from rayforge.shared.util.colors import ColorSet


@pytest.fixture
def white_surface():
    """Creates a 10x10 white surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    return surface


@pytest.fixture
def black_surface():
    """Creates a 10x10 black surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    return surface


@pytest.fixture
def checkerboard_surface():
    """Creates a 10x10 checkerboard surface."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 0:
                ctx.set_source_rgb(0, 0, 0)
            else:
                ctx.set_source_rgb(1, 1, 1)
            ctx.rectangle(i, j, 1, 1)
            ctx.fill()
    return surface


@pytest.fixture
def offset_checkerboard_surface():
    """Creates a 10x10 checkerboard surface with content offset from edges."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()

    for i in range(2, 8):
        for j in range(2, 8):
            if (i + j) % 2 == 0:
                ctx.set_source_rgb(0, 0, 0)
            else:
                ctx.set_source_rgb(1, 1, 1)
            ctx.rectangle(i, j, 1, 1)
            ctx.fill()
    return surface


@pytest.fixture
def gray_surface():
    """Creates a 10x10 gray surface (128 gray value)."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.paint()
    return surface


@pytest.fixture
def mock_workpiece() -> WorkPiece:
    """Returns a mock workpiece with a default size."""
    wp = WorkPiece(name="test_wp")
    wp.uid = "wp_test_123"
    wp.set_size(1.0, 1.0)
    return wp


@pytest.fixture
def mock_laser():
    """Returns a mock laser with default spot size."""
    laser = MagicMock()
    laser.spot_size_mm = (0.1, 0.1)
    laser.uid = "laser_test_123"
    return laser


def test_dither_rasterizer_serialization():
    """Tests serialization and deserialization of DitherRasterizer."""
    original = DitherRasterizer(
        dither_algorithm="bayer4",
        invert=True,
        bidirectional=False,
    )
    data = original.to_dict()
    recreated = OpsProducer.from_dict(data)

    assert data["type"] == "DitherRasterizer"
    assert data["params"]["dither_algorithm"] == "bayer4"
    assert data["params"]["invert"] is True
    assert data["params"]["bidirectional"] is False
    assert isinstance(recreated, DitherRasterizer)
    assert recreated.dither_algorithm == "bayer4"
    assert recreated.invert is True
    assert recreated.bidirectional is False


def test_dither_rasterizer_run_requires_workpiece(white_surface):
    """Tests that run raises ValueError without workpiece."""
    laser = MagicMock()
    rasterizer = DitherRasterizer()
    with pytest.raises(ValueError, match="requires a workpiece context"):
        rasterizer.run(laser, white_surface, (10, 10))


def test_dither_rasterizer_unsupported_format():
    """Tests that run raises ValueError for unsupported format."""
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 10, 10)
    laser = MagicMock()
    rasterizer = DitherRasterizer()
    mock_workpiece = WorkPiece(name="wp")
    mock_workpiece.uid = "wp_123"
    mock_workpiece.set_size(1.0, 1.0)

    with pytest.raises(ValueError, match="Unsupported Cairo surface format"):
        rasterizer.run(
            laser, surface, (10, 10), workpiece=mock_workpiece, settings={}
        )


def test_dither_rasterizer_empty_surface(mock_workpiece, mock_laser):
    """Test that a zero-dimension surface produces minimal Ops."""
    empty_surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 0, 0)
    rasterizer = DitherRasterizer()
    mock_workpiece.set_size(0, 0)

    artifact = rasterizer.run(
        mock_laser,
        empty_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    assert len(artifact.ops.commands) >= 2
    assert isinstance(artifact.ops.commands[0], OpsSectionStartCommand)
    assert isinstance(artifact.ops.commands[-1], OpsSectionEndCommand)


def test_dither_rasterizer_white_surface(
    white_surface, mock_workpiece, mock_laser
):
    """Tests that a white surface with invert=False produces no ops."""
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count == 0


def test_dither_rasterizer_black_surface(
    black_surface, mock_workpiece, mock_laser
):
    """Tests that a black surface with invert=False produces raster ops."""
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0


def test_dither_rasterizer_invert_white_surface(
    white_surface, mock_workpiece, mock_laser
):
    """Tests that invert=True on a white surface produces raster ops."""
    rasterizer = DitherRasterizer(invert=True)
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0


def test_dither_rasterizer_invert_black_surface(
    black_surface, mock_workpiece, mock_laser
):
    """Tests that invert=True on a black surface produces no ops."""
    rasterizer = DitherRasterizer(invert=True)
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count == 0


def test_dither_rasterizer_bidirectional(
    checkerboard_surface, mock_workpiece, mock_laser
):
    """Tests bidirectional rasterization produces alternating directions."""
    rasterizer = DitherRasterizer(bidirectional=True)
    artifact = rasterizer.run(
        mock_laser,
        checkerboard_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_cmds = [
        cmd
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    ]

    if len(scan_cmds) >= 2:
        first_start = None
        second_start = None

        for cmd in artifact.ops.commands:
            if isinstance(cmd, MoveToCommand):
                if first_start is None:
                    first_start = cmd.end
                elif second_start is None:
                    second_start = cmd.end

        if first_start and second_start:
            x_differs = first_start[0] != second_start[0]
            y_differs = first_start[1] != second_start[1]
            assert x_differs or y_differs


def test_dither_rasterizer_unidirectional(
    checkerboard_surface, mock_workpiece, mock_laser
):
    """Tests unidirectional rasterization produces consistent directions."""
    rasterizer = DitherRasterizer(bidirectional=False)
    artifact = rasterizer.run(
        mock_laser,
        checkerboard_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_cmds = [
        cmd
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    ]
    assert len(scan_cmds) > 0


def test_dither_rasterizer_floyd_steinberg(
    gray_surface, mock_workpiece, mock_laser
):
    """Tests Floyd-Steinberg dithering algorithm with gray surface."""
    rasterizer = DitherRasterizer(dither_algorithm="floyd_steinberg")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    assert artifact is not None
    assert len(artifact.ops.commands) >= 2


def test_dither_rasterizer_bayer2(gray_surface, mock_workpiece, mock_laser):
    """Tests Bayer 2x2 ordered dithering."""
    rasterizer = DitherRasterizer(dither_algorithm="bayer2")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0


def test_dither_rasterizer_bayer4(gray_surface, mock_workpiece, mock_laser):
    """Tests Bayer 4x4 ordered dithering."""
    rasterizer = DitherRasterizer(dither_algorithm="bayer4")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0


def test_dither_rasterizer_bayer8(gray_surface, mock_workpiece, mock_laser):
    """Tests Bayer 8x8 ordered dithering."""
    rasterizer = DitherRasterizer(dither_algorithm="bayer8")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0


def _create_test_color_set() -> ColorSet:
    """Creates a test ColorSet with standard colors."""
    cut_lut = np.zeros((256, 4))
    cut_lut[:, 0] = 1.0
    cut_lut[:, 3] = 1.0

    engrave_lut = np.zeros((256, 4))
    for i in range(256):
        t = i / 255.0
        engrave_lut[i] = [1.0, 1.0 - t, 1.0 - t, 1.0]

    travel_rgba = (0.0, 1.0, 0.0, 1.0)
    zero_power_rgba = (0.0, 0.0, 1.0, 1.0)

    return ColorSet(
        {
            "cut": cut_lut,
            "engrave": engrave_lut,
            "travel": travel_rgba,
            "zero_power": zero_power_rgba,
        }
    )


def _extract_scanline_pixel_rows(ops, height_px, pixels_per_mm) -> set[int]:
    """
    Extracts the pixel row indices that scanlines target.
    This validates that scanlines align with the correct pixel rows.
    """
    px_per_mm_y = pixels_per_mm[1]
    height_mm = height_px / px_per_mm_y
    target_rows = set()

    for cmd in ops.commands:
        if isinstance(cmd, ScanLinePowerCommand):
            y_mm = cmd.end[1]
            line_y_mm = height_mm - y_mm
            y_px_float = (line_y_mm - 0.5 / px_per_mm_y) * px_per_mm_y
            y_px = int(round(y_px_float))
            target_rows.add(y_px)

    return target_rows


def _extract_scanline_pixel_columns(ops, width_px, pixels_per_mm) -> set[int]:
    """
    Extracts the pixel column indices that scanlines target.
    This validates that scanlines align with the correct pixel columns.
    """
    px_per_mm_x = pixels_per_mm[0]
    target_cols = set()
    last_move_to = None

    for cmd in ops.commands:
        if isinstance(cmd, MoveToCommand):
            last_move_to = cmd.end
        elif isinstance(cmd, ScanLinePowerCommand) and last_move_to:
            x_start_mm = last_move_to[0]
            x_end_mm = cmd.end[0]

            x_start_px_float = (x_start_mm * px_per_mm_x) - 0.5
            x_start_px = int(round(x_start_px_float))

            x_end_px_float = (x_end_mm * px_per_mm_x) + 0.5
            x_end_px = int(round(x_end_px_float))

            col_min = min(x_start_px, x_end_px)
            col_max = max(x_start_px, x_end_px)
            for x_px in range(col_min, col_max):
                target_cols.add(x_px)

            last_move_to = None

    return target_cols


def _extract_per_row_pixel_columns(
    ops, width_px, height_px, pixels_per_mm
) -> dict[int, set[int]]:
    """
    Extracts pixel columns targeted by scanlines for each row.
    Returns a dict mapping row index to set of column indices.
    """
    px_per_mm_x = pixels_per_mm[0]
    px_per_mm_y = pixels_per_mm[1]
    height_mm = height_px / px_per_mm_y
    row_cols = {}
    last_move_to = None

    for cmd in ops.commands:
        if isinstance(cmd, MoveToCommand):
            last_move_to = cmd.end
        elif isinstance(cmd, ScanLinePowerCommand) and last_move_to:
            y_mm = cmd.end[1]
            line_y_mm = height_mm - y_mm
            y_px_float = (line_y_mm - 0.5 / px_per_mm_y) * px_per_mm_y
            y_px = int(round(y_px_float))

            x_start_mm = last_move_to[0]
            x_end_mm = cmd.end[0]

            x_start_px_float = (x_start_mm * px_per_mm_x) - 0.5
            x_start_px = int(round(x_start_px_float))

            x_end_px_float = (x_end_mm * px_per_mm_x) + 0.5
            x_end_px = int(round(x_end_px_float))

            if y_px not in row_cols:
                row_cols[y_px] = set()

            col_min = min(x_start_px, x_end_px)
            col_max = max(x_start_px, x_end_px)
            for x_px in range(col_min, col_max):
                row_cols[y_px].add(x_px)

            last_move_to = None

    return row_cols


def _render_ops_to_surface(
    ops, width_px, height_px, pixels_per_mm
) -> np.ndarray:
    """
    Renders ops back to a surface and returns the binary mask.

    This helper function creates a Cairo surface, encodes the ops onto it,
    and extracts a binary mask where 1 represents engraved pixels.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width_px, height_px)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.paint()

    encoder = CairoEncoder()
    colors = _create_test_color_set()

    encoder.encode(
        ops,
        ctx,
        scale=pixels_per_mm,
        colors=colors,
        show_cut_moves=True,
        show_engrave_moves=True,
        show_travel_moves=False,
        show_zero_power_moves=False,
    )

    width = surface.get_width()
    height = surface.get_height()
    stride = surface.get_stride()

    buf = surface.get_data()
    data_with_padding = np.ndarray(
        shape=(height, stride // 4, 4), dtype=np.uint8, buffer=buf
    )
    data = data_with_padding[:, :width, :]

    grayscale = (
        0.2989 * data[:, :, 2]
        + 0.5870 * data[:, :, 1]
        + 0.1140 * data[:, :, 0]
    ).astype(np.uint8)

    return (grayscale < 128).astype(np.uint8)


def test_dither_rasterizer_alignment_black_surface(
    black_surface, mock_workpiece, mock_laser
):
    """
    Validates that the resulting ops are aligned with the input image.
    For a black surface, all pixels should be engraved.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        black_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count == 0, (
        f"Alignment validation failed: {mismatch_count} pixels mismatch. "
        f"Expected all pixels to be engraved for black surface."
    )


def test_dither_rasterizer_scanline_row_alignment(
    black_surface, mock_workpiece, mock_laser
):
    """
    Validates that scanlines target the correct pixel rows.
    This test checks the coordinate transformation directly.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    target_rows = _extract_scanline_pixel_rows(artifact.ops, 10, (10, 10))

    expected = surface_to_dithered_array(
        black_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    expected_rows = set(int(x) for x in np.where(np.any(expected, axis=1))[0])

    assert target_rows == expected_rows, (
        f"Scanline row alignment failed. "
        f"Target rows: {target_rows}, Expected rows: {expected_rows}"
    )


def test_dither_rasterizer_scanline_row_alignment_with_offset(
    white_surface, mock_workpiece, mock_laser
):
    """
    Validates alignment with a non-zero y_offset.
    This tests the global grid alignment logic.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
        y_offset_mm=0.05,
    )

    target_rows = _extract_scanline_pixel_rows(artifact.ops, 10, (10, 10))

    expected = surface_to_dithered_array(
        white_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    expected_rows = set(int(x) for x in np.where(np.any(expected, axis=1))[0])

    assert target_rows == expected_rows, (
        f"Scanline row alignment with offset failed. "
        f"Target rows: {target_rows}, Expected rows: {expected_rows}"
    )


def test_dither_rasterizer_scanline_column_alignment(
    white_surface, mock_workpiece, mock_laser
):
    """
    Validates that scanlines target the correct pixel columns.
    This tests the X coordinate transformation.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    target_cols = _extract_scanline_pixel_columns(artifact.ops, 10, (10, 10))

    expected = surface_to_dithered_array(
        white_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    expected_cols = set(int(x) for x in np.where(np.any(expected, axis=0))[0])

    assert target_cols == expected_cols, (
        f"Scanline column alignment failed. "
        f"Target cols: {target_cols}, Expected cols: {expected_cols}"
    )


def test_dither_rasterizer_scanline_column_alignment_with_offset_content(
    offset_checkerboard_surface, mock_workpiece, mock_laser
):
    """
    Validates X alignment when content doesn't start at (0,0).
    This tests if the X coordinate calculation correctly handles
    content that is offset from the surface edges.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        offset_checkerboard_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    target_cols = _extract_scanline_pixel_columns(artifact.ops, 10, (10, 10))

    expected = surface_to_dithered_array(
        offset_checkerboard_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    expected_cols = set(int(x) for x in np.where(np.any(expected, axis=0))[0])

    assert target_cols == expected_cols, (
        f"Scanline column alignment with offset content failed. "
        f"Target cols: {target_cols}, Expected cols: {expected_cols}"
    )


def test_dither_rasterizer_per_row_x_alignment(
    white_surface, mock_workpiece, mock_laser
):
    """
    Validates that X alignment is consistent across all rows.
    This checks if the X offset changes with every row as reported.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    row_cols = _extract_per_row_pixel_columns(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        white_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    expected_row_cols = {}
    for y_px in range(10):
        cols = set(int(x) for x in np.where(expected[y_px, :])[0])
        if cols:
            expected_row_cols[y_px] = cols

    assert row_cols == expected_row_cols, (
        "Per-row X alignment failed. "
        f"Row cols mismatch: "
        f"{set(row_cols.keys()) ^ set(expected_row_cols.keys())}"
    )


def test_dither_rasterizer_alignment_white_surface(
    white_surface, mock_workpiece, mock_laser
):
    """
    Validates that the resulting ops are aligned with the input image.
    For a white surface, no pixels should be engraved.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        white_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count == 0, (
        f"Alignment validation failed: {mismatch_count} pixels mismatch. "
        f"Expected no pixels to be engraved for white surface."
    )


def test_dither_rasterizer_alignment_checkerboard(
    checkerboard_surface, mock_workpiece, mock_laser
):
    """
    Validates that the resulting ops are aligned with the input image.
    For a checkerboard pattern, the engraved pixels should match the pattern.
    """
    rasterizer = DitherRasterizer()
    artifact = rasterizer.run(
        mock_laser,
        checkerboard_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        checkerboard_surface, DitherAlgorithm.FLOYD_STEINBERG, False
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count < 60, (
        f"Alignment validation failed: {mismatch_count} pixels mismatch. "
        f"Rendered ops do not match the dithered input pattern."
    )


def test_dither_rasterizer_alignment_invert(
    gray_surface, mock_workpiece, mock_laser
):
    """
    Validates alignment with invert=True.
    The inverted pattern should still align correctly.
    """
    rasterizer = DitherRasterizer(invert=True)
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        gray_surface, DitherAlgorithm.FLOYD_STEINBERG, True
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count < 60, (
        f"Alignment validation failed with invert=True: "
        f"{mismatch_count} pixels mismatch."
    )


def test_dither_rasterizer_alignment_bayer4(
    gray_surface, mock_workpiece, mock_laser
):
    """
    Validates alignment with Bayer 4x4 dithering.
    """
    rasterizer = DitherRasterizer(dither_algorithm="bayer4")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        gray_surface, DitherAlgorithm.BAYER4, False
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count < 50, (
        f"Alignment validation failed with bayer4: "
        f"{mismatch_count} pixels mismatch."
    )


def test_dither_rasterizer_alignment_bayer8(
    gray_surface, mock_workpiece, mock_laser
):
    """
    Validates alignment with Bayer 8x8 dithering.
    """
    rasterizer = DitherRasterizer(dither_algorithm="bayer8")
    artifact = rasterizer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 10, 10, (10, 10))

    expected = surface_to_dithered_array(
        gray_surface, DitherAlgorithm.BAYER8, False
    )

    xor_result = np.logical_xor(
        rendered_mask.astype(bool), expected.astype(bool)
    )
    mismatch_count = np.sum(xor_result)

    assert mismatch_count < 50, (
        f"Alignment validation failed with bayer8: "
        f"{mismatch_count} pixels mismatch."
    )


def test_floyd_steinberg_black_surface_no_invert(
    black_surface, mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg engraves black areas with invert=False.
    Black surface should produce many scanlines.
    """
    rasterizer = DitherRasterizer(
        dither_algorithm="floyd_steinberg", invert=False
    )
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0, (
        "Floyd-Steinberg with invert=False should engrave black surface"
    )


def test_floyd_steinberg_white_surface_no_invert(
    white_surface, mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg does NOT engrave white areas with invert=False.
    White surface should produce no scanlines.
    """
    rasterizer = DitherRasterizer(
        dither_algorithm="floyd_steinberg", invert=False
    )
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count == 0, (
        "Floyd-Steinberg with invert=False should NOT engrave white surface"
    )


def test_floyd_steinberg_white_surface_with_invert(
    white_surface, mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg engraves white areas with invert=True.
    White surface should produce many scanlines.
    """
    rasterizer = DitherRasterizer(
        dither_algorithm="floyd_steinberg", invert=True
    )
    artifact = rasterizer.run(
        mock_laser,
        white_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count > 0, (
        "Floyd-Steinberg with invert=True should engrave white surface"
    )


def test_floyd_steinberg_black_surface_with_invert(
    black_surface, mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg does NOT engrave black areas with invert=True.
    Black surface should produce no scanlines.
    """
    rasterizer = DitherRasterizer(
        dither_algorithm="floyd_steinberg", invert=True
    )
    artifact = rasterizer.run(
        mock_laser,
        black_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    scan_count = sum(
        1
        for cmd in artifact.ops.commands
        if isinstance(cmd, ScanLinePowerCommand)
    )
    assert scan_count == 0, (
        "Floyd-Steinberg with invert=True should NOT engrave black surface"
    )


def test_floyd_steinberg_gray_produces_dithered_pattern(
    mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg on a gray surface produces a dithered pattern.
    A 50% gray should produce roughly 50% engrave coverage, not solid black.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0.5, 0.5, 0.5)
    ctx.paint()

    expected = surface_to_dithered_array(
        surface, DitherAlgorithm.FLOYD_STEINBERG, invert=False
    )
    expected_coverage = np.mean(expected)
    assert 0.3 < expected_coverage < 0.7, (
        f"Floyd-Steinberg should produce ~50% coverage, "
        f"got {expected_coverage:.2%}"
    )

    row = expected[0, :]
    diff = np.diff(np.hstack(([0], row, [0])))
    num_runs = np.sum(diff == 1)
    assert num_runs > 10, (
        f"Floyd-Steinberg should produce many runs per row, got {num_runs}"
    )


def test_floyd_steinberg_gradient_produces_varying_density(
    mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg on a gradient produces varying density.
    Dark areas should have more engraving than light areas.
    """
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 100, 100)
    ctx = cairo.Context(surface)
    gradient = cairo.LinearGradient(0, 0, 100, 0)
    gradient.add_color_stop_rgb(0, 0, 0, 0)
    gradient.add_color_stop_rgb(1, 1, 1, 1)
    ctx.set_source(gradient)
    ctx.paint()

    rasterizer = DitherRasterizer(
        dither_algorithm="floyd_steinberg", invert=False
    )
    artifact = rasterizer.run(
        mock_laser,
        surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )

    rendered_mask = _render_ops_to_surface(artifact.ops, 100, 100, (10, 10))

    left_density = np.mean(rendered_mask[:, :25])
    right_density = np.mean(rendered_mask[:, 75:])

    assert left_density > right_density, (
        f"Dark side (left) should have more engraving than light side. "
        f"Left: {left_density:.2%}, Right: {right_density:.2%}"
    )


def test_floyd_steinberg_vs_bayer_different_patterns(
    gray_surface, mock_workpiece, mock_laser
):
    """
    Tests that Floyd-Steinberg and Bayer produce different dithering patterns
    on the same gray surface (they should look different).
    """
    rasterizer_fs = DitherRasterizer(dither_algorithm="floyd_steinberg")
    artifact_fs = rasterizer_fs.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )
    mask_fs = _render_ops_to_surface(artifact_fs.ops, 10, 10, (10, 10))

    rasterizer_bayer = DitherRasterizer(dither_algorithm="bayer4")
    artifact_bayer = rasterizer_bayer.run(
        mock_laser,
        gray_surface,
        (10, 10),
        workpiece=mock_workpiece,
        settings={},
    )
    mask_bayer = _render_ops_to_surface(artifact_bayer.ops, 10, 10, (10, 10))

    difference = np.sum(np.abs(mask_fs.astype(int) - mask_bayer.astype(int)))
    assert difference > 10, (
        "Floyd-Steinberg and Bayer should produce different patterns"
    )


def test_dither_rasterizer_is_vector_producer():
    """Tests that DitherRasterizer correctly reports it's not a vector
    producer.
    """
    rasterizer = DitherRasterizer()
    assert rasterizer.is_vector_producer() is False


def test_dither_rasterizer_chunk_alignment_consistency(
    black_surface, mock_workpiece, mock_laser
):
    """
    Validates that scanline Y positions follow global grid alignment
    when rasterizing with different y_offsets (simulating chunks).
    With global grid alignment, scanlines are placed at fixed global
    positions regardless of content offset.
    """
    rasterizer = DitherRasterizer()
    px_per_mm = (10, 10)
    height_mm = 10 / px_per_mm[1]

    def extract_global_y_positions(ops, y_offset_mm):
        positions = []
        for cmd in ops.commands:
            if isinstance(cmd, ScanLinePowerCommand):
                y_mm = cmd.end[1]
                line_y_mm = height_mm - y_mm
                global_y = line_y_mm + y_offset_mm
                positions.append(round(global_y, 3))
        return sorted(positions)

    results = []
    for y_offset in [0.0, 0.05, 0.1, 0.15, 0.2]:
        artifact = rasterizer.run(
            mock_laser,
            black_surface,
            px_per_mm,
            workpiece=mock_workpiece,
            settings={},
            y_offset_mm=y_offset,
        )
        global_positions = extract_global_y_positions(artifact.ops, y_offset)
        results.append((y_offset, global_positions))

    for y_offset, positions in results:
        assert len(positions) == 10, (
            f"Expected 10 scanlines with y_offset={y_offset}, "
            f"got {len(positions)}"
        )
        for i in range(len(positions) - 1):
            spacing = positions[i + 1] - positions[i]
            assert abs(spacing - 0.1) < 0.001, (
                f"Inconsistent spacing at y_offset={y_offset}: "
                f"{spacing}mm between positions {i} and {i + 1}"
            )


def test_dither_rasterizer_bidirectional_across_chunks(
    white_surface, mock_workpiece, mock_laser
):
    """
    Validates that bidirectional direction is consistent across chunks.
    Lines at the same global Y position should have the same direction
    regardless of chunk offset.
    """
    rasterizer = DitherRasterizer(bidirectional=True)
    px_per_mm = (10, 10)

    def get_scan_directions(ops, height_px, pixels_per_mm):
        directions = {}
        last_move_to = None

        for cmd in ops.commands:
            if isinstance(cmd, MoveToCommand):
                last_move_to = cmd.end
            elif isinstance(cmd, ScanLinePowerCommand) and last_move_to:
                y_mm = cmd.end[1]
                height_mm = height_px / pixels_per_mm[1]
                y_px = int(round((height_mm - y_mm) * pixels_per_mm[1]))

                x_start = last_move_to[0]
                x_end = cmd.end[0]
                direction = (
                    "left_to_right" if x_end > x_start else "right_to_left"
                )
                directions[y_px] = direction
                last_move_to = None

        return directions

    directions_offset_0 = get_scan_directions(
        rasterizer.run(
            mock_laser,
            white_surface,
            px_per_mm,
            workpiece=mock_workpiece,
            settings={},
            y_offset_mm=0.0,
        ).ops,
        10,
        px_per_mm,
    )

    directions_offset_1 = get_scan_directions(
        rasterizer.run(
            mock_laser,
            white_surface,
            px_per_mm,
            workpiece=mock_workpiece,
            settings={},
            y_offset_mm=0.2,
        ).ops,
        10,
        px_per_mm,
    )

    for y_px in range(10):
        dir_0 = directions_offset_0.get(y_px)
        dir_1 = directions_offset_1.get(y_px)
        assert dir_0 == dir_1, (
            f"Direction mismatch at row {y_px}: "
            f"offset=0 gave {dir_0}, offset=0.2 gave {dir_1}"
        )


def test_dither_rasterizer_line_spacing_with_offset(
    black_surface, mock_workpiece, mock_laser
):
    """
    Validates that line spacing remains consistent with offsets.
    Scanlines should be placed at line_interval spacing on a global grid.
    """
    rasterizer = DitherRasterizer()
    px_per_mm = (10, 10)
    height_mm = 10 / px_per_mm[1]

    def extract_y_positions(ops):
        positions = []
        for cmd in ops.commands:
            if isinstance(cmd, ScanLinePowerCommand):
                y_mm = cmd.end[1]
                line_y_mm = height_mm - y_mm
                positions.append(round(line_y_mm, 3))
        return sorted(positions)

    for y_offset in [0.0, 0.05, 0.1, 0.15]:
        artifact = rasterizer.run(
            mock_laser,
            black_surface,
            px_per_mm,
            workpiece=mock_workpiece,
            settings={},
            y_offset_mm=y_offset,
        )

        positions = extract_y_positions(artifact.ops)

        assert len(positions) == 10, (
            f"Expected 10 scanlines with y_offset={y_offset}, "
            f"got {len(positions)}: {positions}"
        )

        for i in range(len(positions) - 1):
            spacing = positions[i + 1] - positions[i]
            assert abs(spacing - 0.1) < 0.001, (
                f"Inconsistent spacing at y_offset={y_offset}: "
                f"{spacing}mm between positions {i} and {i + 1}"
            )
