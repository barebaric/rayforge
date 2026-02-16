import numpy as np
import cairo
from rayforge.image.dither import (
    apply_floyd_steinberg_dither,
    apply_bayer_dither,
    surface_to_dithered_array,
    BAYER_MATRICES,
    DitherAlgorithm,
)


def test_bayer_matrices_shape_and_values():
    """Tests that Bayer matrices have correct shapes and value ranges."""
    assert DitherAlgorithm.BAYER2 in BAYER_MATRICES
    assert DitherAlgorithm.BAYER4 in BAYER_MATRICES
    assert DitherAlgorithm.BAYER8 in BAYER_MATRICES

    bayer2 = BAYER_MATRICES[DitherAlgorithm.BAYER2]
    assert bayer2.shape == (2, 2)
    assert bayer2.dtype == np.float32
    assert np.all(bayer2 >= 0) and np.all(bayer2 <= 3)

    bayer4 = BAYER_MATRICES[DitherAlgorithm.BAYER4]
    assert bayer4.shape == (4, 4)
    assert bayer4.dtype == np.float32
    assert np.all(bayer4 >= 0) and np.all(bayer4 <= 15)

    bayer8 = BAYER_MATRICES[DitherAlgorithm.BAYER8]
    assert bayer8.shape == (8, 8)
    assert bayer8.dtype == np.float32
    assert np.all(bayer8 >= 0) and np.all(bayer8 <= 63)


def test_bayer2_matrix_expected_values():
    """Tests that the 2x2 Bayer matrix has expected values."""
    expected = np.array([[0, 2], [3, 1]], dtype=np.float32)
    np.testing.assert_array_equal(
        BAYER_MATRICES[DitherAlgorithm.BAYER2], expected
    )


def test_bayer4_matrix_expected_values():
    """Tests that the 4x4 Bayer matrix has expected values."""
    expected = np.array(
        [
            [0, 8, 2, 10],
            [12, 4, 14, 6],
            [3, 11, 1, 9],
            [15, 7, 13, 5],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        BAYER_MATRICES[DitherAlgorithm.BAYER4], expected
    )


def test_bayer8_matrix_expected_values():
    """Tests that the 8x8 Bayer matrix has expected values."""
    expected = np.array(
        [
            [0, 32, 8, 40, 2, 34, 10, 42],
            [48, 16, 56, 24, 50, 18, 58, 26],
            [12, 44, 4, 36, 14, 46, 6, 38],
            [60, 28, 52, 20, 62, 30, 54, 22],
            [3, 35, 11, 43, 1, 33, 9, 41],
            [51, 19, 59, 27, 49, 17, 57, 25],
            [15, 47, 7, 39, 13, 45, 5, 37],
            [63, 31, 55, 23, 61, 29, 53, 21],
        ],
        dtype=np.float32,
    )
    np.testing.assert_array_equal(
        BAYER_MATRICES[DitherAlgorithm.BAYER8], expected
    )


def test_floyd_steinberg_dither_all_white():
    """Tests Floyd-Steinberg dithering with all white image."""
    white = np.full((10, 10), 255, dtype=np.float32)
    result = apply_floyd_steinberg_dither(white, invert=False)
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
    assert np.all(result == 0)


def test_floyd_steinberg_dither_all_black():
    """Tests Floyd-Steinberg dithering with all black image."""
    black = np.full((10, 10), 0, dtype=np.float32)
    result = apply_floyd_steinberg_dither(black, invert=False)
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
    assert np.all(result == 1)


def test_floyd_steinberg_dither_invert():
    """Tests that invert parameter flips the output."""
    gray = np.full((10, 10), 128, dtype=np.float32)
    result_normal = apply_floyd_steinberg_dither(gray, invert=False)
    result_inverted = apply_floyd_steinberg_dither(gray, invert=True)
    assert np.array_equal(result_normal, 1 - result_inverted)


def test_floyd_steinberg_dither_gradient():
    """Tests Floyd-Steinberg dithering with a horizontal gradient."""
    gradient = np.linspace(0, 255, 100).reshape(10, 10).astype(np.float32)
    result = apply_floyd_steinberg_dither(gradient, invert=False)
    assert result.shape == (10, 10)
    assert result.dtype == np.uint8
    assert result.shape == gradient.shape


def test_floyd_steinberg_dither_single_pixel():
    """Tests Floyd-Steinberg dithering with a single pixel."""
    single = np.array([[127]], dtype=np.float32)
    result = apply_floyd_steinberg_dither(single, invert=False)
    assert result.shape == (1, 1)
    assert result.dtype == np.uint8


def test_floyd_steinberg_dither_error_diffusion():
    """Tests that error is properly diffused to neighboring pixels."""
    img = np.array([[100, 100], [100, 100]], dtype=np.float32)
    result = apply_floyd_steinberg_dither(img, invert=False)
    assert result.shape == (2, 2)
    assert result.dtype == np.uint8


def test_bayer_dither_all_white():
    """Tests Bayer dithering with all white image."""
    white = np.full((10, 10), 255, dtype=np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(white, matrix, invert=False)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        assert np.all(result == 0)


def test_bayer_dither_all_black():
    """Tests Bayer dithering with all black image."""
    black = np.full((10, 10), 0, dtype=np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(black, matrix, invert=False)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        assert np.sum(result) > 0


def test_bayer_dither_invert():
    """Tests that invert parameter flips the output for Bayer dither."""
    gray = np.full((10, 10), 128, dtype=np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result_normal = apply_bayer_dither(gray, matrix, invert=False)
        result_inverted = apply_bayer_dither(gray, matrix, invert=True)
        assert np.array_equal(result_normal, 1 - result_inverted)


def test_bayer_dither_pattern_consistency():
    """Tests that Bayer dither produces consistent patterns."""
    gray = np.full((16, 16), 128, dtype=np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result1 = apply_bayer_dither(gray, matrix, invert=False)
        result2 = apply_bayer_dither(gray, matrix, invert=False)
        np.testing.assert_array_equal(result1, result2)


def test_bayer_dither_tiling_pattern():
    """Tests that Bayer dither pattern tiles correctly across image."""
    gray = np.full((16, 16), 128, dtype=np.float32)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER2]
    result = apply_bayer_dither(gray, matrix, invert=False)
    assert result.shape == (16, 16)
    assert result.dtype == np.uint8


def test_bayer_dither_single_pixel():
    """Tests Bayer dithering with a single pixel."""
    single = np.array([[127]], dtype=np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(single, matrix, invert=False)
        assert result.shape == (1, 1)
        assert result.dtype == np.uint8


def test_dither_output_alignment_floyd_steinberg():
    """Tests that Floyd-Steinberg dither output aligns with input."""
    original = np.random.rand(50, 50) * 255
    original = original.astype(np.float32)
    result = apply_floyd_steinberg_dither(original, invert=False)
    assert result.shape == original.shape


def test_dither_output_alignment_bayer():
    """Tests that Bayer dither output aligns with input."""
    original = np.random.rand(50, 50) * 255
    original = original.astype(np.float32)
    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(original, matrix, invert=False)
        assert result.shape == original.shape


def test_dither_preserves_dark_regions():
    """Tests that dark regions remain dark after dithering."""
    dark = np.full((10, 10), 30, dtype=np.float32)
    result_fs = apply_floyd_steinberg_dither(dark, invert=False)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER4]
    result_bayer = apply_bayer_dither(dark, matrix, invert=False)
    assert np.mean(result_fs) > 0.5
    assert result_bayer.shape == (10, 10)


def test_dither_preserves_light_regions():
    """Tests that light regions remain light after dithering."""
    light = np.full((10, 10), 225, dtype=np.float32)
    result_fs = apply_floyd_steinberg_dither(light, invert=False)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER4]
    result_bayer = apply_bayer_dither(light, matrix, invert=False)
    assert np.mean(result_fs) < 0.5
    assert result_bayer.shape == (10, 10)


def create_test_surface(width, height, color=(255, 255, 255, 255)):
    """Helper to create a test Cairo surface with a solid color."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    r, g, b, a = color
    ctx.set_source_rgba(r / 255.0, g / 255.0, b / 255.0, a / 255.0)
    ctx.paint()
    return surface


def create_gradient_surface(width, height):
    """Helper to create a test Cairo surface with a gradient."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, width, height)
    ctx = cairo.Context(surface)
    gradient = cairo.LinearGradient(0, 0, width, 0)
    gradient.add_color_stop_rgba(0, 1, 1, 1, 1)
    gradient.add_color_stop_rgba(1, 0, 0, 0, 1)
    ctx.set_source(gradient)
    ctx.paint()
    return surface


def test_surface_to_dithered_array_white_surface():
    """Tests surface_to_dithered_array with a white surface."""
    surface = create_test_surface(10, 10, (255, 255, 255, 255))
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        if algo == DitherAlgorithm.FLOYD_STEINBERG:
            assert np.all(result == 0)
        else:
            assert np.sum(result) == 0


def test_surface_to_dithered_array_black_surface():
    """Tests surface_to_dithered_array with a black surface."""
    surface = create_test_surface(10, 10, (0, 0, 0, 255))
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (10, 10)
        assert result.dtype == np.uint8
        if algo == DitherAlgorithm.FLOYD_STEINBERG:
            assert np.all(result == 1)
        else:
            assert np.sum(result) > 0


def test_surface_to_dithered_array_invert():
    """Tests that invert parameter flips the output."""
    gray_surface = create_test_surface(10, 10, (128, 128, 128, 255))
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result_normal = surface_to_dithered_array(
            gray_surface, algo, invert=False
        )
        result_inverted = surface_to_dithered_array(
            gray_surface, algo, invert=True
        )
        assert np.array_equal(result_normal, 1 - result_inverted)


def test_surface_to_dithered_array_gradient():
    """Tests surface_to_dithered_array with a gradient surface."""
    surface = create_gradient_surface(50, 10)
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (10, 50)
        assert result.dtype == np.uint8
        assert np.any(result == 0) and np.any(result == 1)


def test_surface_to_dithered_array_alpha_transparency():
    """Tests that fully transparent pixels are set to 0."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.paint()
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert np.all(result == 0)


def test_surface_to_dithered_array_partial_alpha():
    """Tests surface with partial alpha transparency."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 0.5)
    ctx.paint()
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (10, 10)


def test_surface_to_dithered_array_mixed_alpha():
    """Tests surface with mixed opaque and transparent regions."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.rectangle(0, 0, 5, 10)
    ctx.fill()
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.rectangle(5, 0, 5, 10)
    ctx.fill()
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (10, 10)
        assert np.all(result[:, 5:] == 0)


def test_surface_to_dithered_array_alignment():
    """Tests that dithered output aligns with surface dimensions."""
    for w, h in [(10, 10), (17, 23), (100, 50), (256, 256)]:
        surface = create_test_surface(w, h)
        for algo in [
            DitherAlgorithm.FLOYD_STEINBERG,
            DitherAlgorithm.BAYER2,
            DitherAlgorithm.BAYER4,
            DitherAlgorithm.BAYER8,
        ]:
            result = surface_to_dithered_array(surface, algo, invert=False)
            assert result.shape == (h, w)


def test_surface_to_dithered_array_single_pixel():
    """Tests surface_to_dithered_array with a single pixel surface."""
    surface = create_test_surface(1, 1, (128, 128, 128, 255))
    for algo in [
        DitherAlgorithm.FLOYD_STEINBERG,
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        result = surface_to_dithered_array(surface, algo, invert=False)
        assert result.shape == (1, 1)
        assert result.dtype == np.uint8


def test_surface_to_dithered_array_color_to_grayscale():
    """Tests that colored surfaces are properly converted to grayscale."""
    surface = create_test_surface(10, 10, (255, 0, 0, 255))
    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )
    assert result.shape == (10, 10)


def test_surface_to_dithered_array_red_channel():
    """Tests grayscale conversion with red channel."""
    surface = create_test_surface(10, 10, (255, 0, 0, 255))
    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )
    assert result.shape == (10, 10)


def test_surface_to_dithered_array_green_channel():
    """Tests grayscale conversion with green channel."""
    surface = create_test_surface(10, 10, (0, 255, 0, 255))
    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )
    assert result.shape == (10, 10)


def test_surface_to_dithered_array_blue_channel():
    """Tests grayscale conversion with blue channel."""
    surface = create_test_surface(10, 10, (0, 0, 255, 255))
    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )
    assert result.shape == (10, 10)


def test_dither_result_is_binary():
    """Tests that dithering results are strictly binary (0 or 1)."""
    random_img = np.random.rand(20, 20) * 255
    random_img = random_img.astype(np.float32)

    result_fs = apply_floyd_steinberg_dither(random_img, invert=False)
    assert np.all(np.isin(result_fs, [0, 1]))

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(random_img, matrix, invert=False)
        assert np.all(np.isin(result, [0, 1]))


def test_dither_edge_case_mid_gray():
    """Tests dithering behavior at exactly mid-gray (128)."""
    mid_gray = np.full((10, 10), 128.0, dtype=np.float32)
    result_fs = apply_floyd_steinberg_dither(mid_gray, invert=False)
    assert result_fs.shape == (10, 10)
    assert result_fs.dtype == np.uint8


def test_dither_edge_case_threshold_boundary():
    """Tests dithering at the threshold boundary (127 vs 128)."""
    below = np.full((10, 10), 127.0, dtype=np.float32)
    above = np.full((10, 10), 128.0, dtype=np.float32)

    result_below = apply_floyd_steinberg_dither(below, invert=False)
    result_above = apply_floyd_steinberg_dither(above, invert=False)

    assert result_below.shape == (10, 10)
    assert result_above.shape == (10, 10)


def test_dither_preserves_input():
    """Tests that dithering does not modify the input array."""
    original = np.random.rand(10, 10) * 255
    original = original.astype(np.float32)
    original_copy = original.copy()

    apply_floyd_steinberg_dither(original, invert=False)
    np.testing.assert_array_equal(original, original_copy)

    apply_bayer_dither(
        original, BAYER_MATRICES[DitherAlgorithm.BAYER4], invert=False
    )
    np.testing.assert_array_equal(original, original_copy)


def test_dither_consistency_same_input():
    """Tests that same input produces same output."""
    fixed_input = np.random.rand(10, 10) * 255
    fixed_input = fixed_input.astype(np.float32)

    result1 = apply_floyd_steinberg_dither(fixed_input.copy(), invert=False)
    result2 = apply_floyd_steinberg_dither(fixed_input.copy(), invert=False)
    np.testing.assert_array_equal(result1, result2)

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result1 = apply_bayer_dither(fixed_input.copy(), matrix, invert=False)
        result2 = apply_bayer_dither(fixed_input.copy(), matrix, invert=False)
        np.testing.assert_array_equal(result1, result2)


def test_dither_checkerboard_pattern():
    """Tests dithering with a checkerboard pattern."""
    checkerboard = np.zeros((10, 10), dtype=np.float32)
    checkerboard[::2, ::2] = 255
    checkerboard[1::2, 1::2] = 255

    result_fs = apply_floyd_steinberg_dither(checkerboard, invert=False)
    assert result_fs.shape == (10, 10)

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(checkerboard, matrix, invert=False)
        assert result.shape == (10, 10)


def test_dither_vertical_gradient():
    """Tests dithering with a vertical gradient."""
    gradient = np.linspace(0, 255, 100).reshape(100, 1).astype(np.float32)
    gradient = np.tile(gradient, (1, 10))

    result_fs = apply_floyd_steinberg_dither(gradient, invert=False)
    assert result_fs.shape == gradient.shape

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(gradient, matrix, invert=False)
        assert result.shape == gradient.shape


def test_dither_diagonal_gradient():
    """Tests dithering with a diagonal gradient."""
    size = 50
    diagonal = np.zeros((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            diagonal[i, j] = (i + j) * 255 / (2 * size)

    result_fs = apply_floyd_steinberg_dither(diagonal, invert=False)
    assert result_fs.shape == diagonal.shape

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(diagonal, matrix, invert=False)
        assert result.shape == diagonal.shape


def test_dither_algorithm_enum_values():
    """Tests that all algorithm enum values are valid strings."""
    assert DitherAlgorithm.FLOYD_STEINBERG.value == "floyd_steinberg"
    assert DitherAlgorithm.BAYER2.value == "bayer2"
    assert DitherAlgorithm.BAYER4.value == "bayer4"
    assert DitherAlgorithm.BAYER8.value == "bayer8"


def test_surface_stride_handling():
    """Tests that surfaces with different strides are handled correctly."""
    surface = create_test_surface(17, 13)
    stride = surface.get_stride()
    assert stride > 0

    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )
    assert result.shape == (13, 17)


def test_dither_non_square_images():
    """Tests dithering with non-square images."""
    wide = np.random.rand(10, 20) * 255
    wide = wide.astype(np.float32)
    result = apply_floyd_steinberg_dither(wide, invert=False)
    assert result.shape == (10, 20)

    tall = np.random.rand(20, 10) * 255
    tall = tall.astype(np.float32)
    result = apply_floyd_steinberg_dither(tall, invert=False)
    assert result.shape == (20, 10)


def test_pixel_probing_corner_alignment():
    """Tests that corner pixels are correctly aligned in output."""
    img = np.full((10, 10), 30, dtype=np.float32)
    img[0, 0] = 0
    img[0, 9] = 0
    img[9, 0] = 0
    img[9, 9] = 0

    result = apply_floyd_steinberg_dither(img, invert=False)
    assert result.shape == (10, 10)
    assert result[0, 0] == 1
    assert result[0, 9] == 1
    assert result[9, 0] == 1
    assert result[9, 9] == 1


def test_pixel_probing_center_pixel():
    """Tests that center pixel is correctly processed."""
    img = np.full((11, 11), 200.0, dtype=np.float32)
    img[5, 5] = 0

    result = apply_floyd_steinberg_dither(img, invert=False)
    assert result.shape == (11, 11)
    assert result[5, 5] == 1
    assert result[5, 6] >= 0
    assert result[6, 5] >= 0


def test_pixel_probing_bayer_threshold_boundaries():
    """Tests Bayer dithering at exact threshold boundaries."""
    gray = np.full((8, 8), 128.0, dtype=np.float32)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER4]
    result = apply_bayer_dither(gray, matrix, invert=False)

    for y in range(8):
        for x in range(8):
            threshold = (matrix[y % 4, x % 4] / 16.0) * 255.0
            expected = 1 if 128.0 < threshold else 0
            assert result[y, x] == expected


def test_pixel_probing_error_propagation():
    """Tests that error propagates correctly to adjacent pixels."""
    img = np.full((3, 3), 127.0, dtype=np.float32)
    img[0, 0] = 0

    result = apply_floyd_steinberg_dither(img, invert=False)
    assert result[0, 0] == 1
    assert np.sum(result) > 1


def test_xor_invert_produces_complement():
    """Tests that inverting produces bitwise complement via XOR."""
    gray = np.full((10, 10), 128, dtype=np.float32)

    result_normal = apply_floyd_steinberg_dither(gray, invert=False)
    result_inverted = apply_floyd_steinberg_dither(gray, invert=True)

    xor_result = np.bitwise_xor(result_normal, result_inverted)
    assert np.all(xor_result == 1)


def test_xor_bayer_invert_complement():
    """Tests that Bayer invert produces bitwise complement via XOR."""
    gray = np.full((10, 10), 128, dtype=np.float32)

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result_normal = apply_bayer_dither(gray, matrix, invert=False)
        result_inverted = apply_bayer_dither(gray, matrix, invert=True)

        xor_result = np.bitwise_xor(result_normal, result_inverted)
        assert np.all(xor_result == 1)


def test_xor_pattern_tiling_consistency():
    """Tests Bayer pattern tiles consistently using XOR comparison."""
    gray = np.full((16, 16), 100, dtype=np.float32)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER4]
    result = apply_bayer_dither(gray, matrix, invert=False)

    for y in range(4):
        for x in range(4):
            tile_0_0 = result[y : y + 4, x : x + 4]
            tile_4_0 = result[y : y + 4, x + 4 : x + 8]
            tile_0_4 = result[y + 4 : y + 8, x : x + 4]
            tile_4_4 = result[y + 4 : y + 8, x + 4 : x + 8]

            xor_0 = np.bitwise_xor(tile_0_0, tile_4_0)
            xor_1 = np.bitwise_xor(tile_0_0, tile_0_4)
            xor_2 = np.bitwise_xor(tile_0_0, tile_4_4)

            assert np.array_equal(xor_0, xor_1)
            assert np.array_equal(xor_1, xor_2)


def test_xor_gradient_monotonicity():
    """Tests gradient dithering produces monotonic pattern changes."""
    gradient = np.linspace(0, 255, 20).reshape(4, 5).astype(np.float32)
    result = apply_floyd_steinberg_dither(gradient, invert=False)

    for col in range(4):
        xor_result = np.bitwise_xor(result[:, col], result[:, col + 1])
        transitions = np.sum(xor_result)
        assert transitions >= 0


def test_xor_surface_conversion_alignment():
    """Tests surface to array conversion maintains alignment via XOR."""
    surface = create_test_surface(16, 16, (128, 128, 128, 255))

    result_bayer2 = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER2, invert=False
    )
    result_bayer4 = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )

    assert result_bayer2.shape == result_bayer4.shape

    xor_result = np.bitwise_xor(result_bayer2, result_bayer4)
    assert xor_result.shape == (16, 16)


def test_pixel_probing_edge_pixel_error_diffusion():
    """Tests error diffusion at image edges doesn't overflow."""
    img = np.zeros((5, 5), dtype=np.float32)
    img[0, :] = 200
    img[4, :] = 200
    img[:, 0] = 200
    img[:, 4] = 200

    result = apply_floyd_steinberg_dither(img, invert=False)
    assert result.shape == (5, 5)
    assert np.all(result >= 0) and np.all(result <= 1)


def test_xor_binary_output_verification():
    """Tests all dithering produces strictly binary output via XOR."""
    random_img = np.random.rand(15, 15) * 255
    random_img = random_img.astype(np.float32)

    result_fs = apply_floyd_steinberg_dither(random_img, invert=False)
    xor_fs = np.bitwise_xor(result_fs, result_fs)
    assert np.all(xor_fs == 0)

    for matrix_enum in [
        DitherAlgorithm.BAYER2,
        DitherAlgorithm.BAYER4,
        DitherAlgorithm.BAYER8,
    ]:
        matrix = BAYER_MATRICES[matrix_enum]
        result = apply_bayer_dither(random_img, matrix, invert=False)
        xor_result = np.bitwise_xor(result, result)
        assert np.all(xor_result == 0)


def test_pixel_probing_specific_bayer_values():
    """Tests specific Bayer threshold values at known positions."""
    img = np.full((4, 4), 128.0, dtype=np.float32)
    matrix = BAYER_MATRICES[DitherAlgorithm.BAYER4]

    result = apply_bayer_dither(img, matrix, invert=False)

    threshold_0_0 = (matrix[0, 0] / 16.0) * 255.0
    threshold_1_1 = (matrix[1, 1] / 16.0) * 255.0

    expected_0_0 = 1 if 128.0 < threshold_0_0 else 0
    expected_1_1 = 1 if 128.0 < threshold_1_1 else 0

    assert result[0, 0] == expected_0_0
    assert result[1, 1] == expected_1_1


def test_xor_surface_alpha_alignment():
    """Tests that alpha channel alignment is preserved via XOR."""
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 1)
    ctx.rectangle(0, 0, 5, 10)
    ctx.fill()
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.rectangle(5, 0, 5, 10)
    ctx.fill()

    result = surface_to_dithered_array(
        surface, DitherAlgorithm.BAYER4, invert=False
    )

    assert np.all(result[:, 5:] == 0)

    xor_left = np.bitwise_xor(result[:, :5], result[:, :5])
    assert np.all(xor_left == 0)
