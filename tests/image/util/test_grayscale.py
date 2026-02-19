import cairo
import pytest
import numpy as np

from rayforge.image.util import grayscale


def test_surface_to_grayscale_black_surface():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    gray, alpha = grayscale.surface_to_grayscale(surface)
    assert gray.shape == (10, 10)
    assert alpha.shape == (10, 10)
    assert np.all(gray == 0)
    assert np.all(alpha == 1.0)


def test_surface_to_grayscale_white_surface():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    gray, alpha = grayscale.surface_to_grayscale(surface)
    assert np.allclose(gray, 255, atol=1)
    assert np.all(alpha == 1.0)


def test_surface_to_grayscale_transparent_surface():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0.5, 0.5, 0.5, 0)
    ctx.paint()
    gray, alpha = grayscale.surface_to_grayscale(surface)
    assert np.all(alpha == 0.0)


def test_surface_to_binary_black_surface_becomes_all_ones():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(0, 0, 0)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128)
    assert binary.shape == (10, 10)
    assert np.all(binary == 1)


def test_surface_to_binary_white_surface_becomes_all_zeros():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128)
    assert np.all(binary == 0)


def test_surface_to_binary_threshold_behavior():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 50
    arr[:, :, 1] = 50
    arr[:, :, 2] = 50
    arr[:, :, 3] = 255
    surface.mark_dirty()

    binary_low = grayscale.surface_to_binary(surface, threshold=40)
    assert np.all(binary_low == 0)

    binary_high = grayscale.surface_to_binary(surface, threshold=60)
    assert np.all(binary_high == 1)


def test_surface_to_binary_invert_mode():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgb(1, 1, 1)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128, invert=True)
    assert np.all(binary == 1)

    surface2 = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx2 = cairo.Context(surface2)
    ctx2.set_source_rgb(0, 0, 0)
    ctx2.paint()
    binary2 = grayscale.surface_to_binary(surface2, threshold=128, invert=True)
    assert np.all(binary2 == 0)


def test_surface_to_binary_transparent_becomes_zero():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(0, 0, 0, 0)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128)
    assert np.all(binary == 0)


def test_surface_to_binary_transparent_ignored_in_invert_mode():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    ctx.set_source_rgba(1, 1, 1, 0)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128, invert=True)
    assert np.all(binary == 0)


def test_surface_to_binary_raises_for_non_argb32():
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
    with pytest.raises(ValueError, match="Unsupported Cairo surface format"):
        grayscale.surface_to_binary(surface)


def test_surface_to_binary_partial_opacity_preserved():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 0
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = 128
    surface.mark_dirty()

    binary = grayscale.surface_to_binary(surface, threshold=128)
    assert np.all(binary == 1)


def test_surface_to_binary_output_is_binary():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 10, 10)
    ctx = cairo.Context(surface)
    gradient = cairo.LinearGradient(0, 0, 10, 10)
    gradient.add_color_stop_rgb(0, 0, 0, 0)
    gradient.add_color_stop_rgb(1, 1, 1, 1)
    ctx.set_source(gradient)
    ctx.paint()
    binary = grayscale.surface_to_binary(surface, threshold=128)
    unique_values = np.unique(binary)
    assert all(v in [0, 1] for v in unique_values)


def test_convert_surface_to_grayscale_inplace_converts_to_grayscale():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 255
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = 255
    surface.mark_dirty()

    grayscale.convert_surface_to_grayscale_inplace(surface)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 0] == arr[0, 0, 1]
    assert arr[0, 0, 1] == arr[0, 0, 2]


def test_convert_surface_to_grayscale_inplace_raises_for_non_argb32():
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
    with pytest.raises(ValueError, match="Unsupported Cairo surface format"):
        grayscale.convert_surface_to_grayscale_inplace(surface)


def test_convert_surface_to_grayscale_inplace_preserves_alpha():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 100
    arr[:, :, 1] = 150
    arr[:, :, 2] = 200
    arr[:, :, 3] = 128
    surface.mark_dirty()

    grayscale.convert_surface_to_grayscale_inplace(surface)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 128


def test_normalize_grayscale_default_params_no_change():
    gray = np.array([[0, 128, 255]], dtype=np.uint8)
    result = grayscale.normalize_grayscale(gray)
    np.testing.assert_array_equal(result, gray)


def test_normalize_grayscale_stretches_contrast():
    gray = np.array([[50, 100, 150]], dtype=np.uint8)
    result = grayscale.normalize_grayscale(
        gray, black_point=50, white_point=150
    )
    assert result[0, 0] == 0
    assert result[0, 1] == 127
    assert result[0, 2] == 255


def test_normalize_grayscale_clips_below_black_point():
    gray = np.array([[0, 25, 50]], dtype=np.uint8)
    result = grayscale.normalize_grayscale(
        gray, black_point=50, white_point=200
    )
    np.testing.assert_array_equal(result[0, :2], [0, 0])


def test_normalize_grayscale_clips_above_white_point():
    gray = np.array([[200, 225, 255]], dtype=np.uint8)
    result = grayscale.normalize_grayscale(
        gray, black_point=0, white_point=200
    )
    np.testing.assert_array_equal(result[0, :], [255, 255, 255])


def test_normalize_grayscale_raises_for_invalid_points():
    gray = np.array([[128]], dtype=np.uint8)
    with pytest.raises(ValueError, match="must be less than"):
        grayscale.normalize_grayscale(gray, black_point=128, white_point=128)

    with pytest.raises(ValueError, match="must be less than"):
        grayscale.normalize_grayscale(gray, black_point=200, white_point=100)


def test_normalize_grayscale_preserves_shape():
    gray = np.random.randint(0, 256, (10, 20), dtype=np.uint8)
    result = grayscale.normalize_grayscale(
        gray, black_point=30, white_point=200
    )
    assert result.shape == gray.shape


def test_normalize_grayscale_extreme_stretch():
    gray = np.array([[100]], dtype=np.uint8)
    result = grayscale.normalize_grayscale(
        gray, black_point=100, white_point=101
    )
    assert result[0, 0] == 0

    gray2 = np.array([[101]], dtype=np.uint8)
    result2 = grayscale.normalize_grayscale(
        gray2, black_point=100, white_point=101
    )
    assert result2[0, 0] == 255
