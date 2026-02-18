import cairo
import pytest
import numpy as np

from rayforge.image.util import transparency


def test_make_surface_transparent_makes_white_pixels_transparent():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 255
    arr[:, :, 1] = 255
    arr[:, :, 2] = 255
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_surface_transparent(surface, threshold=250)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 0


def test_make_surface_transparent_keeps_dark_pixels_opaque():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 0
    arr[:, :, 1] = 0
    arr[:, :, 2] = 0
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_surface_transparent(surface, threshold=250)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 255


def test_make_surface_transparent_raises_for_non_argb32():
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
    with pytest.raises(
        ValueError, match="Surface must be in ARGB32 format"
    ):
        transparency.make_surface_transparent(surface)


def test_make_surface_transparent_custom_threshold():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 200
    arr[:, :, 1] = 200
    arr[:, :, 2] = 200
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_surface_transparent(surface, threshold=150)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 0


def test_make_surface_transparent_threshold_boundary():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 200
    arr[:, :, 1] = 200
    arr[:, :, 2] = 200
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_surface_transparent(surface, threshold=201)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 255


def test_make_transparent_except_color_keeps_target_color_opaque():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 100
    arr[:, :, 1] = 150
    arr[:, :, 2] = 200
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_transparent_except_color(surface, 200, 150, 100)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 255


def test_make_transparent_except_color_makes_other_colors_transparent():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 50
    arr[:, :, 1] = 100
    arr[:, :, 2] = 150
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_transparent_except_color(surface, 200, 150, 100)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 0


def test_make_transparent_except_color_raises_for_non_argb32():
    surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
    with pytest.raises(
        ValueError, match="Surface must be in ARGB32 format"
    ):
        transparency.make_transparent_except_color(surface, 255, 0, 0)


def test_make_transparent_except_color_mixed_colors():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[0, 0, 0] = 0
    arr[0, 0, 1] = 255
    arr[0, 0, 2] = 0
    arr[0, 0, 3] = 255
    arr[0, 1, 0] = 255
    arr[0, 1, 1] = 0
    arr[0, 1, 2] = 0
    arr[0, 1, 3] = 255
    arr[1, 0, 0] = 0
    arr[1, 0, 1] = 255
    arr[1, 0, 2] = 0
    arr[1, 0, 3] = 255
    arr[1, 1, 0] = 0
    arr[1, 1, 1] = 0
    arr[1, 1, 2] = 255
    arr[1, 1, 3] = 255
    surface.mark_dirty()

    transparency.make_transparent_except_color(surface, 0, 255, 0)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 3] == 255
    assert arr[0, 1, 3] == 0
    assert arr[1, 0, 3] == 255
    assert arr[1, 1, 3] == 0


def test_make_transparent_except_color_preserves_color_channels():
    surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    arr[:, :, 0] = 50
    arr[:, :, 1] = 100
    arr[:, :, 2] = 200
    arr[:, :, 3] = 255
    surface.mark_dirty()

    transparency.make_transparent_except_color(surface, 255, 0, 0)

    data = surface.get_data()
    arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
    assert arr[0, 0, 0] == 50
    assert arr[0, 0, 1] == 100
    assert arr[0, 0, 2] == 200
    assert arr[0, 0, 3] == 0
