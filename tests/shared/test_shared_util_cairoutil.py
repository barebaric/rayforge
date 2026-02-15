import cairo
import numpy as np
import pytest

from rayforge.shared.util.cairoutil import (
    convert_surface_to_grayscale,
    make_transparent,
)


class TestConvertSurfaceToGrayscale:
    def test_converts_to_grayscale(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 255
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        arr[:, :, 3] = 255
        surface.mark_dirty()

        result = convert_surface_to_grayscale(surface)

        assert result is surface
        data = result.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 0] == arr[0, 0, 1]
        assert arr[0, 0, 1] == arr[0, 0, 2]

    def test_raises_for_non_argb32(self):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
        with pytest.raises(
            ValueError, match="Unsupported Cairo surface format"
        ):
            convert_surface_to_grayscale(surface)

    def test_preserves_alpha(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 100
        arr[:, :, 1] = 150
        arr[:, :, 2] = 200
        arr[:, :, 3] = 128
        surface.mark_dirty()

        convert_surface_to_grayscale(surface)

        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 3] == 128


class TestMakeTransparent:
    def test_makes_white_pixels_transparent(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 255
        arr[:, :, 1] = 255
        arr[:, :, 2] = 255
        arr[:, :, 3] = 255
        surface.mark_dirty()

        make_transparent(surface, threshold=250)

        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 3] == 0

    def test_keeps_dark_pixels_opaque(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 0
        arr[:, :, 1] = 0
        arr[:, :, 2] = 0
        arr[:, :, 3] = 255
        surface.mark_dirty()

        make_transparent(surface, threshold=250)

        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 3] == 255

    def test_raises_for_non_argb32(self):
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, 2, 2)
        with pytest.raises(
            ValueError, match="Surface must be in ARGB32 format"
        ):
            make_transparent(surface)

    def test_custom_threshold(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 200
        arr[:, :, 1] = 200
        arr[:, :, 2] = 200
        arr[:, :, 3] = 255
        surface.mark_dirty()

        make_transparent(surface, threshold=150)

        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 3] == 0

    def test_threshold_boundary(self):
        surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, 2, 2)
        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        arr[:, :, 0] = 200
        arr[:, :, 1] = 200
        arr[:, :, 2] = 200
        arr[:, :, 3] = 255
        surface.mark_dirty()

        make_transparent(surface, threshold=201)

        data = surface.get_data()
        arr = np.frombuffer(data, dtype=np.uint8).reshape((2, 2, 4))
        assert arr[0, 0, 3] == 255
