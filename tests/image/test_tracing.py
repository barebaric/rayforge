import numpy as np
import cairo
from unittest.mock import MagicMock, ANY
from rayforge.core.geo.constants import COL_TYPE, CMD_TYPE_MOVE, CMD_TYPE_LINE
from rayforge.image.tracing import trace_surface, Geometry


def _create_test_surface(array: np.ndarray) -> cairo.ImageSurface:
    """Creates a Cairo ARGB32 surface from a 2D grayscale numpy array."""
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    h, w = array.shape
    # Convert grayscale (0-255) to BGRA for Cairo with full alpha
    bgra_array = np.zeros((h, w, 4), dtype=np.uint8)
    gray_3channel = np.stack([array] * 3, axis=-1)
    bgra_array[..., :3] = gray_3channel[..., ::-1]  # RGB -> BGR
    bgra_array[..., 3] = 255  # Full alpha

    return cairo.ImageSurface.create_for_data(
        memoryview(bgra_array), cairo.FORMAT_ARGB32, w, h
    )


def _create_test_surface_with_alpha(array: np.ndarray) -> cairo.ImageSurface:
    """Creates a Cairo ARGB32 surface from a 4-channel BGRA numpy array."""
    if array.ndim != 3 or array.shape[2] != 4:
        raise ValueError("Input array must be HxWx4")
    h, w, _ = array.shape
    # Ensure contiguous memory for Cairo
    contiguous_array = np.ascontiguousarray(array)
    return cairo.ImageSurface.create_for_data(
        memoryview(contiguous_array), cairo.FORMAT_ARGB32, w, h
    )


def test_trace_surface_clean_path():
    """Tests tracing a simple, clean image. Should produce one vector path."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[25:75, 25:75] = 0
    surface = _create_test_surface(img)
    geometries = trace_surface(surface)
    assert len(geometries) == 1
    assert isinstance(geometries[0], Geometry)


def test_trace_transparent_png_avoids_framing():
    """
    Tests tracing a transparent image to ensure no framing artifacts are
    created from an incorrectly handled border.
    """
    # Create a 50x50 BGRA array, fully transparent
    bgra_array = np.zeros((50, 50, 4), dtype=np.uint8)
    # Draw an opaque black square in the middle. B,G,R,A
    bgra_array[10:40, 10:40] = [0, 0, 0, 255]

    surface = _create_test_surface_with_alpha(bgra_array)
    geometries = trace_surface(surface)

    # With the fix, only the central square should be traced.
    # Without the fix, this would find 2 or 3 geometries (the square + frames).
    assert len(geometries) == 1


def test_trace_surface_denoising_path():
    """
    Tests tracing a noisy image. Should filter noise and trace the
    main shape.
    """
    img = np.full((100, 100), 255, dtype=np.uint8)
    # Main shape
    img[25:75, 25:75] = 0
    # Add a 1px noise component
    img[10, 10] = 0
    # Add a 2px noise component
    img[12, 12] = 0
    img[12, 13] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface)
    # The adaptive filter should calculate a threshold of 3 and remove all
    # 1px and 2px components, leaving only the main square.
    assert len(geometries) == 1


def test_trace_surface_empty_result():
    """Tests that an all-white or all-noise image returns an empty list."""
    # Create a deterministic grid of 1px noise to avoid flaky tests.
    img = np.full((100, 100), 255, dtype=np.uint8)
    # Every third pixel is black, ensuring no components are connected,
    # even diagonally. All components will have an area of 1px.
    img[::3, ::3] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface)
    assert geometries == []


def test_trace_surface_hull_fallback_path(monkeypatch):
    """
    Tests the fallback to convex hulls when vector count is too high.
    A checkerboard is perfect for generating many distinct vectors.
    """
    monkeypatch.setattr("rayforge.image.tracing.MAX_VECTORS_LIMIT", 50)

    # Create a white canvas
    img = np.full((100, 100), 255, dtype=np.uint8)
    # Draw smaller, non-touching squares to ensure 50 distinct components.
    for i in range(10):
        for j in range(10):
            if (i + j) % 2 == 1:
                # Draw an 8x8 black square inside each 10x10 cell
                start_row, start_col = i * 10 + 1, j * 10 + 1
                img[start_row : start_row + 8, start_col : start_col + 8] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface)

    # There are 50 non-touching black squares (components).
    assert len(geometries) == 50

    # A convex hull of a square has exactly 5 commands: M, L, L, L, L (close)
    # Note: len(geo) counts rows in the numpy array.
    assert len(geometries[0]) == 5

    # Check command types directly from numpy array
    data = geometries[0].data
    assert data is not None
    assert data[0, COL_TYPE] == CMD_TYPE_MOVE
    assert np.all(data[1:, COL_TYPE] == CMD_TYPE_LINE)


def test_trace_surface_edge_touching_shape():
    """
    Tests that a shape touching the edge of the image is still traced
    correctly, verifying that the border handling works.
    """
    img = np.full((100, 100), 255, dtype=np.uint8)
    # A 50x50 black square at (0,0), touching the top and left edges.
    img[0:50, 0:50] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface)

    # The border added during processing should ensure the shape is found.
    assert len(geometries) == 1
    assert isinstance(geometries[0], Geometry)


def test_trace_surface_vtracer_failure_fallback_to_hull(monkeypatch):
    """
    Tests that the correct fallback function (get_enclosing_hull) is called
    if vtracer fails on a non-empty image.
    """
    # Mock vtracer to simulate failure
    mock_vtracer = MagicMock(
        side_effect=Exception("Simulated vtracer failure")
    )
    monkeypatch.setattr("vtracer.convert_raw_image_to_svg", mock_vtracer)

    # Mock the hull function to verify it gets called
    mock_get_hull = MagicMock(return_value="mocked_geometry")
    monkeypatch.setattr(
        "rayforge.image.tracing.get_enclosing_hull", mock_get_hull
    )

    # Create a simple image that would normally trace fine
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[10:30, 10:30] = 0
    surface = _create_test_surface(img)

    # Call the main function
    result = trace_surface(surface)

    # Assert that our mocked hull function was called and its result returned
    mock_get_hull.assert_called_once()
    # Check some of the arguments passed to the mock
    # ANY is used for the numpy array as comparing them directly can be tricky
    mock_get_hull.assert_called_with(
        ANY,
        1.0,  # scale_x is now 1.0 (pixel units)
        1.0,  # scale_y is now 1.0 (pixel units)
        100,
        2,
    )
    assert result == ["mocked_geometry"]
