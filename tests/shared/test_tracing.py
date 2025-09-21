import numpy as np
import cairo
from rayforge.shared.util.tracing import (
    _get_component_areas,
    _find_adaptive_area_threshold,
    _filter_image_by_component_area,
    trace_surface,
    Geometry,
)


def _create_test_surface(array: np.ndarray) -> cairo.ImageSurface:
    """Creates a Cairo ARGB32 surface from a 2D grayscale numpy array."""
    if array.ndim != 2:
        raise ValueError("Input array must be 2D")

    h, w = array.shape
    # Convert grayscale (0-255) to BGRA for Cairo
    bgra_array = np.zeros((h, w, 4), dtype=np.uint8)
    gray_3channel = np.stack([array] * 3, axis=-1)
    bgra_array[..., :3] = gray_3channel[..., ::-1]  # RGB -> BGR
    bgra_array[..., 3] = 255  # Full alpha

    # Pass a memoryview to satisfy stricter type checkers like Pylance
    return cairo.ImageSurface.create_for_data(
        memoryview(bgra_array), cairo.FORMAT_ARGB32, w, h
    )


def test_get_component_areas():
    """Tests if component areas are correctly identified."""
    img = np.array(
        [
            [255, 255, 255, 255, 255, 255, 255],
            [255, 0, 0, 255, 0, 255, 255],
            [255, 0, 0, 255, 255, 255, 255],
            [255, 255, 255, 255, 255, 255, 255],
            [255, 255, 0, 0, 0, 255, 255],
            [255, 255, 0, 0, 0, 255, 255],
        ],
        dtype=np.uint8,
    )
    boolean_image = img < 128
    areas = _get_component_areas(boolean_image)
    assert sorted(areas.tolist()) == [1, 4, 6]


def test_get_component_areas_empty_image():
    """Tests behavior with an image containing no components."""
    img = np.full((10, 10), 255, dtype=np.uint8)
    boolean_image = img < 128
    areas = _get_component_areas(boolean_image)
    assert areas.size == 0


def test_find_adaptive_area_threshold_noisy_image():
    """Tests threshold finding for a typically noisy image histogram."""
    areas = np.concatenate(
        [
            np.ones(1000, dtype=int),
            np.full(500, 2, dtype=int),
            np.full(5, 10, dtype=int),
            np.full(2, 100, dtype=int),
        ]
    )
    # The robust heuristic finds the last significant noise area is 2.
    # Therefore, the threshold to keep components should be 3.
    assert _find_adaptive_area_threshold(areas) == 3


def test_find_adaptive_area_threshold_clean_image():
    """Tests threshold finding for a clean image with only large components."""
    areas = np.array([100, 150, 200, 500])
    # The code now detects this is a clean image (peak is at area > 10)
    # and correctly returns the minimum default.
    assert _find_adaptive_area_threshold(areas) == 2


def test_find_adaptive_area_threshold_empty():
    """Tests threshold finding with no areas."""
    assert _find_adaptive_area_threshold(np.array([])) == 0


def test_filter_image_by_component_area():
    """Tests if small components are correctly removed from an image."""
    img = np.array(
        [
            [255, 255, 255, 255, 255],
            [255, 0, 255, 0, 0],
            [255, 255, 255, 0, 0],
            [255, 255, 255, 0, 0],
        ],
        dtype=np.uint8,
    )
    boolean_image = img < 128
    filtered = _filter_image_by_component_area(boolean_image, min_area=5)
    assert np.sum(filtered) == 6
    assert np.sum(boolean_image) == 7


def test_trace_surface_clean_path():
    """Tests tracing a simple, clean image. Should produce one vector path."""
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[25:75, 25:75] = 0
    surface = _create_test_surface(img)
    geometries = trace_surface(surface, pixels_per_mm=(10.0, 10.0))
    assert len(geometries) == 1
    assert isinstance(geometries[0], Geometry)


def test_trace_surface_denoising_path():
    """
    Tests tracing a noisy image. Should filter noise and trace the
    main shape. This test is now deterministic.
    """
    img = np.full((100, 100), 255, dtype=np.uint8)
    img[25:75, 25:75] = 0
    # Add a deterministic grid of 1px and 2px noise clumps
    img[::5, ::5] = 0  # 1px noise
    img[1::10, 1::10] = 0  # 2px noise (horizontal)
    img[1::10, 2::10] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface, pixels_per_mm=(10.0, 10.0))
    # The adaptive filter should calculate a threshold of 3 and remove all
    # 1px and 2px components, leaving only the main square.
    assert len(geometries) == 1


def test_trace_surface_empty_result():
    """Tests that an all-white or all-noise image returns an empty list."""
    # Create a deterministic grid of 1px noise to avoid flaky tests.
    img = np.full((100, 100), 255, dtype=np.uint8)
    # Every second pixel is black, ensuring no components are larger than 1px.
    img[::2, ::2] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface, pixels_per_mm=(10.0, 10.0))
    assert geometries == []


def test_trace_surface_hull_fallback_path(monkeypatch):
    """
    Tests the fallback to convex hulls when vector count is too high.
    A checkerboard is perfect for generating many distinct vectors.
    """
    monkeypatch.setattr("rayforge.shared.util.tracing.MAX_VECTORS_LIMIT", 50)

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
    geometries = trace_surface(surface, pixels_per_mm=(10.0, 10.0))

    # There are 50 non-touching black squares (components).
    assert len(geometries) == 50

    # A convex hull of a square has exactly 5 commands: M, L, L, L, L (close)
    assert len(geometries[0].commands) == 5
    # The test now correctly asserts that close_path() adds a LineToCommand
    command_types = [type(cmd).__name__ for cmd in geometries[0].commands]
    assert command_types == [
        "MoveToCommand",
        "LineToCommand",
        "LineToCommand",
        "LineToCommand",
        "LineToCommand",
    ]


def test_trace_surface_edge_touching_shape():
    """
    Tests that a shape touching the edge of the image is still traced
    correctly, verifying that the border handling works.
    """
    img = np.full((100, 100), 255, dtype=np.uint8)
    # A 50x50 black square at (0,0), touching the top and left edges.
    img[0:50, 0:50] = 0

    surface = _create_test_surface(img)
    geometries = trace_surface(surface, pixels_per_mm=(10.0, 10.0))

    # The border added during processing should ensure the shape is found.
    assert len(geometries) == 1
    assert isinstance(geometries[0], Geometry)
