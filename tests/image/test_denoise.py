import numpy as np
from rayforge.image.denoise import (
    _get_component_areas,
    _find_adaptive_area_threshold,
    _filter_image_by_component_area,
    denoise_boolean_image,
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
    # The new gap-finding algorithm correctly identifies the noise cluster
    # ending at area 10, just before the large gap to area 100.
    # Therefore, the threshold should be 11.
    assert _find_adaptive_area_threshold(areas) == 11


def test_find_adaptive_area_threshold_clean_image():
    """Tests threshold finding for a clean image with only large components."""
    areas = np.array([100, 150, 200, 500])
    # The heuristic for clean images should return the minimum default.
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


def test_denoise_boolean_image_integration():
    """
    Tests the main public denoising function with a noisy image.
    """
    img = np.full((100, 100), 0, dtype=np.uint8)
    # Main shape
    img[25:75, 25:75] = 1
    # Noise
    img[5, 5] = 1
    img[10, 10:12] = 1

    # The denoising function should remove the 1px and 2px noise components,
    # leaving only the large square.
    denoised_image = denoise_boolean_image(img.astype(bool))
    assert np.sum(denoised_image) == 50 * 50
