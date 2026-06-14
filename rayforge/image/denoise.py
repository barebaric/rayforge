import numpy as np
from raygeo.image import filter_components, get_component_areas


def _find_adaptive_area_threshold(areas):
    """
    Analyzes component areas to find a dynamic threshold for separating
    content from noise by identifying the largest gap in the component size
    distribution.
    """
    if len(areas) == 0:
        return 0

    areas_arr = np.array(areas, dtype=int)

    # A simple heuristic for obviously clean images with only large features
    unique_areas, counts = np.unique(areas_arr, return_counts=True)
    if (
        unique_areas.size > 0
        and np.min(unique_areas) > 10
        and np.all(counts < 10)
    ):
        return 2

    bin_counts = np.bincount(areas_arr)
    if len(bin_counts) <= 1:
        return 0

    # Get all area sizes that are actually present in the image
    present_areas = np.where(bin_counts > 0)[0]
    if present_areas.size <= 1:
        # Only one size of component exists (or none), so no noise to filter
        return 2

    # Find the largest gap between consecutive component sizes. This gap
    # likely separates the noise cluster from the content cluster.
    gaps = np.diff(present_areas)
    if gaps.size == 0:
        return 2  # Should not happen if present_areas.size > 1

    largest_gap_idx = np.argmax(gaps)

    # The last "noisy" area is the one just before the largest gap.
    last_noisy_area = present_areas[largest_gap_idx]
    threshold = last_noisy_area + 1

    # Cap the threshold at a sane upper limit for what can be
    # considered "noise" to prevent it from deleting large features.
    # 100 pixels is a generous but safe limit for noise.
    MAX_NOISE_AREA = 100
    capped_threshold = min(threshold, MAX_NOISE_AREA)

    return max(2, capped_threshold)


def denoise_boolean_image(boolean_image):
    """
    Applies an adaptive denoising pipeline to a boolean image to remove small,
    irrelevant features before tracing.
    """
    img_uint8 = boolean_image.astype(np.uint8)
    areas = get_component_areas(img_uint8)
    if len(areas) <= 1:
        return boolean_image

    min_area_threshold = _find_adaptive_area_threshold(areas)
    if min_area_threshold <= 1:
        return boolean_image

    cleaned = filter_components(img_uint8, min_area_threshold)
    return cleaned > 0
