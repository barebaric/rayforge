import numpy as np
import cv2


def _get_component_areas(boolean_image: np.ndarray) -> np.ndarray:
    """
    Analyzes a boolean image and returns the areas (in pixels) of all
    distinct components.

    Args:
        boolean_image: A NumPy array of dtype=bool.

    Returns:
        A NumPy array of integer areas for each detected component.
    """
    if not np.any(boolean_image):
        return np.array([], dtype=int)

    # Convert boolean image to uint8 for OpenCV
    img_uint8 = boolean_image.astype(np.uint8)

    # Find and get stats for each component.
    # The first label (0) is the background.
    output = cv2.connectedComponentsWithStats(img_uint8, connectivity=8)
    _num_labels, _labels, stats, _centroids = output

    # We only care about the areas of the actual components,
    # not the background.
    # stats[0] is the background, so we slice from index 1.
    areas = stats[1:, cv2.CC_STAT_AREA]
    return areas


def _find_adaptive_area_threshold(areas: np.ndarray) -> int:
    """
    Analyzes component areas to find a dynamic threshold for separating
    content from noise by identifying the largest gap in the component size
    distribution.
    """
    if areas.size == 0:
        return 0

    # A simple heuristic for obviously clean images with only large features
    unique_areas, counts = np.unique(areas, return_counts=True)
    if (
        unique_areas.size > 0
        and np.min(unique_areas) > 10
        and np.all(counts < 10)
    ):
        return 2

    bin_counts = np.bincount(areas)
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


def _filter_image_by_component_area(
    boolean_image: np.ndarray, min_area: int
) -> np.ndarray:
    """
    Removes all components from a boolean image that have an area smaller
    than the provided minimum area.

    Args:
        boolean_image: The source boolean image.
        min_area: The minimum number of pixels for a component to be kept.

    Returns:
        A new boolean image with small components removed.
    """
    if min_area <= 1:
        return boolean_image.copy()

    img_uint8 = boolean_image.astype(np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        img_uint8, connectivity=8
    )

    # Create a new blank image
    filtered_image = np.zeros_like(boolean_image, dtype=bool)

    # Iterate through each component (skipping background label 0)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            # If the component is large enough, add it to our new image
            filtered_image[labels == i] = True

    return filtered_image


def denoise_boolean_image(boolean_image: np.ndarray) -> np.ndarray:
    """
    Applies an adaptive denoising pipeline to a boolean image to remove small,
    irrelevant features before tracing.
    """
    component_areas = _get_component_areas(boolean_image)
    if component_areas.size <= 1:
        return boolean_image

    min_area_threshold = _find_adaptive_area_threshold(component_areas)
    cleaned_image = _filter_image_by_component_area(
        boolean_image, min_area_threshold
    )
    return cleaned_image
