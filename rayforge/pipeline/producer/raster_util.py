import math
from dataclasses import dataclass
from typing import Iterator, List, Optional, Tuple

import numpy as np


@dataclass
class ScanLine:
    """
    Represents a single scan line for raster operations.

    A scan line is defined by its start and end positions in millimeters,
    and the pixel coordinates along the line for sampling the source image.
    """

    index: int
    start_mm: Tuple[float, float]
    end_mm: Tuple[float, float]
    pixels: np.ndarray
    line_interval_mm: float

    @property
    def length_mm(self) -> float:
        """Return the length of the scan line in millimeters."""
        dx = self.end_mm[0] - self.start_mm[0]
        dy = self.end_mm[1] - self.start_mm[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def direction(self) -> Tuple[float, float]:
        """Return the normalized direction vector."""
        length = self.length_mm
        if length < 1e-9:
            return (1.0, 0.0)
        dx = (self.end_mm[0] - self.start_mm[0]) / length
        dy = (self.end_mm[1] - self.start_mm[1]) / length
        return (dx, dy)

    def pixel_to_mm(
        self, px: int, py: int, pixels_per_mm: Tuple[float, float]
    ) -> Tuple[float, float]:
        """
        Project a pixel coordinate onto this scan line and return mm coords.

        The pixel is projected onto the line's axis to preserve alignment
        with the global grid.
        """
        px_per_mm_x, px_per_mm_y = pixels_per_mm
        px_mm = px / px_per_mm_x
        py_mm = py / px_per_mm_y

        dx, dy = self.direction
        cx = (self.start_mm[0] + self.end_mm[0]) / 2
        cy = (self.start_mm[1] + self.end_mm[1]) / 2

        t = (px_mm - cx) * dx + (py_mm - cy) * dy
        return (cx + t * dx, cy + t * dy)


def find_bounding_box(
    image: np.ndarray, threshold: int = 255
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bounding box of pixels below a threshold.

    Args:
        image: 2D numpy array.
        threshold: Maximum value to consider "occupied" (default 255 for
            non-white).

    Returns:
        Tuple of (y_min, y_max, x_min, x_max) or None if all pixels are
        above threshold.
    """
    occupied = image < threshold
    occupied_rows = np.any(occupied, axis=1)
    occupied_cols = np.any(occupied, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return None

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def find_mask_bounding_box(
    mask: np.ndarray,
) -> Optional[Tuple[int, int, int, int]]:
    """
    Find the bounding box of non-zero pixels in a mask.

    Args:
        mask: 2D numpy array with binary values.

    Returns:
        Tuple of (y_min, y_max, x_min, x_max) or None if mask is empty.
    """
    occupied_rows = np.any(mask, axis=1)
    occupied_cols = np.any(mask, axis=0)

    if not np.any(occupied_rows) or not np.any(occupied_cols):
        return None

    y_min, y_max = np.where(occupied_rows)[0][[0, -1]]
    x_min, x_max = np.where(occupied_cols)[0][[0, -1]]
    return y_min, y_max, x_min, x_max


def generate_horizontal_scan_positions(
    y_min_px: int,
    y_max_px: int,
    height_px: int,
    pixels_per_mm: Tuple[float, float],
    line_interval_mm: float,
    offset_y_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate Y positions for horizontal scan lines aligned to global grid.

    This returns floating-point Y positions for sub-pixel sampling, aligned
    to a global grid to ensure consistency across chunks.

    Args:
        y_min_px: Minimum Y pixel with content.
        y_max_px: Maximum Y pixel with content.
        height_px: Total image height in pixels.
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.
        line_interval_mm: Distance between scan lines in millimeters.
        offset_y_mm: Global Y offset for alignment.

    Returns:
        Tuple of (y_coords_mm, y_coords_px) arrays.
        y_coords_mm: Y positions in millimeters.
        y_coords_px: Y positions in pixels (floating-point for sub-pixel).
    """
    px_per_mm_y = pixels_per_mm[1]

    y_min_mm = y_min_px / px_per_mm_y
    y_max_mm = (y_max_px + 1) / px_per_mm_y

    global_y_min_mm = offset_y_mm + y_min_mm
    num_intervals = math.ceil(global_y_min_mm / line_interval_mm)
    first_scan_y_mm = num_intervals * line_interval_mm - offset_y_mm

    y_coords_mm = np.arange(first_scan_y_mm, y_max_mm, line_interval_mm)

    if len(y_coords_mm) == 0:
        return np.array([]), np.array([])

    y_coords_px = y_coords_mm * px_per_mm_y
    y_coords_px = np.clip(y_coords_px, 0, height_px - 1)

    return y_coords_mm, y_coords_px


def resample_rows(image: np.ndarray, y_coords_px: np.ndarray) -> np.ndarray:
    """
    Resample image rows with sub-pixel interpolation.

    For each Y coordinate, interpolates between the two nearest rows.

    Args:
        image: 2D numpy array (height, width).
        y_coords_px: Floating-point Y coordinates in pixels.

    Returns:
        2D numpy array of shape (len(y_coords_px), width) with resampled
        values.
    """
    if len(y_coords_px) == 0:
        return np.array([]).reshape(0, image.shape[1])

    height_px = image.shape[0]
    y0 = np.floor(y_coords_px).astype(int)
    y1 = np.clip(np.ceil(y_coords_px).astype(int), 0, height_px - 1)
    y_frac = y_coords_px - y0

    row0 = image[y0, :]
    row1 = image[y1, :]

    return row0 * (1 - y_frac[:, np.newaxis]) + row1 * y_frac[:, np.newaxis]


def line_pixels(
    start: Tuple[float, float],
    end: Tuple[float, float],
    width: int,
    height: int,
) -> np.ndarray:
    """
    Get pixel coordinates along a line using Bresenham's algorithm.

    Args:
        start: (x, y) start coordinates in pixels
        end: (x, y) end coordinates in pixels
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Array of (x, y) pixel coordinates along the line
    """
    x0, y0 = int(round(start[0])), int(round(start[1]))
    x1, y1 = int(round(end[0])), int(round(end[1]))

    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    pixels = []
    x, y = x0, y0

    while True:
        if 0 <= x < width and 0 <= y < height:
            pixels.append((x, y))
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy

    return np.array(pixels, dtype=np.int32)


def generate_scan_lines(
    bbox: Tuple[int, int, int, int],
    image_size: Tuple[int, int],
    pixels_per_mm: Tuple[float, float],
    line_interval_mm: float,
    direction_degrees: float = 0.0,
    offset_x_mm: float = 0.0,
    offset_y_mm: float = 0.0,
) -> Iterator[ScanLine]:
    """
    Generate scan lines aligned to a global grid.

    This function yields ScanLine objects that cover the bounding box,
    with lines positioned at regular intervals along the perpendicular
    axis to the scan direction. Line positions are aligned to the global
    grid (accounting for offsets) to ensure consistent alignment across
    multiple chunks.

    Args:
        bbox: Bounding box as (y_min, y_max, x_min, x_max) in pixels.
        image_size: Image dimensions as (width, height) in pixels.
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.
        line_interval_mm: Distance between scan lines in millimeters.
        direction_degrees: Scan direction in degrees (0 = horizontal,
            90 = vertical).
        offset_x_mm: Global X offset for line alignment across chunks.
        offset_y_mm: Global Y offset for line alignment across chunks.

    Yields:
        ScanLine objects for each scan line that intersects the bounding box.
    """
    y_min, y_max, x_min, x_max = bbox
    width, height = image_size
    px_per_mm_x, px_per_mm_y = pixels_per_mm

    angle_rad = math.radians(direction_degrees)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    bbox_width_mm = (x_max - x_min + 1) / px_per_mm_x
    bbox_height_mm = (y_max - y_min + 1) / px_per_mm_y

    center_x_mm = (x_min + x_max + 1) / (2 * px_per_mm_x)
    center_y_mm = (y_min + y_max + 1) / (2 * px_per_mm_y)

    diag_mm = math.sqrt(bbox_width_mm**2 + bbox_height_mm**2)

    perp_angle_rad = angle_rad + math.pi / 2
    perp_cos = math.cos(perp_angle_rad)
    perp_sin = math.sin(perp_angle_rad)

    global_center_perp = (center_x_mm + offset_x_mm) * perp_cos + (
        center_y_mm + offset_y_mm
    ) * perp_sin

    perp_extent_start = global_center_perp - diag_mm / 2
    perp_extent_end = global_center_perp + diag_mm / 2

    first_line_index = math.ceil(perp_extent_start / line_interval_mm)
    last_line_index = math.floor(perp_extent_end / line_interval_mm)

    for line_index in range(first_line_index, last_line_index + 1):
        line_global_perp = line_index * line_interval_mm
        line_local_perp = line_global_perp - global_center_perp

        line_center_x_mm = center_x_mm + line_local_perp * perp_cos
        line_center_y_mm = center_y_mm + line_local_perp * perp_sin

        half_diag = diag_mm / 2
        start_x_mm = line_center_x_mm - half_diag * cos_a
        start_y_mm = line_center_y_mm - half_diag * sin_a
        end_x_mm = line_center_x_mm + half_diag * cos_a
        end_y_mm = line_center_y_mm + half_diag * sin_a

        start_x_px = start_x_mm * px_per_mm_x
        start_y_px = start_y_mm * px_per_mm_y
        end_x_px = end_x_mm * px_per_mm_x
        end_y_px = end_y_mm * px_per_mm_y

        pixels = line_pixels(
            (start_x_px, start_y_px),
            (end_x_px, end_y_px),
            width,
            height,
        )

        if len(pixels) == 0:
            continue

        yield ScanLine(
            index=line_index,
            start_mm=(start_x_mm, start_y_mm),
            end_mm=(end_x_mm, end_y_mm),
            pixels=pixels,
            line_interval_mm=line_interval_mm,
        )


def find_segments(values: np.ndarray) -> List[Tuple[int, int]]:
    """
    Find contiguous segments where values are non-zero.

    Args:
        values: 1D numpy array of binary values (0 or non-zero).

    Returns:
        List of (start_index, end_index) tuples for each segment.
        End index is exclusive (Python slice convention).
    """
    if len(values) == 0:
        return []

    padded = np.concatenate(([0], (values != 0).astype(int), [0]))
    diffs = np.diff(padded)
    starts = np.where(diffs == 1)[0]
    ends = np.where(diffs == -1)[0]

    return list(zip(starts.tolist(), ends.tolist()))


def calculate_ymax_mm(
    image_size: Tuple[int, int],
    pixels_per_mm: Tuple[float, float],
) -> float:
    """
    Calculate the Y-max value for coordinate conversion.

    The output coordinate system has Y inverted (0 at bottom), so this
    returns the height in mm for converting from the image's top-left
    origin to the output's bottom-left origin.

    Args:
        image_size: Image dimensions as (width, height) in pixels.
        pixels_per_mm: Resolution as (x, y) pixels per millimeter.

    Returns:
        The Y-max value in millimeters.
    """
    height_px = image_size[1]
    px_per_mm_y = pixels_per_mm[1]
    return height_px / px_per_mm_y


def convert_y_to_output(y_mm: float, ymax_mm: float) -> float:
    """
    Convert Y coordinate from image space to output space.

    Image space: Y=0 at top, Y increases downward.
    Output space: Y=0 at bottom, Y increases upward.

    Args:
        y_mm: Y coordinate in image space (millimeters).
        ymax_mm: Maximum Y value (image height in mm).

    Returns:
        Y coordinate in output space.
    """
    return ymax_mm - y_mm
