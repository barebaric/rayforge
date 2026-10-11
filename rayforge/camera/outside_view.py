"""
Geometry for the optional camera view outside the workspace.

The workspace image is warped from the lens-corrected frame. Lens
correction keeps the frame size, so a strong barrel correction pushes
the frame edges out of the corrected image and drops them. The outside
view therefore samples the raw frame instead: every output pixel is
mapped from world coordinates through the alignment homography into the
lens-corrected image, and from there through the lens distortion model
into the raw frame. Without lens correction the corrected and raw frames
coincide and a plain perspective warp is used.
"""

import math
from dataclasses import dataclass

import cv2
import numpy as np

Pos = tuple[float, float]
Area = tuple[Pos, Pos]

# Upper bound for the normalized undistorted radius searched for the
# limit of the distortion model.
_RADIAL_SEARCH_MAX = 4.0
_RADIAL_SEARCH_STEPS = 4000
# Keep away from the radius where the distortion model folds back.
_RADIAL_SAFETY = 0.98
# Number of grid samples per axis used to measure lens coverage.
_COVERAGE_SAMPLES = 512


@dataclass(frozen=True)
class OutsideView:
    """The camera image around the workspace, for display only.

    Attributes:
        image: Premultiplied BGRA image covering ``area``; pixels without
            camera coverage are fully transparent.
        area: Physical area ((x_min, y_min), (x_max, y_max)) in mm.
        margin_mm: The effective margin around the workspace, in mm.
    """

    image: np.ndarray
    area: Area
    margin_mm: float


@dataclass(frozen=True)
class LensModel:
    """The lens correction applied to a frame, in OpenCV conventions."""

    camera_matrix: np.ndarray
    distortion: np.ndarray

    def key(self) -> tuple:
        return (
            np.asarray(self.camera_matrix, dtype=np.float64).tobytes(),
            np.asarray(self.distortion, dtype=np.float64).tobytes(),
        )


def expand_area(area: Area, margin: float) -> Area:
    (x_min, y_min), (x_max, y_max) = area
    return (x_min - margin, y_min - margin), (x_max + margin, y_max + margin)


def native_output_size(
    H: np.ndarray,
    physical_area: Area,
    margin_mm: float,
    max_dimension: int,
) -> tuple[int, int]:
    """
    Size an expanded output at the camera's pixel density.

    The density is measured along the workspace edges in the corrected
    frame, using the sharper of both axes, since rendering finer than the
    camera resolves adds no detail. It depends only on the alignment, not
    on the display size, so resizing the window does not change it.
    """
    (x_min, y_min), (x_max, y_max) = physical_area
    corners = (
        np.array(
            [
                [x_min, y_min, 1],
                [x_max, y_min, 1],
                [x_max, y_max, 1],
                [x_min, y_max, 1],
            ],
            dtype=np.float64,
        )
        @ H.T
    )
    width_mm = x_max - x_min
    height_mm = y_max - y_min
    density = 0.0
    if np.all(corners[:, 2] > 0):
        px = corners[:, :2] / corners[:, 2:3]
        edges = np.linalg.norm(px - np.roll(px, -1, axis=0), axis=1)
        density = max(
            (edges[0] + edges[2]) / (2 * width_mm),
            (edges[1] + edges[3]) / (2 * height_mm),
        )
    width = width_mm + 2 * margin_mm
    height = height_mm + 2 * margin_mm
    if not math.isfinite(density) or density <= 0:
        density = max_dimension / max(width, height)
    width *= density
    height *= density
    largest = max(width, height)
    if largest > max_dimension:
        width *= max_dimension / largest
        height *= max_dimension / largest
    return max(1, round(width)), max(1, round(height))


def output_to_world(output_size: tuple[int, int], area: Area) -> np.ndarray:
    """The matrix mapping output pixels to world coordinates (Y-up)."""
    (x_min, y_min), (x_max, y_max) = area
    width_px, height_px = output_size
    return np.array(
        [
            [(x_max - x_min) / width_px, 0, x_min],
            [0, -(y_max - y_min) / height_px, y_max],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )


def _distance_beyond(xs: np.ndarray, ys: np.ndarray, area: Area) -> float:
    """The largest distance by which points lie outside an area."""
    if xs.size == 0:
        return 0.0
    (x_min, y_min), (x_max, y_max) = area
    extent = max(
        x_min - xs.min(),
        xs.max() - x_max,
        y_min - ys.min(),
        ys.max() - y_max,
    )
    return max(0.0, float(extent))


def frame_coverage_extent(
    H: np.ndarray, frame_shape: tuple[int, ...], physical_area: Area
) -> float:
    """
    How far, in mm, an undistorted frame reaches beyond the given area.

    Projects the corners of the frame into world coordinates and returns
    the largest distance by which they extend past any side of
    ``physical_area``, or 0 if they do not. A corner at or beyond the
    horizon of the plane counts as unbounded.
    """
    height, width = frame_shape[:2]
    corners = np.array(
        [[0, 0, 1], [width, 0, 1], [width, height, 1], [0, height, 1]],
        dtype=np.float64,
    )
    projected = corners @ np.linalg.inv(H).T
    w = projected[:, 2]
    if np.any(w <= 0) or not np.all(np.isfinite(projected)):
        return math.inf
    return _distance_beyond(
        projected[:, 0] / w, projected[:, 1] / w, physical_area
    )


def radial_limit(distortion: np.ndarray) -> float:
    """
    The largest normalized undistorted radius the radial distortion model
    maps one-to-one.

    Beyond this radius r * (1 + k1 r^2 + k2 r^4 + k3 r^6) stops growing,
    so the model folds back and would mirror image content.
    """
    k1, k2, _p1, _p2, k3 = (list(np.ravel(distortion)) + [0.0] * 5)[:5]
    r = np.linspace(0.0, _RADIAL_SEARCH_MAX, _RADIAL_SEARCH_STEPS)
    r2 = r * r
    slope = 1 + 3 * k1 * r2 + 5 * k2 * r2**2 + 7 * k3 * r2**3
    folded = np.nonzero(slope <= 0)[0]
    if folded.size == 0:
        return math.inf
    return float(r[folded[0]]) * _RADIAL_SAFETY


def world_to_raw(
    H: np.ndarray,
    lens: LensModel,
    xs: np.ndarray,
    ys: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Map world points to raw frame pixel coordinates.

    The homography gives the position in the lens-corrected frame, which
    is then distorted with the lens model, exactly inverting the
    correction applied by ``cv2.undistort`` with the same camera matrix.

    Returns:
        Raw x and y pixel coordinates, and a mask of points for which the
        mapping is valid (in front of the camera and within the range
        the distortion model maps one-to-one).
    """
    u = H[0, 0] * xs + H[0, 1] * ys + H[0, 2]
    v = H[1, 0] * xs + H[1, 1] * ys + H[1, 2]
    w = H[2, 0] * xs + H[2, 1] * ys + H[2, 2]
    valid = w > 1e-12
    w = np.where(valid, w, 1.0)
    u = u / w
    v = v / w

    K = lens.camera_matrix
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = (list(np.ravel(lens.distortion)) + [0.0] * 5)[:5]
    x = (u - cx) / fx
    y = (v - cy) / fy
    r2 = x * x + y * y
    limit = radial_limit(lens.distortion)
    valid &= r2 < limit * limit
    radial = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
    xd = x * radial + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    yd = y * radial + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    return fx * xd + cx, fy * yd + cy, valid


def lens_coverage_extent(
    H: np.ndarray,
    lens: LensModel,
    frame_shape: tuple[int, ...],
    physical_area: Area,
    margin_mm: float,
) -> float:
    """
    How far, in mm and up to ``margin_mm``, the raw frame reaches beyond
    the given area.

    Samples the expanded area on a grid, maps every sample into the raw
    frame and measures the valid samples that land inside it.
    """
    (x_min, y_min), (x_max, y_max) = expand_area(physical_area, margin_mm)
    xs, ys = np.meshgrid(
        np.linspace(x_min, x_max, _COVERAGE_SAMPLES),
        np.linspace(y_min, y_max, _COVERAGE_SAMPLES),
    )
    raw_x, raw_y, valid = world_to_raw(H, lens, xs, ys)
    height, width = frame_shape[:2]
    valid &= (raw_x >= 0) & (raw_x <= width - 1)
    valid &= (raw_y >= 0) & (raw_y <= height - 1)
    extent = _distance_beyond(xs[valid], ys[valid], physical_area)
    return min(extent, margin_mm)


def lens_remap_maps(
    H: np.ndarray,
    lens: LensModel,
    output_size: tuple[int, int],
    area: Area,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build ``cv2.remap`` maps sampling the raw frame for an output area.

    Output pixels without a valid mapping point outside the frame, so
    they come out transparent.
    """
    width_px, height_px = output_size
    cols, rows = np.meshgrid(
        np.arange(width_px, dtype=np.float64),
        np.arange(height_px, dtype=np.float64),
    )
    T = output_to_world(output_size, area)
    xs = T[0, 0] * cols + T[0, 2]
    ys = T[1, 1] * rows + T[1, 2]
    raw_x, raw_y, valid = world_to_raw(H, lens, xs, ys)
    raw_x = np.where(valid, raw_x, -1.0).astype(np.float32)
    raw_y = np.where(valid, raw_y, -1.0).astype(np.float32)
    return cv2.convertMaps(raw_x, raw_y, cv2.CV_16SC2)


def to_opaque_bgra(image: np.ndarray) -> np.ndarray:
    """An opaque BGRA copy of a frame."""
    bgr = np.ascontiguousarray(image[:, :, :3])
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2BGRA)


def remap_transparent(
    image: np.ndarray, maps: tuple[np.ndarray, np.ndarray]
) -> np.ndarray:
    """
    Remap an opaque BGRA image, transparent where there is no source.

    Interpolating an opaque image against a transparent border yields
    premultiplied alpha, including partial edge pixels.
    """
    return cv2.remap(
        image,
        maps[0],
        maps[1],
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )


def warp_transparent(
    image: np.ndarray,
    H: np.ndarray,
    output_size: tuple[int, int],
    area: Area,
) -> np.ndarray:
    """Warp an opaque BGRA image to an area, transparent elsewhere."""
    M = H @ output_to_world(output_size, area)
    return cv2.warpPerspective(
        image,
        np.linalg.inv(M),
        output_size,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0, 0),
    )
