"""Polyline smoothing algorithms — Gaussian kernels, circular smoothing,
and sub-segment smoothing."""

from typing import List, Optional, Tuple

from rayforge.core.geo import Point3D


def compute_gaussian_kernel(amount: int) -> Tuple[List[float], float]:
    """Build a normalised Gaussian convolution kernel.

    Args:
        amount: Kernel half-width.  The full kernel has ``2 * amount + 1``
            elements.

    Returns:
        A tuple ``(kernel, sigma)`` where *kernel* is the list of
        weights and *sigma* is the standard deviation used.
    """
    ...


def smooth_circularly(
    points: List[Point3D],
    kernel: List[float],
) -> List[Point3D]:
    """Smooth a closed polyline using circular convolution.

    The polyline is treated as a loop; wrapping is handled by indexing
    modulo the point count.

    Args:
        points: Closed polyline vertices ``(x, y, z)``.
        kernel: Convolution kernel (e.g. from :func:`compute_gaussian_kernel`).

    Returns:
        Smoothed polyline with the same number of points.
    """
    ...


def smooth_polyline(
    points: List[Point3D],
    amount: int,
    corner_angle_threshold: float,
    is_closed: Optional[float] = ...,
) -> List[Point3D]:
    """Smooth a polyline with corner preservation.

    Applies Gaussian smoothing but skips corners where the angle is
    sharper than *corner_angle_threshold*.

    Args:
        points: Vertices ``(x, y, z)``.
        amount: Smoothing strength (kernel half-width).
        corner_angle_threshold: Angle in radians below which a vertex
            is treated as a sharp corner and left un-smoothed.
        is_closed: Whether the polyline is closed.  ``None`` lets the
            algorithm decide.  Defaults to ``None``.

    Returns:
        Smoothed polyline.
    """
    ...


def smooth_sub_segment(
    points: List[Point3D],
    kernel: List[float],
) -> List[Point3D]:
    """Smooth an open sub-segment of a polyline.

    Unlike :func:`smooth_circularly`, no wrapping is performed.

    Args:
        points: Sub-segment vertices ``(x, y, z)``.
        kernel: Convolution kernel.

    Returns:
        Smoothed sub-segment.
    """
    ...


def resample_polyline(
    points: List[Point3D],
    max_segment_length: float,
    is_closed: bool,
) -> List[Point3D]:
    """Resample a polyline so that no segment exceeds *max_segment_length*.

    New points are inserted by linear interpolation along each segment.

    Args:
        points: Vertices ``(x, y, z)``.
        max_segment_length: Maximum allowed distance between consecutive
            points.
        is_closed: Whether the polyline forms a closed loop.

    Returns:
        Resampled polyline.
    """
    ...
