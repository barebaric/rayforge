import math
from rayforge.core.geo.smooth import (
    compute_gaussian_kernel,
    smooth_sub_segment,
    smooth_circularly,
    smooth_polyline,
)
from rayforge.core.geo.types import Point3D


class TestComputeGaussianKernel:
    """Tests for compute_gaussian_kernel function."""

    def test_zero_amount_returns_unit_kernel(self):
        """Zero amount should return a kernel with a single 1.0 value."""
        kernel, sigma = compute_gaussian_kernel(0)
        assert kernel == [1.0]
        assert sigma == 0.0

    def test_low_amount_produces_small_sigma(self):
        """Low smoothing amount should produce small sigma."""
        kernel, sigma = compute_gaussian_kernel(10)
        assert sigma > 0
        assert sigma < 1.0
        assert len(kernel) > 1

    def test_high_amount_produces_larger_kernel(self):
        """Higher smoothing should produce larger kernel."""
        kernel_low, _ = compute_gaussian_kernel(20)
        kernel_high, _ = compute_gaussian_kernel(80)
        assert len(kernel_high) > len(kernel_low)

    def test_kernel_is_normalized(self):
        """Kernel values should sum to approximately 1.0."""
        for amount in [10, 50, 100]:
            kernel, _ = compute_gaussian_kernel(amount)
            assert abs(sum(kernel) - 1.0) < 1e-10

    def test_kernel_is_symmetric(self):
        """Gaussian kernel should be symmetric around center."""
        kernel, _ = compute_gaussian_kernel(50)
        n = len(kernel)
        for i in range(n // 2):
            assert abs(kernel[i] - kernel[n - 1 - i]) < 1e-10


class TestSmoothSubSegment:
    """Tests for smooth_sub_segment function."""

    def test_preserves_endpoints(self):
        """Endpoints should be preserved exactly."""
        points: list[Point3D] = [(0, 0, 0), (5, 5, 0), (10, 0, 0)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_sub_segment(points, kernel)
        assert smoothed[0] == points[0]
        assert smoothed[-1] == points[-1]

    def test_returns_same_for_short_input(self):
        """Should return input unchanged for less than 3 points."""
        points: list[Point3D] = [(0, 0, 0), (10, 0, 0)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_sub_segment(points, kernel)
        assert smoothed == points

    def test_preserves_z_coordinate(self):
        """Z coordinate should be preserved from original point."""
        points: list[Point3D] = [(0, 0, 5), (5, 5, 5), (10, 0, 5)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_sub_segment(points, kernel)
        for i, point in enumerate(smoothed):
            assert abs(point[2] - 5.0) < 1e-10


class TestSmoothCircularly:
    """Tests for smooth_circularly function."""

    def test_closes_path(self):
        """Result should have first point appended at end."""
        points: list[Point3D] = [(0, 0, 0), (10, 0, 0), (5, 10, 0)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_circularly(points, kernel)
        assert smoothed[0] == smoothed[-1]

    def test_returns_same_for_short_input(self):
        """Should return input unchanged for less than 3 points."""
        points: list[Point3D] = [(0, 0, 0), (10, 0, 0)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_circularly(points, kernel)
        assert smoothed == points

    def test_preserves_z_coordinate(self):
        """Z coordinate should be preserved from original point."""
        points: list[Point3D] = [(0, 0, 3), (10, 0, 3), (5, 10, 3)]
        kernel, _ = compute_gaussian_kernel(50)
        smoothed = smooth_circularly(points, kernel)
        for point in smoothed:
            assert abs(point[2] - 3.0) < 1e-10


class TestSmoothPolyline:
    """Tests for smooth_polyline function."""

    def test_zero_amount_returns_input(self):
        """Zero amount should return input unchanged."""
        points: list[Point3D] = [(0, 0, 0), (10, 0, 0), (20, 10, 0)]
        result = smooth_polyline(points, 0, 45)
        assert result == points

    def test_short_input_returns_unchanged(self):
        """Less than 3 points should return input unchanged."""
        points: list[Point3D] = [(0, 0, 0), (10, 0, 0)]
        result = smooth_polyline(points, 50, 45)
        assert result == points

    def test_preserves_open_path_endpoints(self):
        """Open paths should have exact endpoints preserved."""
        points: list[Point3D] = [
            (0, 0, 0),
            (10, 0, 0),
            (20, 10, 0),
            (30, 0, 0),
            (40, 0, 0),
        ]
        result = smooth_polyline(points, 50, 45, is_closed=False)
        assert result[0] == points[0]
        assert result[-1] == points[-1]

    def test_closed_path_detection(self):
        """Should auto-detect closed path when start == end."""
        points: list[Point3D] = [
            (0, 0, 0),
            (10, 0, 0),
            (10, 10, 0),
            (0, 10, 0),
            (0, 0, 0),
        ]
        result = smooth_polyline(points, 30, 120, is_closed=None)
        assert result[0] == result[-1]

    def test_sharp_corner_preserved(self):
        """Sharp corners below threshold should be preserved."""
        points: list[Point3D] = [
            (0, 50, 0),
            (50, 0, 0),
            (100, 50, 0),
        ]
        result = smooth_polyline(points, 30, 95, is_closed=False)
        corner_point = (50, 0, 0)
        closest = min(result, key=lambda p: math.dist(p, corner_point))
        dist = math.dist(closest, corner_point)
        assert dist < 1.0, "Sharp corner should be preserved"

    def test_dull_corner_smoothed(self):
        """Dull corners above threshold should be smoothed."""
        points: list[Point3D] = [
            (0, 50, 0),
            (50, 0, 0),
            (100, 50, 0),
            (150, 50, 0),
        ]
        result = smooth_polyline(points, 40, 95, is_closed=False)
        dull_corner = (100, 50, 0)
        closest = min(result, key=lambda p: math.dist(p, dull_corner))
        dist = math.dist(closest, dull_corner)
        assert dist > 0.1, "Dull corner should be smoothed"

    def test_closed_loop_smoothing(self):
        """Closed loops should be smoothed circularly."""
        points: list[Point3D] = [
            (0, 0, 0),
            (10, 0, 0),
            (10, 10, 0),
            (0, 10, 0),
        ]
        result = smooth_polyline(points, 50, 45, is_closed=True)
        assert len(result) >= 3
        assert result[0] == result[-1]

    def test_z_preserved_through_smoothing(self):
        """Z coordinates should be preserved during smoothing."""
        points: list[Point3D] = [
            (0, 0, 7),
            (10, 0, 7),
            (20, 10, 7),
            (30, 0, 7),
            (40, 0, 7),
        ]
        result = smooth_polyline(points, 50, 45, is_closed=False)
        for point in result:
            assert abs(point[2] - 7.0) < 1e-10
