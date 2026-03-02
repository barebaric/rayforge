"""
Tests for rayforge.core.geo.minkowski module.
"""

import pytest

from rayforge.core.geo.minkowski import (
    convolve_two_segments,
    convolve_point_sequences,
    calculate_input_scale,
)


class TestConvolveTwoSegments:
    """Tests for convolve_two_segments function."""

    def test_horizontal_vertical(self):
        a1 = (0, 0)
        a2 = (10, 0)
        b1 = (0, 0)
        b2 = (0, 10)
        result = convolve_two_segments(a1, a2, b1, b2)
        assert len(result) == 4
        assert result == [(0, 10), (0, 0), (10, 0), (10, 10)]

    def test_diagonal_segments(self):
        a1 = (0, 0)
        a2 = (5, 5)
        b1 = (0, 0)
        b2 = (3, 0)
        result = convolve_two_segments(a1, a2, b1, b2)
        assert len(result) == 4
        expected = [(3, 0), (0, 0), (5, 5), (8, 5)]
        assert result == expected

    def test_zero_length_segment(self):
        a1 = (5, 5)
        a2 = (5, 5)
        b1 = (0, 0)
        b2 = (10, 10)
        result = convolve_two_segments(a1, a2, b1, b2)
        assert len(result) == 4
        assert result == [(15, 15), (5, 5), (5, 5), (15, 15)]

    def test_negative_coordinates(self):
        a1 = (-5, -5)
        a2 = (5, 5)
        b1 = (-3, 0)
        b2 = (3, 0)
        result = convolve_two_segments(a1, a2, b1, b2)
        assert len(result) == 4

    def test_large_coordinates(self):
        a1 = (0, 0)
        a2 = (1000000, 0)
        b1 = (0, 0)
        b2 = (0, 1000000)
        result = convolve_two_segments(a1, a2, b1, b2)
        assert len(result) == 4
        assert result[0] == (0, 1000000)
        assert result[2] == (1000000, 0)


class TestConvolvePointSequences:
    """Tests for convolve_point_sequences function."""

    def test_square_square(self):
        square_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        square_b = [(0, 0), (5, 0), (5, 5), (0, 5)]
        result = convolve_point_sequences(square_a, square_b)
        assert len(result) == 16

    def test_triangle_triangle(self):
        tri_a = [(0, 0), (10, 0), (5, 10)]
        tri_b = [(0, 0), (5, 0), (2, 5)]
        result = convolve_point_sequences(tri_a, tri_b)
        assert len(result) == 9

    def test_empty_path_a(self):
        path_b = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = convolve_point_sequences([], path_b)
        assert result == []

    def test_empty_path_b(self):
        path_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = convolve_point_sequences(path_a, [])
        assert result == []

    def test_single_point_path_a(self):
        path_a = [(5, 5)]
        path_b = [(0, 0), (10, 0), (10, 10), (0, 10)]
        result = convolve_point_sequences(path_a, path_b)
        assert result == []

    def test_single_point_path_b(self):
        path_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        path_b = [(5, 5)]
        result = convolve_point_sequences(path_a, path_b)
        assert result == []

    def test_two_point_paths(self):
        path_a = [(0, 0), (10, 0)]
        path_b = [(0, 0), (5, 5)]
        result = convolve_point_sequences(path_a, path_b)
        assert len(result) == 4

    def test_each_parallelogram_has_four_vertices(self):
        square_a = [(0, 0), (10, 0), (10, 10), (0, 10)]
        square_b = [(0, 0), (5, 0), (5, 5), (0, 5)]
        result = convolve_point_sequences(square_a, square_b)
        for parallelogram in result:
            assert len(parallelogram) == 4


class TestCalculateInputScale:
    """Tests for calculate_input_scale function."""

    def test_small_polygons(self):
        polygons = [[(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]]
        scale = calculate_input_scale(polygons)
        assert scale > 0
        assert isinstance(scale, float)

    def test_large_polygons(self):
        polygons = [
            [(0.0, 0.0), (1000.0, 0.0), (1000.0, 1000.0), (0.0, 1000.0)]
        ]
        scale = calculate_input_scale(polygons)
        assert scale > 0
        assert scale < 1e6

    def test_negative_coordinates(self):
        polygons = [
            [
                (-100.0, -100.0),
                (100.0, -100.0),
                (100.0, 100.0),
                (-100.0, 100.0),
            ]
        ]
        scale = calculate_input_scale(polygons)
        assert scale > 0

    def test_multiple_polygons(self):
        polygons = [
            [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)],
            [(50.0, 50.0), (60.0, 50.0), (60.0, 60.0), (50.0, 60.0)],
        ]
        scale = calculate_input_scale(polygons)
        assert scale > 0

    def test_empty_polygon_list(self):
        scale = calculate_input_scale([])
        assert scale == pytest.approx(0.1 * 2147483647)

    def test_custom_max_int(self):
        polygons = [[(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]]
        scale = calculate_input_scale(polygons, max_int=1000000)
        assert scale > 0
        assert scale < 100000

    def test_very_small_coordinates(self):
        polygons = [
            [(0.001, 0.001), (0.002, 0.001), (0.002, 0.002), (0.001, 0.002)]
        ]
        scale = calculate_input_scale(polygons)
        assert scale > 0
        assert scale == pytest.approx(0.1 * 2147483647)

    def test_very_large_coordinates(self):
        polygons = [[(0.0, 0.0), (1e9, 0.0), (1e9, 1e9), (0.0, 1e9)]]
        scale = calculate_input_scale(polygons)
        assert scale > 0
        assert scale < 1

    def test_mixed_polygon_sizes(self):
        polygons = [
            [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            [(0.0, 0.0), (10000.0, 0.0), (10000.0, 10000.0), (0.0, 10000.0)],
        ]
        scale = calculate_input_scale(polygons)
        assert scale > 0
