"""
Tests for rayforge.core.geo.polygon module.
"""

from typing import cast

from rayforge.core.geo.polygon import (
    Polygon,
    almost_equal,
    polygon_area,
    polygon_bounds,
    polygon_centroid,
    rotate_polygon,
    translate_polygon,
    scale_polygon,
    convex_hull,
    clean_polygon,
    polygon_offset,
    polygon_union,
    polygon_intersection,
    polygon_difference,
    point_in_polygon,
    polygons_intersect,
)


def P(*points) -> Polygon:
    """Helper to create a polygon from integer points."""
    return [(float(x), float(y)) for x, y in points]


class TestPolygonArea:
    def test_triangle(self):
        polygon = P((0, 0), (10, 0), (5, 5))
        area = polygon_area(polygon)
        assert abs(area - 25.0) < 0.001

    def test_square(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        area = polygon_area(polygon)
        assert abs(area - 100.0) < 0.001

    def test_ccw_positive(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        area = polygon_area(polygon)
        assert area > 0

    def test_cw_negative(self):
        polygon = P((0, 0), (0, 10), (10, 10), (10, 0))
        area = polygon_area(polygon)
        assert area < 0

    def test_empty(self):
        assert polygon_area(cast(Polygon, [])) == 0.0

    def test_single_point(self):
        assert polygon_area(P((0, 0))) == 0.0

    def test_two_points(self):
        assert polygon_area(P((0, 0), (1, 1))) == 0.0


class TestPolygonBounds:
    def test_basic(self):
        polygon = P((1, 2), (5, 3), (3, 7), (0, 5))
        min_x, min_y, max_x, max_y = polygon_bounds(polygon)
        assert min_x == 0
        assert min_y == 2
        assert max_x == 5
        assert max_y == 7

    def test_empty(self):
        assert polygon_bounds(cast(Polygon, [])) == (0.0, 0.0, 0.0, 0.0)

    def test_single_point(self):
        min_x, min_y, max_x, max_y = polygon_bounds(P((5, 10)))
        assert min_x == max_x == 5
        assert min_y == max_y == 10


class TestPolygonCentroid:
    def test_square(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        cx, cy = polygon_centroid(polygon)
        assert almost_equal(cx, 5.0)
        assert almost_equal(cy, 5.0)

    def test_triangle(self):
        polygon = P((0, 0), (6, 0), (3, 6))
        cx, cy = polygon_centroid(polygon)
        assert almost_equal(cx, 3.0)
        assert almost_equal(cy, 2.0)

    def test_empty(self):
        assert polygon_centroid(cast(Polygon, [])) == (0.0, 0.0)


class TestRotatePolygon:
    def test_90_degrees(self):
        polygon = P((1, 0), (2, 0), (2, 1))
        rotated = rotate_polygon(polygon, 90)
        assert almost_equal(rotated[0][0], 0)
        assert almost_equal(rotated[0][1], 1)

    def test_180_degrees(self):
        polygon = P((1, 0), (2, 0), (2, 1))
        rotated = rotate_polygon(polygon, 180)
        assert almost_equal(rotated[0][0], -1)
        assert almost_equal(rotated[0][1], 0)

    def test_360_degrees(self):
        polygon = P((1, 0), (2, 0), (2, 1))
        rotated = rotate_polygon(polygon, 360)
        assert almost_equal(rotated[0][0], 1)
        assert almost_equal(rotated[0][1], 0)

    def test_0_degrees(self):
        polygon = P((1, 2), (3, 4))
        rotated = rotate_polygon(polygon, 0)
        assert rotated == polygon

    def test_negative_angle(self):
        polygon = P((1, 0), (2, 0), (2, 1))
        rotated = rotate_polygon(polygon, -90)
        assert almost_equal(rotated[0][0], 0)
        assert almost_equal(rotated[0][1], -1)


class TestTranslatePolygon:
    def test_basic(self):
        polygon = P((0, 0), (10, 0), (5, 5))
        translated = translate_polygon(polygon, 5, 10)
        assert translated[0] == (5.0, 10.0)
        assert translated[1] == (15.0, 10.0)
        assert translated[2] == (10.0, 15.0)

    def test_negative(self):
        polygon = P((10, 20), (30, 40))
        translated = translate_polygon(polygon, -5, -10)
        assert translated[0] == (5.0, 10.0)
        assert translated[1] == (25.0, 30.0)

    def test_zero(self):
        polygon = P((1, 2), (3, 4))
        translated = translate_polygon(polygon, 0, 0)
        assert translated == polygon


class TestScalePolygon:
    def test_uniform(self):
        polygon = P((1, 1), (3, 1), (3, 3), (1, 3))
        scaled = scale_polygon(polygon, 2)
        assert scaled[0] == (2.0, 2.0)
        assert scaled[2] == (6.0, 6.0)

    def test_non_uniform(self):
        polygon = P((1, 1), (3, 1), (3, 3), (1, 3))
        scaled = scale_polygon(polygon, 2, 3)
        assert scaled[0] == (2.0, 3.0)
        assert scaled[2] == (6.0, 9.0)

    def test_shrink(self):
        polygon = P((2, 2), (6, 2), (6, 6), (2, 6))
        scaled = scale_polygon(polygon, 0.5)
        assert scaled[0] == (1.0, 1.0)
        assert scaled[2] == (3.0, 3.0)


class TestConvexHull:
    def test_basic(self):
        polygon = P((0, 0), (5, 3), (10, 0), (5, 5), (5, 2))
        hull = convex_hull(polygon)
        assert len(hull) >= 3
        hull_set = set(hull)
        assert (0.0, 0.0) in hull_set
        assert (10.0, 0.0) in hull_set
        assert (5.0, 5.0) in hull_set

    def test_already_convex(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        hull = convex_hull(polygon)
        assert len(hull) == 4

    def test_triangle(self):
        polygon = P((0, 0), (5, 10), (10, 0))
        hull = convex_hull(polygon)
        assert len(hull) == 3


class TestCleanPolygon:
    def test_valid_triangle(self):
        polygon = P((0, 0), (10, 0), (5, 5))
        cleaned = clean_polygon(polygon)
        assert cleaned is not None
        assert len(cleaned) == 3

    def test_empty(self):
        assert clean_polygon(cast(Polygon, [])) is None

    def test_single_point(self):
        assert clean_polygon(P((0, 0))) is None

    def test_two_points(self):
        assert clean_polygon(P((0, 0), (1, 1))) is None


class TestPolygonOffset:
    def test_expand(self):
        polygon = P((0, 0), (10, 0), (5, 10))
        offset_polys = polygon_offset(polygon, 1.0)
        assert len(offset_polys) >= 1
        expanded_area = abs(polygon_area(offset_polys[0]))
        original_area = abs(polygon_area(polygon))
        assert expanded_area > original_area

    def test_shrink(self):
        polygon = P((0, 0), (10, 0), (5, 10))
        offset_polys = polygon_offset(polygon, -0.5)
        assert len(offset_polys) >= 1
        shrunk_area = abs(polygon_area(offset_polys[0]))
        original_area = abs(polygon_area(polygon))
        assert shrunk_area < original_area

    def test_zero_offset(self):
        polygon = P((0, 0), (10, 0), (5, 10))
        offset_polys = polygon_offset(polygon, 0)
        assert len(offset_polys) == 1
        assert offset_polys[0] == polygon

    def test_empty(self):
        assert polygon_offset(cast(Polygon, []), 1.0) == []


class TestPolygonBooleanOps:
    def test_union(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((5, 5), (15, 5), (15, 15), (5, 15))
        result = polygon_union([poly1, poly2])
        assert len(result) >= 1

    def test_intersection(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((5, 5), (15, 5), (15, 15), (5, 15))
        result = polygon_intersection(poly1, poly2)
        assert len(result) >= 1
        expected_area = 25.0
        actual_area = sum(abs(polygon_area(p)) for p in result)
        assert abs(actual_area - expected_area) < 0.1

    def test_no_intersection(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((20, 20), (30, 20), (30, 30), (20, 30))
        result = polygon_intersection(poly1, poly2)
        assert len(result) == 0

    def test_difference(self):
        poly1 = P((0, 0), (20, 0), (20, 20), (0, 20))
        poly2 = P((5, 5), (15, 5), (15, 15), (5, 15))
        result = polygon_difference(poly1, poly2)
        assert len(result) >= 1


class TestPointInPolygon:
    def test_inside(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        assert point_in_polygon((5, 5), polygon) is True

    def test_outside(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        assert point_in_polygon((15, 15), polygon) is False

    def test_on_edge(self):
        polygon = P((0, 0), (10, 0), (10, 10), (0, 10))
        assert point_in_polygon((5, 0), polygon) is True

    def test_empty_polygon(self):
        assert point_in_polygon((5, 5), cast(Polygon, [])) is False


class TestPolygonsIntersect:
    def test_overlapping(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((5, 5), (15, 5), (15, 15), (5, 15))
        assert polygons_intersect(poly1, poly2) is True

    def test_non_overlapping(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((20, 20), (30, 20), (30, 30), (20, 30))
        assert polygons_intersect(poly1, poly2) is False

    def test_touching(self):
        poly1 = P((0, 0), (10, 0), (10, 10), (0, 10))
        poly2 = P((10, 0), (20, 0), (20, 10), (10, 10))
        result = polygons_intersect(poly1, poly2)
        assert result is False

    def test_empty(self):
        assert (
            polygons_intersect(cast(Polygon, []), P((0, 0), (1, 0), (1, 1)))
            is False
        )


class TestAlmostEqual:
    def test_equal(self):
        assert almost_equal(1.0, 1.0) is True

    def test_close(self):
        assert almost_equal(1.0, 1.0 + 1e-10) is True

    def test_not_close(self):
        assert almost_equal(1.0, 1.1) is False

    def test_custom_tolerance(self):
        assert almost_equal(1.0, 1.01, tolerance=0.1) is True
        assert almost_equal(1.0, 1.01, tolerance=0.001) is False
