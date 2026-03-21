import pytest
from rayforge.core.sketcher.entities import Bezier
from rayforge.core.sketcher.registry import EntityRegistry
from rayforge.core.geo.geometry import Geometry


@pytest.fixture
def registry():
    return EntityRegistry()


def test_bezier_serialization_round_trip():
    original_bezier = Bezier(
        id=10,
        start_idx=1,
        end_idx=4,
        construction=True,
    )

    data = original_bezier.to_dict()
    assert data == {
        "id": 10,
        "type": "bezier",
        "construction": True,
        "start_idx": 1,
        "end_idx": 4,
    }

    new_bezier = Bezier.from_dict(data)
    assert isinstance(new_bezier, Bezier)
    assert new_bezier.id == original_bezier.id
    assert new_bezier.start_idx == original_bezier.start_idx
    assert new_bezier.end_idx == original_bezier.end_idx
    assert new_bezier.construction == original_bezier.construction


def test_bezier_get_point_ids(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(p1, p2))
    assert set(bezier.get_point_ids()) == {p1, p2}


def test_bezier_get_endpoint_ids(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(p1, p2))
    assert bezier.get_endpoint_ids() == [p1, p2]


def test_bezier_as_line(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)
    bid = registry.add_bezier(start, end)
    bezier = registry.get_entity(bid)

    assert bezier.is_line(registry) is True

    vertices = bezier.to_polygon_vertices(registry, forward=True)
    assert len(vertices) == 1
    assert vertices[0] == pytest.approx((20.0, 0.0))

    vertices_rev = bezier.to_polygon_vertices(registry, forward=False)
    assert len(vertices_rev) == 1
    assert vertices_rev[0] == pytest.approx((0.0, 0.0))


def test_bezier_with_control_points(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)

    bid = registry.add_bezier(start, end)
    bezier = registry.get_entity(bid)
    bezier.cp1 = (5.0, 10.0)
    bezier.cp2 = (-5.0, 10.0)

    assert bezier.is_line(registry) is False

    vertices = bezier.to_polygon_vertices(registry, forward=True)
    assert len(vertices) == 21
    assert vertices[0] == pytest.approx((0.0, 0.0))
    assert vertices[-1] == pytest.approx((20.0, 0.0))


def test_bezier_get_junction_point_ids(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(p1, p2))
    assert set(bezier.get_junction_point_ids()) == {p1, p2}


def test_bezier_hit_test_as_line(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(start, end))
    threshold = 5.0

    assert bezier.hit_test(0, 0, threshold, registry) is True
    assert bezier.hit_test(20, 0, threshold, registry) is True
    assert bezier.hit_test(10, 0, threshold, registry) is True
    assert bezier.hit_test(10, 10, threshold, registry) is False


def test_bezier_hit_test_with_control_points(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)

    bezier = registry.get_entity(registry.add_bezier(start, end))
    bezier.cp1 = (0.0, 20.0)
    bezier.cp2 = (0.0, -20.0)

    threshold = 5.0

    assert bezier.hit_test(0, 0, threshold, registry) is True
    assert bezier.hit_test(20, 0, threshold, registry) is True


def test_bezier_update_constrained_status(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(20, 0)
    bid = registry.add_bezier(p1, p2)
    bezier = registry.get_entity(bid)

    pt1 = registry.get_point(p1)
    pt2 = registry.get_point(p2)

    pt1.constrained = False
    pt2.constrained = False
    bezier.update_constrained_status(registry, [])
    assert bezier.constrained is False

    pt1.constrained = True
    pt2.constrained = True
    bezier.update_constrained_status(registry, [])
    assert bezier.constrained is True


def test_bezier_to_geometry_as_line(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)
    bezier = registry.get_entity(registry.add_bezier(start, end))
    geo = bezier.to_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) == 2
    assert geo.data is not None
    assert geo.data[0][0] == 1.0
    assert geo.data[1][0] == 2.0


def test_bezier_to_geometry_with_control_points(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)

    bezier = registry.get_entity(registry.add_bezier(start, end))
    bezier.cp1 = (5.0, 10.0)
    bezier.cp2 = (-5.0, 10.0)

    geo = bezier.to_geometry(registry)
    assert isinstance(geo, Geometry)
    assert len(geo) == 2
    assert geo.data is not None
    assert geo.data[0][0] == 1.0
    assert geo.data[1][0] == 4.0


def test_bezier_get_set_state(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(20, 0)
    bid = registry.add_bezier(p1, p2)
    bezier = registry.get_entity(bid)

    state = bezier.get_state()
    assert state == {"construction": False}

    bezier.construction = True

    bezier.set_state(state)
    assert bezier.construction is False


def test_bezier_with_own_control_points(registry):
    start = registry.add_point(0, 0)
    end = registry.add_point(20, 0)

    bid = registry.add_bezier(start, end)
    bezier = registry.get_entity(bid)
    bezier.cp1 = (5.0, 10.0)
    bezier.cp2 = (-5.0, 10.0)

    assert bezier.is_line(registry) is False

    cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points(registry)
    assert cp1_x == pytest.approx(5.0)
    assert cp1_y == pytest.approx(10.0)
    assert cp2_x == pytest.approx(15.0)
    assert cp2_y == pytest.approx(10.0)


def test_bezier_control_points_serialization():
    original = Bezier(
        id=10,
        start_idx=1,
        end_idx=4,
        cp1=(5.0, 10.0),
        cp2=(-3.0, 7.0),
    )

    data = original.to_dict()
    assert data["cp1_dx"] == 5.0
    assert data["cp1_dy"] == 10.0
    assert data["cp2_dx"] == -3.0
    assert data["cp2_dy"] == 7.0

    restored = Bezier.from_dict(data)
    assert restored.cp1 == (5.0, 10.0)
    assert restored.cp2 == (-3.0, 7.0)


def test_bezier_without_control_points_serialization():
    original = Bezier(
        id=10,
        start_idx=1,
        end_idx=4,
    )

    data = original.to_dict()
    assert "cp1_dx" not in data
    assert "cp2_dx" not in data

    restored = Bezier.from_dict(data)
    assert restored.cp1 is None
    assert restored.cp2 is None


def test_bezier_get_control_points_none(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=None, cp2=None)
    )

    cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points(registry)

    assert cp1_x is None
    assert cp1_y is None
    assert cp2_x is None
    assert cp2_y is None


def test_bezier_get_control_points_with_values(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=(20, 10), cp2=(-30, -15))
    )

    cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points(registry)

    assert cp1_x == 20.0
    assert cp1_y == 10.0
    assert cp2_x == 70.0
    assert cp2_y == -15.0


def test_bezier_get_control_points_or_endpoints_with_cps(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=(20, 10), cp2=(-30, -15))
    )

    cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points_or_endpoints(
        registry
    )

    assert cp1_x == 20.0
    assert cp1_y == 10.0
    assert cp2_x == 70.0
    assert cp2_y == -15.0


def test_bezier_get_control_points_or_endpoints_without_cps(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 50)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=None, cp2=None)
    )

    cp1_x, cp1_y, cp2_x, cp2_y = bezier.get_control_points_or_endpoints(
        registry
    )

    assert cp1_x == 0.0
    assert cp1_y == 0.0
    assert cp2_x == 100.0
    assert cp2_y == 50.0


def test_bezier_sample_bezier_linear(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=None, cp2=None)
    )

    points = bezier._sample_bezier(0, 0, 0, 0, 100, 0, 100, 0, 5)

    assert len(points) == 6
    assert points[0] == (0.0, 0.0)
    assert points[5] == (100.0, 0.0)

    for i, (x, y) in enumerate(points):
        assert y == pytest.approx(0.0)


def test_bezier_sample_bezier_curved(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=(0, 100), cp2=(100, 100))
    )

    points = bezier._sample_bezier(0, 0, 0, 100, 100, 100, 100, 0, 10)

    assert len(points) == 11
    assert points[0] == (0.0, 0.0)
    assert points[10] == (100.0, 0.0)

    mid_point = points[5]
    assert mid_point[0] == pytest.approx(50.0)
    assert mid_point[1] == pytest.approx(75.0)


def test_bezier_sample_bezier_quarter_circle_approximation(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(registry.add_bezier(p1, p2))

    k = 0.5522847498
    points = bezier._sample_bezier(
        0, 100, k * 100, 100, 100, k * 100, 100, 0, 20
    )

    assert len(points) == 21

    start = points[0]
    assert start[0] == pytest.approx(0.0)
    assert start[1] == pytest.approx(100.0)

    end = points[20]
    assert end[0] == pytest.approx(100.0)
    assert end[1] == pytest.approx(0.0)


def test_bezier_get_bbox_linear(registry):
    p1 = registry.add_point(10, 20)
    p2 = registry.add_point(50, 60)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=None, cp2=None)
    )

    bbox = bezier._get_bbox(registry)

    assert bbox == (10.0, 20.0, 50.0, 60.0)


def test_bezier_get_bbox_curved(registry):
    p1 = registry.add_point(0, 0)
    p2 = registry.add_point(100, 0)

    bezier = registry.get_entity(
        registry.add_bezier(p1, p2, cp1=(0, 100), cp2=(100, 100))
    )

    bbox = bezier._get_bbox(registry)

    assert bbox[0] == pytest.approx(0.0)
    assert bbox[1] == pytest.approx(0.0)
    assert bbox[2] >= 100.0
    assert bbox[3] > 50.0
