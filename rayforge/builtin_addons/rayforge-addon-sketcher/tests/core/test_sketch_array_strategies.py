import math

import pytest
from raygeo.sketch.arrays import (
    Array,
    CircularArray,
    CircularArrayStrategy,
    CurveAlongArray,
    CurveAlongArrayStrategy,
    InstancePlacement,
    PlacementKind,
    find_array_for_entity,
)
from raygeo.sketch.registry import EntityRegistry


def test_full_circle_rotation_placements():
    strategy = CircularArrayStrategy(
        count=4, total_angle_deg=360.0, center=(0.0, 0.0), rotate_copies=True
    )
    placements = strategy.member_placements((5.0, 0.0))
    assert len(placements) == 3
    angles = [p.angle for p in placements]
    assert angles == [
        pytest.approx(math.pi / 2),
        pytest.approx(math.pi),
        pytest.approx(3 * math.pi / 2),
    ]
    assert all(p.kind == PlacementKind.ROTATION for p in placements)


def test_partial_arc_placements():
    strategy = CircularArrayStrategy(
        count=3, total_angle_deg=90.0, center=(0.0, 0.0), rotate_copies=True
    )
    placements = strategy.member_placements((0.0, 0.0))
    angles = [math.degrees(p.angle) for p in placements]
    assert angles == [pytest.approx(30.0), pytest.approx(60.0)]


def test_translate_mode_placements():
    strategy = CircularArrayStrategy(
        count=3,
        total_angle_deg=180.0,
        center=(0.0, 0.0),
        rotate_copies=False,
    )
    seed_center = (10.0, 0.0)
    placements = strategy.member_placements(seed_center)
    assert all(p.kind == PlacementKind.TRANSLATION for p in placements)

    # First copy: seed center rotated by the 60 deg step around origin.
    dx, dy = placements[0].delta
    assert (seed_center[0] + dx, seed_center[1] + dy) == (
        pytest.approx(5.0),
        pytest.approx(10.0 * math.sin(math.radians(60))),
    )


def test_placement_transform_point():
    placement = InstancePlacement(
        kind=PlacementKind.ROTATION,
        angle=math.pi / 2,
        center=(1.0, 1.0),
    )
    x, y = placement.transform_point(2.0, 1.0)
    assert x == pytest.approx(1.0)
    assert y == pytest.approx(2.0)

    translation = InstancePlacement(
        kind=PlacementKind.TRANSLATION, delta=(3.0, -4.0)
    )
    x, y = translation.transform_point(2.0, 1.0)
    assert (x, y) == (5.0, -3.0)


def test_master_geometry():
    strategy = CircularArrayStrategy(
        count=6,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=15.0,
        rotate_copies=True,
    )
    points, entities, constraints = strategy.create_master_geometry(
        center_pid=1, radius_pt_pid=2
    )
    assert len(points) == 1
    assert points[0].x == pytest.approx(15.0)
    assert len(entities) == 1
    assert entities[0].construction is True
    assert len(constraints) == 1


def test_master_geometry_skipped_without_radius():
    strategy = CircularArrayStrategy(radius=0.0)
    assert strategy.create_master_geometry(1, 2) == ([], [], [])


# ----------------------------------------------------------------------
# Array
# ----------------------------------------------------------------------


def test_definition_serialization_round_trip():
    array = CircularArray(
        uid="abc",
        guide_circle_id=7,
        members=[(0, [1]), (1, [2]), (2, [3])],
        count=8,
        total_angle_deg=270.0,
        rotate_copies=False,
    )
    restored = Array.from_dict(array.to_dict())
    assert isinstance(restored, CircularArray)
    assert restored.uid == "abc"
    assert restored.mode == "circular"
    assert restored.guide_circle_id == 7
    assert restored.members == [(0, [1]), (1, [2]), (2, [3])]
    assert restored.count == 8
    assert restored.total_angle_deg == 270.0
    assert restored.rotate_copies is False


def test_find_array_for_entity():
    p1 = CircularArray("a", 11, [], 6, 360, True)
    p2 = CircularArray("b", 22, [], 6, 360, True)
    arrays: list[Array] = [p1, p2]
    assert find_array_for_entity(arrays, 22) is p2
    assert find_array_for_entity(arrays, 99) is None


def test_living_members_filters_deleted_and_keeps_groups():
    registry = EntityRegistry()
    pts = [registry.add_point(float(i), 0.0, False) for i in range(6)]
    e0 = registry.add_line(pts[0], pts[1], False)
    e1 = registry.add_line(pts[1], pts[2], False)
    e2 = registry.add_line(pts[2], pts[3], False)
    e3 = registry.add_line(pts[3], pts[4], False)
    e4 = registry.add_line(pts[4], pts[5], False)
    registry.remove_entities([e1, e3])
    array = CircularArray(
        "x",
        9,
        members=[(0, [e0, e1]), (1, [e2, e3]), (2, [e4])],
        count=6,
    )
    assert array.living_members(registry) == [
        (0, [e0]),
        (1, [e2]),
        (2, [e4]),
    ]
    assert array.living_entity_ids(registry) == [e0, e2, e4]
    assert array.occupied_slots(registry) == [0, 1, 2]


def test_circular_template_placement_is_identity():
    """Position 0 of a circular array is the drawn position: the
    template placement is the identity, so applying it moves the
    template onto its guide (the constructed circle) without
    translating it."""
    strategy = CircularArrayStrategy(count=4, center=(0.0, 0.0))
    placement = strategy.template_placement((5.0, 3.0))
    assert placement.transform_point(5.0, 3.0) == (5.0, 3.0)
    assert placement.transform_offset(2.0, -1.0) == (2.0, -1.0)


# ----------------------------------------------------------------------
# Array state transitions
# ----------------------------------------------------------------------


def test_circular_snapshot_restore_round_trip():
    array = CircularArray(
        "a",
        7,
        [(0, [1])],
        count=5,
        total_angle_deg=270.0,
        rotate_copies=False,
    )
    state = array.snapshot()
    array.count = 9
    array.total_angle_deg = 45.0
    array.rotate_copies = True
    array.restore(state)
    assert array.count == 5
    assert array.total_angle_deg == 270.0
    assert array.rotate_copies is False
    assert array.members == [(0, [1])]


def test_curve_along_snapshot_restore_round_trip():
    array = CurveAlongArray(
        "b",
        9,
        [(0, [2])],
        count=4,
        path_entity_id=3,
        align_to_tangent=False,
        offset_to_start=2.0,
        spacing=5.0,
        template_anchor=((1.0, 2.0), 0.5),
    )
    state = array.snapshot()
    array.count = 8
    array.path_entity_id = 11
    array.spacing = 1.0
    array.template_anchor = None
    array.restore(state)
    assert array.count == 4
    assert array.path_entity_id == 3
    assert array.align_to_tangent is False
    assert array.offset_to_start == 2.0
    assert array.spacing == 5.0
    assert array.template_anchor == ((1.0, 2.0), 0.5)


def test_circular_commit_and_params_changed():
    array = CircularArray(
        "a",
        7,
        [(0, [1])],
        count=5,
        total_angle_deg=270.0,
        rotate_copies=False,
    )
    unchanged = CircularArrayStrategy(
        count=5, total_angle_deg=270.0, rotate_copies=False
    )
    assert array.params_changed(unchanged) is False

    changed = CircularArrayStrategy(
        count=5, total_angle_deg=90.0, rotate_copies=True
    )
    assert array.params_changed(changed) is True

    array.commit(unchanged)
    assert array.count == 5
    array.commit(changed)
    assert array.count == 5
    assert array.total_angle_deg == 90.0
    assert array.rotate_copies is True


def test_curve_along_commit_and_params_changed():
    array = CurveAlongArray(
        "b", 9, [(0, [2])], count=4, path_entity_id=3, spacing=5.0
    )
    unchanged = CurveAlongArrayStrategy(count=4, path_entity_id=3, spacing=5.0)
    assert array.params_changed(unchanged) is False

    # rotate_copies is intentionally not compared for curve arrays.
    rotated = CurveAlongArrayStrategy(
        count=4, path_entity_id=3, spacing=5.0, rotate_copies=False
    )
    assert array.params_changed(rotated) is False

    changed = CurveAlongArrayStrategy(count=4, path_entity_id=3, spacing=6.0)
    assert array.params_changed(changed) is True

    array.commit(changed)
    assert array.count == 4
    assert array.spacing == 6.0
    # rotate_copies is not written by curve-along commits.
    assert array.rotate_copies is True


# ----------------------------------------------------------------------
# Array sync support
# ----------------------------------------------------------------------


def test_signatures_changed_ignores_missing_caches():
    array = CircularArray("a", 11, [(0, [1])])
    assert array.signatures_changed(((1, 2),), ((3, 4),)) is False
    array.update_caches(((1, 2),), ((3, 4),))
    assert array.signatures_changed(((1, 2),), ((3, 4),)) is False
    assert array.signatures_changed(((9, 9),), ((3, 4),)) is True
    assert array.signatures_changed(((1, 2),), ((9, 9),)) is True


def test_guide_and_template_signatures_from_registry():
    registry = EntityRegistry()
    p0 = registry.add_point(0.0, 0.0, False)
    p1 = registry.add_point(10.0, 0.0, False)
    line = registry.add_line(p0, p1, False)
    array = CurveAlongArray("b", line, [(0, [line])])
    expected = [[0.0, 0.0], [10.0, 0.0]]
    assert array.guide_signature(registry) == expected
    assert array.template_signature(registry) == [expected]


def test_prune_drops_dead_entities_and_reports_dead_master():
    registry = EntityRegistry()
    pts = [registry.add_point(float(i), 0.0, False) for i in range(6)]
    guide = registry.add_circle(pts[0], pts[1], True)
    e0 = registry.add_line(pts[0], pts[1], False)
    e1 = registry.add_line(pts[1], pts[2], False)
    e2 = registry.add_line(pts[2], pts[3], False)
    registry.remove_entities([e1])
    array = CircularArray("a", guide, members=[(0, [e0, e1]), (1, [e2])])
    assert array.prune(registry) is True
    assert array.members == [(0, [e0]), (1, [e2])]
    registry.remove_entities([guide])
    assert array.prune(registry) is False


def test_is_guide_radius_point_only_for_circle_guides():
    registry = EntityRegistry()
    center = registry.add_point(0.0, 0.0, False)
    edge = registry.add_point(5.0, 0.0, False)
    circle = registry.add_circle(center, edge, False)
    array = CircularArray("a", circle, [(0, [circle])])
    assert array.is_guide_radius_point(registry, edge) is True
    assert array.is_guide_radius_point(registry, center) is False
    curve = CurveAlongArray("b", circle, [(0, [circle])])
    assert curve.is_guide_radius_point(registry, edge) is False
