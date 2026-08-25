import math
from typing import cast

import pytest
from sketcher.core.constraints import RotationalConstraint
from sketcher.core.patterns import (
    CircularPatternParams,
    CircularPatternStrategy,
    InstancePlacement,
    PlacementKind,
    SketchArrayMode,
    find_pattern_for_entity,
    make_pattern_strategy,
)
from sketcher.core.patterns.definition import PatternDefinition


def test_factory_returns_circular_strategy():
    params = CircularPatternParams()
    strategy = make_pattern_strategy(SketchArrayMode.CIRCULAR, params)
    assert isinstance(strategy, CircularPatternStrategy)
    assert strategy.needs_center_point is True


def test_factory_rejects_unknown_mode():
    bogus_mode = cast(SketchArrayMode, "bogus")
    with pytest.raises(ValueError):
        make_pattern_strategy(bogus_mode, CircularPatternParams())


def test_full_circle_rotation_placements():
    params = CircularPatternParams(
        count=4, total_angle_deg=360.0, center=(0.0, 0.0), rotate_copies=True
    )
    strategy = CircularPatternStrategy(params)
    placements = strategy.calculate_placements((5.0, 0.0))
    assert len(placements) == 3
    angles = [p.angle for p in placements]
    assert angles == [
        pytest.approx(math.pi / 2),
        pytest.approx(math.pi),
        pytest.approx(3 * math.pi / 2),
    ]
    assert all(p.kind == PlacementKind.ROTATION for p in placements)


def test_partial_arc_placements():
    params = CircularPatternParams(
        count=3, total_angle_deg=90.0, center=(0.0, 0.0), rotate_copies=True
    )
    strategy = CircularPatternStrategy(params)
    placements = strategy.calculate_placements((0.0, 0.0))
    angles = [math.degrees(p.angle) for p in placements]
    assert angles == [pytest.approx(30.0), pytest.approx(60.0)]


def test_translate_mode_placements():
    params = CircularPatternParams(
        count=3,
        total_angle_deg=180.0,
        center=(0.0, 0.0),
        rotate_copies=False,
    )
    strategy = CircularPatternStrategy(params)
    seed_center = (10.0, 0.0)
    placements = strategy.calculate_placements(seed_center)
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


def test_linkage_constraints_rotate_mode():
    params = CircularPatternParams(
        count=3,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        rotate_copies=True,
    )
    strategy = CircularPatternStrategy(params)
    instances = [(1, {11: 21, 12: 22}), (2, {11: 31, 12: 32})]
    constraints = strategy.build_linkage_constraints(instances, 99)
    assert len(constraints) == 4
    step = math.radians(120)
    values = sorted(c.value for c in constraints)
    assert values == pytest.approx(sorted([step, step, 2 * step, 2 * step]))
    for c in constraints:
        assert isinstance(c, RotationalConstraint)
        assert c.center == 99


def test_linkage_constraints_empty_in_translate_mode():
    params = CircularPatternParams(
        count=3, total_angle_deg=360.0, rotate_copies=False
    )
    strategy = CircularPatternStrategy(params)
    assert strategy.build_linkage_constraints([(1, {1: 2})], 9) == []


def test_master_geometry():
    params = CircularPatternParams(
        count=6,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=15.0,
        rotate_copies=True,
    )
    strategy = CircularPatternStrategy(params)
    points, entities, constraints = strategy.create_master_geometry(
        center_pid=1, radius_pt_pid=2
    )
    assert len(points) == 1
    assert points[0].x == pytest.approx(15.0)
    assert len(entities) == 1
    assert entities[0].construction is True
    assert len(constraints) == 1


def test_master_geometry_skipped_without_radius():
    params = CircularPatternParams(radius=0.0)
    strategy = CircularPatternStrategy(params)
    assert strategy.create_master_geometry(1, 2) == ([], [], [])


# ----------------------------------------------------------------------
# PatternDefinition
# ----------------------------------------------------------------------


def test_definition_serialization_round_trip():
    pattern = PatternDefinition(
        uid="abc",
        mode=SketchArrayMode.CIRCULAR,
        guide_circle_id=7,
        members=[(0, [1]), (1, [2]), (2, [3])],
        count=8,
        total_angle_deg=270.0,
        rotate_copies=False,
    )
    restored = PatternDefinition.from_dict(pattern.to_dict())
    assert restored.uid == "abc"
    assert restored.mode == SketchArrayMode.CIRCULAR
    assert restored.guide_circle_id == 7
    assert restored.members == [(0, [1]), (1, [2]), (2, [3])]
    assert restored.count == 8
    assert restored.total_angle_deg == 270.0
    assert restored.rotate_copies is False


def test_definition_legacy_flat_format_migrates():
    restored = PatternDefinition.from_dict(
        {
            "uid": "old",
            "mode": "circular",
            "guide_circle_id": 5,
            "entity_ids": [1, 2],
            "entity_slots": [0, 3],
            "count": 6,
        }
    )
    assert restored.members == [(0, [1]), (3, [2])]


def test_find_pattern_for_entity():
    p1 = PatternDefinition("a", SketchArrayMode.CIRCULAR, 11, [], 6, 360, True)
    p2 = PatternDefinition("b", SketchArrayMode.CIRCULAR, 22, [], 6, 360, True)
    patterns = [p1, p2]
    assert find_pattern_for_entity(patterns, 22) is p2
    assert find_pattern_for_entity(patterns, 99) is None


def test_living_members_filters_deleted_and_keeps_groups():
    class FakeRegistry:
        def get_entity(self, eid):
            return eid if eid in (1, 2, 5) else None

    # Member at slot 1 lost one of its two entities but survives.
    pattern = PatternDefinition(
        "x",
        SketchArrayMode.CIRCULAR,
        9,
        members=[(0, [1, 2]), (1, [3, 4]), (2, [5])],
        count=6,
    )
    assert pattern.living_members(FakeRegistry()) == [
        (0, [1, 2]),
        (2, [5]),
    ]
    assert pattern.living_entity_ids(FakeRegistry()) == [1, 2, 5]
    assert pattern.occupied_slots(FakeRegistry()) == {0, 2}
