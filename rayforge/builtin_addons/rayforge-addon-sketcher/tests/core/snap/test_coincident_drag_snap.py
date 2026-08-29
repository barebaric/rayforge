"""
Regression tests for snap glue on coincident points.

An array places a template point exactly onto guide geometry (e.g. a
curve-along template center on the guide path start). These are
DISTINCT points separated by float noise only, and the guide point is
not part of the dragged group. Its axis snap lines must not glue the
drag to its own start position.
"""

import pytest
from sketcher.core.registry import EntityRegistry
from sketcher.core.snap.engine import SnapEngine
from sketcher.core.snap.producers.centers import CentersProducer
from sketcher.core.snap.producers.entity_points import EntityPointsProducer
from sketcher.core.snap.types import DragContext, SnapLineType


@pytest.fixture
def registry():
    return EntityRegistry()


@pytest.fixture
def engine():
    engine = SnapEngine()
    engine.register_producer(EntityPointsProducer())
    engine.register_producer(CentersProducer())
    return engine


def test_coincident_point_axis_lines_are_suppressed(registry):
    """The guide endpoint sits float-noise from the dragged template
    point: its axis lines must not be produced as snap candidates."""
    dragged = registry.get_point(registry.add_point(10.0, 5.0))
    endpoint = registry.get_point(registry.add_point(10.0 + 1e-9, 5.0 - 1e-9))
    context = DragContext(dragged_point_ids={dragged.id})
    producer = EntityPointsProducer()

    snap_lines = list(producer.produce(registry, (10.0, 5.0), context, 5.0))

    endpoint_lines = [
        sl
        for sl in snap_lines
        if sl.source is endpoint or sl.coordinate in (endpoint.x, endpoint.y)
    ]
    assert endpoint_lines == []


def test_coincident_point_snap_point_is_suppressed(registry):
    dragged = registry.get_point(registry.add_point(10.0, 5.0))
    registry.get_point(registry.add_point(10.0 + 1e-9, 5.0 - 1e-9))
    context = DragContext(dragged_point_ids={dragged.id})
    producer = EntityPointsProducer()

    snap_points = list(
        producer.produce_points(registry, (10.0, 5.0), context, 5.0)
    )

    assert snap_points == []


def test_nearby_point_still_produces_snap_lines(registry):
    """Normal magnet snapping is preserved for points that are close
    but genuinely separate."""
    dragged = registry.get_point(registry.add_point(10.0, 5.0))
    neighbour = registry.get_point(registry.add_point(10.001, 5.0))
    context = DragContext(dragged_point_ids={dragged.id})
    producer = EntityPointsProducer()

    snap_lines = list(producer.produce(registry, (10.0, 5.0), context, 5.0))

    assert any(sl.source is neighbour for sl in snap_lines)


def test_drag_near_coincident_guide_point_does_not_snap(registry, engine):
    """Engine level: querying at the dragged point's position with a
    coincident guide point nearby must not snap."""
    dragged = registry.get_point(registry.add_point(10.0, 5.0))
    registry.get_point(registry.add_point(10.0 + 1e-9, 5.0 - 1e-9))
    context = DragContext(dragged_point_ids={dragged.id})

    result = engine.query(registry, (10.0, 5.0), context)

    assert not result.snapped


def test_drag_still_snaps_to_other_points(registry, engine):
    dragged = registry.get_point(registry.add_point(10.0, 5.0))
    target = registry.get_point(registry.add_point(12.0, 5.0))
    context = DragContext(dragged_point_ids={dragged.id})

    result = engine.query(registry, (12.05, 5.0), context)

    assert result.snapped
    assert result.primary_snap_point is not None
    assert result.primary_snap_point.x == pytest.approx(target.x)
    assert result.primary_snap_point.line_type == SnapLineType.ENTITY_POINT


def test_center_coincident_with_dragged_point_is_suppressed(registry):
    """A circle center sitting on the dragged point (e.g. the guide
    circle center after a re-projection) must not glue the drag."""
    center_pid = registry.add_point(20.0, 30.0)
    radius_pt = registry.add_point(25.0, 30.0)
    registry.add_circle(center_pid, radius_pt)
    dragged = registry.get_point(registry.add_point(20.0 + 1e-9, 30.0))
    context = DragContext(dragged_point_ids={dragged.id})
    producer = CentersProducer()

    snap_lines = list(producer.produce(registry, (20.0, 30.0), context, 5.0))

    assert snap_lines == []


def test_center_elsewhere_still_produces_snap_lines(registry):
    center_pid = registry.add_point(20.0, 30.0)
    radius_pt = registry.add_point(25.0, 30.0)
    registry.add_circle(center_pid, radius_pt)
    dragged = registry.get_point(registry.add_point(10.0, 10.0))
    context = DragContext(dragged_point_ids={dragged.id})
    producer = CentersProducer()

    snap_lines = list(producer.produce(registry, (20.0, 30.0), context, 5.0))

    assert len(snap_lines) == 2


def test_coincides_with_dragged_missing_point_is_safe(registry):
    """A stale dragged point id must not crash the check."""
    context = DragContext(dragged_point_ids={999})

    assert not context.coincides_with_dragged(1.0, 1.0, registry)


def test_index_rebuilds_when_points_move_in_place(registry, engine):
    """
    Array sync moves points in place (re-anchor, re-projection)
    without bumping the registry version. The snap index must follow
    the moved geometry instead of serving stale lines from the old
    positions — those appeared as snap lines crossing no geometry.
    """
    registry.get_point(registry.add_point(10.0, 0.0))
    context = DragContext()

    assert engine.query(registry, (10.0, 3.0), context).snapped

    moved = registry.get_point(0)
    moved.x = 50.0
    moved.y = 20.0

    assert engine.query(registry, (50.0, 23.0), context).snapped
    assert not engine.query(registry, (10.0, 3.0), context).snapped


def test_index_rebuilds_when_entity_is_added(registry, engine):
    context = DragContext()
    assert not engine.query(registry, (10.0, 3.0), context).snapped

    registry.add_point(10.0, 0.0)

    assert engine.query(registry, (10.0, 3.0), context).snapped


def test_snap_result_current_tracks_geometry_changes(registry, engine):
    registry.get_point(registry.add_point(10.0, 0.0))
    context = DragContext()

    engine.query(registry, (10.0, 3.0), context)
    assert engine.is_snap_result_current(registry)

    registry.get_point(0).x = 12.0
    assert not engine.is_snap_result_current(registry)

    engine.query(registry, (12.0, 3.0), context)
    assert engine.is_snap_result_current(registry)


def test_snap_result_current_false_before_first_query(registry, engine):
    registry.add_point(10.0, 0.0)

    assert not engine.is_snap_result_current(registry)
