import pytest
from sketcher.core.registry import EntityRegistry
from sketcher.core.snap.engine import SnapEngine
from sketcher.core.snap.producers import OnEntityProducer
from sketcher.core.snap.types import DragContext, SnapLineType


@pytest.fixture
def engine():
    registry = EntityRegistry()
    a = registry.add_point(0.0, 0.0)
    b = registry.add_point(100.0, 0.0)
    registry.add_line(a, b)
    engine = SnapEngine(threshold=5.0)
    engine.register_producer(OnEntityProducer())
    return engine, registry


def test_primary_stays_engaged_at_threshold_boundary(engine):
    """A candidate wobbling across the plain threshold keeps its snap
    engaged within the exit radius instead of toggling off every
    frame."""
    engine, registry = engine
    context = DragContext()

    first = engine.query(registry, (50.0, 4.5), context)
    assert first.primary_snap_point is not None
    assert first.primary_snap_point.line_type == SnapLineType.ON_ENTITY

    second = engine.query(registry, (50.0, 6.5), context)
    assert second.primary_snap_point is not None
    assert second.primary_snap_point.line_type == SnapLineType.ON_ENTITY
    assert second.position == (50.0, 0.0)

    third = engine.query(registry, (50.0, 9.0), context)
    assert third.primary_snap_point is None


def test_reacquire_binds_to_same_source(engine):
    """The exit-radius leash binds to the previous snap target's
    source, not to whatever else appears nearby."""
    engine, registry = engine
    c = registry.add_point(50.0, 8.0)
    d = registry.add_point(150.0, 8.0)
    registry.add_line(c, d)
    context = DragContext()

    first = engine.query(registry, (20.0, 4.5), context)
    engaged_source = first.primary_snap_point.source

    second = engine.query(registry, (20.0, 6.5), context)
    assert second.primary_snap_point.source is engaged_source


def test_disengagement_allows_fresh_engagement(engine):
    """After the snap fully disengages, a new primary engages
    normally on the next approach."""
    engine, registry = engine
    context = DragContext()

    engine.query(registry, (50.0, 4.5), context)
    gone = engine.query(registry, (50.0, 9.0), context)
    assert gone.primary_snap_point is None

    again = engine.query(registry, (50.0, 3.0), context)
    assert again.primary_snap_point is not None
    assert again.primary_snap_point.line_type == SnapLineType.ON_ENTITY
