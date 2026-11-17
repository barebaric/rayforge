from unittest.mock import Mock

from sketcher.core.constraints import CoincidentConstraint
from sketcher.core.sketch import Sketch
from sketcher.core.snap.producers import OnEntityProducer
from sketcher.core.snap.types import DragContext


def build_coupled_triangle():
    """A closed shape whose corners are distinct points tied by a
    CoincidentConstraint: one edge moves with a drag of another
    without sharing any dragged point."""
    sk = Sketch("test")
    a = sk.add_point(30.0, 30.0)
    b = sk.add_point(120.0, 30.0)
    c = sk.add_point(120.0, 110.0)
    d = sk.add_point(30.0, 110.0)
    ab_id = sk.add_line(a, b)
    sk.add_line(b, c)
    cd_id = sk.add_line(c, d)
    sk.add_line(d, a)
    sk.constraints.append(CoincidentConstraint(d, a))
    ab = sk.registry.get_entity(ab_id)
    cd = sk.registry.get_entity(cd_id)
    assert ab is not None and cd is not None
    return sk, ab, cd, (a, b, c, d)


def test_coupled_edge_excluded_when_scope_points_dragged():
    """The edge that moves with the drag (coupled through the
    coincidence, sharing none of the dragged points) must not act as
    a snap target: its geometry changes under the drag, so snapping
    onto it only flickers."""
    sk, ab, cd, (a, b, _c, d) = build_coupled_triangle()
    producer = OnEntityProducer()
    registry = sk.registry

    narrow = DragContext(
        dragged_point_ids={a, b},
        dragged_entity_ids={ab.id},
    )
    near_coupled_edge = list(
        producer.produce_points(registry, (75.0, 110.0), narrow, 5.0)
    )
    assert cd.id in {
        sp.source.id for sp in near_coupled_edge if sp.source is not None
    }

    scope = {a, b, d}
    expanded = DragContext(
        dragged_point_ids=scope,
        dragged_entity_ids={ab.id},
    )
    candidates = list(
        producer.produce_points(registry, (75.0, 110.0), expanded, 5.0)
    )
    assert candidates == []


def test_entity_drag_query_uses_constraint_scope():
    """The drag snap query carries the full solve scope, so points
    coupled through constraints exclude their entities too."""
    from sketcher.ui_gtk.tools.select_tool import SelectTool

    sk, ab, _cd, (a, b, _c, d) = build_coupled_triangle()
    element = Mock()
    element.sketch = sk
    element.mark_dirty = Mock()
    element.hittester = Mock()
    ident = Mock(
        transform_vector=Mock(side_effect=lambda v: v),
        without_translation=Mock(
            return_value=Mock(transform_vector=Mock(side_effect=lambda v: v))
        ),
    )
    element.get_world_transform = Mock(return_value=ident)
    element.content_transform = ident
    element.canvas = None
    element.selection = Mock()
    element.selection.entity_ids = [ab.id]

    tool = SelectTool(element)
    tool._prepare_entity_drag(ab, 0.0, 0.0)

    assert tool.drag_scope_point_ids == {a, b, d}
