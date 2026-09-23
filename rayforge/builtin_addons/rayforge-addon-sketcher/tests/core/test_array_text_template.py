"""End-to-end tests for arrays whose template member is a text box.

The text box is a compound entity (frame points plus construction
lines and glyph geometry derived from content + font), so these
tests pin down the array behaviour that plain line/circle templates
exercise implicitly.
"""

import math

import pytest
from raygeo.geo.shape.text import FontConfig
from sketcher.core.arrays import CircularArrayStrategy, CurveAlongArrayStrategy
from sketcher.core.commands import (
    CreateArrayCommand,
    EditArrayCommand,
    RemoveItemsCommand,
    TextBoxCommand,
)
from sketcher.core.commands.duplicate import DuplicateCommand
from sketcher.core.constraints import ParallelogramConstraint
from sketcher.core.entities import Circle, Line, TextBoxEntity
from sketcher.core.params import ParameterContext
from sketcher.core.selection import SketchSelection
from sketcher.core.sketch import Sketch


def ui_delete(sketch, entity_ids):
    """Deletes like the DeleteTool: dependency-based removal."""
    selection = SketchSelection()
    selection.entity_ids = list(entity_ids)
    points, entities, constraints = RemoveItemsCommand.calculate_dependencies(
        sketch, selection
    )
    RemoveItemsCommand(
        sketch,
        "",
        points=points,
        entities=entities,
        constraints=constraints,
    ).execute()


def make_strategy(count=4, radius=40.0):
    return CircularArrayStrategy(
        count=count,
        total_angle_deg=360.0,
        center=(0.0, 0.0),
        radius=radius,
        rotate_copies=True,
    )


@pytest.fixture
def text_array():
    """Sketch with a text box turned into a circular array through
    the full create flow."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (30, 0), width=20, height=10)
    cmd.execute()
    assert cmd.text_box_id is not None
    box = sketch.registry.get_entity(cmd.text_box_id)
    assert isinstance(box, TextBoxEntity)
    box.content = "AB"
    box.font_config = FontConfig("sans-serif", 8.0)

    create_cmd = CreateArrayCommand(sketch, make_strategy(), [box.id])
    create_cmd.execute()
    sketch.solve()
    return sketch, create_cmd, box


def frame_distances(points):
    return sorted(
        round(math.dist(a, b), 6)
        for i, a in enumerate(points)
        for b in points[i + 1 :]
    )


def member_frame(sketch, eids):
    registry = sketch.registry
    return [
        (registry.get_point(pid).x, registry.get_point(pid).y)
        for eid in eids
        for pid in registry.get_entity(eid).get_point_ids()
    ]


def test_copies_are_clean_text_boxes(text_array):
    """Copies carry content and font, and no stale construction-line
    references into the template's helper geometry."""
    sketch, cmd, _box = text_array
    registry = sketch.registry
    copies = [registry.get_entity(eid) for eid in cmd.created_entity_ids]
    assert len(copies) == 3
    for clone in copies:
        assert isinstance(clone, TextBoxEntity)
        assert clone.array_copy is True
        assert clone.content == "AB"
        assert clone.font_config.size == 8.0
        assert clone.construction_line_ids == []
        assert clone.get_fourth_corner_id(registry) is None


def test_copies_are_rigid_rotations_of_the_template(text_array):
    sketch, cmd, box = text_array
    registry = sketch.registry
    template_frame = member_frame(sketch, [box.id])
    for _slot, eids in cmd.array.living_members(registry)[1:]:
        copy_frame = member_frame(sketch, eids)
        assert frame_distances(copy_frame) == frame_distances(template_frame)


def test_content_edit_propagates_to_copies(text_array):
    """The text box signature covers content and font, so editing the
    template's text re-derives every copy on the next solve."""
    sketch, cmd, box = text_array
    registry = sketch.registry
    box.content = "XYZ"
    sketch.solve()
    for eid in cmd.array.living_entity_ids(registry):
        assert registry.get_entity(eid).content == "XYZ"


def test_font_edit_propagates_to_copies(text_array):
    sketch, cmd, box = text_array
    registry = sketch.registry
    box.font_config = FontConfig("sans-serif", 14.0, bold=True)
    sketch.solve()
    for eid in cmd.array.living_entity_ids(registry):
        font = registry.get_entity(eid).font_config
        assert font.size == 14.0
        assert font.bold is True


def test_deleting_a_copy_and_editing_regenerates_it(text_array):
    sketch, cmd, box = text_array
    registry = sketch.registry
    _slot, victim_eids = cmd.array.living_members(registry)[2]
    ui_delete(sketch, victim_eids)
    assert len(cmd.array.living_entity_ids(registry)) == 3

    EditArrayCommand(sketch, cmd.array, make_strategy()).execute()
    sketch.solve()
    assert len(cmd.array.living_entity_ids(registry)) == 4
    for eid in cmd.array.living_entity_ids(registry):
        entity = registry.get_entity(eid)
        assert entity.content == "AB"
        if entity is not box:
            assert entity.construction_line_ids == []


def test_undoing_the_create_restores_the_template(text_array):
    sketch, cmd, box = text_array
    registry = sketch.registry
    copies_before = len(cmd.created_entity_ids)

    cmd.undo()
    sketch.solve()

    assert cmd.array not in sketch.arrays
    assert registry.get_entity(box.id) is not None
    assert box.construction_line_ids, "helpers restored"
    remaining = registry.get_entity(box.id)
    for eid in [box.id]:
        assert registry.get_entity(eid) is remaining
    assert (
        len([e for e in registry.entities if isinstance(e, TextBoxEntity)])
        == 1
    )
    assert copies_before == 3


def test_duplicating_a_text_box_does_not_share_the_font(text_array):
    """Deep-copying a text box needs FontConfig copy support; the
    duplicate must own its font and helper lines."""
    sketch, _cmd, box = text_array
    selection = SketchSelection()
    selection.entity_ids = [box.id]
    dup_cmd = DuplicateCommand(sketch, selection)
    dup_cmd.execute()
    registry = sketch.registry
    dupes = [
        registry.get_entity(eid)
        for eid in dup_cmd.new_entity_ids
        if isinstance(registry.get_entity(eid), TextBoxEntity)
    ]
    assert len(dupes) == 1
    assert dupes[0].content == "AB"
    assert dupes[0].font_config is not box.font_config
    assert dupes[0].construction_line_ids != box.construction_line_ids


def test_serialization_round_trip_keeps_text_arrays(text_array):
    sketch, _cmd, _box = text_array
    data = sketch.to_dict()
    restored = Sketch.from_dict(data)
    restored.solve()
    assert len(restored.arrays) == 1
    assert restored.arrays[0].mode == "circular"
    text_boxes = [
        e for e in restored.registry.entities if isinstance(e, TextBoxEntity)
    ]
    assert len(text_boxes) == 4
    assert all(e.content == "AB" for e in text_boxes)


# ----------------------------------------------------------------------
# Guide drags: the template's fourth frame corner is referenced only by
# its construction lines (helper geometry), so every rigid motion of a
# member must carry the helpers along. If one is left behind, the
# template's ParallelogramConstraint is violated until a global solve
# repairs it — the corner visibly lags during the drag and snaps
# (rotating the box) when it ends.
# ----------------------------------------------------------------------


def corner_residual(sketch, box):
    """How far the fourth corner is from the parallelogram identity
    p4 = p_width + p_height - p_origin."""
    registry = sketch.registry
    o = registry.get_point(box.origin_id)
    w = registry.get_point(box.width_id)
    h = registry.get_point(box.height_id)
    p4 = registry.get_point(box.get_fourth_corner_id(registry))
    return math.hypot(p4.x - (w.x + h.x - o.x), p4.y - (w.y + h.y - o.y))


def parallelogram_residual(sketch):
    ctx = ParameterContext()
    worst = 0.0
    for constr in sketch.constraints:
        if not isinstance(constr, ParallelogramConstraint):
            continue
        err = constr.error(sketch.registry, ctx)
        worst = max(worst, max(abs(v) for v in err))
    return worst


def frame_angle(sketch, box):
    """Orientation of the template frame, in degrees."""
    registry = sketch.registry
    o = registry.get_point(box.origin_id)
    w = registry.get_point(box.width_id)
    return math.degrees(math.atan2(w.y - o.y, w.x - o.x))


@pytest.fixture
def text_curve_array():
    """Sketch with a text box arrayed along a bezier guide, built
    through the full create flow."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (30, 0), width=20, height=10)
    cmd.execute()
    assert cmd.text_box_id is not None
    box = sketch.registry.get_entity(cmd.text_box_id)
    assert isinstance(box, TextBoxEntity)
    box.content = "AB"
    box.font_config = FontConfig("sans-serif", 8.0)

    g0 = sketch.registry.add_point(0.0, 0.0)
    g1 = sketch.registry.add_point(40.0, 0.0)
    guide = sketch.registry.add_bezier(
        g0, g1, cp1=(0.0, 15.0), cp2=(40.0, -15.0)
    )
    strategy = CurveAlongArrayStrategy(
        count=4, path_entity_id=guide, align_to_tangent=True
    )
    create_cmd = CreateArrayCommand(sketch, strategy, [box.id])
    create_cmd.execute()
    sketch.solve()
    return sketch, guide, create_cmd, box


def test_corner_follows_the_frame_while_the_guide_is_dragged(
    text_curve_array,
):
    """Dragging the guide re-anchors the template every frame; the
    fourth corner must move with it instead of lagging behind."""
    sketch, guide, _cmd, box = text_curve_array
    bezier = sketch.registry.get_entity(guide)

    for frame in range(4):
        bezier.cp1 = (bezier.cp1[0], bezier.cp1[1] - 4.0)
        sketch.solve()
        assert corner_residual(sketch, box) < 1e-6, f"frame {frame}"
        assert parallelogram_residual(sketch) < 1e-6, f"frame {frame}"


def test_settling_solve_after_a_drag_does_not_rotate_the_template(
    text_curve_array,
):
    """The drag-end global solve finds every template constraint
    already satisfied: the orientation the re-anchor chose must
    survive it unchanged."""
    sketch, guide, _cmd, box = text_curve_array
    bezier = sketch.registry.get_entity(guide)

    for _frame in range(3):
        bezier.cp1 = (bezier.cp1[0], bezier.cp1[1] - 4.0)
        sketch.solve()

    angle_before = frame_angle(sketch, box)
    sketch.solve()
    assert frame_angle(sketch, box) == pytest.approx(angle_before, abs=1e-6)
    assert corner_residual(sketch, box) < 1e-6


def test_guide_center_drag_keeps_corner_on_circular_members():
    """Circular arrays move members by translation and radial
    re-projection (apply_frame); the construction lines must ride
    along there too."""
    sketch = Sketch()
    cmd = TextBoxCommand(sketch, (30, 0), width=20, height=10)
    cmd.execute()
    assert cmd.text_box_id is not None
    box = sketch.registry.get_entity(cmd.text_box_id)
    assert isinstance(box, TextBoxEntity)
    box.content = "AB"
    box.font_config = FontConfig("sans-serif", 8.0)

    create_cmd = CreateArrayCommand(sketch, make_strategy(), [box.id])
    create_cmd.execute()
    sketch.solve()
    array = create_cmd.array
    assert array is not None
    circle = sketch.registry.get_entity(array.guide_circle_id)
    assert isinstance(circle, Circle)
    center_pt = sketch.registry.get_point(circle.center_idx)

    for _frame in range(3):
        center_pt.x += 3.0
        center_pt.y -= 2.0
        sketch.solve()
        assert corner_residual(sketch, box) < 1e-6
        assert parallelogram_residual(sketch) < 1e-6

    # The construction lines must still connect the frame corners
    # after the move: the helpers were carried, not left behind.
    registry = sketch.registry
    corners = {
        (
            round(registry.get_point(pid).x, 6),
            round(registry.get_point(pid).y, 6),
        )
        for pid in box.get_all_frame_point_ids(registry)
    }
    for eid in box.construction_line_ids:
        line = registry.get_entity(eid)
        assert isinstance(line, Line)
        for pid in line.get_point_ids():
            pt = registry.get_point(pid)
            assert (round(pt.x, 6), round(pt.y, 6)) in corners
