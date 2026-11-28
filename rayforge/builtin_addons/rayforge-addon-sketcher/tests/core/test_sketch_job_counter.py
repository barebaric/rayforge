from typing import cast

from raygeo.sketch.entities import TextBox as TextBoxEntity
from raygeo.template import CounterSettings
from sketcher.core import Sketch

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece


def _sketch_with_text(content: str) -> tuple[Sketch, TextBoxEntity]:
    sketch = Sketch(name="plate")
    origin = sketch.add_point(0, 0)
    width_pt = sketch.add_point(10, 0)
    height_pt = sketch.add_point(0, 10)
    box_id = sketch.registry.add_text_box(
        origin, width_pt, height_pt, content, "sans-serif", 10.0
    )
    return sketch, cast(TextBoxEntity, sketch.registry.get_entity(box_id))


def _place(doc: Doc, sketch: Sketch, offset: int = 0) -> WorkPiece:
    doc.add_asset(sketch)
    wp = WorkPiece.from_geometry_provider(sketch)
    wp.variable_offset = offset
    doc.active_setup.add_child(wp)
    return wp


def _resolved(wp: WorkPiece, box: TextBoxEntity) -> str:
    return wp._resolved_text_cache[box.id][1]


def test_variables_used_reports_counter():
    sketch, _box = _sketch_with_text("No. {counter:03d}")
    assert sketch.variables_used() == {"counter"}


def test_plain_sketch_uses_no_variables():
    sketch, _box = _sketch_with_text("Hello")
    assert sketch.variables_used() == set()


def test_counter_is_resolved_in_document():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=7)
    sketch, box = _sketch_with_text("No. {counter:03d}")
    wp = _place(doc, sketch)
    assert _resolved(wp, box) == "No. 007"


def test_copies_show_consecutive_numbers():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=41)
    sketch, box = _sketch_with_text("{counter}")
    first = _place(doc, sketch)
    second = first.duplicate()
    assert isinstance(second, WorkPiece)
    second.variable_offset = 1
    doc.active_setup.add_child(second)
    assert _resolved(first, box) == "41"
    assert _resolved(second, box) == "42"


def test_counter_change_updates_text():
    doc = Doc()
    sketch, box = _sketch_with_text("{counter:04d}")
    wp = _place(doc, sketch)
    doc.job_counter.counter = CounterSettings(current=12)
    assert _resolved(wp, box) == "0012"


def test_counter_change_keeps_uuid_of_other_box():
    doc = Doc()
    sketch, counter_box = _sketch_with_text("{counter}")
    uuid_origin = sketch.add_point(0, 20)
    uuid_width = sketch.add_point(10, 20)
    uuid_height = sketch.add_point(0, 30)
    uuid_box_id = sketch.registry.add_text_box(
        uuid_origin,
        uuid_width,
        uuid_height,
        "ID {uuid8()}",
        "sans-serif",
        10.0,
    )
    uuid_box = cast(TextBoxEntity, sketch.registry.get_entity(uuid_box_id))
    wp = _place(doc, sketch)
    before = _resolved(wp, uuid_box)
    doc.job_counter.counter = CounterSettings(current=5)
    assert _resolved(wp, counter_box) == "5"
    assert _resolved(wp, uuid_box) == before


def test_uuid_next_to_counter_in_one_box_survives():
    doc = Doc()
    sketch, box = _sketch_with_text("ID {uuid8()} No. {counter}")
    wp = _place(doc, sketch)
    before = _resolved(wp, box)
    doc.job_counter.counter = CounterSettings(current=9)
    after = _resolved(wp, box)
    assert before.split()[1] == after.split()[1]
    assert "No. 9" in after


def test_saved_text_is_kept_on_load():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=5)
    sketch, box = _sketch_with_text("{counter}-{uuid8()}")
    wp = _place(doc, sketch)
    before = _resolved(wp, box)
    restored = Doc.from_dict(doc.to_dict())
    restored_wp = restored.all_workpieces[0]
    assert restored_wp.boundaries is not None
    assert _resolved(restored_wp, box) == before
