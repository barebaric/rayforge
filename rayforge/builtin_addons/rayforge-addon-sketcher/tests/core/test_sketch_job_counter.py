from typing import cast

from raygeo.geo.shape.text import FontConfig
from sketcher.core import Sketch
from sketcher.core.entities.text_box import TextBoxEntity

from rayforge.core.doc import Doc
from rayforge.core.job_counter import CounterSettings
from rayforge.core.workpiece import WorkPiece


def _sketch_with_text(content: str) -> tuple[Sketch, TextBoxEntity]:
    sketch = Sketch(name="plate")
    origin = sketch.add_point(0, 0)
    width_pt = sketch.add_point(10, 0)
    height_pt = sketch.add_point(0, 10)
    box_id = sketch.registry.add_text_box(
        origin, width_pt, height_pt, content, FontConfig()
    )
    return sketch, cast(TextBoxEntity, sketch.registry.get_entity(box_id))


def _resolved(wp: WorkPiece, box: TextBoxEntity) -> str:
    entry = wp._resolved_text_cache[box.id]
    return entry[1]


def _place(doc: Doc, sketch: Sketch, offset: int = 0) -> WorkPiece:
    doc.add_asset(sketch)
    wp = WorkPiece.from_geometry_provider(sketch)
    wp.variable_offset = offset
    doc.active_layer.add_child(wp)
    return wp


def test_template_sources_lists_text_box_contents():
    sketch, _box = _sketch_with_text("No. {counter:03d}")
    assert sketch.get_template_sources() == ["No. {counter:03d}"]


def test_variables_used_reports_read_names():
    sketch, _box = _sketch_with_text("No. {counter:03d}")
    assert sketch.variables_used() == {"counter"}


def test_plain_sketch_uses_no_variables():
    sketch, _box = _sketch_with_text("Hello")
    assert sketch.variables_used() == set()


def test_geometry_is_reused_for_same_values():
    sketch, _box = _sketch_with_text("{counter}")
    first, _fills = sketch.get_geometry({"counter": 1})
    again, _fills = sketch.get_geometry({"counter": 1})
    assert again is first
    other, _fills = sketch.get_geometry({"counter": 2})
    assert other is not first


def test_sketch_change_drops_cached_geometry():
    sketch, _box = _sketch_with_text("{counter}")
    first, _fills = sketch.get_geometry({"counter": 1})
    sketch.notify_update()
    again, _fills = sketch.get_geometry({"counter": 1})
    assert again is not first


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
    doc.active_layer.add_child(second)
    assert _resolved(first, box) == "41"
    assert _resolved(second, box) == "42"


def test_counter_change_updates_text():
    doc = Doc()
    sketch, box = _sketch_with_text("{counter:04d}")
    wp = _place(doc, sketch)
    doc.job_counter.counter = CounterSettings(current=12)
    assert _resolved(wp, box) == "0012"


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
