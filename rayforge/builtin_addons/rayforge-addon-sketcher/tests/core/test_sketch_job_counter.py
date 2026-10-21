from typing import cast

import numpy as np
from raygeo.geo.shape.text import FontConfig
from sketcher.core import Sketch
from sketcher.core.entities.text_box import TextBoxEntity

from rayforge.core.doc import Doc
from rayforge.core.job_counter import CounterSettings
from rayforge.core.undo import ChangePropertyCommand
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


def _count_solves(monkeypatch, sketch: Sketch) -> list[Sketch]:
    """Records every clone the sketch solves for a geometry request."""
    clones: list[Sketch] = []
    clone_for_geometry = sketch._clone_for_geometry

    def counting_clone() -> Sketch:
        clone = clone_for_geometry()
        clones.append(clone)
        return clone

    monkeypatch.setattr(sketch, "_clone_for_geometry", counting_clone)
    return clones


def test_geometry_is_reused_for_same_values(monkeypatch):
    sketch, _box = _sketch_with_text("{counter}")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    sketch.get_geometry({"counter": 1})
    assert len(solves) == 1
    sketch.get_geometry({"counter": 2})
    assert len(solves) == 2


def test_reused_geometry_is_a_copy():
    sketch, _box = _sketch_with_text("{counter}")
    first, _fills = sketch.get_geometry({"counter": 1})
    rect = first.rect()
    first.transform(np.diag([0.5, 0.5, 1.0, 1.0]))
    again, _fills = sketch.get_geometry({"counter": 1})
    assert again is not first
    assert again.rect() == rect


def test_unread_variable_does_not_regenerate(monkeypatch):
    sketch, _box = _sketch_with_text("Hello")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    sketch.get_geometry({"counter": 2})
    assert len(solves) == 1


def test_variable_read_by_a_parameter_regenerates(monkeypatch):
    sketch, _box = _sketch_with_text("Hello")
    sketch.params.set("gap", "counter * 2")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    sketch.get_geometry({"counter": 2})
    assert len(solves) == 2


def test_variable_read_by_a_constraint_regenerates(monkeypatch):
    sketch, _box = _sketch_with_text("Hello")
    start = sketch.add_point(0, 20)
    end = sketch.add_point(10, 20)
    sketch.constrain_distance(start, end, "counter * 10")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    sketch.get_geometry({"counter": 2})
    assert len(solves) == 2


def test_sketch_change_drops_cached_geometry(monkeypatch):
    sketch, _box = _sketch_with_text("{counter}")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    sketch.notify_update()
    sketch.get_geometry({"counter": 1})
    assert len(solves) == 2


def test_solve_drops_cached_geometry(monkeypatch):
    sketch, box = _sketch_with_text("{counter}")
    solves = _count_solves(monkeypatch, sketch)
    sketch.get_geometry({"counter": 1})
    box.content = "{counter}{counter}"
    sketch.solve()
    sketch.get_geometry({"counter": 1})
    assert len(solves) == 2


def test_volatile_text_is_not_shared_between_callers(monkeypatch):
    sketch, box = _sketch_with_text("{uuid8()}")
    solves = _count_solves(monkeypatch, sketch)
    first: dict = {}
    second: dict = {}
    sketch.get_geometry(resolved_text_cache=first)
    sketch.get_geometry(resolved_text_cache=second)
    assert first[box.id] != second[box.id]
    assert len(solves) == 2


def test_volatile_text_reuses_geometry_for_its_caller(monkeypatch):
    sketch, _box = _sketch_with_text("{uuid8()}")
    solves = _count_solves(monkeypatch, sketch)
    cache: dict = {}
    sketch.get_geometry(resolved_text_cache=cache)
    sketch.get_geometry(resolved_text_cache=cache)
    assert len(solves) == 1


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


def _sketch_with_boxes(*contents: str) -> tuple[Sketch, list[TextBoxEntity]]:
    sketch = Sketch(name="plate")
    boxes = []
    for i, content in enumerate(contents):
        origin = sketch.add_point(0, i * 20)
        width_pt = sketch.add_point(10, i * 20)
        height_pt = sketch.add_point(0, i * 20 + 10)
        box_id = sketch.registry.add_text_box(
            origin, width_pt, height_pt, content, FontConfig()
        )
        boxes.append(cast(TextBoxEntity, sketch.registry.get_entity(box_id)))
    return sketch, boxes


def test_counter_change_keeps_uuid_of_other_sketch():
    doc = Doc()
    counter_sketch, _counter_box = _sketch_with_text("{counter}")
    _place(doc, counter_sketch)
    sketch, box = _sketch_with_text("ID {uuid8()}")
    wp = _place(doc, sketch)
    before = _resolved(wp, box)
    doc.job_counter.counter = CounterSettings(current=2)
    assert _resolved(wp, box) == before
    doc.job_counter.counter = CounterSettings(current=1)
    assert _resolved(wp, box) == before


def test_counter_change_keeps_uuid_next_to_counter():
    doc = Doc()
    sketch, (counter_box, uuid_box) = _sketch_with_boxes(
        "{counter}", "ID {uuid8()}"
    )
    wp = _place(doc, sketch)
    before = _resolved(wp, uuid_box)
    doc.job_counter.counter = CounterSettings(current=5)
    assert _resolved(wp, counter_box) == "5"
    assert _resolved(wp, uuid_box) == before
    doc.job_counter.counter = CounterSettings(current=1)
    assert _resolved(wp, counter_box) == "1"
    assert _resolved(wp, uuid_box) == before


def test_uuid_survives_next_and_undo():
    doc = Doc()
    counter_sketch, counter_box = _sketch_with_text("{counter}")
    counter_wp = _place(doc, counter_sketch)
    sketch, box = _sketch_with_text("ID {uuid8()}")
    wp = _place(doc, sketch)
    before = _resolved(wp, box)
    doc.history_manager.execute(
        ChangePropertyCommand(
            target=doc.job_counter,
            property_name="counter",
            new_value=doc.job_counter.counter.advanced(1),
        )
    )
    assert _resolved(counter_wp, counter_box) == "2"
    assert _resolved(wp, box) == before
    doc.history_manager.undo()
    assert _resolved(counter_wp, counter_box) == "1"
    assert _resolved(wp, box) == before
