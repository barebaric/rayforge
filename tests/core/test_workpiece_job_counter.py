from types import SimpleNamespace

from raygeo.template import CounterSettings

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece
from tests.core.template_provider import place


def _resolved(wp: WorkPiece) -> str:
    return wp._resolved_text_cache[1][1]


def test_offset_round_trip():
    wp = WorkPiece("part")
    wp.variable_offset = 3
    restored = WorkPiece.from_dict(wp.to_dict())
    assert restored.variable_offset == 3


def test_default_offset_is_zero():
    assert WorkPiece("part").variable_offset == 0


def test_counter_is_resolved_in_document():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=7)
    wp, _provider = place(doc, "No. {counter:03d}")
    assert _resolved(wp) == "No. 007"


def test_copies_show_consecutive_numbers():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=41)
    first, _provider = place(doc, "{counter}")
    second, _provider = place(doc, "{counter}", offset=1)
    assert _resolved(first) == "41"
    assert _resolved(second) == "42"


def test_counter_change_updates_text():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=5)
    wp, _provider = place(doc, "{counter:04d}")
    assert _resolved(wp) == "0005"
    doc.job_counter.counter = CounterSettings(current=12)
    assert _resolved(wp) == "0012"


def test_text_growth_keeps_scale():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=7)
    wp, _provider = place(doc, "{counter}")
    wp.set_size(5.0, 5.0)
    old_w = wp.natural_width_mm
    doc.job_counter.counter = CounterSettings(current=77)
    assert wp.natural_width_mm > old_w
    assert wp.size[0] == wp.natural_width_mm * 5.0 / old_w


def test_in_world_carries_offset_and_counter():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=9)
    wp, _provider = place(doc, "{counter}", offset=2)
    world = wp.in_world()
    assert world.variable_offset == 2
    assert world._text_values_override == {"counter": 11}
    assert world._resolved_text_cache == wp._resolved_text_cache


def test_counter_and_offset_survive_save_and_load():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=3, step=2)
    _wp, _provider = place(doc, "{counter}", offset=4)
    restored = Doc.from_dict(doc.to_dict())
    assert restored.job_counter.counter == doc.job_counter.counter
    restored_wp = restored.all_workpieces[0]
    assert restored_wp.variable_offset == 4
    assert restored_wp._resolved_text_cache[1][1] == "11"


def test_new_copy_is_numbered_automatically():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=5)
    first, _provider = place(doc, "{counter}")
    second, _provider = place(doc, "{counter}")
    assert first.variable_offset == 0
    assert second.variable_offset == 1
    assert _resolved(first) == "5"
    assert _resolved(second) == "6"


def test_saved_copy_keeps_its_explicit_offset():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=5)
    place(doc, "{counter}")
    second, _provider = place(doc, "{counter}", offset=4)
    assert second.variable_offset == 4


def test_readded_copy_keeps_its_offset():
    doc = Doc()
    doc.job_counter.counter = CounterSettings(current=5)
    first, _provider = place(doc, "{counter}")
    second, _provider = place(doc, "{counter}")
    layer = doc.active_layer
    layer.remove_child(first)
    layer.add_child(first)
    assert first.variable_offset == 0
    assert second.variable_offset == 1
    assert _resolved(first) == "5"
    assert _resolved(second) == "6"


def test_provider_without_counter_is_not_refreshed():
    doc = Doc()
    _wp, provider = place(doc, "Hello")
    calls_before = len(provider.calls)
    doc.job_counter.counter = CounterSettings(current=2)
    assert len(provider.calls) == calls_before


def test_gcode_only_counter_counts_all_workpieces():
    doc = Doc()
    place(doc, "Hello", offset=0)
    place(doc, "Hello", offset=1)
    machine = SimpleNamespace(macros={}, hookmacros={}, dialect=None)
    assert doc.job_counter_copies(machine) == 0
    machine.macros = {"mark": SimpleNamespace(code=["; Job {counter:04d}"])}
    assert doc.job_counter_copies(machine) == 2
