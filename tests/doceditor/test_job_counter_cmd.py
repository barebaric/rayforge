from raygeo.template import CounterSettings

from rayforge.machine.job_status import JobKind, JobOutcome
from tests.core.template_provider import place


def _two_copies(doc_editor):
    place(doc_editor.doc, "{counter}", offset=0)
    place(doc_editor.doc, "{counter}", offset=1)


def test_set_counter_is_undoable(doc_editor):
    doc_editor.job_counter.set_counter(CounterSettings(current=7))
    assert doc_editor.doc.job_counter.counter.current == 7
    doc_editor.history_manager.undo()
    assert doc_editor.doc.job_counter.counter.current == 1


def test_next_previous_reset(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(CounterSettings(current=5))
    doc_editor.job_counter.next_counter()
    assert doc_editor.doc.job_counter.counter.current == 7
    doc_editor.job_counter.previous_counter()
    assert doc_editor.doc.job_counter.counter.current == 5
    doc_editor.job_counter.reset_counter()
    assert doc_editor.doc.job_counter.counter.current == 1


def test_preview_values(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(CounterSettings(current=5))
    assert doc_editor.job_counter.preview_counter_values() == [5, 6]


def test_advance_after_job_counts_copies(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(CounterSettings(current=5))
    assert doc_editor.job_counter.advance_after_job() is True
    assert doc_editor.doc.job_counter.counter.current == 7


def test_advance_ignores_disabled_auto_advance(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(
        CounterSettings(current=5, auto_advance=False)
    )
    assert doc_editor.job_counter.advance_after_job() is False
    assert doc_editor.doc.job_counter.counter.current == 5


def test_job_finished_only_advances_completed_send(doc_editor):
    _two_copies(doc_editor)
    doc_editor.job_counter.set_counter(CounterSettings(current=5))
    doc_editor.job_counter.on_job_finished(
        None, kind=JobKind.FRAME, outcome=JobOutcome.COMPLETED
    )
    assert doc_editor.doc.job_counter.counter.current == 5
    doc_editor.job_counter.on_job_finished(
        None, kind=JobKind.SEND, outcome=JobOutcome.CANCELLED
    )
    assert doc_editor.doc.job_counter.counter.current == 5
    doc_editor.job_counter.on_job_finished(
        None, kind=JobKind.SEND, outcome=JobOutcome.COMPLETED
    )
    assert doc_editor.doc.job_counter.counter.current == 7


def test_number_copies_uses_reading_order(doc_editor):
    positions = [(0.0, 0.0), (20.0, 0.0), (0.0, 20.0), (20.0, 20.0)]
    workpieces = []
    for x, y in positions:
        wp, _provider = place(doc_editor.doc, "{counter}")
        wp.set_size(10.0, 10.0)
        wp.pos = (x, y)
        workpieces.append(wp)

    doc_editor.job_counter.number_copies(workpieces)

    offsets = {(wp.pos[0], wp.pos[1]): wp.variable_offset for wp in workpieces}
    assert offsets[(0.0, 20.0)] == 0
    assert offsets[(20.0, 20.0)] == 1
    assert offsets[(0.0, 0.0)] == 2
    assert offsets[(20.0, 0.0)] == 3
