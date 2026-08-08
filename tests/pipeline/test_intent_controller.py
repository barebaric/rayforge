"""
Tests for :mod:`rayforge.pipeline.intent_controller`.

These tests use the existing :class:`~rayforge.core.doc.Doc` /
:class:`Step` / :class:`WorkPiece` classes (real signal wiring) and a
fake :class:`TaskManager` so they do not require a running GTK event
loop.
"""

from typing import Any, Callable, Optional

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Machine
from rayforge.pipeline.intent_builder import (
    job_encode_key,
    job_key,
    step_key,
    workpiece_key,
)
from rayforge.pipeline.intent_controller import (
    REBUILD_DEBOUNCE_MS,
    IntentController,
)


class _TestStep(Step):
    """Concrete ``Step`` for tests; controls the position-sensitive
    flag without pulling in the transformer addon registry."""

    def __init__(self, name: str = "test", position_sensitive: bool = False):
        super().__init__(typelabel="test", name=name)
        self._position_sensitive = position_sensitive

    def is_position_sensitive(self) -> bool:
        return self._position_sensitive


class FakeCancelHandle:
    def __init__(self):
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @property
    def cancelled(self):
        return self._cancelled


class _ScheduledCall(FakeCancelHandle):
    def __init__(self, delay: int, fn: Callable[[], None]):
        super().__init__()
        self.delay = delay
        self.fn = fn
        self._fired = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def fire(self):
        if not self._cancelled and not self._fired:
            self._fired = True
            self.fn()


class FakeTaskManager:
    """
    Drop-in replacement for :class:`TaskManager` that records scheduled
    calls and lets tests fire them deterministically.
    """

    def __init__(self):
        self.delayed: list[_ScheduledCall] = []
        self.main_thread_calls: list[Callable[..., Any]] = []

    def schedule_on_main_thread(
        self,
        callback: Callable[..., Any],
        *_args: Any,
        **_kw: Any,
    ) -> FakeCancelHandle:
        self.main_thread_calls.append(callback)
        return FakeCancelHandle()

    def schedule_delayed_on_main_thread(
        self,
        delay_ms: int,
        callback: Callable[..., Any],
        *_args,
        **_kw,
    ) -> FakeCancelHandle:
        call = _ScheduledCall(delay_ms, callback)
        self.delayed.append(call)
        return call

    def run_thread(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[..., Any]] = None,
        **kwargs: Any,
    ) -> Any:
        func(*args, **kwargs)
        if when_done:
            when_done(None)
        return None

    def fire_latest(self) -> None:
        """Fire the most recently scheduled delayed call and remove it
        from the pending list.

        Any main-thread calls produced as a side-effect of the
        rebuild (e.g. ``_emit_rebuild_finished``) are cleared so the
        caller only sees calls it produces itself.
        """
        assert self.delayed, "no delayed call scheduled"
        call = self.delayed.pop()
        call.fire()
        self.main_thread_calls.clear()


class _StubNode:
    """Minimal stand-in for ``raygeo.CompletedNode`` for tests."""

    def __init__(
        self,
        key: str,
        generation_id: int,
        output: Any = None,
        error: Optional[str] = None,
    ):
        self.key = key
        self.generation_id = generation_id
        self.output = output
        self.error = error
        self.error_kind = None


def _make_doc(step: _TestStep, *workpieces: WorkPiece) -> Doc:
    doc = Doc()
    layer = doc.active_layer
    workflow = layer.workflow
    assert workflow is not None
    workflow.add_child(step)
    for wp in workpieces:
        layer.add_child(wp)
    return doc


# ----------------------------------------------------------------------
# Construction
# ----------------------------------------------------------------------


def test_raygeo_pipeline_default_constructed(isolated_machine):
    doc = _make_doc(_TestStep(name="s1"), WorkPiece(name="wp"))
    ctrl = IntentController(doc, FakeTaskManager(), machine=isolated_machine)
    assert ctrl.raygeo_pipeline is not None


# ----------------------------------------------------------------------
# Debounced rebuild
# ----------------------------------------------------------------------


def test_signal_triggers_debounced_rebuild(isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()

    # Trigger a change and verify a debounced call is scheduled.
    wp.updated.send(wp)
    assert len(tm.delayed) == 1
    assert tm.delayed[0].delay == REBUILD_DEBOUNCE_MS

    # Fire the debounced callback and verify the intent was built.
    tm.fire_latest()
    assert ctrl.intent is not None
    assert ctrl.generation_id == 1
    ctrl.shutdown()


def test_second_change_reschedules_debounce(isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()

    wp.updated.send(wp)
    timer = ctrl._rebuild_timer
    assert timer is not None

    wp.updated.send(wp)  # immediately sends again — should cancel first
    assert timer.cancelled
    assert len(tm.delayed) == 2
    tm.fire_latest()
    assert ctrl.generation_id == 1
    ctrl.shutdown()


# ----------------------------------------------------------------------
# intent.update semantics
# ----------------------------------------------------------------------


def test_intent_updates_on_subsequent_rebuilds(isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()

    wp.updated.send(wp)
    tm.fire_latest()
    intent_first = ctrl.intent
    assert intent_first is not None
    gen_first = ctrl.generation_id

    step.cut_speed = 4321
    # No signal fired — the controller does not rebuild on plain
    # attribute assignment that bypasses the Step's signal-emitting
    # setters.
    assert not tm.delayed
    assert ctrl.generation_id == gen_first

    # Now fire a signal and rebuild.
    step.updated.send(step)
    assert tm.delayed
    tm.fire_latest()
    assert ctrl.generation_id > gen_first
    assert ctrl.intent is intent_first  # updated in place
    ctrl.shutdown()


# ----------------------------------------------------------------------
# run_intent is always called
# ----------------------------------------------------------------------


def test_run_intent_called(monkeypatch, isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()

    run_calls: list[Any] = []

    def _capture_run(
        intent, on_completed=None, on_batch_progress=None, pipeline=None
    ):
        run_calls.append((intent, on_completed, pipeline))

    monkeypatch.setattr(
        "rayforge.pipeline.intent_controller.run_intent", _capture_run
    )
    wp.updated.send(wp)
    tm.fire_latest()
    assert len(run_calls) == 1
    intent, on_completed, pipeline = run_calls[0]
    assert intent is ctrl.intent
    assert on_completed == ctrl._on_completed
    assert pipeline is ctrl.raygeo_pipeline
    ctrl.shutdown()


# ----------------------------------------------------------------------
# Epoch filter
# ----------------------------------------------------------------------


def _make_controller_for_completed_test(
    monkeypatch,
    idle_calls: Optional[list] = None,
    machine: Optional["Machine"] = None,
):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    idle_calls = idle_calls if idle_calls is not None else []

    def _capture(callback: Callable[..., Any], *args: Any, **_kw: Any):
        idle_calls.append((callback, args))
        return FakeCancelHandle()

    tm.schedule_on_main_thread = _capture
    ctrl = IntentController(doc, tm, machine=machine)
    ctrl.connect()

    # Build once so the key map is populated.
    monkeypatch.setattr(
        "rayforge.pipeline.intent_controller.run_intent",
        lambda *a, **kw: None,
    )
    wp.updated.send(wp)
    tm.fire_latest()
    idle_calls.clear()
    return ctrl, wp, step


def test_on_completed_superseded_generation_discarded(
    monkeypatch, isolated_machine
):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    wpk = workpiece_key(wp.uid, step.uid)

    # Simulate a stale result (older generation).
    stale = _StubNode(key=wpk, generation_id=0, output="stale")
    ctrl._on_completed(stale)
    assert idle_calls == []

    # Simulate a current result.
    current = _StubNode(
        key=wpk, generation_id=ctrl.generation_id, output="fresh"
    )
    ctrl._on_completed(current)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    assert isinstance(args, tuple) and len(args) == 3
    ctrl.shutdown()


def test_on_completed_unknown_key_skipped(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    # Generate a key not in the map.
    node = _StubNode(key="nonexistent", generation_id=ctrl.generation_id)
    ctrl._on_completed(node)
    assert idle_calls == []
    ctrl.shutdown()


def test_on_completed_reaches_correct_doc_item(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    # Verify the key->DocItem map includes workpiece, step, and job keys.
    wpk = workpiece_key(wp.uid, step.uid)
    assert ctrl._key_to_item[wpk] is wp

    sk = step_key(step.uid)
    assert ctrl._key_to_item[sk] is step

    jk = job_key()
    assert ctrl._key_to_item[jk] is ctrl._doc

    # Fire a completion for the workpiece key.
    node = _StubNode(key=wpk, generation_id=ctrl.generation_id, output="ok")
    ctrl._on_completed(node)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    key, item, output = args
    assert key == wpk
    assert item is wp
    assert output == "ok"
    ctrl.shutdown()


# ----------------------------------------------------------------------
# Reattachment → signals (B2.4)
# ----------------------------------------------------------------------


def test_reattach_workpiece_emits_signal(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    wpk = workpiece_key(wp.uid, step.uid)

    received = []

    def _on_wp(sender, **kw):
        received.append(kw)

    ctrl.workpiece_artifact_ready.connect(_on_wp)

    node = _StubNode(key=wpk, generation_id=ctrl.generation_id, output="ops")
    ctrl._on_completed(node)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(received) == 1
    payload = received[0]
    assert payload["step"] is step
    assert payload["workpiece"] is wp
    assert payload["output"] == "ops"
    ctrl.shutdown()


def test_reattach_step_emits_signal(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    sk = step_key(step.uid)

    received = []

    def _on_step(sender, **kw):
        received.append(kw)

    ctrl.step_artifact_ready.connect(_on_step)

    node = _StubNode(key=sk, generation_id=ctrl.generation_id, output="agg")
    ctrl._on_completed(node)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(received) == 1
    assert received[0]["step"] is step
    assert received[0]["output"] == "agg"
    ctrl.shutdown()


def test_reattach_job_emits_aggregate_and_time(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    class _AggOutput:
        time_estimate = 12.5

    agg_received = []
    time_received = []

    def _on_agg(sender, **kw):
        agg_received.append(kw)

    def _on_time(sender, **kw):
        time_received.append(kw)

    ctrl.job_aggregate_ready.connect(_on_agg)
    ctrl.job_time_updated.connect(_on_time)

    node = _StubNode(
        key=job_key(), generation_id=ctrl.generation_id, output=_AggOutput()
    )
    ctrl._on_completed(node)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(agg_received) == 1
    assert len(time_received) == 1
    assert time_received[0]["total_seconds"] == 12.5
    ctrl.shutdown()


def test_reattach_job_encode_emits_finished(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    # The controller has no machine, so the builder doesn't emit a
    # job:encode node; inject the key manually so _on_completed
    # routes it.
    assert ctrl._doc is not None
    ctrl._key_to_item[job_encode_key()] = ctrl._doc

    received = []

    def _on_finished(sender, **kw):
        received.append(kw)

    ctrl.job_generation_finished.connect(_on_finished)

    node = _StubNode(
        key=job_encode_key(),
        generation_id=ctrl.generation_id,
        output="encoded",
    )
    ctrl._on_completed(node)
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(received) == 1
    assert received[0]["handle"] == "encoded"
    assert received[0]["task_status"] == "completed"
    ctrl.shutdown()


def test_on_batch_progress_emits_progress_changed(
    monkeypatch, isolated_machine
):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    received = []

    def _on_progress(sender, **kw):
        received.append(kw)

    ctrl.progress_changed.connect(_on_progress)

    ctrl._on_batch_progress(0.5, "halfway")
    # _on_batch_progress marshals onto the main thread via
    # schedule_on_main_thread, which the fake captures.
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(received) == 1
    assert received[0]["fraction"] == 0.5
    assert received[0]["message"] == "halfway"
    ctrl.shutdown()


def test_on_completed_with_warnings_emits_pipeline_warnings(
    monkeypatch, isolated_machine
):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    wpk = workpiece_key(wp.uid, step.uid)

    received = []

    def _on_warnings(sender, **kw):
        received.append(kw)

    ctrl.pipeline_warnings.connect(_on_warnings)

    class _OutputWithWarnings:
        warnings = ["warn1", "warn2"]

    node = _StubNode(
        key=wpk,
        generation_id=ctrl.generation_id,
        output=_OutputWithWarnings(),
    )
    ctrl._on_completed(node)

    # Two main-thread callbacks: the reattach and the warnings emit.
    assert len(idle_calls) == 2
    warning_emit = next(
        (fn, args)
        for fn, args in idle_calls
        if fn.__name__ == "_emit_pipeline_warnings"
    )
    fn, args = warning_emit
    fn(*args)
    assert len(received) == 1
    assert received[0]["warnings"] == ["warn1", "warn2"]
    ctrl.shutdown()


def test_on_completed_without_warnings_skips_emit(
    monkeypatch, isolated_machine
):
    idle_calls: list = []
    ctrl, wp, step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )
    wpk = workpiece_key(wp.uid, step.uid)

    received = []

    def _on_warnings(sender, **kw):
        received.append(kw)

    ctrl.pipeline_warnings.connect(_on_warnings)

    node = _StubNode(
        key=wpk,
        generation_id=ctrl.generation_id,
        output="plain-output",
    )
    ctrl._on_completed(node)

    # Only the reattach callback is scheduled; no warnings emit.
    assert len(idle_calls) == 1
    fn, _ = idle_calls[0]
    assert fn.__name__ == "_reattach"
    assert received == []
    ctrl.shutdown()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------


def test_shutdown_cancels_pending_timer(isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()

    wp.updated.send(wp)
    timer = ctrl._rebuild_timer
    assert timer is not None
    ctrl.shutdown()
    assert ctrl._rebuild_timer is None


def test_disconnect_prevents_further_rebuilds(isolated_machine):
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc = _make_doc(step, wp)
    tm = FakeTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    ctrl.connect()
    ctrl.disconnect()

    wp.updated.send(wp)
    assert tm.delayed == []
