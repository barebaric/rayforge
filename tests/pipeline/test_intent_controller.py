"""
Tests for :mod:`rayforge.pipeline.intent_controller`.

These tests use the existing :class:`~rayforge.core.doc.Doc` /
:class:`Step` / :class:`WorkPiece` classes (real signal wiring) and a
fake :class:`TaskManager` so they do not require a running GTK event
loop.
"""

from collections.abc import Callable
from typing import Any, ClassVar, Optional

from raygeo.geo import Geometry, Matrix

from rayforge.core.doc import Doc
from rayforge.core.step import Step
from rayforge.core.stock import StockItem
from rayforge.core.stock_asset import StockAsset
from rayforge.core.workpiece import WorkPiece
from rayforge.machine.models.machine import Machine
from rayforge.machine.models.machine_panel import PanelOrientation
from rayforge.machine.models.rotary_module import RotaryModule
from rayforge.pipeline.intent_builder import (
    IntentBuilder,
    job_encode_key,
    job_key,
    step_key,
    stock_key,
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
        key: Any | None = None,
        when_done: Callable[..., Any] | None = None,
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


class ImmediateMainThreadTaskManager(FakeTaskManager):
    """Fake scheduler that executes main-thread callbacks immediately."""

    def schedule_on_main_thread(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> FakeCancelHandle:
        self.main_thread_calls.append(callback)
        callback(*args, **kwargs)
        return FakeCancelHandle()


class _StubNode:
    """Minimal stand-in for ``raygeo.CompletedNode`` for tests."""

    def __init__(
        self,
        key: str,
        generation_id: int,
        output: Any = None,
        error: str | None = None,
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


def test_rotary_panel_error_is_reported(isolated_machine):
    doc = _make_doc(_TestStep(name="s1"), WorkPiece(name="wp"))
    doc.active_layer.set_rotary_enabled(True)
    isolated_machine.set_panel_orientation(PanelOrientation.ROTATED_LEFT)
    tm = ImmediateMainThreadTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    messages: list[str] = []
    ctrl.pipeline_error.connect(
        lambda _sender, *, message: messages.append(message), weak=False
    )

    ctrl.force_rebuild()

    assert messages == [
        (
            "Rotary layers require the Native panel orientation. "
            "Set Machine → Hardware → Panel Orientation to Native."
        )
    ]
    assert ctrl.intent is not None
    ctrl.shutdown()


def test_unexpected_build_error_completes_job_waiters(
    monkeypatch, isolated_machine
):
    """An unexpected exception during intent building must still emit
    ``job_generation_finished`` (with a failure status), so callers
    awaiting job generation are never left blocked forever."""
    doc = _make_doc(_TestStep(name="s1"), WorkPiece(name="wp"))
    tm = ImmediateMainThreadTaskManager()
    ctrl = IntentController(doc, tm, machine=isolated_machine)
    finished: list[tuple] = []
    ctrl.job_generation_finished.connect(
        lambda _sender, **kw: finished.append(
            (kw.get("handle"), kw.get("task_status"))
        ),
        weak=False,
    )

    def _boom(self, doc):
        raise KeyError("simulated serialization bug")

    monkeypatch.setattr(IntentBuilder, "build", _boom)

    ctrl.force_rebuild()

    assert ("failed" in [status for _, status in finished]) or (
        None,
        "failed",
    ) in finished
    assert (None, "failed") in finished
    ctrl.shutdown()


# ----------------------------------------------------------------------
# Epoch filter
# ----------------------------------------------------------------------


def _make_controller_for_completed_test(
    monkeypatch,
    idle_calls: list | None = None,
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
    _fn, args = idle_calls[0]
    assert isinstance(args, tuple) and len(args) == 3
    ctrl.shutdown()


def test_on_completed_unknown_key_skipped(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, _wp, _step = _make_controller_for_completed_test(
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
    _fn, args = idle_calls[0]
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
    ctrl, _wp, step = _make_controller_for_completed_test(
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
    ctrl, _wp, _step = _make_controller_for_completed_test(
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
    ctrl, _wp, _step = _make_controller_for_completed_test(
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
    ctrl, _wp, _step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    received = []

    def _on_progress(sender, **kw):
        received.append(kw)

    ctrl.progress_changed.connect(_on_progress)

    ctrl._on_batch_progress(0.5, "job:encode")
    # _on_batch_progress marshals onto the main thread via
    # schedule_on_main_thread, which the fake captures.
    assert len(idle_calls) == 1
    fn, args = idle_calls[0]
    fn(*args)
    assert len(received) == 1
    assert received[0]["fraction"] == 0.5
    assert received[0]["message"] == "job:encode"
    ctrl.shutdown()


def test_on_batch_progress_updates_rebuild_task(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, _wp, _step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    class _FakeTask:
        def __init__(self):
            self.updates: list[tuple] = []

        def update(self, progress=None, message=None):
            self.updates.append((progress, message))

    fake_task = _FakeTask()
    ctrl._rebuild_task = fake_task

    ctrl._on_batch_progress(0.5, "job:encode")
    fn, args = idle_calls[0]
    fn(*args)
    assert fake_task.updates == [(0.5, "Generating machine code")]
    ctrl.shutdown()


def test_batch_progress_shows_parallel_tasks(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, _wp, _step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    class _FakeTask:
        def __init__(self):
            self.updates: list[tuple] = []

        def update(self, progress=None, message=None):
            self.updates.append((progress, message))

    fake_task = _FakeTask()
    ctrl._rebuild_task = fake_task

    ctrl._on_batch_progress(0.4, "job")
    ctrl._on_batch_progress(0.5, "job:encode")
    while idle_calls:
        fn, args = idle_calls.pop(0)
        fn(*args)
    assert fake_task.updates[-1] == (
        0.5,
        "Aggregating job\nGenerating machine code",
    )
    ctrl.shutdown()


def test_batch_progress_completion_removes_node(monkeypatch, isolated_machine):
    idle_calls: list = []
    ctrl, _wp, _step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    class _FakeTask:
        def __init__(self):
            self.updates: list[tuple] = []

        def update(self, progress=None, message=None):
            self.updates.append((progress, message))

    fake_task = _FakeTask()
    ctrl._rebuild_task = fake_task

    ctrl._on_batch_progress(0.4, "job")
    ctrl._on_batch_progress(0.5, "\tjob")
    while idle_calls:
        fn, args = idle_calls.pop(0)
        fn(*args)
    assert fake_task.updates[-1] == (0.5, "")
    ctrl.shutdown()


def test_batch_progress_final_tick_clears_window(
    monkeypatch, isolated_machine
):
    idle_calls: list = []
    ctrl, _wp, _step = _make_controller_for_completed_test(
        monkeypatch,
        idle_calls=idle_calls,
        machine=isolated_machine,
    )

    class _FakeTask:
        def __init__(self):
            self.updates: list[tuple] = []

        def update(self, progress=None, message=None):
            self.updates.append((progress, message))

    fake_task = _FakeTask()
    ctrl._rebuild_task = fake_task

    ctrl._on_batch_progress(0.4, "job")
    ctrl._on_batch_progress(1.0, "")
    while idle_calls:
        fn, args = idle_calls.pop(0)
        fn(*args)
    assert fake_task.updates[-1] == (1.0, "")
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
        warnings: ClassVar[list[str]] = ["warn1", "warn2"]

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


# ----------------------------------------------------------------------
# Stock fold reattachment
# ----------------------------------------------------------------------


def _make_doc_with_stock(
    step: _TestStep, wp: WorkPiece
) -> tuple[Doc, StockItem]:
    """Build a Doc with a stock item overlapping *wp*; return
    ``(doc, stock_item)``."""
    # Give the workpiece a 50×50 mm world geometry at the origin so it
    # overlaps the stock (AABB intersection drives fold-node emission).
    geo_wp = Geometry()
    geo_wp.move_to(0, 0)
    geo_wp.line_to(1, 0)
    geo_wp.line_to(1, 1)
    geo_wp.line_to(0, 1)
    geo_wp.close_path()
    wp._boundaries_cache = geo_wp
    wp.matrix = Matrix.scale(50.0, 50.0)

    doc = _make_doc(step, wp)
    asset = StockAsset(name="sheet")
    asset.set_thickness(18.0)
    geo = Geometry()
    geo.move_to(0, 0)
    geo.line_to(100, 0)
    geo.line_to(100, 80)
    geo.line_to(0, 80)
    geo.close_path()
    asset.geometry = geo
    doc.add_asset(asset)
    item = StockItem(stock_asset_uid=asset.uid, name="sheet")
    doc.add_child(item)
    return doc, item


def _make_controller_with_stock(monkeypatch, machine, step, wp, stock_item):
    """Build a controller whose key map includes ``stock:{uid}``."""
    doc = stock_item.parent
    tm = ImmediateMainThreadTaskManager()
    ctrl = IntentController(doc, tm, machine=machine)
    ctrl.connect()
    # Build once so the key map is populated; run_intent is stubbed
    # so no real pipeline work happens.
    monkeypatch.setattr(
        "rayforge.pipeline.intent_controller.run_intent",
        lambda *a, **kw: None,
    )
    wp.updated.send(wp)
    tm.fire_latest()
    # Clear any rebuild-finished main-thread calls queued above so the
    # stock tests only see their own.
    tm.main_thread_calls.clear()
    return ctrl


def test_stock_key_in_reattach_map(monkeypatch, isolated_machine):
    """A ``stock:{uid}`` key is mapped to its StockItem in the
    reattachment map."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    _doc, item = _make_doc_with_stock(step, wp)
    ctrl = _make_controller_with_stock(
        monkeypatch, isolated_machine, step, wp, item
    )
    sk = stock_key(item.uid)
    assert sk in ctrl._key_to_item
    assert ctrl._key_to_item[sk] is item
    ctrl.shutdown()


def test_material_state_ready_emitted_for_current_generation(
    monkeypatch, isolated_machine
):
    """A current-generation ``stock:`` result emits
    ``material_state_ready`` with the stock item and output."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    _doc, item = _make_doc_with_stock(step, wp)
    ctrl = _make_controller_with_stock(
        monkeypatch, isolated_machine, step, wp, item
    )

    received: list = []
    ctrl.material_state_ready.connect(
        lambda _sender, *, item, output, generation_id: received.append(
            (item, output, generation_id)
        ),
        weak=False,
    )

    sk = stock_key(item.uid)
    node = _StubNode(key=sk, generation_id=ctrl.generation_id, output="state")
    ctrl._on_completed(node)
    assert len(received) == 1
    assert received[0][0] is item
    assert received[0][1] == "state"
    assert received[0][2] == ctrl.generation_id
    ctrl.shutdown()


def test_superseded_stock_generation_discarded(monkeypatch, isolated_machine):
    """A stale ``stock:`` result (older generation) is discarded."""
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    _doc, item = _make_doc_with_stock(step, wp)
    ctrl = _make_controller_with_stock(
        monkeypatch, isolated_machine, step, wp, item
    )

    received: list = []
    ctrl.material_state_ready.connect(
        lambda _sender, *, item, output, generation_id: received.append(
            (item, output, generation_id)
        ),
        weak=False,
    )

    sk = stock_key(item.uid)
    stale = _StubNode(key=sk, generation_id=0, output="stale")
    ctrl._on_completed(stale)
    assert received == []
    ctrl.shutdown()


# ----------------------------------------------------------------------
# Rotary layer fold reattachment
# ----------------------------------------------------------------------


def _make_doc_with_rotary_layer(step: _TestStep, wp: WorkPiece):
    """Build a Doc whose layer is rotary-enabled with a workpiece in
    the unrolled domain; return ``(doc, layer)``."""
    geo_wp = Geometry()
    geo_wp.move_to(0, 0)
    geo_wp.line_to(1, 0)
    geo_wp.line_to(1, 1)
    geo_wp.line_to(0, 1)
    geo_wp.close_path()
    wp._boundaries_cache = geo_wp
    wp.matrix = Matrix.scale(50.0, 50.0)

    doc = _make_doc(step, wp)
    layer = doc.active_layer
    layer.rotary_enabled = True
    layer.rotary_diameter = 50.0
    return doc, layer


def _make_controller(monkeypatch, machine, doc):
    tm = ImmediateMainThreadTaskManager()
    ctrl = IntentController(doc, tm, machine=machine)
    ctrl.connect()
    monkeypatch.setattr(
        "rayforge.pipeline.intent_controller.run_intent",
        lambda *a, **kw: None,
    )
    return ctrl, tm


def _with_default_rotary(machine):
    module = RotaryModule()
    module.max_workpiece_length = 300.0
    machine.rotary_modules[module.uid] = module
    machine.default_rotary_module_uid = module.uid
    return module


def test_rotary_layer_key_maps_to_layer(monkeypatch, isolated_machine):
    """A ``stock:{layer_uid}`` key is mapped to its Layer."""
    _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc, layer = _make_doc_with_rotary_layer(step, wp)
    ctrl, tm = _make_controller(monkeypatch, isolated_machine, doc)
    wp.updated.send(wp)
    tm.fire_latest()

    sk = stock_key(layer.uid)
    assert sk in ctrl._key_to_item
    assert ctrl._key_to_item[sk] is layer
    ctrl.shutdown()


def test_material_state_ready_emitted_for_rotary_layer(
    monkeypatch, isolated_machine
):
    """A ``stock:{layer_uid}`` result emits ``material_state_ready``
    with the layer as the item."""
    _with_default_rotary(isolated_machine)
    step = _TestStep(name="s1")
    wp = WorkPiece(name="wp")
    doc, layer = _make_doc_with_rotary_layer(step, wp)
    ctrl, tm = _make_controller(monkeypatch, isolated_machine, doc)
    wp.updated.send(wp)
    tm.fire_latest()
    tm.main_thread_calls.clear()

    received: list = []
    ctrl.material_state_ready.connect(
        lambda _sender, *, item, output, generation_id: received.append(
            (item, output, generation_id)
        ),
        weak=False,
    )
    sk = stock_key(layer.uid)
    node = _StubNode(key=sk, generation_id=ctrl.generation_id, output="state")
    ctrl._on_completed(node)
    assert len(received) == 1
    assert received[0][0] is layer
    assert received[0][1] == "state"
    ctrl.shutdown()
