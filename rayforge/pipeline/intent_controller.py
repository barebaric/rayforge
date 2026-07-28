"""
Intent controller for the raygeo-backed pipeline.

The :class:`IntentController` listens to the same Doc signals that
:class:`~rayforge.pipeline.pipeline.Pipeline` already listens to
(``descendant_updated``, ``descendant_transform_changed``,
``descendant_added``, ``descendant_removed``, ``job_assembly_invalidated``)
and rebuilds a raygeo :class:`Intent` whenever the document changes.

On each debounced rebuild:

1. :class:`~rayforge.pipeline.intent_builder.IntentBuilder` is called
   to produce a fresh list of :class:`NodeRequest` objects from the
   current :class:`Doc`.
2. The new list is wrapped into a raygeo :class:`Intent` via
   :func:`create_intent_from_nodes`.
3. :meth:`Intent.update` diffs the previous intent against the new one
   using the ``version_token`` values and evicts any stale cache entries
   on the shared :class:`~raygeo.pipeline.execute.Pipeline`.
4. The new intent is executed via :func:`run_intent`; the
   ``on_completed`` callback performs the epoch filter (discarding
   results whose ``generation_id`` is older than the controller's
   current generation) and then marshals a DOM reattachment back to the
   application main thread via the shared task manager.
"""

from __future__ import annotations

import logging
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Protocol,
    runtime_checkable,
)

from blinker import Signal
from raygeo.cnc.execution.intent import (
    Intent,
    create_intent_from_nodes,
    run_intent,
)
from raygeo.pipeline.execute import Pipeline as RaygeoPipeline
from raygeo.pipeline.request import NodeRequest

from .intent_builder import IntentBuilder, parse_workpiece_key

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..core.item import DocItem
    from ..core.step import Step
    from ..core.workpiece import WorkPiece
    from ..machine.models.machine import Machine

logger = logging.getLogger(__name__)


# Debounce window for signal-driven intent rebuilds (milliseconds).
REBUILD_DEBOUNCE_MS = 200


@runtime_checkable
class _DelayedScheduler(Protocol):
    """The subset of :class:`TaskManager` the controller depends on.

    Decoupling from the concrete :class:`TaskManager` lets tests supply
    a minimal fake without needing the asyncio loop / worker pool the
    real one wires up.
    """

    def schedule_delayed_on_main_thread(
        self,
        delay_ms: int,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def schedule_on_main_thread(
        self,
        callback: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any: ...

    def run_thread(
        self,
        func: Callable[..., Any],
        *args: Any,
        key: Optional[Any] = None,
        when_done: Optional[Callable[[Any], None]] = None,
        **kwargs: Any,
    ) -> Any: ...


class IntentController:
    """
    Owns a raygeo :class:`Intent` and the surrounding rebuild lifecycle.

    Pairs with the existing :class:`~rayforge.pipeline.pipeline.Pipeline`
    instance; it consumes the same signals but generates a parallel,
    cache-aware Intent that the future pipeline cutover will use.
    """

    def __init__(
        self,
        doc: "Optional[Doc]",
        task_manager: "_DelayedScheduler",
        machine: "Optional[Machine]" = None,
        raygeo_pipeline: Optional[RaygeoPipeline] = None,
    ):
        self._doc: Optional[Doc] = doc
        self._task_manager = task_manager
        self._machine = machine
        self._raygeo_pipeline: RaygeoPipeline = (
            raygeo_pipeline or RaygeoPipeline()
        )
        self._intent: Optional[Intent] = None
        self._generation_id: int = 0
        self._rebuild_timer: Optional[Any] = None
        self._rebuilding: bool = False
        self._rebuild_pending: bool = False
        self._pause_count: int = 0
        self._auto_rebuild: bool = True
        self._data_stale_flag: bool = False
        # Flat map from node key back to the originating :class:`DocItem`
        # for DOM reattachment.  Rebuilt on every successful
        # ``IntentBuilder.build`` call.
        self._key_to_item: Dict[str, DocItem] = {}
        self._workpieces_by_uid: Dict[str, "WorkPiece"] = {}
        self._steps_by_uid: Dict[str, "Step"] = {}

        # Signals for notifying the UI of generation progress.
        self.workpiece_artifact_ready = Signal()
        self.step_artifact_ready = Signal()
        self.job_aggregate_ready = Signal()
        self.job_generation_finished = Signal()
        self.job_time_updated = Signal()
        self.progress_changed = Signal()
        self.rebuild_started = Signal()
        self.rebuild_finished = Signal()
        self.data_stale = Signal()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def raygeo_pipeline(self) -> RaygeoPipeline:
        return self._raygeo_pipeline

    @property
    def intent(self) -> Optional[Intent]:
        return self._intent

    @property
    def generation_id(self) -> int:
        return self._generation_id

    @property
    def is_paused(self) -> bool:
        return self._pause_count > 0

    @property
    def is_rebuild_pending(self) -> bool:
        return self._rebuild_timer is not None or self._rebuilding

    @property
    def is_data_stale(self) -> bool:
        return self._data_stale_flag

    @property
    def auto_rebuild(self) -> bool:
        return self._auto_rebuild

    @auto_rebuild.setter
    def auto_rebuild(self, value: bool) -> None:
        if self._auto_rebuild == value:
            return
        self._auto_rebuild = value
        if value and self._data_stale_flag:
            self._data_stale_flag = False
            self._schedule_rebuild()

    def pause(self) -> None:
        self._pause_count += 1

    def resume(self) -> None:
        if self._pause_count == 0:
            return
        self._pause_count -= 1
        if self._pause_count == 0 and self._data_stale_flag:
            self._data_stale_flag = False
            if self._auto_rebuild:
                self._schedule_rebuild()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the document's bubbled signals."""
        if self._doc is None:
            return
        doc = self._doc
        doc.descendant_updated.connect(self._on_doc_changed)
        doc.descendant_transform_changed.connect(self._on_doc_changed)
        doc.descendant_added.connect(self._on_doc_changed)
        doc.descendant_removed.connect(self._on_doc_changed)
        doc.job_assembly_invalidated.connect(self._on_doc_changed)

    def disconnect(self) -> None:
        """Disconnect from the document's signals."""
        if self._doc is None:
            return
        doc = self._doc
        doc.descendant_updated.disconnect(self._on_doc_changed)
        doc.descendant_transform_changed.disconnect(self._on_doc_changed)
        doc.descendant_added.disconnect(self._on_doc_changed)
        doc.descendant_removed.disconnect(self._on_doc_changed)
        doc.job_assembly_invalidated.disconnect(self._on_doc_changed)

    # ------------------------------------------------------------------
    # Signal handling (debounced)
    # ------------------------------------------------------------------

    def _on_doc_changed(self, *args: Any, **kwargs: Any) -> None:
        """Trigger a debounced intent rebuild on any doc change."""
        if self._pause_count > 0 or not self._auto_rebuild:
            if not self._data_stale_flag:
                self._data_stale_flag = True
                self.data_stale.send(self)
            return
        self._schedule_rebuild()

    def set_doc(self, doc: "Optional[Doc]") -> None:
        """Replace the document and trigger a rebuild.

        Preserves the existing :class:`RaygeoPipeline` and
        :class:`~raygeo.Intent` so cache entries survive the doc
        swap.
        """
        self.disconnect()
        self._doc = doc
        if doc is not None:
            self.connect()
        self.force_rebuild()

    def set_machine(self, machine: "Optional[Machine]") -> None:
        """Replace the machine and trigger a rebuild.

        Preserves the existing :class:`RaygeoPipeline` and
        :class:`~raygeo.Intent` so cache entries survive the machine
        swap.
        """
        self._machine = machine
        self.force_rebuild()

    def force_rebuild(self) -> None:
        """Cancel any pending debounce and rebuild immediately.

        If a rebuild is already running on the background thread, the
        in-flight run is signalled to cancel so the next generation
        starts as soon as possible.
        """
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
            self._rebuild_timer = None
        if self._rebuilding:
            self._rebuild_pending = True
            if self._intent is not None:
                self._intent.cancel()
            return
        self._rebuild()

    def _schedule_rebuild(self) -> None:
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
        if self._rebuilding:
            self._rebuild_pending = True
            if self._intent is not None:
                self._intent.cancel()
            return
        self._rebuild_timer = (
            self._task_manager.schedule_delayed_on_main_thread(
                REBUILD_DEBOUNCE_MS,
                self._rebuild,
            )
        )

    def _rebuild(self) -> None:
        """Build a fresh intent from the doc and execute it.

        The heavy work (intent construction including raster rendering,
        and pipeline execution) runs on a background thread via the
        task manager so the GTK main loop stays responsive.
        ``rebuild_started`` fires before the thread starts;
        ``rebuild_finished`` fires on the main thread after the thread
        completes.
        """
        self._rebuild_timer = None
        self._generation_id += 1
        self._rebuilding = True
        gen = self._generation_id
        self.rebuild_started.send(self)

        def _worker() -> None:
            if self._doc is None:
                return
            builder = IntentBuilder(machine=self._machine, generation_id=gen)
            nodes = builder.build(self._doc)
            self._refresh_key_to_item_map(nodes)
            new_intent = create_intent_from_nodes(nodes)
            if self._intent is None:
                self._intent = new_intent
            else:
                self._intent.update(new_intent, pipeline=self._raygeo_pipeline)
            if nodes:
                try:
                    run_intent(
                        self._intent,
                        on_completed=self._on_completed,
                        on_batch_progress=self._on_batch_progress,
                        pipeline=self._raygeo_pipeline,
                    )
                except RuntimeError as exc:
                    logger.debug("run_intent failed: %s", exc)

        def _on_done(_task: Any) -> None:
            self._rebuilding = False
            if self._rebuild_pending:
                self._rebuild_pending = False
                self._task_manager.schedule_on_main_thread(self._rebuild)
            else:
                self._task_manager.schedule_on_main_thread(
                    self._emit_rebuild_finished
                )

        self._task_manager.run_thread(
            _worker, when_done=_on_done, key="intent-rebuild"
        )

    def _emit_rebuild_finished(self) -> None:
        """Emit ``rebuild_finished`` on the main thread."""
        self.rebuild_finished.send(self)

    # ------------------------------------------------------------------
    # on_completed → epoch filter → DOM reattachment via main-thread
    # schedule
    # ------------------------------------------------------------------

    def _on_completed(self, node: Any) -> None:
        """
        raygeo ``on_completed`` callback.

        Invoked on a rayon worker thread with the GIL held.  We check
        the node's ``generation_id`` against the controller's current
        generation (epoch filter) and, if still current, schedule a
        DOM reattachment onto the application main thread via the
        shared task manager.
        """
        gen = node.generation_id
        if gen < self._generation_id:
            logger.debug(
                "Discarding superseded result for %s (gen %s < %s)",
                node.key,
                gen,
                self._generation_id,
            )
            return
        key = node.key
        item = self._key_to_item.get(key)
        if item is None:
            logger.debug(
                "No DocItem mapped for completed node %s; skipping",
                key,
            )
            return
        output = node.output
        self._task_manager.schedule_on_main_thread(
            self._reattach, key, item, output
        )

    def _on_batch_progress(self, fraction: float, message: str) -> None:
        """raygeo ``on_batch_progress`` callback.

        Invoked on a rayon worker thread with the GIL held.  Relays
        the aggregate progress fraction and status message to
        listeners via :attr:`progress_changed` (marshalled onto the
        application main thread so signal handlers never run on a
        worker).
        """
        self._task_manager.schedule_on_main_thread(
            self._emit_progress, fraction, message
        )

    def _emit_progress(self, fraction: float, message: str) -> None:
        """Emit :attr:`progress_changed` on the main thread."""
        self.progress_changed.send(self, fraction=fraction, message=message)

    def _reattach(self, key: str, item: "DocItem", output: Any) -> None:
        """
        Reattach a completed node's output onto the owning DocItem and
        emit the corresponding signal so the UI can update.

        Runs on the application main thread.  Dispatches on the node
        key shape:

        * ``workpiece:{wp_uid}:{step_uid}`` →
          :attr:`workpiece_artifact_ready`
        * ``step:{step_uid}`` → :attr:`step_artifact_ready`
        * ``job`` → :attr:`job_aggregate_ready`
        * ``job:encode`` → :attr:`job_generation_finished` (and
          :attr:`job_time_updated` when a time estimate is available)
        """
        gen = self._generation_id
        if key.startswith("workpiece:"):
            parsed = parse_workpiece_key(key)
            if parsed is None:
                logger.warning("Malformed workpiece key: %s", key)
                return
            wp_uid, step_uid = parsed
            workpiece = self._find_workpiece(wp_uid)
            step = self._find_step(step_uid)
            if workpiece is not None and step is not None:
                self.workpiece_artifact_ready.send(
                    self,
                    step=step,
                    workpiece=workpiece,
                    output=output,
                    generation_id=gen,
                )
        elif key.startswith("step:"):
            step = self._find_step(key.split(":", 1)[1])
            if step is not None:
                self.step_artifact_ready.send(
                    self, step=step, output=output, generation_id=gen
                )
        elif key == "job":
            self.job_aggregate_ready.send(
                self, output=output, generation_id=gen
            )
            time_estimate = (
                output.time_estimate if output is not None else None
            )
            self.job_time_updated.send(self, total_seconds=time_estimate)
        elif key == "job:encode":
            self.job_generation_finished.send(
                self, handle=output, task_status="completed"
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _refresh_key_to_item_map(self, nodes: List[NodeRequest]) -> None:
        """
        Build a flat ``key -> DocItem`` map from the freshly built
        ``NodeRequest`` list so the ``on_completed`` epoch-filtered
        callback can reattach outputs onto the originating WorkPiece or
        Step without needing to re-walk the Doc.
        """
        from ..core.step import Step
        from ..core.workpiece import WorkPiece

        self._key_to_item = {}
        if self._doc is None:
            return
        # Index workpieces and steps by uid for fast lookup.  Kept on
        # the instance so :meth:`_reattach` can resolve the owning
        # DocItem for a node key without re-walking the doc.
        workpieces: Dict[str, WorkPiece] = {}
        steps: Dict[str, Step] = {}
        for layer in self._doc.layers:
            for wp in layer.all_workpieces:
                workpieces[wp.uid] = wp
            if layer.workflow:
                for step in layer.workflow.steps:
                    steps[step.uid] = step
        self._workpieces_by_uid = workpieces
        self._steps_by_uid = steps

        for n in nodes:
            key = n.key
            # ``workpiece:{wp_uid}:{step_uid}``
            if key.startswith("workpiece:"):
                parsed = parse_workpiece_key(key)
                if parsed is None:
                    raise ValueError(
                        f"Malformed workpiece key in node key map: {key!r}"
                    )
                wp_uid, _step_uid = parsed
                wp = workpieces.get(wp_uid)
                if wp is not None:
                    self._key_to_item[key] = wp
            # ``step:{step_uid}``
            elif key.startswith("step:"):
                _, s_uid = key.split(":")
                step = steps.get(s_uid)
                if step is not None:
                    self._key_to_item[key] = step
            # ``job`` or ``job:encode``
            elif key == "job" or key == "job:encode":
                self._key_to_item[key] = self._doc

    def _find_workpiece(self, uid: str) -> "Optional[WorkPiece]":
        return self._workpieces_by_uid.get(uid)

    def _find_step(self, uid: str) -> "Optional[Step]":
        return self._steps_by_uid.get(uid)

    def shutdown(self) -> None:
        """Cancel any pending rebuild timer and disconnect signals."""
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
            self._rebuild_timer = None
        try:
            self.disconnect()
        except Exception:
            logger.warning(
                "Error during IntentController shutdown",
                exc_info=True,
            )
