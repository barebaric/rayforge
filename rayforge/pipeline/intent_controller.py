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
4. When :attr:`dispatch` is ``True`` the new intent is also executed via
   :func:`run_intent`; the ``on_completed`` callback performs the
   epoch filter (discarding results whose ``generation_id`` is older
   than the controller's current generation) and then marshals a DOM
   reattachment back to the application main thread via the shared
   task manager.

The controller is constructed by application bootstrapping (see
``app.py``) and lives alongside the legacy multiprocessing path.  The
legacy path remains authoritative while ``dispatch`` is ``False``; in
that mode the controller only maintains the cache-invalidation diff.
Setting ``dispatch`` to ``True`` also drives raygeo :func:`run_intent`
so the new pipeline runs end-to-end.
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

from raygeo.cnc.execution.intent import (
    Intent,
    create_intent_from_nodes,
    run_intent,
)
from raygeo.pipeline.execute import Pipeline as RaygeoPipeline
from raygeo.pipeline.request import NodeRequest

from .intent_builder import IntentBuilder

if TYPE_CHECKING:
    from ..core.doc import Doc
    from ..core.item import DocItem

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


class IntentController:
    """
    Owns a raygeo :class:`Intent` and the surrounding rebuild lifecycle.

    Pairs with the existing :class:`~rayforge.pipeline.pipeline.Pipeline`
    instance; it consumes the same signals but generates a parallel,
    cache-aware Intent that the future pipeline cutover will use.
    """

    def __init__(
        self,
        doc: "Doc",
        task_manager: "_DelayedScheduler",
        raygeo_pipeline: Optional[RaygeoPipeline] = None,
        dispatch: bool = False,
    ):
        self._doc = doc
        self._task_manager = task_manager
        self._raygeo_pipeline: RaygeoPipeline = (
            raygeo_pipeline or RaygeoPipeline()
        )
        self._dispatch: bool = dispatch
        self._intent: Optional[Intent] = None
        self._generation_id: int = 0
        self._rebuild_timer: Optional[Any] = None
        # Flat map from node key back to the originating :class:`DocItem`
        # for DOM reattachment.  Rebuilt on every successful
        # ``IntentBuilder.build`` call.
        self._key_to_item: Dict[str, DocItem] = {}

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def dispatch(self) -> bool:
        """Whether to execute the intent after each rebuild."""
        return self._dispatch

    @dispatch.setter
    def dispatch(self, value: bool) -> None:
        self._dispatch = bool(value)

    @property
    def raygeo_pipeline(self) -> RaygeoPipeline:
        return self._raygeo_pipeline

    @property
    def intent(self) -> Optional[Intent]:
        return self._intent

    @property
    def generation_id(self) -> int:
        return self._generation_id

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to the document's bubbled signals."""
        doc = self._doc
        doc.descendant_updated.connect(self._on_doc_changed)
        doc.descendant_transform_changed.connect(self._on_doc_changed)
        doc.descendant_added.connect(self._on_doc_changed)
        doc.descendant_removed.connect(self._on_doc_changed)
        doc.job_assembly_invalidated.connect(self._on_doc_changed)

    def disconnect(self) -> None:
        """Disconnect from the document's signals."""
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
        self._schedule_rebuild()

    def _schedule_rebuild(self) -> None:
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
        self._rebuild_timer = (
            self._task_manager.schedule_delayed_on_main_thread(
                REBUILD_DEBOUNCE_MS,
                self._rebuild,
            )
        )

    def _rebuild(self) -> None:
        """Build a fresh intent from the doc and update the cache."""
        self._rebuild_timer = None
        self._generation_id += 1
        builder = IntentBuilder(generation_id=self._generation_id)
        nodes = builder.build(self._doc)
        self._refresh_key_to_item_map(nodes)
        new_intent = create_intent_from_nodes(nodes)
        if self._intent is None:
            self._intent = new_intent
        else:
            self._intent.update(new_intent, pipeline=self._raygeo_pipeline)
        if self._dispatch:
            run_intent(
                self._intent,
                on_completed=self._on_completed,
                pipeline=self._raygeo_pipeline,
            )

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
        gen = getattr(node, "generation_id", -1)
        if gen < self._generation_id:
            logger.debug(
                "Discarding superseded result for %s (gen %s < %s)",
                getattr(node, "key", "?"),
                gen,
                self._generation_id,
            )
            return
        key = getattr(node, "key", "")
        item = self._key_to_item.get(key)
        if item is None:
            logger.debug(
                "No DocItem mapped for completed node %s; skipping",
                key,
            )
            return
        output = getattr(node, "output", None)
        self._task_manager.schedule_on_main_thread(
            self._reattach, key, item, output
        )

    def _reattach(self, key: str, item: "DocItem", output: Any) -> None:
        """
        Reattach a completed node's output onto the owning DocItem.

        Runs on the GTK main loop.  The default implementation is a
        no-op stub; concrete reattachment is supplied by callers or
        subclasses once the new pipeline becomes authoritative.
        """
        logger.debug(
            "Reattaching output for %s onto %s", key, type(item).__name__
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
        # Index workpieces and steps by uid for fast lookup.
        workpieces: Dict[str, WorkPiece] = {}
        steps: Dict[str, Step] = {}
        for layer in self._doc.layers:
            for wp in layer.all_workpieces:
                workpieces[wp.uid] = wp
            if layer.workflow:
                for step in layer.workflow.steps:
                    steps[step.uid] = step

        for n in nodes:
            key = n.key
            # ``workpiece:{wp_uid}:{step_uid}``
            if key.startswith("workpiece:"):
                _, wp_uid, _step_uid = key.split(":")
                wp = workpieces.get(wp_uid)
                if wp is not None:
                    self._key_to_item[key] = wp
            # ``step:{step_uid}``
            elif key.startswith("step:"):
                _, s_uid = key.split(":")
                step = steps.get(s_uid)
                if step is not None:
                    self._key_to_item[key] = step
            # ``job``
            elif key == "job":
                self._key_to_item[key] = self._doc

    def shutdown(self) -> None:
        """Cancel any pending rebuild timer and disconnect signals."""
        if self._rebuild_timer is not None:
            self._rebuild_timer.cancel()
            self._rebuild_timer = None
        try:
            self.disconnect()
        except Exception:
            pass
