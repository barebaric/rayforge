from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Generator,
    Optional,
    Tuple,
)

from blinker import Signal

from ..core.doc import Doc
from ..core.workpiece import WorkPiece
from .artifact import (
    BaseArtifactHandle,
    JobArtifact,
    StepOpsArtifact,
    WorkPieceArtifact,
)
from .artifact.store import ArtifactStore
from .coord import CoordinateSystem
from .encoder.base import EncodedOutput, MachineCodeOpMap
from .intent_controller import IntentController

if TYPE_CHECKING:
    from ..core.step import Step
    from ..machine.models.machine import Machine
    from ..shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


class Pipeline:
    """
    Public facade over the raygeo-backed intent pipeline.

    Owns the :class:`ArtifactStore` integration: translates the raw
    raygeo outputs emitted by its internal :class:`IntentController`
    into refcounted artifact handles that the UI and export paths
    consume, and exposes the signal/property surface the rest of the
    application expects.

    Consumers (:class:`~rayforge.doceditor.editor.DocEditor`,
    :class:`ViewManager`, UI widgets, test code) should depend on this
    class only.  :class:`IntentController` and
    :class:`~rayforge.pipeline.intent_builder.IntentBuilder` are
    implementation details of the facade and may change without
    notice.
    """

    def __init__(
        self,
        doc: Optional[Doc],
        task_manager: "TaskManager",
        artifact_store: ArtifactStore,
        machine: Optional["Machine"],
    ):
        if machine is None:
            raise RuntimeError("Machine is not configured in context")

        self._doc: Optional[Doc] = doc
        self._task_manager = task_manager
        self._store = artifact_store
        self._machine = machine
        self._is_shutting_down = False
        self._last_known_busy = False

        self._wp_handles: Dict[Tuple[str, str], BaseArtifactHandle] = {}
        self._step_handles: Dict[str, BaseArtifactHandle] = {}
        self._last_aggregate_output: Any = None
        self._last_job_handle: Optional[BaseArtifactHandle] = None

        self.processing_state_changed = Signal()
        self.workpiece_starting = Signal()
        self.workpiece_artifact_ready = Signal()
        self.workpiece_artifact_adopted = Signal()
        self.step_assembly_starting = Signal()
        self.job_generation_finished = Signal()
        self.job_time_updated = Signal()
        self.visual_chunk_available = Signal()
        self.data_stale = Signal()

        self._intent_ctl = IntentController(
            doc=doc,
            task_manager=task_manager,
            machine=machine,
            dispatch=True,
        )
        self._connect_ctl_signals()

        if doc:
            self._intent_ctl.connect()
            if self._doc and self._has_workflow_content():
                self._intent_ctl._schedule_rebuild()

    def _has_workflow_content(self) -> bool:
        if not self._doc:
            return False
        for layer in self._doc.layers:
            if layer.workflow and layer.workflow.steps:
                if layer.all_workpieces:
                    return True
        return False

    def _connect_ctl_signals(self) -> None:
        ctl = self._intent_ctl
        ctl.workpiece_artifact_ready.connect(self._on_wp_output)
        ctl.step_artifact_ready.connect(self._on_step_output)
        ctl.job_aggregate_ready.connect(self._on_job_aggregate)
        ctl.job_generation_finished.connect(self._on_job_encoded)
        ctl.job_time_updated.connect(self._job_time_relay)
        ctl.rebuild_started.connect(self._on_rebuild_started)
        ctl.rebuild_finished.connect(self._on_rebuild_finished)
        ctl.data_stale.connect(self._on_data_stale)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def doc(self) -> Optional[Doc]:
        return self._doc

    @doc.setter
    def doc(self, new_doc: Optional[Doc]) -> None:
        if self._doc is new_doc:
            return
        self._intent_ctl.shutdown()
        self._doc = new_doc
        self._wp_handles.clear()
        self._step_handles.clear()
        self._last_job_handle = None
        self._last_aggregate_output = None
        self._intent_ctl = IntentController(
            doc=new_doc,
            task_manager=self._task_manager,
            machine=self._machine,
            dispatch=True,
        )
        self._connect_ctl_signals()
        if new_doc:
            self._intent_ctl.connect()
            if self._has_workflow_content():
                self._intent_ctl._schedule_rebuild()

    @property
    def machine(self) -> Optional["Machine"]:
        return self._machine

    @property
    def task_manager(self) -> "TaskManager":
        return self._task_manager

    @property
    def artifact_store(self) -> ArtifactStore:
        return self._store

    @property
    def data_generation_id(self) -> int:
        return self._intent_ctl.generation_id

    @property
    def last_completed_handle(self) -> Optional[BaseArtifactHandle]:
        return self._last_job_handle

    @property
    def auto_pipeline(self) -> bool:
        return self._intent_ctl.auto_rebuild

    @auto_pipeline.setter
    def auto_pipeline(self, value: bool) -> None:
        self._intent_ctl.auto_rebuild = value

    @property
    def is_paused(self) -> bool:
        return self._intent_ctl.is_paused

    @property
    def is_data_stale(self) -> bool:
        return self._intent_ctl.is_data_stale

    @property
    def is_busy(self) -> bool:
        return (
            self._intent_ctl.is_rebuild_pending
            or self._task_manager.has_tasks()
        )

    # ------------------------------------------------------------------
    # Pause / resume
    # ------------------------------------------------------------------

    def pause(self) -> None:
        self._intent_ctl.pause()

    def resume(self) -> None:
        self._intent_ctl.resume()

    @contextmanager
    def paused(self) -> Generator[None, None, None]:
        self.pause()
        try:
            yield
        finally:
            self.resume()

    # ------------------------------------------------------------------
    # Machine
    # ------------------------------------------------------------------

    def set_machine(self, machine: "Machine") -> None:
        if self._machine is machine:
            return
        self._machine = machine
        self._intent_ctl.shutdown()
        self._intent_ctl = IntentController(
            doc=self._doc,
            task_manager=self._task_manager,
            machine=machine,
            dispatch=True,
        )
        self._connect_ctl_signals()
        if self._doc:
            self._intent_ctl.connect()
            if self._has_workflow_content():
                self._intent_ctl._schedule_rebuild()

    # ------------------------------------------------------------------
    # Recalculate
    # ------------------------------------------------------------------

    def recalculate(self, force: bool = False) -> None:
        self._intent_ctl.force_rebuild()

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def shutdown(self) -> None:
        self._is_shutting_down = True
        self._intent_ctl.shutdown()
        self._wp_handles.clear()
        self._step_handles.clear()
        self._last_job_handle = None
        self._last_aggregate_output = None

    # ------------------------------------------------------------------
    # IntentController signal handlers
    # ------------------------------------------------------------------

    def _on_rebuild_started(self, sender) -> None:
        self._set_busy(True)

    def _on_rebuild_finished(self, sender) -> None:
        self._task_manager.schedule_on_main_thread(self._check_busy)

    def _on_data_stale(self, sender) -> None:
        self.data_stale.send(self)

    def _check_busy(self) -> None:
        self._set_busy(self.is_busy)

    def _set_busy(self, busy: bool) -> None:
        if self._last_known_busy != busy:
            self._last_known_busy = busy
            self.processing_state_changed.send(self, is_processing=busy)

    def _on_wp_output(
        self, sender, *, step, workpiece, output, generation_id
    ) -> None:
        if self._is_shutting_down or output is None:
            return
        coord_sys = (
            CoordinateSystem.MILLIMETER_SPACE
            if output.is_scalable
            else CoordinateSystem.PIXEL_SPACE
        )
        source_dims = output.source_dimensions
        gen_size = workpiece.size if workpiece else (0.0, 0.0)
        artifact = WorkPieceArtifact(
            ops=output.ops,
            is_scalable=output.is_scalable,
            source_coordinate_system=coord_sys,
            generation_size=gen_size,
            generation_id=generation_id,
            source_dimensions=source_dims,
        )
        old = self._wp_handles.pop((workpiece.uid, step.uid), None)
        if old is not None:
            self._store.release(old)
        handle = self._store.put(artifact, "wp")
        self._wp_handles[(workpiece.uid, step.uid)] = handle
        self.workpiece_artifact_ready.send(
            self,
            step=step,
            workpiece=workpiece,
            handle=handle,
            generation_id=generation_id,
        )

    def _on_step_output(self, sender, *, step, output, generation_id) -> None:
        if self._is_shutting_down or output is None:
            return
        artifact = StepOpsArtifact(
            ops=output.ops,
            generation_id=generation_id,
        )
        old = self._step_handles.pop(step.uid, None)
        if old is not None:
            self._store.release(old)
        handle = self._store.put(artifact, "step")
        self._step_handles[step.uid] = handle

    def _on_job_aggregate(self, sender, *, output, generation_id) -> None:
        if self._is_shutting_down:
            return
        if output is not None:
            self._last_aggregate_output = output
            time_est = output.time_estimate
            self.job_time_updated.send(self, total_seconds=time_est)

    def _on_job_encoded(self, sender, *, handle, task_status) -> None:
        if self._is_shutting_down:
            return
        agg = self._last_aggregate_output
        if agg is None:
            logger.debug("Encode finished but no aggregate output cached")
            return

        text = handle.text or ""
        op_to_mc = handle.op_to_machine_code or {}
        mc_to_op = handle.machine_code_to_op or {}
        encoded = EncodedOutput(
            text=text,
            op_map=MachineCodeOpMap(
                op_to_machine_code=dict(op_to_mc),
                machine_code_to_op=dict(mc_to_op),
            ),
        )

        ops = agg.ops
        distance = ops.distance() if ops else 0.0
        artifact = JobArtifact(
            ops=ops,
            distance=distance,
            generation_id=self._intent_ctl.generation_id,
            time_estimate=agg.time_estimate,
            encoded_output=encoded,
        )
        if self._last_job_handle is not None:
            self._store.release(self._last_job_handle)
        job_handle = self._store.put(artifact, "job")
        self._last_job_handle = job_handle
        self.job_generation_finished.send(
            self, handle=job_handle, task_status=task_status
        )

    def _job_time_relay(self, sender, *, total_seconds) -> None:
        self.job_time_updated.send(self, total_seconds=total_seconds)

    # ------------------------------------------------------------------
    # Public API for artifact access
    # ------------------------------------------------------------------

    def get_artifact_handle(
        self, step_uid: str, workpiece_uid: str
    ) -> Optional[BaseArtifactHandle]:
        return self._wp_handles.get((workpiece_uid, step_uid))

    def get_step_ops_artifact_handle(
        self, step_uid: str
    ) -> Optional[BaseArtifactHandle]:
        return self._step_handles.get(step_uid)

    def get_artifact(self, step: "Step", workpiece: "WorkPiece") -> Any:
        handle = self._wp_handles.get((workpiece.uid, step.uid))
        if handle is None:
            return None
        return self._store.get(handle)

    def get_existing_job_handle(self) -> Optional[BaseArtifactHandle]:
        return self._last_job_handle

    # ------------------------------------------------------------------
    # Job generation
    # ------------------------------------------------------------------

    def generate_job(self) -> None:
        def no_op(handle, error):
            if error:
                logger.error(f"Fire-and-forget job generation failed: {error}")

        self.generate_job_artifact(when_done=no_op)

    def generate_job_artifact(
        self,
        when_done: Callable[
            [Optional[BaseArtifactHandle], Optional[Exception]], None
        ],
    ):
        if not self._doc:
            when_done(None, RuntimeError("No document is loaded."))
            return

        if self._last_job_handle is not None:
            when_done(self._last_job_handle, None)
            return

        def _on_finished(sender, *, handle, task_status):
            self.job_generation_finished.disconnect(_on_finished)
            when_done(handle, None)

        self.job_generation_finished.connect(_on_finished)
        self._intent_ctl.force_rebuild()

    async def generate_job_artifact_async(
        self,
    ) -> Optional[BaseArtifactHandle]:
        future = asyncio.get_running_loop().create_future()

        def _when_done(handle, error):
            if not future.done():
                if error:
                    future.set_exception(error)
                else:
                    future.set_result(handle)

        self.generate_job_artifact(when_done=_when_done)
        return await future
