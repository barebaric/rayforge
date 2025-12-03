from __future__ import annotations
import logging
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from blinker import Signal
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.vectorization_spec import VectorizationSpec
from ..pipeline.pipeline import Pipeline
from ..machine.cmd import MachineCmd
from ..pipeline.artifact import JobArtifactHandle, JobArtifact
from .asset_cmd import AssetCmd
from .edit_cmd import EditCmd
from .split_cmd import SplitCmd
from .file_cmd import FileCmd
from .group_cmd import GroupCmd
from .layer_cmd import LayerCmd
from .layout_cmd import LayoutCmd
from .material_test_cmd import MaterialTestCmd
from .stock_cmd import StockCmd
from .step_cmd import StepCmd
from .tab_cmd import TabCmd
from .transform_cmd import TransformCmd

if TYPE_CHECKING:
    from ..undo import HistoryManager
    from ..shared.tasker.manager import TaskManager
    from ..context import RayforgeContext
    from ..core.workpiece import WorkPiece
    from ..core.tab import Tab


logger = logging.getLogger(__name__)


class DocEditor:
    """
    The central, non-UI controller for document state and operations.

    This class owns the core data models (Doc, Pipeline) and provides a
    structured API for all document manipulations, which are organized into
    namespaced command handlers. It is instantiated with its dependencies
    (task_manager, config_manager) to be a self-contained unit.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        context: "RayforgeContext",
        doc: Doc | None = None,
    ):
        """
        Initializes the DocEditor.

        Args:
            task_manager: The application's TaskManager instance.
            config_manager: The application's ConfigManager instance.
            doc: An optional existing Doc object. If None, a new one is
                 created.
        """
        self.context = context
        self.task_manager = task_manager
        self._config_manager = context.config_mgr
        self.doc = doc or Doc()
        self.pipeline = Pipeline(self.doc, self.task_manager)
        self.history_manager: "HistoryManager" = self.doc.history_manager

        # A set to track temporary artifacts (e.g., for job previews)
        # that don't live in the Pipeline cache.
        self._transient_artifact_handles: set[JobArtifactHandle] = set()

        # Track the number of active background tasks initiated by editor
        # commands
        self._busy_task_count: int = 0

        # Signals for monitoring document processing state
        self.processing_state_changed = Signal()
        self.document_settled = Signal()  # Fires when processing finishes
        self.notification_requested = Signal()  # For UI feedback
        self.pipeline.processing_state_changed.connect(
            self._on_processing_state_changed
        )

        # Instantiate and link command handlers, passing dependencies.
        self.asset = AssetCmd(self)
        self.edit = EditCmd(self)
        self.file = FileCmd(self, self.task_manager)
        self.group = GroupCmd(self, self.task_manager)
        self.layer = LayerCmd(self)
        self.layout = LayoutCmd(self, self.task_manager)
        self.material_test = MaterialTestCmd(self)
        self.split = SplitCmd(self)
        self.stock = StockCmd(self)
        self.step = StepCmd(self)
        self.tab = TabCmd(self)
        self.transform = TransformCmd(self)
        self.machine = MachineCmd(self)

    def cleanup(self):
        """
        Shuts down owned long-running services, like the Pipeline, to
        ensure cleanup of resources (e.g., shared memory).
        """
        # This is the safety net for any transient job artifacts that were
        # in-flight when the application was closed.
        logger.info(
            f"Releasing {len(self._transient_artifact_handles)} "
            "transient job artifacts..."
        )
        artifact_store = self.context.artifact_store
        for handle in list(self._transient_artifact_handles):
            artifact_store.release(handle)
        self._transient_artifact_handles.clear()

        self.pipeline.shutdown()

    def add_tab_from_context(self, context: Dict[str, Any]):
        """
        Public handler for the 'add_tab' action, using context from the UI.
        """
        workpiece: "WorkPiece" = context["workpiece"]
        location: Dict[str, Any] = context["location"]
        segment_index = location["segment_index"]
        pos = location["pos"]

        self.tab.add_single_tab(
            workpiece=workpiece, segment_index=segment_index, pos=pos
        )

    def remove_tab_from_context(self, context: Dict[str, Any]):
        """
        Public handler for the 'remove_tab' action, using context from the UI.
        """
        workpiece: "WorkPiece" = context["workpiece"]
        tab_to_remove: "Tab" = context["tab_data"]

        self.tab.remove_single_tab(
            workpiece=workpiece, tab_to_remove=tab_to_remove
        )

    @property
    def machine_dimensions(self) -> Optional[Tuple[float, float]]:
        """Returns the configured machine's dimensions, or None."""
        config = self.context.config
        if config and config.machine:
            return config.machine.dimensions
        return None

    @property
    def default_workpiece_layer(self) -> Layer:
        """
        Determines the most appropriate layer for adding new workpieces.
        """
        return self.doc.active_layer

    async def wait_until_settled(self, timeout: float = 10.0) -> None:
        """
        Waits until the internal Pipeline has finished all background
        processing and the document state is stable.
        """
        if not self.is_processing:
            return

        settled_future = asyncio.get_running_loop().create_future()

        # The signal sends `is_processing`, but the handler only needs
        # `sender`.
        def on_settled(sender, is_processing: bool):
            if not is_processing and not settled_future.done():
                settled_future.set_result(True)

        self.processing_state_changed.connect(on_settled)
        try:
            await asyncio.wait_for(settled_future, timeout)
        finally:
            self.processing_state_changed.disconnect(on_settled)

    async def import_file_from_path(
        self,
        filename: Path,
        mime_type: Optional[str],
        vectorization_spec: Optional[VectorizationSpec],
    ) -> None:
        """
        Imports a file from the specified path and waits for the operation
        to complete.
        """
        # Step 1: Run the importer
        payload = await self.file._load_file_async(
            filename, mime_type, vectorization_spec
        )
        if not payload or not payload.items:
            logger.warning(
                f"Test import of {filename.name} produced no items."
            )
            return

        # Step 2: Run the finalizer on the main thread.
        self.file._finalize_import_on_main_thread(
            payload, filename, position_mm=None
        )

    async def export_gcode_to_path(self, output_path: "Path") -> None:
        """
        Exports the current document to a G-code file at the specified path
        and waits for the operation to complete. This awaitable version is
        useful for tests.
        """
        export_future = asyncio.get_running_loop().create_future()
        artifact_store = self.context.artifact_store

        def _on_export_assembly_done(
            handle: Optional[JobArtifactHandle], error: Optional[Exception]
        ):
            try:
                if error:
                    export_future.set_exception(error)
                    return
                if not handle:
                    exc = ValueError("Assembly process returned no artifact.")
                    export_future.set_exception(exc)
                    return

                artifact = artifact_store.get(handle)
                assert isinstance(artifact, JobArtifact)
                if artifact.machine_code_bytes is None:
                    exc = ValueError("Final artifact is missing G-code data.")
                    export_future.set_exception(exc)
                    return

                gcode_str = artifact.machine_code_bytes.tobytes().decode(
                    "utf-8"
                )
                output_path.write_text(gcode_str, encoding="utf-8")

                logger.info(f"Test export successful to {output_path}")
                export_future.set_result(True)

            except Exception as e:
                if not export_future.done():
                    export_future.set_exception(e)
            finally:
                if handle:
                    artifact_store.release(handle)

        # Call the non-blocking method and provide our callback to bridge it
        self.file.assemble_job_in_background(
            when_done=_on_export_assembly_done
        )
        await export_future

    def set_doc(self, new_doc: Doc):
        """
        Assigns a new document to the editor, re-initializing the core
        components like the Pipeline.
        """
        self.pipeline.processing_state_changed.disconnect(
            self._on_processing_state_changed
        )

        logger.debug("DocEditor is setting a new document.")
        self.doc = new_doc
        self.history_manager = self.doc.history_manager
        # The Pipeline's setter handles cleanup and reconnection
        self.pipeline.doc = new_doc

        self.pipeline.processing_state_changed.connect(
            self._on_processing_state_changed
        )

    @property
    def is_processing(self) -> bool:
        """Returns True if the document is currently generating operations."""
        # The editor is busy if the pipeline is active OR if there are
        # outstanding background tasks (like grouping calculations)
        # running.
        return self.pipeline.is_busy or self._busy_task_count > 0

    def notify_task_started(self):
        """
        Notifies the editor that a background task (e.g. calculation) has
        started.
        This prevents wait_until_settled from returning prematurely.
        """
        was_processing = self.is_processing
        self._busy_task_count += 1

        # If we transitioned from idle to busy, emit the signal.
        if not was_processing:
            self.processing_state_changed.send(self, is_processing=True)

    def notify_task_ended(self):
        """
        Notifies the editor that a background task has ended.
        """
        if self._busy_task_count > 0:
            self._busy_task_count -= 1

        # If we transitioned from busy to idle, emit the signals.
        if not self.is_processing:
            self.processing_state_changed.send(self, is_processing=False)
            self.document_settled.send(self)

    def _on_processing_state_changed(self, sender, is_processing: bool):
        """Proxies the signal from the Pipeline."""
        # Use the effective state (pipeline + tasks) rather than just
        # pipeline state
        effective_state = self.is_processing
        self.processing_state_changed.send(self, is_processing=effective_state)
        if not effective_state:
            self.document_settled.send(self)
