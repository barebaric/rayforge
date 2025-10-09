from __future__ import annotations
import logging
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from blinker import Signal
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.vectorization_config import TraceConfig
from ..pipeline.generator import OpsGenerator
from ..machine.cmd import MachineCmd
from ..pipeline.artifact.handle import ArtifactHandle
from ..pipeline.artifact.store import ArtifactStore
from .edit_cmd import EditCmd
from .file_cmd import FileCmd
from .group_cmd import GroupCmd
from .layer_cmd import LayerCmd
from .layout_cmd import LayoutCmd
from .transform_cmd import TransformCmd
from .stock_cmd import StockCmd
from .step_cmd import StepCmd
from .tab_cmd import TabCmd

if TYPE_CHECKING:
    from ..undo import HistoryManager
    from ..shared.tasker.manager import TaskManager
    from ..config import ConfigManager
    from ..core.workpiece import WorkPiece
    from ..core.tab import Tab


logger = logging.getLogger(__name__)


class DocEditor:
    """
    The central, non-UI controller for document state and operations.

    This class owns the core data models (Doc, OpsGenerator) and provides a
    structured API for all document manipulations, which are organized into
    namespaced command handlers. It is instantiated with its dependencies
    (task_manager, config_manager) to be a self-contained unit.
    """

    def __init__(
        self,
        task_manager: "TaskManager",
        config_manager: "ConfigManager",
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
        self.task_manager = task_manager
        self._config_manager = config_manager
        self.doc = doc or Doc()
        self.ops_generator = OpsGenerator(self.doc, self.task_manager)
        self.history_manager: "HistoryManager" = self.doc.history_manager

        # A set to track temporary artifacts (e.g., for job previews)
        # that don't live in the OpsGenerator cache.
        self._transient_artifact_handles: set[ArtifactHandle] = set()

        # Signals for monitoring document processing state
        self.processing_state_changed = Signal()
        self.document_settled = Signal()  # Fires when processing finishes
        self.notification_requested = Signal()  # For UI feedback
        self.ops_generator.processing_state_changed.connect(
            self._on_processing_state_changed
        )

        # Instantiate and link command handlers, passing dependencies.
        self.edit = EditCmd(self)
        self.file = FileCmd(self, self.task_manager, self._config_manager)
        self.group = GroupCmd(self, self.task_manager)
        self.layer = LayerCmd(self)
        self.layout = LayoutCmd(self, self.task_manager)
        self.transform = TransformCmd(self)
        self.stock = StockCmd(self)
        self.step = StepCmd(self)
        self.tab = TabCmd(self)
        self.machine = MachineCmd(self)

    def cleanup(self):
        """
        Shuts down owned long-running services, like the OpsGenerator, to
        ensure cleanup of resources (e.g., shared memory).
        """
        # This is the safety net for any transient job artifacts that were
        # in-flight when the application was closed.
        logger.info(
            f"Releasing {len(self._transient_artifact_handles)} "
            "transient job artifacts..."
        )
        for handle in list(self._transient_artifact_handles):
            ArtifactStore.release(handle)
        self._transient_artifact_handles.clear()

        self.ops_generator.shutdown()

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
        machine = self._config_manager.config.machine
        if machine:
            return machine.dimensions
        return None

    @property
    def default_workpiece_layer(self) -> Layer:
        """
        Determines the most appropriate layer for adding new workpieces.
        """
        return self.doc.active_layer

    async def wait_until_settled(self, timeout: float = 10.0) -> None:
        """
        Waits until the internal OpsGenerator has finished all background
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
        vector_config: Optional[TraceConfig],
    ) -> None:
        """
        Imports a file from the specified path and waits for the operation
        to complete.
        """
        # Directly await the private async method, which is self-contained.
        await self.file._load_file_async(filename, mime_type, vector_config)

    async def export_gcode_to_path(self, output_path: "Path") -> None:
        """
        Exports the current document to a G-code file at the specified path
        and waits for the operation to complete. This awaitable version is
        useful for tests.
        """
        export_future = asyncio.get_running_loop().create_future()

        def _on_export_assembly_done(
            handle: Optional[ArtifactHandle], error: Optional[Exception]
        ):
            try:
                if error:
                    export_future.set_exception(error)
                    return
                if not handle:
                    exc = ValueError("Assembly process returned no artifact.")
                    export_future.set_exception(exc)
                    return

                artifact = ArtifactStore.get(handle)
                if artifact.gcode_bytes is None:
                    exc = ValueError("Final artifact is missing G-code data.")
                    export_future.set_exception(exc)
                    return

                gcode_str = artifact.gcode_bytes.tobytes().decode("utf-8")
                output_path.write_text(gcode_str, encoding="utf-8")

                logger.info(f"Test export successful to {output_path}")
                export_future.set_result(True)

            except Exception as e:
                if not export_future.done():
                    export_future.set_exception(e)
            finally:
                if handle:
                    ArtifactStore.release(handle)

        # Call the non-blocking method and provide our callback to bridge it
        self.file.assemble_job_in_background(
            when_done=_on_export_assembly_done
        )
        await export_future

    def set_doc(self, new_doc: Doc):
        """
        Assigns a new document to the editor, re-initializing the core
        components like the OpsGenerator.
        """
        self.ops_generator.processing_state_changed.disconnect(
            self._on_processing_state_changed
        )

        logger.debug("DocEditor is setting a new document.")
        self.doc = new_doc
        self.history_manager = self.doc.history_manager
        # The OpsGenerator's setter handles cleanup and reconnection
        self.ops_generator.doc = new_doc

        self.ops_generator.processing_state_changed.connect(
            self._on_processing_state_changed
        )

    @property
    def is_processing(self) -> bool:
        """Returns True if the document is currently generating operations."""
        return self.ops_generator.is_busy

    def _on_processing_state_changed(self, sender, is_processing: bool):
        """Proxies the signal from the OpsGenerator."""
        self.processing_state_changed.send(self, is_processing=is_processing)
        if not is_processing:
            self.document_settled.send(self)
