from __future__ import annotations
import logging
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple, Dict, Any

from blinker import Signal
from ..core.doc import Doc
from ..core.layer import Layer
from ..core.stocklayer import StockLayer
from ..core.vectorization_config import TraceConfig
from ..pipeline.generator import OpsGenerator
from ..machine.cmd import MachineCmd
from .edit_cmd import EditCmd
from .file_cmd import FileCmd
from .group_cmd import GroupCmd
from .layer_cmd import LayerCmd
from .layout_cmd import LayoutCmd
from .transform_cmd import TransformCmd
from .stock_cmd import StockCmd
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
        self._task_manager = task_manager
        self._config_manager = config_manager
        self.doc = doc or Doc()
        self.ops_generator = OpsGenerator(self.doc, self._task_manager)
        self.history_manager: "HistoryManager" = self.doc.history_manager

        # Signals for monitoring document processing state
        self.processing_state_changed = Signal()
        self.document_settled = Signal()  # Fires when processing finishes
        self.notification_requested = Signal()  # For UI feedback
        self.ops_generator.processing_state_changed.connect(
            self._on_processing_state_changed
        )

        # Instantiate and link command handlers, passing dependencies.
        self.edit = EditCmd(self)
        self.file = FileCmd(self, self._task_manager, self._config_manager)
        self.group = GroupCmd(self, self._task_manager)
        self.layer = LayerCmd(self)
        self.layout = LayoutCmd(self, self._task_manager)
        self.transform = TransformCmd(self)
        self.stock = StockCmd(self)
        self.tab = TabCmd(self)
        self.machine = MachineCmd(self)

    def add_tab_from_context(self, context: Dict[str, Any]):
        """
        Public handler for the 'add_tab' action, using context from the UI.
        """
        workpiece: "WorkPiece" = context["workpiece"]
        location: Dict[str, Any] = context["location"]
        segment_index = location["segment_index"]
        t = location["t"]

        self.tab.add_single_tab(
            workpiece=workpiece, segment_index=segment_index, t=t
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
        - If the active layer is a standard layer, returns it.
        - If the active layer is a stock layer, returns the topmost standard
          layer.
        - If no standard layers exist, it creates one.
        """
        active_layer = self.doc.active_layer
        if not isinstance(active_layer, StockLayer):
            return active_layer

        # Active layer is stock, find the top-most standard layer
        for child in reversed(self.doc.children):
            if isinstance(child, Layer) and not isinstance(child, StockLayer):
                return child

        # No standard layer found, so create one.
        # This is an edge case, but good to handle.
        logger.warning("No standard layer found; creating a new one.")
        new_layer = Layer(_("Layer 1"))
        self.doc.add_layer(new_layer)
        return new_layer

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
        import_future = asyncio.get_running_loop().create_future()

        def when_done_callback(task):
            try:
                task.result()  # Re-raises exceptions from the task
                import_future.set_result(True)
            except Exception as e:
                import_future.set_exception(e)

        self.file.load_file_from_path(
            filename, mime_type, vector_config, when_done=when_done_callback
        )
        await import_future

    async def export_gcode_to_path(self, output_path: "Path") -> None:
        """
        Exports the current document to a G-code file at the specified path
        and waits for the operation to complete.
        """
        export_future = asyncio.get_running_loop().create_future()

        def when_done_callback(task):
            try:
                task.result()  # Re-raises exceptions from the task
                export_future.set_result(True)
            except Exception as e:
                export_future.set_exception(e)

        self.file.export_gcode_to_path(
            output_path, when_done=when_done_callback
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
