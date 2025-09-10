from __future__ import annotations
import logging
from typing import TYPE_CHECKING
from ..core.doc import Doc
from ..pipeline.generator import OpsGenerator
from .edit_cmd import EditCmd
from .file_cmd import FileCmd
from .group_cmd import GroupCmd
from .layer_cmd import LayerCmd
from .layout_cmd import LayoutCmd
from .transform_cmd import TransformCmd
from ..machine.cmd import MachineCmd
from ..workbench.view_mode_cmd import ViewModeCmd

if TYPE_CHECKING:
    from ..undo import HistoryManager

logger = logging.getLogger(__name__)


class DocEditor:
    """
    The central, non-UI controller for document state and operations.

    This class owns the core data models (Doc, OpsGenerator) and provides a
    structured API for all document manipulations, which are organized into
    namespaced command handlers.
    """

    def __init__(self, doc: Doc | None = None):
        """
        Initializes the DocEditor.

        Args:
            doc: An optional existing Doc object. If None, a new one is
                 created.
        """
        self.doc = doc or Doc()
        self.ops_generator = OpsGenerator(self.doc)
        self.history_manager: "HistoryManager" = self.doc.history_manager

        # Instantiate and link command handlers, passing a reference to self.
        self.edit = EditCmd(self)
        self.file = FileCmd(self)
        self.group = GroupCmd(self)
        self.layer = LayerCmd(self)
        self.layout = LayoutCmd(self)
        self.transform = TransformCmd(self)
        self.machine = MachineCmd(self)
        self.view = ViewModeCmd(self)

    def set_doc(self, new_doc: Doc):
        """
        Assigns a new document to the editor, re-initializing the core
        components like the OpsGenerator.
        """
        logger.debug("DocEditor is setting a new document.")
        self.doc = new_doc
        self.history_manager = self.doc.history_manager
        # The OpsGenerator's setter handles cleanup and reconnection
        self.ops_generator.doc = new_doc
