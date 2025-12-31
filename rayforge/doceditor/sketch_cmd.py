from __future__ import annotations
import logging
from typing import TYPE_CHECKING, List, Dict, Any
from ..core.undo import ChangePropertyCommand
from ..core.workpiece import WorkPiece

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)


class SketchCmd:
    """Handles commands related to sketch-based workpieces."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    def set_workpiece_parameters(
        self, workpieces: List[WorkPiece], new_params: Dict[str, Any]
    ):
        """
        Updates the sketch parameters for one or more workpiece instances
        in a single undoable transaction.
        """
        if not workpieces or not new_params:
            return

        history = self._editor.history_manager

        with history.transaction(_("Change Sketch Parameters")) as t:
            for wp in workpieces:
                if not wp.sketch_uid:
                    continue

                # Create the full new dictionary of parameters by merging
                # the old with the new.
                old_params = wp.sketch_params.copy()
                updated_params = old_params.copy()
                updated_params.update(new_params)

                if old_params == updated_params:
                    continue

                # Use a property change command, which will trigger the
                # setter on the WorkPiece, causing it to regenerate.
                cmd = ChangePropertyCommand(
                    target=wp,
                    property_name="sketch_params",
                    new_value=updated_params,
                    old_value=old_params,
                )
                t.execute(cmd)
