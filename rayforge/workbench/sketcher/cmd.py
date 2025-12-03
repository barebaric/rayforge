import logging
from typing import Dict, Any, TYPE_CHECKING
from rayforge.undo import Command
from rayforge.core.sketcher import Sketch

if TYPE_CHECKING:
    from rayforge.core.doc import Doc


logger = logging.getLogger(__name__)


class UpdateSketchCommand(Command):
    """
    A command that updates a Sketch template in the document's central
    registry. It also recalculates geometry and resizes all WorkPiece
    instances that use this sketch.
    """

    def __init__(
        self,
        doc: "Doc",
        sketch_uid: str,
        new_sketch_dict: Dict[str, Any],
        name: str = "Edit Sketch",
    ):
        super().__init__(name)
        self.doc = doc
        self.sketch_uid = sketch_uid

        # --- Store old state for undo ---
        old_sketch = doc.get_asset_by_uid(sketch_uid)
        if not old_sketch:
            raise ValueError(f"Sketch with UID {sketch_uid} not found.")
        self.old_sketch_dict = old_sketch.to_dict()
        self.new_sketch_dict = new_sketch_dict

        # Store old matrices of all affected workpieces for a perfect undo
        self.old_matrices = {
            wp.uid: wp.matrix.copy()
            for wp in doc.all_workpieces
            if wp.sketch_uid == sketch_uid
        }

    def _apply_sketch_state(self, sketch_dict: Dict[str, Any]):
        """Helper to apply a sketch dictionary to the document state."""

        # 1. Update the sketch template in the document registry
        sketch_instance = Sketch.from_dict(sketch_dict)
        self.doc.sketches[self.sketch_uid] = sketch_instance

        # 2. Calculate new geometry and dimensions from the solved sketch
        sketch_instance.solve()
        geo = sketch_instance.to_geometry()

        if geo.is_empty():
            new_width = 50.0  # Default for empty sketch
            new_height = 50.0
        else:
            min_x, min_y, max_x, max_y = geo.rect()
            new_width = max(max_x - min_x, 1e-9)
            new_height = max(max_y - min_y, 1e-9)

        # 3. Update all associated workpieces
        for workpiece in self.doc.all_workpieces:
            if workpiece.sketch_uid == self.sketch_uid:
                # Update the workpiece's own dimension attributes
                workpiece.natural_width_mm = new_width
                workpiece.natural_height_mm = new_height

                # This resizes the workpiece's matrix while preserving its
                # center
                workpiece.set_size(new_width, new_height)

                # This clears _boundaries_cache and _render_cache
                workpiece.clear_render_cache()

                # Signal for UI to redraw this specific workpiece
                workpiece.updated.send(workpiece)

        # Send a general doc update signal for pipeline, etc.
        self.doc.updated.send(self.doc)

    def execute(self):
        self._do_execute()

    def undo(self):
        self._do_undo()

    def _do_execute(self):
        logger.debug(
            f"Executing UpdateSketchCommand for sketch {self.sketch_uid}"
        )
        self._apply_sketch_state(self.new_sketch_dict)

    def _do_undo(self):
        logger.debug(
            f"Undoing UpdateSketchCommand for sketch {self.sketch_uid}"
        )
        # Re-apply the old sketch definition, which will call set_size
        self._apply_sketch_state(self.old_sketch_dict)

        # `set_size` preserves center, which might not be what we want for
        # undo.
        # To guarantee a perfect undo, explicitly restore original matrices.
        for wp in self.doc.all_workpieces:
            if wp.uid in self.old_matrices:
                wp.matrix = self.old_matrices[wp.uid].copy()
