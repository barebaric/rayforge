import logging
from typing import Optional, cast, TYPE_CHECKING
from ...core.workpiece import WorkPiece
from ...core.workflow import Step
from ...core.ops import Ops
from ..canvas import CanvasElement
from .ops import WorkPieceOpsElement

if TYPE_CHECKING:
    from ...pipeline.generator import OpsGenerator


logger = logging.getLogger(__name__)


class StepElement(CanvasElement):
    """
    StepElements display the result of a Step on the
    WorkSurface. The output represents the laser path.
    """

    def __init__(
        self,
        step: Step,
        ops_generator: "OpsGenerator",
        x: float,
        y: float,
        width: float,
        height: float,
        show_travel_moves: bool = False,
        **kwargs,
    ):
        """
        Initializes a StepElement with pixel dimensions.

        Args:
            step: The Step data object.
            ops_generator: The central generator for pipeline operations.
            x: The x-coordinate (pixel) relative to the parent.
            y: The y-coordinate (pixel) relative to the parent.
            width: The width (pixel).
            height: The height (pixel).
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        super().__init__(
            x, y, width, height, data=step, selectable=False, **kwargs
        )
        self.show_travel_moves = show_travel_moves
        self.ops_generator = ops_generator

        # Connect to model signals for visibility and structural changes.
        step.changed.connect(self._on_step_model_changed)
        step.visibility_changed.connect(self._on_step_model_changed)

        # Connect to the OpsGenerator's signals for data pipeline events.
        self.ops_generator.ops_generation_starting.connect(
            self._on_ops_generation_starting
        )
        self.ops_generator.ops_chunk_available.connect(
            self._on_ops_chunk_available
        )
        self.ops_generator.ops_generation_finished.connect(
            self._on_ops_generation_finished
        )

    def add_workpiece(self, workpiece) -> WorkPieceOpsElement:
        """
        Adds a WorkPieceOpsElement for the given workpiece if it doesn't exist.
        Returns the existing or newly created element.
        """
        elem = self.find_by_data(workpiece)
        if elem:
            elem.mark_dirty()
            return cast(WorkPieceOpsElement, elem)

        elem = WorkPieceOpsElement(
            workpiece,
            show_travel_moves=self.show_travel_moves,
            canvas=self.canvas,
            parent=self,
        )
        self.add(elem)
        return elem

    def set_show_travel_moves(self, show: bool):
        """Sets travel move visibility for all child Ops elements."""
        if self.show_travel_moves == show:
            return
        self.show_travel_moves = show
        for child in self.children:
            if isinstance(child, WorkPieceOpsElement):
                child.set_show_travel_moves(show)

    def _on_step_model_changed(self, step: Step, **kwargs):
        """
        Handles any change from the Step model to sync the view.
        This includes visibility and pruning child elements for workpieces
        that are no longer in the parent layer.
        """
        assert self.canvas and self.parent and self.parent.data, (
            "Received step change, but element has no canvas"
            " or parent context"
        )

        # Sync visibility
        self.set_visible(step.visible)

        # Sync the child ops elements with the model's workpiece list.
        # The list of workpieces comes from the parent LayerElement's data.
        # This is crucial for handling undo/redo of add/remove workpiece,
        current_wp_elems = {child.data: child for child in self.children}
        model_workpieces = set(self.parent.data.workpieces)

        # Remove ops elements for workpieces that are no longer in the model.
        # Iterate over a copy of the keys to safely modify the underlying
        # list of children during iteration.
        for workpiece in list(current_wp_elems.keys()):
            if workpiece not in model_workpieces:
                # Get the element from the dictionary and remove it
                current_wp_elems[workpiece].remove()

        if self.canvas:
            self.canvas.queue_draw()

    def _find_or_add_workpiece_elem(
        self, workpiece: WorkPiece
    ) -> WorkPieceOpsElement:
        """Finds the element for a workpiece, creating if necessary."""
        elem = cast(
            Optional[WorkPieceOpsElement], self.find_by_data(workpiece)
        )
        if not elem:
            logger.debug(f"Adding workpiece to step: {workpiece.name}")
            elem = self.add_workpiece(workpiece)
        return elem

    def _on_ops_generation_starting(
        self,
        sender: Step,
        workpiece: WorkPiece,
        generation_id: int,
    ):
        """Called before ops generation starts for a workpiece."""
        # Only handle events for the step this element represents
        if sender is not self.data:
            return

        logger.debug(
            f"StepElem '{sender.name}': Received ops_generation_starting "
            f"for {workpiece.name}"
        )
        assert self.canvas and self.parent and self.parent.data, (
            "Received ops_start, but element has no canvas or parent context"
        )

        if workpiece not in self.parent.data.workpieces:
            elem = self.find_by_data(workpiece)
            if elem:
                elem.remove()
            return

        elem = self._find_or_add_workpiece_elem(workpiece)
        elem.clear_ops(generation_id=generation_id)

    def _on_ops_chunk_available(
        self,
        sender: Step,
        workpiece: WorkPiece,
        chunk: Ops,
        generation_id: int,
    ):
        """Called when a chunk of ops is available for a workpiece."""
        # Only handle events for the step this element represents
        if sender is not self.data:
            return

        logger.debug(
            f"StepElem '{sender.name}': Received ops_chunk_available for "
            f"{workpiece.name} (chunk size: {len(chunk)}, pos={workpiece.pos})"
        )
        assert self.canvas and self.parent and self.parent.data, (
            "Received update, but element has no canvas or parent context"
        )

        if workpiece not in self.parent.data.workpieces:
            elem = self.find_by_data(workpiece)
            if elem:
                elem.remove()
            return

        elem = self._find_or_add_workpiece_elem(workpiece)
        elem.add_ops(chunk, generation_id=generation_id)

    def _on_ops_generation_finished(
        self,
        sender: Step,
        workpiece: WorkPiece,
        generation_id: int,
    ):
        """
        Called when ops generation is finished. This handler ensures a final,
        guaranteed redraw of the element's complete state.
        """
        # Only handle events for the step this element represents
        if sender is not self.data:
            return

        logger.debug(
            f"StepElem '{sender.name}': Received ops_generation_finished "
            f"for {workpiece.name}"
        )
        assert self.canvas and self.parent and self.parent.data, (
            "Received ops_finished, but element has no canvas or parent "
            "context"
        )

        if workpiece not in self.parent.data.workpieces:
            elem = self.find_by_data(workpiece)
            if elem:
                elem.remove()
            return

        elem = self._find_or_add_workpiece_elem(workpiece)
        final_ops = self.ops_generator.get_ops(sender, workpiece)
        elem.set_ops(final_ops, generation_id=generation_id)
