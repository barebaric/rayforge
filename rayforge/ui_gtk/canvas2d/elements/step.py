import logging
from typing import TYPE_CHECKING, cast

from ....core.workflow import Step
from ...canvas import CanvasElement
from .group import GroupElement
from .workpiece import WorkPieceElement

if TYPE_CHECKING:
    from ....pipeline.pipeline import Pipeline


logger = logging.getLogger(__name__)


class StepElement(CanvasElement):
    """
    A non-rendering CanvasElement that manages the view-state for a Step.

    This element is the "controller" for a Step in the view. Its primary job
    is to listen to its model for visibility changes and then broadcast that
    state to all sibling WorkPieceElements within the same layer.
    Its lifecycle is automatically managed by its parent LayerElement.
    """

    def __init__(
        self,
        step: Step,
        pipeline: "Pipeline",
        **kwargs,
    ):
        """
        Initializes a StepElement.

        Args:
            step: The Step data object.
            pipeline: The central generator for pipeline operations.
            **kwargs: Additional keyword arguments for CanvasElement.
        """
        super().__init__(
            x=0,
            y=0,
            width=0,
            height=0,  # No dimensions needed
            data=step,
            selectable=False,
            visible=self._effective_visible(step),  # Sync initial visibility
            **kwargs,
        )
        self.pipeline = pipeline

        # Connect to the model signals that drive its behavior
        step.visibility_changed.connect(self._on_visibility_changed)
        workflow = step.workflow
        if workflow is not None:
            workflow.updated.connect(self._on_workflow_updated)

    def remove(self):
        """Disconnects signals before removing the element."""
        step = cast(Step, self.data)
        step.visibility_changed.disconnect(self._on_visibility_changed)
        workflow = step.workflow
        if workflow is not None:
            workflow.updated.disconnect(self._on_workflow_updated)
        super().remove()

    @staticmethod
    def _effective_visible(step: Step) -> bool:
        """A step is visible on the canvas when it and its workflow are on."""
        workflow = step.workflow
        return step.visible and (workflow is None or workflow.enabled)

    def _on_visibility_changed(self, step: Step):
        """
        Handles visibility changes from the model. It updates its own state
        and then broadcasts the change to its siblings.
        """
        self._sync_visibility()

    def _on_workflow_updated(self, workflow):
        """Re-evaluates visibility when the workflow is enabled or disabled."""
        self._sync_visibility()

    def _sync_visibility(self):
        """Applies the effective step visibility and broadcasts it."""
        step = cast(Step, self.data)
        visible = self._effective_visible(step)
        if self.visible != visible:
            self.set_visible(visible)
            self._update_sibling_ops_visibility()

    def _update_sibling_ops_visibility(self):
        """
        THE CORE LOGIC: Finds all WorkPieceElement siblings in the same parent
        (LayerElement) and tells them to update the visibility of the ops
        layer corresponding to this step.
        """
        if not self.parent:
            return
        parent = cast("CanvasElement", self.parent)

        step_uid = self.data.uid
        is_visible = self.visible

        logger.debug(
            f"StepElement '{self.data.name}' broadcasting visibility "
            f"({is_visible}) to siblings."
        )

        # Iterate through all children of the parent (the LayerElement)
        for child in parent.children:
            if isinstance(child, (WorkPieceElement, GroupElement)):
                child.set_ops_visibility(step_uid, is_visible)
