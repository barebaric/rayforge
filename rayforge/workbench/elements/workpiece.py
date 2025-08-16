import logging
import cairo
from typing import Optional, TYPE_CHECKING
from ...core.workpiece import WorkPiece
from ..canvas import CanvasElement
from ...core.matrix import Matrix

if TYPE_CHECKING:
    from ..surface import WorkSurface


logger = logging.getLogger(__name__)


class WorkPieceElement(CanvasElement):
    """
    A CanvasElement that displays a WorkPiece. Its transform and content
    are driven entirely by signals from the WorkPiece data model.
    """

    # This constant matrix corrects for the Y-down orientation of the
    # cairo surface content. It performs a vertical flip within the
    # element's local 1x1 coordinate space.
    CONTENT_FLIP_MATRIX = Matrix.translation(0, 1) @ Matrix.scale(1, -1)

    def __init__(self, workpiece: WorkPiece, **kwargs):
        self.canvas: Optional["WorkSurface"]
        self.data: WorkPiece = workpiece
        # The element's local geometry is a fixed 1x1 unit square.
        super().__init__(
            0.0,
            0.0,
            1.0,
            1.0,
            data=workpiece,
            clip=False,
            buffered=True,
            pixel_perfect_hit=True,
            **kwargs,
        )
        # DECLARE the content orientation. The base class will handle the rest.
        self.content_transform = self.CONTENT_FLIP_MATRIX

        workpiece.updated.connect(self._on_model_content_changed)
        workpiece.transform_changed.connect(self._on_transform_changed)

        # Set the initial state from the model upon creation.
        self._on_transform_changed(workpiece)
        self.trigger_update()

    def remove(self):
        """Disconnects signals before removing the element."""
        self.data.updated.disconnect(self._on_model_content_changed)
        self.data.transform_changed.disconnect(self._on_transform_changed)
        super().remove()

    def render_to_surface(
        self, width: int, height: int
    ) -> Optional[cairo.ImageSurface]:
        """
        Renders the workpiece's visual data to a pixel buffer.
        """
        return self.data.importer.render_to_pixels(
            width=width,
            height=height,
        )

    def _on_model_content_changed(self, workpiece: WorkPiece):
        """
        Handles content-only changes from the model.
        """
        self.trigger_update()

    def _on_transform_changed(self, workpiece: WorkPiece):
        """
        Handles all geometric changes from the model by updating the
        element's GEOMETRIC transformation matrix.
        """
        # Only update the view if the model's matrix is actually
        # different from the element's current visual transform.
        if not self.canvas or self.transform == workpiece.matrix:
            return

        self.set_transform(workpiece.matrix)
        self.canvas.queue_draw()
