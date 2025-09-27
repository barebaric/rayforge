import logging
from typing import Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
from ...core.matrix import Matrix
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)
from ...core.geo import Geometry

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece

logger = logging.getLogger(__name__)


class FrameProducer(OpsProducer):
    """
    Generates a simple rectangular frame around the workpiece content with a
    specified offset.

    This producer operates on the workpiece's bounding box metadata and does
    not require a raster image.
    """

    def __init__(self, offset: float = 1.0):
        """
        Initializes the FrameProducer.

        Args:
            offset: The distance in millimeters to offset the frame from the
                    content's bounding box. A positive value expands the frame
                    outwards.
        """
        super().__init__()
        self.offset = offset

    def run(
        self,
        laser,
        surface,  # Unused
        pixels_per_mm,  # Unused
        *,
        workpiece: "Optional[WorkPiece]" = None,
        y_offset_mm: float = 0.0,
    ) -> PipelineArtifact:
        if workpiece is None:
            raise ValueError("FrameProducer requires a workpiece context.")

        # 1. Get the workpiece's current final size in millimeters.
        final_w, final_h = workpiece.size

        # 2. Define the frame around a 1x1 normalized box.
        #    The offset is in final mm, so we need to "un-scale" it.
        offset_x_norm = self.offset / final_w if final_w > 1e-9 else 0
        offset_y_norm = self.offset / final_h if final_h > 1e-9 else 0

        frame_x0 = -offset_x_norm
        frame_y0 = -offset_y_norm
        frame_x1 = 1 + offset_x_norm
        frame_y1 = 1 + offset_y_norm

        # 3. Create the normalized rectangular geometry.
        geo = Geometry()
        geo.move_to(frame_x0, frame_y0)
        geo.line_to(frame_x1, frame_y0)
        geo.line_to(frame_x1, frame_y1)
        geo.line_to(frame_x0, frame_y1)
        geo.close_path()
        frame_ops = Ops.from_geometry(geo)

        # 4. Apply the workpiece's scale to the normalized frame.
        sx, sy = workpiece.matrix.get_abs_scale()
        scaling_matrix = Matrix.scale(sx, sy)
        frame_ops.transform(scaling_matrix.to_4x4_numpy())

        logger.info(
            f"Generated frame with final geometry. Rect: {frame_ops.rect()}"
        )

        # Build the final Ops object
        final_ops = Ops()
        final_ops.add(
            OpsSectionStartCommand(SectionType.VECTOR_OUTLINE, workpiece.uid)
        )
        final_ops.extend(frame_ops)
        final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        # 5. Return a NON-SCALABLE artifact. The ops inside are already
        #    scaled to the correct final size.
        return PipelineArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

    def can_scale(self) -> bool:
        """
        Returns True to indicate the *generation process* is vector-based
        and can run without chunked rendering. The *output artifact* itself
        is marked as non-scalable.
        """
        return True

    @property
    def requires_full_render(self) -> bool:
        """
        This producer only needs the workpiece's metadata, not its
        rendered pixel data.
        """
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {"offset": self.offset},
        }
