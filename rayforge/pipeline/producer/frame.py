import logging
from typing import Optional, TYPE_CHECKING
from .base import OpsProducer, PipelineArtifact, CoordinateSystem
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
        """
        Generates the frame operations. This is treated as a non-scalable
        operation, as the frame's final dimensions depend on the workpiece's
        current size.
        """
        if workpiece is None:
            raise ValueError("FrameProducer requires a workpiece context.")

        if not workpiece.vectors:
            return PipelineArtifact(
                ops=Ops(),
                is_scalable=False,
                source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            )

        # 1. Get the unscaled bounding box from the source vector geometry.
        bbox = workpiece.vectors.rect()
        if not bbox:
            return PipelineArtifact(
                ops=Ops(),
                is_scalable=False,
                source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            )
        vx0, vy0, vx1, vy1 = bbox
        vw_unscaled = vx1 - vx0
        vh_unscaled = vy1 - vy0

        # 2. Get the workpiece's current final size in millimeters.
        final_w, final_h = workpiece.size

        # 3. Calculate the scale factors needed to transform the source
        #    geometry to its current size.
        sx = final_w / vw_unscaled if vw_unscaled > 1e-9 else 1.0
        sy = final_h / vh_unscaled if vh_unscaled > 1e-9 else 1.0

        # 4. Calculate the frame coordinates around the *scaled* content box.
        frame_x0 = (vx0 * sx) - self.offset
        frame_y0 = (vy0 * sy) - self.offset
        frame_x1 = (vx1 * sx) + self.offset
        frame_y1 = (vy1 * sy) + self.offset

        # 5. Create the rectangular geometry in final millimeter coordinates.
        geo = Geometry()
        geo.move_to(frame_x0, frame_y0)
        geo.line_to(frame_x1, frame_y0)
        geo.line_to(frame_x1, frame_y1)
        geo.line_to(frame_x0, frame_y1)
        geo.close_path()
        frame_ops = Ops.from_geometry(geo)

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

        # 6. Return a non-scalable artifact. The OpsGenerator will cache this
        #    and regenerate it if the workpiece size changes.
        return PipelineArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
        )

    def can_scale(self) -> bool:
        """
        While the output is vector, its dimensions depend on the workpiece's
        current transform, so it should be regenerated on resize. We return
        True here to ensure it uses the direct-vector path in the steprunner,
        but the returned artifact will correctly be marked as not scalable.
        """
        return True

    @property
    def requires_full_render(self) -> bool:
        """
        This producer only needs the workpiece's bounding box, not its
        rendered pixel data.
        """
        return False

    def to_dict(self) -> dict:
        """Serializes the producer configuration."""
        return {
            "type": self.__class__.__name__,
            "params": {"offset": self.offset},
        }
