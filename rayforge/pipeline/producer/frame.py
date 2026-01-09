import logging
from typing import Optional, TYPE_CHECKING, Dict, Any
from ...core.geo import Geometry
from ...core.ops import (
    Ops,
    OpsSectionStartCommand,
    OpsSectionEndCommand,
    SectionType,
)
from ..artifact import WorkPieceArtifact
from ..coord import CoordinateSystem
from .base import OpsProducer, CutSide

if TYPE_CHECKING:
    from ...core.workpiece import WorkPiece
    from ...machine.models.laser import Laser
    from ...shared.tasker.proxy import BaseExecutionContext

logger = logging.getLogger(__name__)


class FrameProducer(OpsProducer):
    """
    Generates a simple rectangular frame around the workpiece content with a
    specified offset.

    This producer operates on the workpiece's bounding box metadata and does
    not require a raster image.
    """

    def __init__(
        self,
        path_offset_mm: float = 0.0,
        cut_side: CutSide = CutSide.CENTERLINE,
    ):
        """
        Initializes the FrameProducer.

        Args:
            path_offset_mm: An absolute distance to offset the frame from the
                            content's bounding box.
            cut_side: The rule for determining the final cut side. A frame
                      is typically an OUTSIDE cut.
        """
        super().__init__()
        self.path_offset_mm = path_offset_mm
        self.cut_side = cut_side

    def run(
        self,
        laser: "Laser",
        surface,  # Unused
        pixels_per_mm,  # Unused
        *,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        proxy: Optional["BaseExecutionContext"] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("FrameProducer requires a workpiece context.")

        final_ops = Ops()

        # 1. Calculate total offset
        kerf_mm = (settings or {}).get("kerf_mm", laser.spot_size_mm[0])
        kerf_compensation = kerf_mm / 2.0
        total_offset = 0.0
        if self.cut_side == CutSide.CENTERLINE:
            total_offset = 0.0  # Centerline ignores path offset
        elif self.cut_side == CutSide.OUTSIDE:
            # For a frame, OUTSIDE means expanding the boundary
            total_offset = self.path_offset_mm + kerf_compensation
        elif self.cut_side == CutSide.INSIDE:
            # For a frame, INSIDE means shrinking the boundary
            total_offset = -self.path_offset_mm - kerf_compensation

        # 2. Get the workpiece's final size in millimeters.
        final_w, final_h = workpiece.size

        # 3. Create a rectangular geometry at the workpiece's final size.
        geo = Geometry()
        geo.move_to(0, 0)
        geo.line_to(final_w, 0)
        geo.line_to(final_w, final_h)
        geo.line_to(0, final_h)
        geo.close_path()

        # 4. Apply the final offset in millimeter space.
        if abs(total_offset) > 1e-6:
            # The rectangle is CCW, so a positive offset expands it, and
            # a negative offset shrinks it. This aligns with our calculation.
            geo = geo.grow(total_offset)

        if not geo.is_empty():
            frame_ops = Ops.from_geometry(geo)
            logger.info(
                f"Generated frame with final geometry. "
                f"Rect: {frame_ops.rect()}"
            )
            # Build the final Ops object
            final_ops.set_laser(laser.uid)
            final_ops.add(
                OpsSectionStartCommand(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
            )
            final_ops.set_power((settings or {}).get("power", 0))
            final_ops.extend(frame_ops)
            final_ops.add(OpsSectionEndCommand(SectionType.VECTOR_OUTLINE))

        # 5. Return a NON-SCALABLE artifact. The ops are already at the correct
        #    final size, ready for positioning.
        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
        )

    @property
    def supports_kerf(self) -> bool:
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
            "params": {
                "path_offset_mm": self.path_offset_mm,
                "cut_side": self.cut_side.name,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameProducer":
        """Deserializes a dictionary into a FrameProducer instance."""
        params = data.get("params", {})
        cut_side_str = params.get(
            "cut_side", params.get("kerf_mode", "CENTERLINE")
        )
        try:
            cut_side = CutSide[cut_side_str]
        except KeyError:
            cut_side = CutSide.CENTERLINE

        # For backward compatibility with old configs
        path_offset_mm = params.get(
            "path_offset_mm",
            params.get("offset_mm", params.get("offset", 0.0)),
        )

        return cls(path_offset_mm=path_offset_mm, cut_side=cut_side)
