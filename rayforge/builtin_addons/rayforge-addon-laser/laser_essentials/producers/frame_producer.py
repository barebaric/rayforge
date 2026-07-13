import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from raygeo.ops.assembly.frame import frame

from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.producer.base import CutSide, OpsProducer
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector,
    wrap_assembler_result,
)
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser

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
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("FrameProducer requires a workpiece context.")

        settings = settings or {}
        kerf_mm = settings.get("kerf_mm", laser.spot_size_mm[0])

        # 2. Build a Part from the workpiece and use the frame assembler.
        #    The assembler computes the total offset from kerf, path
        #    offset, and cut side internally.
        part = build_part_vector(workpiece)
        if part is None:
            raise ValueError("FrameProducer: workpiece has no vector geometry")

        result = frame(
            part,
            kerf_mm=kerf_mm,
            path_offset_mm=self.path_offset_mm,
            cut_side=self.cut_side.name.lower(),
        )

        return wrap_assembler_result(
            result,
            workpiece,
            laser,
            generation_id,
            set_power=settings.get("power", 0),
        )

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
