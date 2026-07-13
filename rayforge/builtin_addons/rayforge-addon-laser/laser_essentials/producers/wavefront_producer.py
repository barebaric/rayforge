import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

from raygeo.ops import Ops
from raygeo.ops.assembly.wavefront import adaptive_wavefronts_multi_pocket
from raygeo.ops.types import SectionType

from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.pipeline.stage.assembler_helpers import (
    build_part_vector,
    make_artifact,
)
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser

logger = logging.getLogger(__name__)


class WavefrontProducer(OpsProducer):
    """Uses raygeo adaptive wavefront to generate toolpaths for pockets.

    Delegates pocket decomposition and wavefront expansion to
    ``adaptive_wavefronts_multi_pocket``.
    """

    def __init__(
        self,
        step_over_mm: Optional[float] = None,
        offset_mm: float = 0.0,
        area_tolerance: float = 0.01,
    ):
        super().__init__()
        self.step_over_mm = step_over_mm
        self.offset_mm = offset_mm
        self.area_tolerance = area_tolerance

    def run(
        self,
        laser: "Laser",
        surface,
        pixels_per_mm,
        *,
        generation_id: int,
        workpiece: "Optional[WorkPiece]" = None,
        settings: Optional[Dict[str, Any]] = None,
        y_offset_mm: float = 0.0,
        context: Optional[ProgressContext] = None,
    ) -> WorkPieceArtifact:
        if workpiece is None:
            raise ValueError("WavefrontProducer requires a workpiece context.")

        tool_radius = laser.spot_size_mm[0] / 2.0
        step_over = (
            self.step_over_mm
            if self.step_over_mm is not None
            else laser.spot_size_mm[0]
        )

        cut_feed_rate = settings.get("cut_speed", 500) if settings else 500
        cut_power = settings.get("power", 1.0) if settings else 1.0
        precision = settings.get("arc_tolerance", 0.03) if settings else 0.03

        part = build_part_vector(
            workpiece,
            surface=surface,
            normalize_windings=True,
        )
        if part is None or not part.has_geometry():
            logger.error(
                "WavefrontProducer: build_part_vector returned no geometry"
                " for '%s'",
                workpiece.name,
            )
            raise ValueError(
                "WavefrontProducer could not derive vector geometry. "
                "No workpiece boundaries and surface tracing yielded "
                "no contours."
            )

        logger.info(
            "WavefrontProducer: calling adaptive_wavefronts_multi_pocket for"
            " '%s' (tr=%.2f, step=%.2f, off=%.2f, tol=%.4f, prec=%.4f,"
            " feed=%d, pwr=%.2f)",
            workpiece.name,
            tool_radius,
            step_over,
            self.offset_mm,
            self.area_tolerance,
            precision,
            cut_feed_rate,
            cut_power,
        )
        try:
            result = adaptive_wavefronts_multi_pocket(
                part,
                tool_radius=tool_radius,
                step_over=step_over,
                offset_mm=self.offset_mm,
                area_tolerance=self.area_tolerance,
                precision=precision,
                cut_feed_rate=cut_feed_rate,
                cut_power=cut_power,
            )
            result_ops = result.ops
            logger.info(
                "WavefrontProducer: returned %d ops for '%s'",
                result_ops.len(),
                workpiece.name,
            )
        except Exception:
            logger.exception(
                "WavefrontProducer: adaptive_wavefronts_multi_pocket failed "
                "for workpiece '%s'",
                workpiece.name,
            )
            result_ops = Ops()

        final_ops = Ops()
        final_ops.set_head(laser.uid)
        final_ops.ops_section_start(
            SectionType.VECTOR_OUTLINE, workpiece.uid
        )
        final_ops.extend(result_ops)
        final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        return make_artifact(
            final_ops, workpiece, generation_id, is_vector=True
        )

    def to_dict(self) -> dict:
        return {
            "type": self.__class__.__name__,
            "params": {
                "step_over_mm": self.step_over_mm,
                "offset_mm": self.offset_mm,
                "area_tolerance": self.area_tolerance,
            },
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WavefrontProducer":
        params = data.get("params", {})
        tolerance = params.get("area_tolerance")
        if tolerance is None or tolerance == 1.0:
            tolerance = 0.01
        return cls(
            step_over_mm=params.get("step_over_mm", None),
            offset_mm=params.get("offset_mm", 0.0),
            area_tolerance=tolerance,
        )
