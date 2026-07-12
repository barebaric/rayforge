import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raygeo.cnc.machining.plan import Workplan
from raygeo.cnc.machining.wavefront import build_wavefront_workplan
from raygeo.geo import Geometry
from raygeo.geo.shape.polygon import is_point_inside_polygon
from raygeo.ops import Ops
from raygeo.ops.part import Part
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
    """Uses raygeo HSM adaptive wavefront to generate toolpaths for pockets.

    Removes material from a pocket via helical/spiral entry and wavefront
    passes. Z height is fixed for laser cutting — no Z motion.
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
        travel_rapid_rate = (
            settings.get("travel_speed", 5000) if settings else 5000
        )
        cut_power = settings.get("power", 1.0) if settings else 1.0

        part = build_part_vector(
            workpiece,
            surface=surface,
            normalize_windings=True,
        )
        if part is None or not part.has_geometry():
            raise ValueError(
                "WavefrontProducer could not derive vector geometry. "
                "No workpiece boundaries and surface tracing yielded "
                "no contours."
            )

        assert part.geometry is not None
        scaled_geo = part.geometry

        if self.offset_mm > 0:
            scaled_geo = scaled_geo.grow(-self.offset_mm)

        contours = scaled_geo.split_into_contours()
        closed = [c for c in contours if c.is_closed()]
        if not closed:
            raise ValueError("No closed contours found in workpiece geometry.")

        closed_geo = Geometry()
        for c in closed:
            closed_geo.extend(c)

        inner_list, outer_list = closed_geo.split_inner_and_outer_contours()

        if not outer_list:
            raise ValueError(
                "No outer boundary contour found in workpiece geometry."
            )

        outer_polys = [g.to_polygons(tolerance=0.01)[0] for g in outer_list]
        inner_polys = (
            [g.to_polygons(tolerance=0.01)[0] for g in inner_list]
            if inner_list
            else []
        )

        pockets = self._associate_pockets(outer_polys, inner_polys)

        final_ops = Ops()
        final_ops.set_head(laser.uid)

        precision = settings.get("arc_tolerance", 0.03) if settings else 0.03
        for idx, (boundary, islands) in enumerate(pockets):
            logger.info(
                "Processing pocket %d/%d: %d pts, %d islands",
                idx + 1,
                len(pockets),
                len(boundary),
                len(islands),
            )
            pocket_ops = self._process_pocket(
                boundary,
                islands,
                tool_radius,
                step_over,
                cut_feed_rate,
                travel_rapid_rate,
                cut_power,
                workpiece,
                precision=precision,
            )
            if pocket_ops.len() > 0:
                final_ops.ops_section_start(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
                final_ops.extend(pocket_ops)
                final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        return make_artifact(
            final_ops, workpiece, generation_id, is_vector=True
        )

    @staticmethod
    def _associate_pockets(
        outer_polys: List[List[Tuple[float, float]]],
        inner_polys: List[List[Tuple[float, float]]],
    ) -> List[
        Tuple[
            List[Tuple[float, float]],
            List[List[Tuple[float, float]]],
        ]
    ]:
        """Associate each outer contour with its nested inner contours.

        Returns a list of (boundary, islands) pairs.
        """
        used_inners: set = set()
        pockets = []
        for outer in outer_polys:
            islands = []
            for j, inner in enumerate(inner_polys):
                if j in used_inners:
                    continue
                if not inner:
                    continue
                cx = sum(p[0] for p in inner) / len(inner)
                cy = sum(p[1] for p in inner) / len(inner)
                if is_point_inside_polygon((cx, cy), outer):
                    islands.append(inner)
                    used_inners.add(j)
            pockets.append((outer, islands))
        return pockets

    def _process_pocket(
        self,
        pocket_boundary: List[Tuple[float, float]],
        islands: List[List[Tuple[float, float]]],
        tool_radius: float,
        step_over: float,
        cut_feed_rate: int,
        travel_rapid_rate: int,
        cut_power: float,
        workpiece: "WorkPiece",
        precision: float = 0.03,
    ) -> Ops:
        """Run wavefront passes for a single pocket."""
        part = Part.from_polygons(pocket_boundary, islands)
        try:
            wp = Workplan.from_part(part, safe_z=0.0)
        except ValueError:
            logger.warning(
                "Workplan.from_part failed for pocket of '%s' — no boundary.",
                workpiece.name,
            )
            return Ops()

        try:
            wf_steps = build_wavefront_workplan(
                pocket_boundary=pocket_boundary,
                islands=islands if islands else None,
                tool_radius=tool_radius,
                step_over=step_over,
                target_z=0.0,
                area_tolerance=self.area_tolerance,
                precision=precision,
            )
            wp.extend(wf_steps)
            result = wp.execute(
                cut_feed_rate=cut_feed_rate,
                cut_power=cut_power,
            )
            logger.info(
                "  pocket: %d ops (tool_radius=%.3f)",
                result.ops.len(),
                tool_radius,
            )
            return result.ops
        except Exception:
            logger.exception(
                "workplan failed for workpiece '%s'",
                workpiece.name,
            )
            return Ops()

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
