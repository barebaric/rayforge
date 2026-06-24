import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from raygeo.geo import Geometry
from raygeo.geo.algo.cleared_area import ClearedArea
from raygeo.geo.shape.polygon import is_point_inside_polygon
from raygeo.ops import Ops
from raygeo.ops.assembly.hsm import (
    adaptive_entry,
    adaptive_peeling,
)
from raygeo.ops.types import SectionType

from rayforge.core.matrix import Matrix
from rayforge.pipeline.artifact import WorkPieceArtifact
from rayforge.pipeline.coord import CoordinateSystem
from rayforge.pipeline.producer.base import OpsProducer
from rayforge.shared.tasker.progress import ProgressContext

if TYPE_CHECKING:
    from rayforge.core.workpiece import WorkPiece
    from rayforge.machine.models.laser import Laser

logger = logging.getLogger(__name__)


class AdaptiveClearingProducer(OpsProducer):
    """Uses raygeo HSM adaptive clearing to generate toolpaths for pockets.

    Removes material from a pocket via helical/spiral entry and trochoidal
    peeling passes. Z height is fixed for laser cutting — no Z motion.
    """

    def __init__(
        self,
        step_over_mm: Optional[float] = None,
        offset_mm: float = 0.0,
        area_tolerance: float = 1.0,
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
            raise ValueError(
                "AdaptiveClearingProducer requires a workpiece context."
            )

        has_vector_source = (
            workpiece
            and workpiece.boundaries
            and not workpiece.boundaries.is_empty()
        )
        if not has_vector_source:
            raise ValueError(
                "AdaptiveClearingProducer requires vector geometry. "
                "No workpiece boundaries found."
            )

        assert workpiece.boundaries is not None
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

        scaled_geo = workpiece.boundaries.copy()
        width_mm, height_mm = workpiece.size
        scaling_matrix = Matrix.scale(width_mm, height_mm)
        scaled_geo.transform(scaling_matrix.to_4x4_numpy())
        scaled_geo.normalize_winding_orders()

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
            )
            if pocket_ops.len() > 0:
                final_ops.ops_section_start(
                    SectionType.VECTOR_OUTLINE, workpiece.uid
                )
                final_ops.extend(pocket_ops)
                final_ops.ops_section_end(SectionType.VECTOR_OUTLINE)

        return WorkPieceArtifact(
            ops=final_ops,
            is_scalable=False,
            source_coordinate_system=CoordinateSystem.MILLIMETER_SPACE,
            source_dimensions=workpiece.size,
            generation_size=workpiece.size,
            generation_id=generation_id,
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
    ) -> Ops:
        """Run entry and peeling passes for a single pocket."""
        z_cut = 0.0
        z_safe = 2.0
        plunge_pitch = 1.0

        entry_ops = Ops()
        cleared_polys = []
        try:
            entry_ops, cleared_polys = adaptive_entry(
                pocket_boundary=pocket_boundary,
                islands=islands,
                tool_radius=tool_radius,
                step_over=step_over,
                target_z=z_cut,
                safe_z=z_safe,
                plunge_pitch=plunge_pitch,
                cut_feed_rate=cut_feed_rate,
                cut_power=cut_power,
            )
        except Exception:
            logger.exception(
                "adaptive_entry failed for workpiece '%s', "
                "continuing with peeling only",
                workpiece.name,
            )

        logger.info(
            "  entry: %d ops, %d cleared polys (tool_radius=%.3f)",
            entry_ops.len(),
            len(cleared_polys) if cleared_polys else 0,
            tool_radius,
        )

        cleared = ClearedArea()
        if cleared_polys:
            cleared.add_cleared_polygons(cleared_polys)

        if not cleared.fragments():
            seed_radius = max(tool_radius, step_over * 0.5)
            cx = sum(p[0] for p in pocket_boundary) / len(pocket_boundary)
            cy = sum(p[1] for p in pocket_boundary) / len(pocket_boundary)
            seed_circle = [
                (
                    cx + seed_radius * math.cos(2 * math.pi * i / 16),
                    cy + seed_radius * math.sin(2 * math.pi * i / 16),
                )
                for i in range(16)
            ]
            cleared.add_cleared_polygons([seed_circle])

        peeled_ops = Ops()
        try:
            peeled_ops = adaptive_peeling(
                cleared=cleared,
                pocket_boundary=pocket_boundary,
                islands=islands,
                tool_radius=tool_radius,
                step_over=step_over,
                cut_z=z_cut,
                safe_z=z_safe,
                wall_margin=self.offset_mm,
                cut_feed_rate=cut_feed_rate,
                travel_rapid_rate=travel_rapid_rate,
                cut_power=cut_power,
            )
        except Exception:
            logger.exception(
                "adaptive_peeling failed for workpiece '%s'",
                workpiece.name,
            )

        logger.info("  peeling: %d ops", peeled_ops.len())

        result = Ops()
        for part in (entry_ops, peeled_ops):
            if part.len() > 0:
                result.extend(part)
        return result

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
    def from_dict(cls, data: Dict[str, Any]) -> "AdaptiveClearingProducer":
        params = data.get("params", {})
        return cls(
            step_over_mm=params.get("step_over_mm", None),
            offset_mm=params.get("offset_mm", 0.0),
            area_tolerance=params.get("area_tolerance", 1.0),
        )
