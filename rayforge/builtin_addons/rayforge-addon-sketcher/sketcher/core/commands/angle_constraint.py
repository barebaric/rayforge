from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raygeo.geo.shape.arc import normalize_angle
from raygeo.geo.shape.line import get_line_line_intersection
from raygeo.sketch.entities.line import Line
from raygeo.sketch.types import EntityID

if TYPE_CHECKING:
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


@dataclass
class AngleConstraintParams:
    anchor_id: EntityID
    other_id: EntityID
    value_deg: float
    anchor_far_idx: EntityID
    other_far_idx: EntityID


class AngleConstraintCommand:
    @staticmethod
    def calculate_constraint_params(
        registry: EntityRegistry,
        e1_id: EntityID,
        e2_id: EntityID,
    ) -> AngleConstraintParams | None:
        e1 = registry.get_entity_object(e1_id)
        e2 = registry.get_entity_object(e2_id)

        if not (isinstance(e1, Line) and isinstance(e2, Line)):
            logger.warning("Angle constraint requires exactly 2 lines.")
            return None

        p1 = registry.get_point(e1.p1_idx)
        p2 = registry.get_point(e1.p2_idx)
        p3 = registry.get_point(e2.p1_idx)
        p4 = registry.get_point(e2.p2_idx)

        if not (p1 and p2 and p3 and p4):
            return None

        intersection = get_line_line_intersection(p1, p2, p3, p4)

        if intersection is None:
            logger.warning(
                "Lines are parallel, cannot create angle constraint."
            )
            return None

        ix, iy = intersection

        def get_far_point(pid1, pid2, point1, point2):
            d1 = (point1[0] - ix) ** 2 + (point1[1] - iy) ** 2
            d2 = (point2[0] - ix) ** 2 + (point2[1] - iy) ** 2
            return pid1 if d1 > d2 else pid2

        far1 = get_far_point(e1.p1_idx, e1.p2_idx, p1, p2)
        far2 = get_far_point(e2.p1_idx, e2.p2_idx, p3, p4)

        far1_pt = p1 if far1 == e1.p1_idx else p2
        far2_pt = p3 if far2 == e2.p1_idx else p4

        dir1 = math.atan2(far1_pt[1] - iy, far1_pt[0] - ix)
        dir2 = math.atan2(far2_pt[1] - iy, far2_pt[0] - ix)

        cw_e1_to_e2 = normalize_angle(dir1 - dir2)
        cw_e1_to_e2_deg = math.degrees(cw_e1_to_e2)

        if cw_e1_to_e2 <= math.pi:
            anchor_id = e1_id
            other_id = e2_id
            value_deg = cw_e1_to_e2_deg
            anchor_far_idx = far1
            other_far_idx = far2
        else:
            anchor_id = e2_id
            other_id = e1_id
            value_deg = 360 - cw_e1_to_e2_deg
            anchor_far_idx = far2
            other_far_idx = far1

        logger.debug(
            f"calculate_constraint_params: e1_id={e1_id}, e2_id={e2_id}, "
            f"e1.p1_idx={e1.p1_idx}, e1.p2_idx={e1.p2_idx}, "
            f"e2.p1_idx={e2.p1_idx}, e2.p2_idx={e2.p2_idx}, "
            f"far1={far1}, far2={far2}, "
            f"cw_e1_to_e2_deg={cw_e1_to_e2_deg:.1f}°, "
            f"anchor_id={anchor_id}, other_id={other_id}, "
            f"value_deg={value_deg:.1f}°, "
            f"anchor_far_idx={anchor_far_idx}, other_far_idx={other_far_idx}"
        )

        return AngleConstraintParams(
            anchor_id=anchor_id,
            other_id=other_id,
            value_deg=value_deg,
            anchor_far_idx=anchor_far_idx,
            other_far_idx=other_far_idx,
        )
