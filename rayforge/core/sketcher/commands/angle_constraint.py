from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

from ....core.geo.primitives import line_intersection, normalize_angle
from ..entities import Line

if TYPE_CHECKING:
    from ..registry import EntityRegistry

logger = logging.getLogger(__name__)


@dataclass
class AngleConstraintParams:
    anchor_id: int
    other_id: int
    value_deg: float
    anchor_far_idx: int
    other_far_idx: int


class AngleConstraintCommand:
    @staticmethod
    def calculate_constraint_params(
        registry: EntityRegistry,
        e1_id: int,
        e2_id: int,
    ) -> Optional[AngleConstraintParams]:
        e1 = registry.get_entity(e1_id)
        e2 = registry.get_entity(e2_id)

        if not (isinstance(e1, Line) and isinstance(e2, Line)):
            logger.warning("Angle constraint requires exactly 2 lines.")
            return None

        p1 = registry.get_point(e1.p1_idx)
        p2 = registry.get_point(e1.p2_idx)
        p3 = registry.get_point(e2.p1_idx)
        p4 = registry.get_point(e2.p2_idx)

        if not (p1 and p2 and p3 and p4):
            return None

        intersection = line_intersection(
            (p1.x, p1.y), (p2.x, p2.y), (p3.x, p3.y), (p4.x, p4.y)
        )

        if intersection is None:
            logger.warning(
                "Lines are parallel, cannot create angle constraint."
            )
            return None

        ix, iy = intersection

        def get_far_point(px1, px2):
            d1 = (px1.x - ix) ** 2 + (px1.y - iy) ** 2
            d2 = (px2.x - ix) ** 2 + (px2.y - iy) ** 2
            return px1 if d1 > d2 else px2

        far1 = get_far_point(p1, p2)
        far2 = get_far_point(p3, p4)

        dir1 = math.atan2(far1.y - iy, far1.x - ix)
        dir2 = math.atan2(far2.y - iy, far2.x - ix)

        cw_e1_to_e2 = normalize_angle(dir1 - dir2)
        cw_e1_to_e2_deg = math.degrees(cw_e1_to_e2)

        if cw_e1_to_e2 <= math.pi:
            anchor_id = e1_id
            other_id = e2_id
            value_deg = cw_e1_to_e2_deg
            anchor_far_idx = far1.id
            other_far_idx = far2.id
        else:
            anchor_id = e2_id
            other_id = e1_id
            value_deg = 360 - cw_e1_to_e2_deg
            anchor_far_idx = far2.id
            other_far_idx = far1.id

        logger.debug(
            f"calculate_constraint_params: e1_id={e1_id}, e2_id={e2_id}, "
            f"e1.p1_idx={e1.p1_idx}, e1.p2_idx={e1.p2_idx}, "
            f"e2.p1_idx={e2.p1_idx}, e2.p2_idx={e2.p2_idx}, "
            f"far1.id={far1.id}, far2.id={far2.id}, "
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
