from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

from ..entities import Line
from ..types import EntityID

if TYPE_CHECKING:
    from ..registry import EntityRegistry
    from ..entities import Point


@dataclass
class DistanceConstraintParams:
    p1_id: EntityID
    p2_id: EntityID
    distance: float


class DistanceConstraintCommand:
    @staticmethod
    def calculate_distance(
        registry: EntityRegistry,
        point_ids: List[EntityID],
        entity_ids: List[EntityID],
    ) -> Optional[DistanceConstraintParams]:
        if len(point_ids) == 2:
            p1 = registry.get_point(point_ids[0])
            p2 = registry.get_point(point_ids[1])
            if p1 and p2:
                dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                return DistanceConstraintParams(
                    p1_id=p1.id,
                    p2_id=p2.id,
                    distance=dist,
                )

        if len(entity_ids) == 1:
            eid = entity_ids[0]
            e = registry.get_entity(eid)
            if isinstance(e, Line):
                p1 = registry.get_point(e.p1_idx)
                p2 = registry.get_point(e.p2_idx)
                if p1 and p2:
                    dist = math.hypot(p1.x - p2.x, p1.y - p2.y)
                    return DistanceConstraintParams(
                        p1_id=p1.id,
                        p2_id=p2.id,
                        distance=dist,
                    )

        return None

    @staticmethod
    def calculate_distance_from_points(
        p1: Point,
        p2: Point,
    ) -> float:
        return math.hypot(p1.x - p2.x, p1.y - p2.y)
