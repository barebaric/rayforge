from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from raygeo.sketch.entities.line import Line

from ..types import EntityID

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry


@dataclass
class DistanceConstraintParams:
    p1_id: EntityID
    p2_id: EntityID
    distance: float


class DistanceConstraintCommand:
    @staticmethod
    def calculate_distance(
        registry: EntityRegistry,
        point_ids: list[EntityID],
        entity_ids: list[EntityID],
    ) -> DistanceConstraintParams | None:
        if len(point_ids) == 2:
            p1 = registry.get_point(point_ids[0])
            p2 = registry.get_point(point_ids[1])
            if p1 and p2:
                dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                return DistanceConstraintParams(
                    p1_id=point_ids[0],
                    p2_id=point_ids[1],
                    distance=dist,
                )

        if len(entity_ids) == 1:
            eid = entity_ids[0]
            entity = registry.get_entity_object(eid)
            if isinstance(entity, Line):
                p1 = registry.get_point(entity.p1_idx)
                p2 = registry.get_point(entity.p2_idx)
                if p1 and p2:
                    dist = math.hypot(p1[0] - p2[0], p1[1] - p2[1])
                    return DistanceConstraintParams(
                        p1_id=entity.p1_idx,
                        p2_id=entity.p2_idx,
                        distance=dist,
                    )

        return None

    @staticmethod
    def calculate_distance_from_points(
        p1: tuple[float, float],
        p2: tuple[float, float],
    ) -> float:
        return math.hypot(p1[0] - p2[0], p1[1] - p2[1])
