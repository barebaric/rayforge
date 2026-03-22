from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

from ..types import EntityID

if TYPE_CHECKING:
    pass


@dataclass
class SymmetryConstraintParams:
    p1_id: EntityID
    p2_id: EntityID
    center_id: Optional[EntityID] = None
    axis_id: Optional[EntityID] = None


class SymmetryConstraintCommand:
    @staticmethod
    def determine_constraint_params(
        point_ids: List[EntityID],
        entity_ids: List[EntityID],
    ) -> Optional[SymmetryConstraintParams]:
        if len(point_ids) == 3 and not entity_ids:
            return SymmetryConstraintParams(
                p1_id=point_ids[0],
                p2_id=point_ids[1],
                center_id=point_ids[2],
            )
        elif len(point_ids) == 2 and len(entity_ids) == 1:
            return SymmetryConstraintParams(
                p1_id=point_ids[0],
                p2_id=point_ids[1],
                axis_id=entity_ids[0],
            )
        return None
