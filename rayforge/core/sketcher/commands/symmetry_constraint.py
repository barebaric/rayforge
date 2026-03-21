from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    pass


@dataclass
class SymmetryConstraintParams:
    p1_id: int
    p2_id: int
    center_id: Optional[int] = None
    axis_id: Optional[int] = None


class SymmetryConstraintCommand:
    @staticmethod
    def determine_constraint_params(
        point_ids: list[int],
        entity_ids: list[int],
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
