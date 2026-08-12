from dataclasses import dataclass

from ..types import EntityID


@dataclass
class SymmetryConstraintParams:
    p1_id: EntityID
    p2_id: EntityID
    center_id: EntityID | None = None
    axis_id: EntityID | None = None


class SymmetryConstraintCommand:
    @staticmethod
    def determine_constraint_params(
        point_ids: list[EntityID],
        entity_ids: list[EntityID],
    ) -> SymmetryConstraintParams | None:
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
