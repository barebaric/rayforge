from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raygeo.sketch.constraints.equal_length import EqualLengthConstraint

if TYPE_CHECKING:
    from ..sketch import Sketch


@dataclass
class EqualConstraintMergeResult:
    final_entity_ids: list[int]
    constraints_to_remove: list[Any]


class EqualConstraintCommand:
    @staticmethod
    def find_and_merge_constraints(
        sketch: Sketch,
        selected_entity_ids: list[int],
    ) -> EqualConstraintMergeResult | None:
        selected_ids = set(selected_entity_ids)
        existing_constraints_to_merge: list[Any] = []
        final_ids = set(selected_ids)

        for constr in sketch.constraints:
            if isinstance(constr, EqualLengthConstraint):
                member_ids = {pid for pair in constr.pairs for pid in pair}
                if member_ids and not selected_ids.isdisjoint(
                    constraint_info_entities(sketch, constr)
                ):
                    existing_constraints_to_merge.append(constr)
                    final_ids.update(constraint_info_entities(sketch, constr))

        return EqualConstraintMergeResult(
            final_entity_ids=list(final_ids),
            constraints_to_remove=existing_constraints_to_merge,
        )


def constraint_info_entities(sketch, constr) -> set[int]:
    from .. import constraint_info

    return constraint_info.referenced_entity_ids(sketch.registry, constr)
