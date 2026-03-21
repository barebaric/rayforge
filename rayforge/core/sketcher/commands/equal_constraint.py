from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

from ..constraints import EqualLengthConstraint

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..sketch import Sketch


@dataclass
class EqualConstraintMergeResult:
    final_entity_ids: List[int]
    constraints_to_remove: List[Constraint]


class EqualConstraintCommand:
    @staticmethod
    def find_and_merge_constraints(
        sketch: Sketch,
        selected_entity_ids: List[int],
    ) -> Optional[EqualConstraintMergeResult]:
        selected_ids = set(selected_entity_ids)
        existing_constraints_to_merge: List[Constraint] = []
        final_ids = set(selected_ids)

        for constr in sketch.constraints:
            if isinstance(constr, EqualLengthConstraint):
                if not selected_ids.isdisjoint(constr.entity_ids):
                    existing_constraints_to_merge.append(constr)
                    final_ids.update(constr.entity_ids)

        return EqualConstraintMergeResult(
            final_entity_ids=list(final_ids),
            constraints_to_remove=existing_constraints_to_merge,
        )
