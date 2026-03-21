from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional, List

from ..entities import Arc, Circle, Line
from ..types import EntityID

if TYPE_CHECKING:
    from ..registry import EntityRegistry


@dataclass
class TangentConstraintParams:
    line_id: EntityID
    shape_id: EntityID


class TangentConstraintCommand:
    @staticmethod
    def identify_entities(
        registry: EntityRegistry,
        entity_ids: List[EntityID],
    ) -> Optional[TangentConstraintParams]:
        sel_line: Optional[Line] = None
        sel_shape: Optional[Arc | Circle] = None

        for eid in entity_ids:
            e = registry.get_entity(eid)
            if isinstance(e, Line):
                sel_line = e
            elif isinstance(e, (Arc, Circle)):
                sel_shape = e

        if sel_line and sel_shape:
            return TangentConstraintParams(
                line_id=sel_line.id,
                shape_id=sel_shape.id,
            )
        return None
