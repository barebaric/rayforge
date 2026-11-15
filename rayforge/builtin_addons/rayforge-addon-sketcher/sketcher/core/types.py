from typing import TypeAlias

from raygeo.sketch.constraints import Constraint
from raygeo.sketch.entities import Entity
from raygeo.sketch.types import EntityID

HitTarget: TypeAlias = Constraint | Entity | EntityID | None
