from typing import TypeAlias

from raygeo.sketch.constraints import Constraint
from raygeo.sketch.entities import Entity

EntityID: TypeAlias = int

HitTarget: TypeAlias = Constraint | Entity | EntityID | None
