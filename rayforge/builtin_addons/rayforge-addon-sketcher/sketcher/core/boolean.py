"""Boolean operations over a sketch selection."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

from raygeo.sketch.boolean import BooleanOp as RustBooleanOp
from raygeo.sketch.boolean import BooleanRegion, BooleanSolid
from raygeo.sketch.boolean import (
    apply_boolean as _apply_boolean,
)
from raygeo.sketch.boolean import (
    build_boolean_regions as _build_boolean_regions,
)

from .contour import coincident_pairs

if TYPE_CHECKING:
    from .sketch import Sketch


class BooleanOp(Enum):
    UNION = "union"
    DIFFERENCE = "difference"
    INTERSECTION = "intersection"
    EXCLUDE = "exclude"


_OP_MAP = {
    BooleanOp.UNION: RustBooleanOp.UNION,
    BooleanOp.DIFFERENCE: RustBooleanOp.DIFFERENCE,
    BooleanOp.INTERSECTION: RustBooleanOp.INTERSECTION,
    BooleanOp.EXCLUDE: RustBooleanOp.EXCLUDE,
}


def build_boolean_regions(
    sketch: Sketch, entity_ids: list[int]
) -> list[BooleanRegion] | None:
    """Partitions the selection into closed candidate regions."""
    pairs = coincident_pairs(sketch, entity_ids)
    return _build_boolean_regions(sketch.registry, entity_ids, pairs)


def apply_boolean(
    op: BooleanOp, regions: list[BooleanRegion]
) -> list[BooleanSolid] | None:
    """Runs the boolean operation over the regions."""
    return _apply_boolean(_OP_MAP[op], regions)
