from __future__ import annotations

from typing import TypeAlias, TypeVar

from raygeo.sketch.arrays import (
    CircularArrayStrategy,
    CurveAlongArrayStrategy,
)

ArrayStrategy: TypeAlias = CircularArrayStrategy | CurveAlongArrayStrategy

StrategyT = TypeVar(
    "StrategyT", CircularArrayStrategy, CurveAlongArrayStrategy
)
