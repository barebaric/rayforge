from __future__ import annotations

from .base import InstancePlacement, PatternStrategy, PlacementKind
from .circular import CircularPatternStrategy
from .definition import PatternDefinition, find_pattern_for_entity
from .params import CircularPatternParams, SketchArrayMode

_STRATEGIES = {
    SketchArrayMode.CIRCULAR: CircularPatternStrategy,
}


def make_pattern_strategy(
    mode: SketchArrayMode, params: CircularPatternParams
) -> PatternStrategy:
    """Creates the pattern strategy for the given array mode."""
    strategy_cls = _STRATEGIES.get(mode)
    if strategy_cls is None:
        raise ValueError(f"Unsupported sketch array mode: {mode}")
    return strategy_cls(params)


__all__ = [
    "CircularPatternParams",
    "CircularPatternStrategy",
    "InstancePlacement",
    "PatternDefinition",
    "PatternStrategy",
    "PlacementKind",
    "SketchArrayMode",
    "find_pattern_for_entity",
    "make_pattern_strategy",
]
