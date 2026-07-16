"""
Array strategies: pure-geometry calculators that turn array parameters
into a list of world-space transformation matrices (one per array
instance).

A strategy performs no document mutation. The first instance in the
returned list is always the identity matrix, representing the original
selection that stays in place.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from raygeo.geo import Matrix
from raygeo.geo.types import Rect


class ArrayStrategy(ABC):
    """Computes world-space delta matrices for an array arrangement.

    Each delta, when applied to every item of the source selection,
    places a copy at one array instance. Instance 0 is the identity
    (the original selection).

    The *anchor* is a relative point ``(u, v)`` within the selection's
    collective bounding box (the unit square ``[0, 1] × [0, 1]``).
    The default ``(0.5, 0.5)`` is the centre; subclasses may override
    ``_default_anchor`` for strategy-specific behaviour (e.g. the grid
    uses the origin corner ``(0, 0)``).
    When the selection is moved, the bbox changes and the anchor
    follows automatically.
    """

    def __init__(
        self,
        unit_bbox: Rect,
        anchor: Optional[Tuple[float, float]] = None,
    ):
        """
        Args:
            unit_bbox: The collective world-space bounding box
                ``(min_x, min_y, max_x, max_y)`` of the source
                selection, treated as one rigid unit.
            anchor: Optional custom anchor as a *local* ``(u, v)``
                pair inside the bbox's unit square.  ``(0, 0)`` is the
                bbox origin corner, ``(1, 1)`` the far corner.  When
                ``None`` the strategy-specific default is used.
        """
        self.unit_bbox: Rect = unit_bbox
        self._custom_anchor: Optional[Tuple[float, float]] = anchor

    @property
    def anchor(self) -> Tuple[float, float]:
        """The effective LOCAL anchor ``(u, v)`` for this strategy."""
        if self._custom_anchor is not None:
            return self._custom_anchor
        return self._default_anchor

    @property
    def anchor_world(self) -> Tuple[float, float]:
        """The anchor evaluated in world coordinates for the current
        ``unit_bbox``."""
        u, v = self.anchor
        min_x, min_y, max_x, max_y = self.unit_bbox
        w = max_x - min_x
        h = max_y - min_y
        return (min_x + u * w, min_y + v * h)

    @property
    def _default_anchor(self) -> Tuple[float, float]:
        """Default anchor — the bbox centre ``(0.5, 0.5)``."""
        return (0.5, 0.5)

    @property
    def _unit_center(self) -> Tuple[float, float]:
        min_x, min_y, max_x, max_y = self.unit_bbox
        return ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

    @property
    def _unit_size(self) -> Tuple[float, float]:
        min_x, min_y, max_x, max_y = self.unit_bbox
        return (max_x - min_x, max_y - min_y)

    @abstractmethod
    def calculate_placements(self) -> List[Matrix]:
        """Return one world-space delta Matrix per array instance."""
        raise NotImplementedError

    @staticmethod
    def distribute_angles(count: int, total_angle_deg: float) -> List[float]:
        """Distributes ``count`` angular offsets over ``total_angle_deg``.

        The step is always ``total / count`` so that every offset is
        strictly less than ``total_angle_deg`` and no two copies ever
        coincide with the original position (offset 0). The final copy
        sits at ``(count - 1) * total / count``.
        """
        if count <= 1:
            return [0.0]
        if count == 0:
            return []
        step = total_angle_deg / count
        return [i * step for i in range(count)]
