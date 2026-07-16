"""Point-rotation array strategy."""

from __future__ import annotations

from typing import List

from raygeo.geo import Matrix

from .base import ArrayStrategy
from .params import PointRotationParams


class PointRotationStrategy(ArrayStrategy):
    """Rotates copies in place around the selection's own center.

    Each copy is the selection rotated by its angular offset around the
    unit center, so all copies share the same position and differ only
    in orientation. Instance 0 is the identity (the original).
    """

    def __init__(
        self,
        unit_bbox,
        params: PointRotationParams,
    ):
        super().__init__(unit_bbox)
        self.params = params

    def calculate_placements(self) -> List[Matrix]:
        p = self.params
        count = max(1, int(p.count))
        ax, ay = self.anchor_world
        offsets = self.distribute_angles(count, p.total_angle_deg)

        placements: List[Matrix] = []
        for i, ang_offset in enumerate(offsets):
            # Instance 0 is always the identity: the original stays put.
            if i == 0:
                placements.append(Matrix.identity())
                continue
            placements.append(Matrix.rotation(ang_offset, center=(ax, ay)))
        return placements
