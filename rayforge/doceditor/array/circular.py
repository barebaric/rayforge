"""Circular array strategy."""

from __future__ import annotations

import math
from typing import List

from raygeo.geo import Matrix

from .base import ArrayStrategy
from .params import CircularArrayParams


class CircularArrayStrategy(ArrayStrategy):
    """Places copies along a circular arc around a center.

    Copies orbit ``center_mm`` at ``radius_mm``. With ``rotate_copies``
    each copy is also spun by its angular offset around the selection's
    own center. Instance 0 is the identity (the original stays in
    place).
    """

    def __init__(
        self,
        unit_bbox,
        params: CircularArrayParams,
    ):
        super().__init__(unit_bbox)
        self.params = params

    def calculate_placements(self) -> List[Matrix]:
        p = self.params
        count = max(1, int(p.count))

        cx, cy = p.center_mm
        ux, uy = self.anchor_world  # workpiece position on the circle
        radius = p.radius_mm

        # Base angle of the original selection relative to the center.
        base_angle = math.atan2(uy - cy, ux - cx)
        offsets = self.distribute_angles(count, p.total_angle_deg)

        placements: List[Matrix] = []
        for i, ang_offset in enumerate(offsets):
            # Instance 0 is always the identity: the original selection
            # stays in place as the anchor of the array.
            if i == 0:
                placements.append(Matrix.identity())
                continue

            angle = base_angle + math.radians(ang_offset)
            new_center_x = cx + radius * math.cos(angle)
            new_center_y = cy + radius * math.sin(angle)

            dx = new_center_x - ux
            dy = new_center_y - uy
            delta = Matrix.translation(dx, dy)

            if p.rotate_copies and abs(ang_offset) > 1e-9:
                # Spin the copy in place around the selection's own
                # center by the angular offset.
                spin = Matrix.rotation(ang_offset, center=(ux, uy))
                delta = delta @ spin

            placements.append(delta)
        return placements
