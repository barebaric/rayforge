"""Grid (rows x columns) array strategy."""

from __future__ import annotations

from typing import List, Tuple

from raygeo.geo import Matrix

from .base import ArrayStrategy
from .params import GridArrayParams, SpacingMode


class GridArrayStrategy(ArrayStrategy):
    """Arranges copies in a regular 2D grid.

    The anchor defaults to the bounding-box origin ``(min_x, min_y)``
    which preserves the current behaviour: cell ``(0, 0)`` is identity
    (the original selection), and subsequent cells extend in the grid
    pattern relative to that corner.
    """

    def __init__(
        self,
        unit_bbox,
        params: GridArrayParams,
        anchor=None,
    ):
        super().__init__(unit_bbox, anchor)
        self.params = params

    @property
    def _default_anchor(self) -> Tuple[float, float]:
        """Grid's default anchor is the bbox origin corner."""
        return (0.0, 0.0)

    def _resolve_pitch(self) -> Tuple[float, float]:
        """Returns the (x, y) center-to-center pitch."""
        p = self.params
        if p.spacing_mode == SpacingMode.GAP:
            unit_w, unit_h = self._unit_size
            pitch_x = unit_w + p.col_spacing_mm
            pitch_y = unit_h + p.row_spacing_mm
        else:
            pitch_x = p.col_spacing_mm
            pitch_y = p.row_spacing_mm
        return pitch_x, pitch_y

    def calculate_placements(self) -> List[Matrix]:
        p = self.params
        rows = max(1, int(p.rows))
        cols = max(1, int(p.cols))
        pitch_x, pitch_y = self._resolve_pitch()
        ax, ay = self.anchor_world
        ox, oy = self.unit_bbox[0], self.unit_bbox[1]

        placements: List[Matrix] = []
        for row in range(rows):
            for col in range(cols):
                dx = (ax - ox) + col * pitch_x
                dy = (ay - oy) - row * pitch_y
                placements.append(Matrix.translation(dx, dy))
        return placements
