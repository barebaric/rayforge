from __future__ import annotations

from dataclasses import dataclass

from ...core.ops.axis import Axis


@dataclass(frozen=True)
class CanvasProjection:
    horizontal_axis: Axis = Axis.X
    vertical_axis: Axis = Axis.Y

    @property
    def circumferential_axis(self) -> Axis:
        return self.vertical_axis
