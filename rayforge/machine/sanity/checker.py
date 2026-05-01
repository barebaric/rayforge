from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, TYPE_CHECKING

from ...core.geo.types import Rect
from ...core.ops.container import Ops
from ..models.zone import Zone
from .result import CheckMode, SanityReport

if TYPE_CHECKING:
    from ..models.machine import Machine

from .checks.extent_2d import ExtentCheck2D
from .checks.nogo_zones_2d import NoGoZoneCheck2D
from .checks.workarea_2d import WorkareaCheck2D


@dataclass
class SanityContext:
    ops: Ops
    machine: Machine
    work_area: Rect
    axis_extents: Tuple[float, float]
    enabled_zones: Dict[str, Zone]


class SanityChecker:
    FAST_CHECKS = [
        WorkareaCheck2D,
        ExtentCheck2D,
        NoGoZoneCheck2D,
    ]
    COMPLETE_CHECKS = [
        WorkareaCheck2D,
        ExtentCheck2D,
        NoGoZoneCheck2D,
    ]

    def __init__(self, machine: Machine):
        self._machine = machine

    def check(
        self, ops: Ops, mode: CheckMode = CheckMode.FAST
    ) -> SanityReport:
        ctx = self._build_context(ops)
        classes = (
            self.FAST_CHECKS
            if mode == CheckMode.FAST
            else self.COMPLETE_CHECKS
        )
        report = SanityReport(mode=mode)
        for cls in classes:
            report.issues.extend(cls().run(ctx))
        return report

    def _build_context(self, ops: Ops) -> SanityContext:
        return SanityContext(
            ops=ops,
            machine=self._machine,
            work_area=self._machine.work_area,
            axis_extents=self._machine.axis_extents,
            enabled_zones={
                k: v for k, v in self._machine.nogo_zones.items() if v.enabled
            },
        )
