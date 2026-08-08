from typing import TYPE_CHECKING

from raygeo.ops.types import CommandCategory

from ..result import IssueCategory, IssueSeverity, SanityIssue
from .base import BaseCheck

if TYPE_CHECKING:
    from ..checker import SanityContext

_BOUNDARY_TOLERANCE = 1e-6


class WorkareaCheck2D(BaseCheck):
    @property
    def category(self) -> IssueCategory:
        return IssueCategory.WORKAREA

    def run(self, context: "SanityContext") -> list[SanityIssue]:
        wa = context.work_area
        x_min, y_min = wa[0], wa[1]
        x_max, y_max = wa[0] + wa[2], wa[1] + wa[3]
        seen: set[str] = set()
        issues: list[SanityIssue] = []
        ops = context.ops
        for i in range(ops.len()):
            if ops.category(i) != CommandCategory.MOVING:
                continue
            end = ops.endpoint(i)
            self._check_point(issues, seen, end, x_min, y_min, x_max, y_max)
        return issues

    @staticmethod
    def _check_point(
        issues: list[SanityIssue],
        seen: set[str],
        point,
        x_min: float,
        y_min: float,
        x_max: float,
        y_max: float,
    ) -> None:
        for axis, val, limit in (
            ("X-", point[0], x_min),
            ("X+", point[0], x_max),
            ("Y-", point[1], y_min),
            ("Y+", point[1], y_max),
        ):
            if axis.endswith("-") and val >= limit - _BOUNDARY_TOLERANCE:
                continue
            if axis.endswith("+") and val <= limit + _BOUNDARY_TOLERANCE:
                continue
            if axis in seen:
                continue
            seen.add(axis)
            issues.append(
                SanityIssue(
                    category=IssueCategory.WORKAREA,
                    severity=IssueSeverity.WARNING,
                    message=_workarea_msg(axis, val, limit),
                )
            )


def _workarea_msg(axis: str, val: float, limit: float) -> str:
    axis_name = axis[0]
    if axis[1] == "-":
        return f"{axis_name}={val:.1f} < {limit:.1f} (work area)"
    return f"{axis_name}={val:.1f} > {limit:.1f} (work area)"
