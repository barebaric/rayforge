from typing import TYPE_CHECKING, List, Set

from ..result import IssueCategory, IssueSeverity, SanityIssue
from .base import BaseCheck

if TYPE_CHECKING:
    from ..checker import SanityContext


class WorkareaCheck2D(BaseCheck):

    @property
    def category(self) -> IssueCategory:
        return IssueCategory.WORKAREA

    def run(self, context: "SanityContext") -> List[SanityIssue]:
        wa = context.work_area
        x_min, y_min = wa[0], wa[1]
        x_max, y_max = wa[0] + wa[2], wa[1] + wa[3]
        seen: Set[str] = set()
        issues: List[SanityIssue] = []
        for cmd in context.ops.commands:
            end = getattr(cmd, "end", None)
            if end is None:
                continue
            self._check_point(
                issues, seen, end, x_min, y_min, x_max, y_max
            )
        return issues

    @staticmethod
    def _check_point(
        issues: List[SanityIssue],
        seen: Set[str],
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
            if axis.endswith("-") and val >= limit:
                continue
            if axis.endswith("+") and val <= limit:
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
        return "{}={:.1f} < {:.1f} (work area)".format(
            axis_name, val, limit
        )
    return "{}={:.1f} > {:.1f} (work area)".format(
        axis_name, val, limit
    )
