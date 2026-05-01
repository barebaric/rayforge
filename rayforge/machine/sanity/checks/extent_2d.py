from typing import TYPE_CHECKING, List, Set

from ..result import IssueCategory, IssueSeverity, SanityIssue
from .base import BaseCheck

if TYPE_CHECKING:
    from ..checker import SanityContext


class ExtentCheck2D(BaseCheck):

    @property
    def category(self) -> IssueCategory:
        return IssueCategory.MACHINE_EXTENT

    def run(self, context: "SanityContext") -> List[SanityIssue]:
        max_x, max_y = context.axis_extents
        seen: Set[str] = set()
        issues: List[SanityIssue] = []
        for cmd in context.ops.commands:
            end = getattr(cmd, "end", None)
            if end is None:
                continue
            for point in (end,):
                self._check_point(
                    issues, seen, point, max_x, max_y
                )
        return issues

    @staticmethod
    def _check_point(
        issues: List[SanityIssue],
        seen: Set[str],
        point,
        max_x: float,
        max_y: float,
    ) -> None:
        for axis, val, limit in (
            ("X-", point[0], 0),
            ("X+", point[0], max_x),
            ("Y-", point[1], 0),
            ("Y+", point[1], max_y),
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
                    category=IssueCategory.MACHINE_EXTENT,
                    severity=IssueSeverity.ERROR,
                    message=_extent_msg(axis, val, limit),
                )
            )


def _extent_msg(axis: str, val: float, limit: float) -> str:
    axis_name = axis[0]
    if axis[1] == "-":
        return f"{axis_name}={val:.1f} < 0"
    return f"{axis_name}={val:.1f} > {limit:.1f}"
