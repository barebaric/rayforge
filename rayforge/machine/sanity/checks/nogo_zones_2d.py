from typing import TYPE_CHECKING, List, Set

from ....core.ops import CommandCategory, CommandType
from ..result import IssueCategory, IssueSeverity, SanityIssue
from .base import BaseCheck

if TYPE_CHECKING:
    from ..checker import SanityContext


class NoGoZoneCheck2D(BaseCheck):
    @property
    def category(self) -> IssueCategory:
        return IssueCategory.NOGO_ZONE

    def run(self, context: "SanityContext") -> List[SanityIssue]:
        if not context.enabled_zones:
            return []

        hit_zones: Set[str] = set()
        issues: List[SanityIssue] = []
        pos = None
        ops = context.ops
        for i in range(ops.len()):
            if ops.category(i) != CommandCategory.MOVING:
                continue
            end = ops.endpoint(i)
            if pos is not None:
                start_2d = (pos[0], pos[1])
                end_2d = (end[0], end[1])
                for uid, zone in context.enabled_zones.items():
                    if uid in hit_zones:
                        continue
                    if self._check_segment(zone, start_2d, end_2d, i, ops):
                        hit_zones.add(uid)
                        issues.append(
                            SanityIssue(
                                category=IssueCategory.NOGO_ZONE,
                                severity=IssueSeverity.ERROR,
                                message='No-Go Zone "{}" entered'.format(
                                    zone.name
                                ),
                                zone_uid=uid,
                                zone_name=zone.name,
                            )
                        )
            pos = end
        return issues

    @staticmethod
    def _check_segment(zone, start_2d, end_2d, idx, ops) -> bool:
        if ops.command_type(idx) == CommandType.ARC_TO:
            i, j, cw = ops.arc_params(idx)
            arc_center = (start_2d[0] + i, start_2d[1] + j)
            return zone.collides_with_arc(
                start_2d,
                end_2d,
                arc_center,
                cw,
            )
        return zone.collides_with_line(start_2d, end_2d)
