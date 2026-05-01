from typing import TYPE_CHECKING, List, Set

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
        for cmd in context.ops.commands:
            end = getattr(cmd, "end", None)
            if end is not None and pos is not None:
                start_2d = (pos[0], pos[1])
                end_2d = (end[0], end[1])
                for uid, zone in context.enabled_zones.items():
                    if uid in hit_zones:
                        continue
                    if self._check_segment(zone, start_2d, end_2d, cmd):
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
            if end is not None:
                pos = end
        return issues

    @staticmethod
    def _check_segment(zone, start_2d, end_2d, cmd) -> bool:
        if hasattr(cmd, "center_offset"):
            arc_center = (
                start_2d[0] + cmd.center_offset[0],
                start_2d[1] + cmd.center_offset[1],
            )
            return zone.collides_with_arc(
                start_2d,
                end_2d,
                arc_center,
                cmd.clockwise,
            )
        return zone.collides_with_line(start_2d, end_2d)
