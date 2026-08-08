from dataclasses import dataclass, field
from enum import Enum
from gettext import gettext as _

from raygeo.geo.types import Point


class CheckMode(Enum):
    FAST = "fast"
    COMPLETE = "complete"


class IssueSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"


class IssueCategory(Enum):
    NOGO_ZONE = "nogo_zone"
    WORKAREA = "workarea"
    MACHINE_EXTENT = "machine_extent"


ISSUE_CATEGORY_LABELS = {
    IssueCategory.NOGO_ZONE: _("No-Go Zone"),
    IssueCategory.WORKAREA: _("Outside Work Area"),
    IssueCategory.MACHINE_EXTENT: _("Machine Extent"),
}


@dataclass
class SanityIssue:
    category: IssueCategory
    severity: IssueSeverity
    message: str
    zone_uid: str | None = None
    zone_name: str | None = None
    segment_start: Point | None = None
    segment_end: Point | None = None
    command_index: int | None = None


@dataclass
class SanityReport:
    mode: CheckMode
    issues: list[SanityIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return any(i.severity == IssueSeverity.ERROR for i in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(i.severity == IssueSeverity.WARNING for i in self.issues)

    @property
    def is_clean(self) -> bool:
        return len(self.issues) == 0
