from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

from ..result import IssueCategory, SanityIssue

if TYPE_CHECKING:
    from ..checker import SanityContext


class BaseCheck(ABC):
    @property
    @abstractmethod
    def category(self) -> IssueCategory: ...

    @abstractmethod
    def run(self, context: "SanityContext") -> List[SanityIssue]: ...
