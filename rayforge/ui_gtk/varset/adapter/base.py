import re
from abc import ABC, abstractmethod
from gettext import gettext as _
from typing import Any, List, Optional, Tuple, Union

from gi.repository import Adw

from ....core.varset import Var

NULL_CHOICE_LABEL = _("None Selected")


def escape_title(text: str) -> str:
    return text.replace("&", "&&")


def natural_sort_key(s: str) -> List[Union[int, str]]:
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)
    ]


class RowAdapter(ABC):
    """
    Base class for row value adapters.

    Each adapter owns both the row widget creation and the value
    read/write logic. VarSetWidget uses adapters exclusively —
    it never dispatches on row/var type itself.

    Subclasses must implement create(), get_value(), and set_value().
    Register the subclass in _ADAPTER_MAP to associate it with a Var
    subclass.
    """

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "RowAdapter"]:
        raise NotImplementedError

    @abstractmethod
    def get_value(self) -> Optional[Any]:
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value: Any) -> None:
        raise NotImplementedError
