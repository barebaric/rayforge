import re
from abc import ABC, abstractmethod
from gettext import gettext as _
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
)

from blinker import Signal
from gi.repository import Adw

from ....core.varset import Var

NULL_CHOICE_LABEL = _("None Selected")

_ADAPTER_REGISTRY: Dict[Type[Var], Type["RowAdapter"]] = {}

_A = TypeVar("_A", bound="RowAdapter")


def register_adapter(*var_classes: Type[Var]):
    """
    Decorator to register a RowAdapter for one or more Var subclasses.
    Lookup uses MRO, so only the most-specific Var class needs
    registration — subclasses inherit the adapter automatically.
    """

    def decorator(adapter_cls: Type[_A]) -> Type[_A]:
        for var_cls in var_classes:
            _ADAPTER_REGISTRY[var_cls] = adapter_cls
        return adapter_cls

    return decorator


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
    Use the @register_adapter decorator to associate with Var subclasses.

    Convention: adapters store their row as self._row so that
    update_from_var can operate on it.
    """

    changed: Signal
    has_natural_commit = False

    def __init__(self):
        self.changed = Signal()

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

    def needs_rebuild(self, old_var: Var, new_var: Var) -> bool:
        """Return True if the row must be recreated for the new var."""
        return type(old_var) is not type(new_var)

    def update_from_var(self, var: Var):
        pass
