import re
from abc import ABC, abstractmethod
from gettext import gettext as _
from typing import (
    Any,
    TypeVar,
)

from blinker import Signal
from gi.repository import Adw

from ....core.varset import Var

NULL_CHOICE_LABEL = _("None Selected")

_ADAPTER_REGISTRY: dict[type[Var], type["RowAdapter"]] = {}

_A = TypeVar("_A", bound="RowAdapter")


def register_adapter(*var_classes: type[Var]):
    """
    Decorator to register a RowAdapter for one or more Var subclasses.
    Lookup uses MRO, so only the most-specific Var class needs
    registration — subclasses inherit the adapter automatically.
    """

    def decorator(adapter_cls: type[_A]) -> type[_A]:
        for var_cls in var_classes:
            _ADAPTER_REGISTRY[var_cls] = adapter_cls
        return adapter_cls

    return decorator


def escape_title(text: str) -> str:
    return text.replace("&", "&&")


def natural_sort_key(s: str) -> list[int | str]:
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

    Composite (multi-key) adapters declare ``related_keys`` for the
    additional step attributes their row edits alongside the primary
    var key. The manager maps all those keys to the same adapter, skips
    creating separate rows for them, and emits ``data_changed`` for
    every key when the adapter fires.
    """

    changed: Signal
    has_natural_commit = False

    #: Additional keys (besides the primary var key) that this
    #: adapter's row reads or writes. Empty for single-key adapters.
    related_keys: tuple[str, ...] = ()

    def __init__(self):
        self.changed = Signal()

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> tuple[Adw.PreferencesRow, "RowAdapter"]:
        raise NotImplementedError

    @abstractmethod
    def get_value(self) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def set_value(self, value: Any) -> None:
        raise NotImplementedError

    def get_value_for_key(self, key: str) -> Any | None:
        """The value for a specific key this adapter manages.

        Single-key adapters only manage the primary key and delegate
        to :meth:`get_value`. Composite adapters override this to
        dispatch per key.
        """
        return self.get_value()

    def set_value_for_key(self, key: str, value: Any) -> None:
        """Set the value for a specific key this adapter manages.

        Single-key adapters only manage the primary key and delegate
        to :meth:`set_value`. Composite adapters override this to
        dispatch per key.
        """
        self.set_value(value)

    def needs_rebuild(self, old_var: Var, new_var: Var) -> bool:
        """Return True if the row must be recreated for the new var."""
        return type(old_var) is not type(new_var)

    def update_from_var(self, var: Var):
        pass

    def update_from_values(self, values: dict[str, Any]) -> None:
        """Refresh the row from the widget's current sibling values.

        Called by the row manager after every ``data_changed``
        emission (and after populate) with a dict of all current
        values keyed by var key. Adapters whose row depends on other
        vars (e.g. a preview driven by a sibling switch) override
        this. The default does nothing.
        """
