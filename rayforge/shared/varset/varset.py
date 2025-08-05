from typing import Dict, Optional, Iterator, Any, List, KeysView
from .var import Var


class VarSet:
    """
    A collection of Var objects, representing a logical group of settings or
    parameters.
    """

    def __init__(
        self,
        vars: Optional[List[Var]] = None,
        title: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Initializes a new VarSet.

        Args:
            vars: An optional list of Var objects to populate the set with.
            title: An optional title for the group of variables.
            description: An optional description for the group.
        """
        self.title = title
        self.description = description
        self._vars: Dict[str, Var] = {}
        if vars:
            for var in vars:
                self.add(var)

    def add(self, var: Var):
        """Adds a Var to the set. Raises KeyError if the key exists."""
        if var.key in self._vars:
            raise KeyError(
                f"Var with key '{var.key}' already exists in this VarSet."
            )
        self._vars[var.key] = var

    def get(self, key: str) -> Optional[Var]:
        """Gets a Var by its key, or None if not found."""
        return self._vars.get(key)

    def __getitem__(self, key: str) -> Var:
        """Gets a Var by its key. Raises KeyError if not found."""
        return self._vars[key]

    def __setitem__(self, key: str, value: Any):
        """Sets the value of an existing Var by its key."""
        if key not in self._vars:
            raise KeyError(
                f"No Var with key '{key}' in this VarSet. "
                "Use add() to add a new Var."
            )
        self._vars[key].value = value

    def __iter__(self) -> Iterator[Var]:
        """Iterates over the Var objects in insertion order."""
        return iter(self._vars.values())

    def __len__(self) -> int:
        """Returns the number of Var objects in the set."""
        return len(self._vars)

    def keys(self) -> KeysView[str]:
        """Returns a view of the Var keys."""
        return self._vars.keys()

    def get_values(self) -> Dict[str, Any]:
        """Returns a dictionary of all keys and their current values."""
        return {key: var.value for key, var in self._vars.items()}

    def set_values(self, values: Dict[str, Any]):
        """
        Sets the values for multiple Vars from a dictionary.
        Ignores keys that are not in the VarSet.
        """
        for key, value in values.items():
            if key in self._vars:
                # Use the setter to trigger validation and coercion
                self[key] = value

    def clear(self):
        """Removes all Var objects from the set."""
        self._vars.clear()

    def validate(self):
        """
        Validates all Var objects in the set.

        Raises:
            ValidationError: On the first validation failure.
        """
        for var in self:
            var.validate()

    def __repr__(self) -> str:
        return f"VarSet(title='{self.title}', count={len(self)})"
