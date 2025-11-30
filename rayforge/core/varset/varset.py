from typing import Dict, Optional, Iterator, Any, List, KeysView, Type
from blinker import Signal
from .var import Var


class VarSet:
    """
    A collection of Var objects, representing a logical group of settings or
    parameters. This class is observable via blinker signals.
    """

    var_added = Signal()
    """
    Signal sent when a new Var is added to the set.
    Sender: The VarSet instance.
    Args:
        var (Var): The Var instance that was added.
    """

    var_removed = Signal()
    """
    Signal sent when a Var is removed from the set.
    Sender: The VarSet instance.
    Args:
        var (Var): The Var instance that was removed.
    """

    cleared = Signal()
    """
    Signal sent when the VarSet is cleared of all Vars.
    Sender: The VarSet instance.
    """

    var_value_changed = Signal()
    """
    Signal sent when a contained Var's value changes (bubbled up).
    Sender: The VarSet instance.
    Args:
        var (Var): The child Var instance whose value changed.
        **kwargs: The original arguments from Var.value_changed
          (e.g., new_value).
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

    def _on_child_var_changed(self, var: Var, **kwargs):
        """Handler for bubbling up value changes from contained Vars."""
        self.var_value_changed.send(self, var=var, **kwargs)

    @staticmethod
    def _create_var_from_dict(data: Dict[str, Any]) -> Var:
        """
        Internal factory to instantiate a Var subclass from its serialized
        definition.
        """
        from .baudratevar import BaudrateVar
        from .boolvar import BoolVar
        from .choicevar import ChoiceVar
        from .floatvar import FloatVar, SliderFloatVar
        from .hostnamevar import HostnameVar
        from .intvar import IntVar
        from .portvar import PortVar
        from .serialportvar import SerialPortVar
        from .textareavar import TextAreaVar

        _CLASS_MAP: Dict[str, Type[Var]] = {
            "BaudrateVar": BaudrateVar,
            "BoolVar": BoolVar,
            "ChoiceVar": ChoiceVar,
            "FloatVar": FloatVar,
            "HostnameVar": HostnameVar,
            "IntVar": IntVar,
            "PortVar": PortVar,
            "SerialPortVar": SerialPortVar,
            "SliderFloatVar": SliderFloatVar,
            "TextAreaVar": TextAreaVar,
            "Var": Var,
        }

        data_copy = data.copy()
        class_name = data_copy.pop("class", None)
        if not class_name:
            raise ValueError(
                "Var definition dictionary is missing 'class' key."
            )
        if class_name not in _CLASS_MAP:
            raise ValueError(
                f"Unknown Var class '{class_name}' in definition."
            )
        VarClass = _CLASS_MAP[class_name]
        if "value" in data_copy:
            del data_copy["value"]
        return VarClass(**data_copy)

    @property
    def vars(self) -> List[Var]:
        """Returns the list of Var objects in the set."""
        return list(self._vars.values())

    def add(self, var: Var):
        """Adds a Var to the set. Raises KeyError if the key exists."""
        if var.key in self._vars:
            raise KeyError(
                f"Var with key '{var.key}' already exists in this VarSet."
            )
        self._vars[var.key] = var
        # Use weak=False to ensure the bound method is not garbage collected
        # prematurely. We are responsible for disconnecting it manually.
        Var.value_changed.connect(
            self._on_child_var_changed, sender=var, weak=False
        )
        self.var_added.send(self, var=var)

    def remove(self, key: str) -> Optional[Var]:
        """Removes a Var from the set by its key and returns it."""
        var = self._vars.pop(key, None)
        if var:
            Var.value_changed.disconnect(
                self._on_child_var_changed, sender=var
            )
            self.var_removed.send(self, var=var)
        return var

    def get(self, key: str) -> Optional[Var]:
        """Gets a Var by its key, or None if not found."""
        return self._vars.get(key)

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        """Serializes the VarSet's full definition to a dictionary."""
        return {
            "title": self.title,
            "description": self.description,
            "vars": [var.to_dict(include_value=include_value) for var in self],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "VarSet":
        """Deserializes a dictionary into a full VarSet instance."""
        new_set = cls(
            title=data.get("title"), description=data.get("description")
        )
        var_definitions = data.get("vars", [])
        for var_data in var_definitions:
            try:
                new_var = cls._create_var_from_dict(var_data)
                new_set.add(new_var)
            except Exception as e:
                print(f"Warning: Could not deserialize var: {e}")
        return new_set

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
                self[key] = value

    def clear(self):
        """Removes all Var objects from the set."""
        for var in self:
            Var.value_changed.disconnect(
                self._on_child_var_changed, sender=var
            )
        self._vars.clear()
        self.cleared.send(self)

    def validate(self):
        """
        Validates all Var objects in the set.
        Raises: ValidationError on the first validation failure.
        """
        for var in self:
            var.validate()

    def __repr__(self) -> str:
        return f"VarSet(title='{self.title}', count={len(self)})"
