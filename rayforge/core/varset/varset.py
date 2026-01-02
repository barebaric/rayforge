import logging
from typing import Dict, Optional, Iterator, Any, List, KeysView, Type
from blinker import Signal
from .var import Var

logger = logging.getLogger(__name__)


class VarSet:
    """
    A collection of Var objects, representing a logical group of settings or
    parameters. This class is observable via blinker signals.
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
        self._order: List[str] = []  # Explicit order tracking

        self.var_added = Signal()
        self.var_removed = Signal()
        self.cleared = Signal()
        self.var_value_changed = Signal()
        self.var_definition_changed = Signal()

        if vars:
            for var in vars:
                self.add(var)

    def _on_child_var_changed(self, var: Var, **kwargs):
        """Handler for bubbling up value changes from contained Vars."""
        logger.debug(
            f"Signal bubble-up: var_value_changed for var '{var.key}' "
            f"to '{kwargs.get('new_value')}'"
        )
        self.var_value_changed.send(self, var=var, **kwargs)

    def _on_child_var_definition_changed(self, var: Var, **kwargs):
        """
        Handler for bubbling up definition changes from contained Vars.
        """
        if kwargs.get("property") == "key":
            old_key = None
            for k, v in self._vars.items():
                if v is var:
                    old_key = k
                    break

            if old_key is not None and old_key != var.key:
                logger.debug(
                    f"Resyncing VarSet dictionary for key rename: "
                    f"'{old_key}' -> '{var.key}'"
                )
                # Update the dictionary key
                self._vars[var.key] = self._vars.pop(old_key)
                # Update the explicit order list
                try:
                    idx = self._order.index(old_key)
                    self._order[idx] = var.key
                except ValueError:
                    # Should not happen if state is consistent
                    pass

        logger.debug(
            f"Signal bubble-up: var_definition_changed for var '{var.key}' "
            f"(prop: {kwargs.get('property')})"
        )
        self.var_definition_changed.send(self, var=var, **kwargs)

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
        # Allow 'value' to be passed to constructor if present in dict
        return VarClass(**data_copy)

    @property
    def vars(self) -> List[Var]:
        """Returns the list of Var objects in the set in order."""
        return [self._vars[key] for key in self._order]

    def add(self, var: Var):
        """Adds a Var to the set. Raises KeyError if the key exists."""
        if var.key in self._vars:
            raise KeyError(
                f"Var with key '{var.key}' already exists in this VarSet."
            )
        self._vars[var.key] = var
        self._order.append(var.key)

        # Connect directly to the var's instance signal.
        # weak=False ensures the bound method is not garbage collected
        # prematurely. We are responsible for disconnecting it manually.
        var.value_changed.connect(self._on_child_var_changed, weak=False)
        var.definition_changed.connect(
            self._on_child_var_definition_changed, weak=False
        )
        logger.debug(f"Emitting signal: var_added for var '{var.key}'")
        self.var_added.send(self, var=var)

    def remove(self, key: str) -> Optional[Var]:
        """Removes a Var from the set by its key and returns it."""
        var = self._vars.pop(key, None)
        if var:
            if key in self._order:
                self._order.remove(key)
            # Disconnect from the specific instance signal.
            var.value_changed.disconnect(self._on_child_var_changed)
            var.definition_changed.disconnect(
                self._on_child_var_definition_changed
            )
            logger.debug(f"Emitting signal: var_removed for var '{var.key}'")
            self.var_removed.send(self, var=var)
        return var

    def get(self, key: str) -> Optional[Var]:
        """Gets a Var by its key, or None if not found."""
        return self._vars.get(key)

    def move_var(self, key: str, new_index: int):
        """
        Moves the variable with the given key to a new index in the list.
        """
        if key not in self._order:
            return

        # Clamp index
        if new_index < 0:
            new_index = 0
        if new_index >= len(self._order):
            new_index = len(self._order) - 1

        current_index = self._order.index(key)
        if current_index == new_index:
            return

        self._order.pop(current_index)
        self._order.insert(new_index, key)

    def to_dict(
        self, include_value: bool = False, include_metadata: bool = True
    ) -> Dict[str, Any]:
        """
        Serializes the VarSet's definition to a dictionary.

        Args:
            include_value: If True, include the current value of each Var.
            include_metadata: If True, include the VarSet's title and
                              description.
        """
        data: Dict[str, Any] = {
            "vars": [
                self._vars[key].to_dict(include_value=include_value)
                for key in self._order
            ],
        }
        if include_metadata:
            data["title"] = self.title
            data["description"] = self.description
        return data

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
        """Iterates over the Var objects in insertion/defined order."""
        for key in self._order:
            yield self._vars[key]

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
        for var in list(self._vars.values()):
            var.value_changed.disconnect(self._on_child_var_changed)
            var.definition_changed.disconnect(
                self._on_child_var_definition_changed
            )
        self._vars.clear()
        self._order.clear()
        logger.debug("Emitting signal: cleared")
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
