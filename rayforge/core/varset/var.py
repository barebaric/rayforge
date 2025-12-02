from typing import Optional, Type, Callable, Generic, TypeVar, Dict, Any
from blinker import Signal


T = TypeVar("T")


class ValidationError(ValueError):
    """Custom exception for validation failures in Var."""

    pass


class Var(Generic[T]):
    """
    Represents a single typed variable with metadata for UI generation,
    validation, and data handling.
    """

    def __init__(
        self,
        key: str,
        label: str,
        var_type: Type[T],
        description: Optional[str] = None,
        default: Optional[T] = None,
        value: Optional[T] = None,
        validator: Optional[Callable[[Optional[T]], None]] = None,
    ):
        """
        Initializes a new Var instance.

        Args:
            key: The unique machine-readable identifier for the variable.
            label: The human-readable name for the variable (e.g., for UI).
            var_type: The expected Python type of the variable's value.
            description: A longer, human-readable description.
            default: The default value.
            value: The initial value. If provided, it overrides the default.
            validator: An optional callable that raises an exception if a new
                       value is invalid.
        """
        self._key = key
        self._label = label
        self.var_type = var_type
        self._description = description
        self._default = default
        self.validator = validator
        self._value: Optional[T] = None

        # Signal sent when the Var's value or default value changes.
        self.value_changed = Signal()
        # Signal sent when the Var's definition (key, label, etc.) changes.
        self.definition_changed = Signal()

        # Set initial explicit value ONLY if provided.
        if value is not None:
            self.value = value  # Use the public setter

    @property
    def key(self) -> str:
        """The unique machine-readable identifier for the variable."""
        return self._key

    @key.setter
    def key(self, new_key: str):
        if self._key != new_key:
            self._key = new_key
            self.definition_changed.send(self, property="key")

    @property
    def label(self) -> str:
        """The human-readable name for the variable (e.g., for UI)."""
        return self._label

    @label.setter
    def label(self, new_label: str):
        if self._label != new_label:
            self._label = new_label
            self.definition_changed.send(self, property="label")

    @property
    def description(self) -> Optional[str]:
        """A longer, human-readable description."""
        return self._description

    @description.setter
    def description(self, new_description: Optional[str]):
        if self._description != new_description:
            self._description = new_description
            self.definition_changed.send(self, property="description")

    def validate(self) -> None:
        """
        Runs the validator on the current effective value.

        Raises:
            ValidationError: If validation fails.
        """
        if self.validator:
            try:
                # The validator checks the effective value.
                self.validator(self.value)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Validation failed for key '{self.key}' with value "
                    f"'{self.value}': {e}"
                ) from e

    @property
    def default(self) -> Optional[T]:
        """The default value of the variable."""
        return self._default

    @default.setter
    def default(self, new_default: Optional[T]):
        """
        Sets the default value, triggering updates if effective value changes.
        """
        old_effective_value = self.value
        old_default = self._default

        if old_default == new_default:
            return  # Nothing to do

        self._default = new_default
        new_effective_value = self.value

        # A change in default is always a change in definition.
        self.definition_changed.send(self, property="default")

        # If the effective value was also changed, send that signal too.
        if old_effective_value != new_effective_value:
            self.value_changed.send(
                self,
                new_value=new_effective_value,
                old_value=old_effective_value,
            )

    @property
    def raw_value(self) -> Optional[T]:
        """The explicitly set value, or None if the default is being used."""
        return self._value

    @property
    def value(self) -> Optional[T]:
        """
        The effective value of the variable (returns explicit value if set,
        otherwise default).
        """
        if self._value is not None:
            return self._value
        return self.default

    @value.setter
    def value(self, new_value: Optional[T]):
        """
        Sets the explicit override value for the variable.
        """
        old_effective_value = self.value
        coerced_value: Optional[T]

        # 1. Coerce value if not None
        if new_value is None:
            coerced_value = None
        else:
            try:
                if self.var_type is int:
                    coerced_value = int(float(new_value))  # type: ignore
                elif self.var_type is bool:
                    if isinstance(new_value, str):
                        val_lower = new_value.lower()
                        if val_lower in ("true", "1", "on", "yes"):
                            coerced_value = True  # type: ignore
                        elif val_lower in ("false", "0", "off", "no"):
                            coerced_value = False  # type: ignore
                        else:
                            raise ValueError(
                                f"Cannot convert string '{new_value}' to bool."
                            )
                    else:
                        coerced_value = bool(new_value)  # type: ignore
                else:
                    coerced_value = self.var_type(new_value)  # type: ignore
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Value '{new_value}' for key '{self.key}' cannot be "
                    f"coerced to type {self.var_type.__name__}"
                ) from e

        # 2. Assign the coerced value to the explicit storage.
        self._value = coerced_value

        # 3. Emit signal if the *effective* value changed.
        new_effective_value = self.value
        if old_effective_value != new_effective_value:
            self.value_changed.send(
                self,
                new_value=new_effective_value,
                old_value=old_effective_value,
            )

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        """
        Serializes the Var's definition to a dictionary.

        Args:
            include_value: If True, the current value of the Var is included
                           in the output. Defaults to False.
        """
        data = {
            "class": self.__class__.__name__,
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "default": self.default,
        }
        if include_value:
            # Always serialize the effective value
            data["value"] = self.value
        return data

    def __repr__(self) -> str:
        return (
            f"Var(key='{self.key}', value={self.value}, "
            f"type={self.var_type.__name__})"
        )
