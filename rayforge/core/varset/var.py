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

    value_changed = Signal()
    """
    Signal sent when the Var's value changes.
    Sender: The Var instance.
    Args:
        new_value (Optional[T]): The new value.
        old_value (Optional[T]): The previous value.
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
        self.key = key
        self.label = label
        self.var_type = var_type
        self.description = description
        self.default = default
        self.validator = validator
        self._value: Optional[T] = None  # Initialize attribute

        # Set initial value, preferring explicit `value` over `default`.
        self.value = value if value is not None else default

    def validate(self) -> None:
        """
        Runs the validator on the current value.

        Raises:
            ValidationError: If validation fails.
        """
        if self.validator:
            try:
                # The validator receives self._value which might be None.
                # It's the validator's job to handle this.
                self.validator(self._value)
            except ValidationError:
                raise
            except Exception as e:
                raise ValidationError(
                    f"Validation failed for key '{self.key}' with value "
                    f"'{self._value}': {e}"
                ) from e

    @property
    def value(self) -> Optional[T]:
        """The current value of the variable."""
        return self._value

    @value.setter
    def value(self, new_value: Optional[T]):
        """
        Sets the variable's value, coercing it to the correct type and
        emitting a signal if the value has changed.
        """
        old_value = self._value
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

        # 2. Assign the coerced value and emit signal if changed.
        if old_value != coerced_value:
            self._value = coerced_value
            self.value_changed.send(
                self, new_value=self._value, old_value=old_value
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
            data["value"] = self.value
        return data

    def __repr__(self) -> str:
        return (
            f"Var(key='{self.key}', value={self.value}, "
            f"type={self.var_type.__name__})"
        )
