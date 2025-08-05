from typing import Optional, Type, Callable, Generic, TypeVar


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
        Sets the variable's value, coercing it to the correct type.
        Validation is now handled exclusively by the `validate()` method.
        """
        # 1. Coerce value if not None
        value: Optional[T]
        if new_value is None:
            self._value = None
            return

        try:
            if self.var_type is int:
                # We coerce via float() to handle strings, floats, and ints.
                value = int(float(new_value))  # type: ignore

            # Handle coercion to bool, which can come from ints or various
            # string forms.
            elif self.var_type is bool:
                if isinstance(new_value, str):
                    val_lower = new_value.lower()
                    if val_lower in ("true", "1", "on", "yes"):
                        value = True  # type: ignore
                    elif val_lower in ("false", "0", "off", "no"):
                        value = False  # type: ignore
                    else:
                        raise ValueError(
                            f"Cannot convert string '{new_value}' to bool."
                        )
                else:
                    # Coerce from other types like int or float using standard
                    # bool()
                    value = bool(new_value)  # type: ignore
            else:
                # For other types (float, str, etc.), use direct coercion.
                value = self.var_type(new_value)  # type: ignore[call-arg]
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Value '{new_value}' for key '{self.key}' cannot be "
                f"coerced to type {self.var_type.__name__}"
            ) from e

        # 2. Assign the coerced value. Validation is NOT performed here.
        self._value = value

    def to_dict(self):
        """Returns a dict of constructor arguments for easy copying."""
        return {
            "key": self.key,
            "label": self.label,
            "var_type": self.var_type,
            "description": self.description,
            "default": self.default,
            "value": self.value,
            "validator": self.validator,
        }

    def __repr__(self) -> str:
        return (
            f"Var(key='{self.key}', value={self.value}, "
            f"type={self.var_type.__name__})"
        )
