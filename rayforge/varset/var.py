from typing import Optional, Type, Callable, Generic, TypeVar


T = TypeVar("T")


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
        validator: Optional[Callable[[T], None]] = None,
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

    @property
    def value(self) -> Optional[T]:
        """The current value of the variable."""
        return self._value

    @value.setter
    def value(self, new_value: Optional[T]):
        if new_value is None:
            self._value = None
            return

        value: T
        try:
            # Special case: allow int for bool type (0/1)
            if self.var_type is bool and isinstance(new_value, int):
                value = bool(new_value)
            else:
                # This dynamic cast is correct for the intended types (str,
                # int, float, and str-subclasses), but Pylance cannot verify
                # it for a generic T. Hence, we suppress the 'call-arg' error.
                value = self.var_type(new_value)  # type: ignore[call-arg]
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Value '{new_value}' for key '{self.key}' cannot be coerced "
                f"to type {self.var_type.__name__}"
            ) from e

        if self.validator:
            try:
                self.validator(value)
            except Exception as e:
                raise ValueError(
                    f"Validation failed for key '{self.key}' with value "
                    f"'{value}': {e}"
                ) from e

        self._value = value

    def __repr__(self) -> str:
        return (
            f"Var(key='{self.key}', value={self.value}, "
            f"type={self.var_type.__name__})"
        )
