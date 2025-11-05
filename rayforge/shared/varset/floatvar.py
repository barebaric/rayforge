from typing import Optional, Callable
from .var import Var, ValidationError


class FloatVar(Var[float]):
    """A Var subclass for float values with optional bounds."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[float] = None,
        value: Optional[float] = None,
        min_val: Optional[float] = None,
        max_val: Optional[float] = None,
        extra_validator: Optional[Callable[[float], None]] = None,
    ):
        self.min_val = min_val
        self.max_val = max_val

        def validator(v: Optional[float]):
            # A None value is valid for an unset optional field.
            if v is None:
                return

            if self.min_val is not None and v < self.min_val:
                raise ValidationError(
                    _("Value must be at least {min_val}.").format(
                        min_val=self.min_val
                    )
                )
            if self.max_val is not None and v > self.max_val:
                raise ValidationError(
                    _("Value must be at most {max_val}.").format(
                        max_val=self.max_val
                    )
                )
            if extra_validator:
                extra_validator(v)

        super().__init__(
            key=key,
            label=label,
            var_type=float,
            description=description,
            default=default,
            value=value,
            validator=validator,
        )
