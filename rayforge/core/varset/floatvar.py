from typing import Optional, Callable, Dict, Any
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

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update({"min_val": self.min_val, "max_val": self.max_val})
        return data


class SliderFloatVar(FloatVar):
    """
    A FloatVar subclass that hints to the UI that it should be represented
    by a slider rather than a spinbox.
    The value is typically expected to be in a normalized 0.0-1.0 range,
    which the UI will display as 0-100.
    """

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
        show_value: bool = True,
    ):
        self.show_value = show_value
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=min_val,
            max_val=max_val,
            extra_validator=extra_validator,
        )

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update({"show_value": self.show_value})
        return data
