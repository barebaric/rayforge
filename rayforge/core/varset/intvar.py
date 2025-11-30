from typing import Optional, Callable, Dict, Any
from .var import Var, ValidationError


class IntVar(Var[int]):
    """A Var subclass for integer values with optional bounds."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
        value: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        validator: Optional[Callable[[Optional[int]], None]] = None,
    ):
        self.min_val = min_val
        self.max_val = max_val

        def thevalidator(v: Optional[int]):
            if self.min_val is not None and v is not None and v < self.min_val:
                raise ValidationError(
                    _("Value must be at least {min_val}.").format(
                        min_val=self.min_val
                    )
                )
            if self.max_val is not None and v is not None and v > self.max_val:
                raise ValidationError(
                    _("Value must be at most {max_val}.").format(
                        max_val=self.max_val
                    )
                )
            if validator:
                validator(v)

        super().__init__(
            key=key,
            label=label,
            var_type=int,
            description=description,
            default=default,
            value=value,
            validator=thevalidator,
        )

    def to_dict(self, include_value: bool = False) -> Dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data.update({"min_val": self.min_val, "max_val": self.max_val})
        return data
