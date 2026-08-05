from typing import Callable, Optional

from .floatvar import FloatVar


class LengthVar(FloatVar):
    """
    A FloatVar representing a length value (e.g. offset, overcut).

    Values are always stored in base units (mm). Hints the UI to apply
    unit conversion via LengthSpinRow.
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
    ):
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
