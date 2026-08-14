from collections.abc import Callable
from typing import Any

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
        description: str | None = None,
        default: float | None = None,
        value: float | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
        extra_validator: Callable[[float], None] | None = None,
        digits: int | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
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
            digits=digits,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )
