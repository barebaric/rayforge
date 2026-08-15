from collections.abc import Callable
from typing import Any

from .floatvar import FloatVar


class AngleVar(FloatVar):
    """A FloatVar representing an angle in degrees.

    Hints the UI to render an :class:`AngleSpinRow` (degree-range spin
    button) rather than a plain spin.
    """

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: float | None = None,
        value: float | None = None,
        min_val: float | None = -360.0,
        max_val: float | None = 360.0,
        extra_validator: Callable[[float], None] | None = None,
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
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )
