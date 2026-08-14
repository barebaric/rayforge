from collections.abc import Callable
from typing import Any

from .intvar import IntVar


class SpeedVar(IntVar):
    """
    An IntVar representing a speed value (e.g. cut speed, travel speed).

    Hints the UI to apply unit conversion via SpeedSpinRow.
    """

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: int | None = None,
        value: int | None = None,
        min_val: int | None = None,
        max_val: int | None = None,
        role: str = "cut",
        validator: Callable[[int | None], None] | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        sensitive_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        self.role = role
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=min_val,
            max_val=max_val,
            validator=validator,
            visible_when=visible_when,
            sensitive_when=sensitive_when,
        )
