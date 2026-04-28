from typing import Optional, Callable
from .intvar import IntVar


class SpeedVar(IntVar):
    """
    An IntVar representing a speed value (e.g. cut speed, travel speed).

    Hints the UI to apply unit conversion via UnitSpinRowHelper.
    """

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
        value: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
        role: str = "cut",
        validator: Optional[Callable[[Optional[int]], None]] = None,
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
        )
