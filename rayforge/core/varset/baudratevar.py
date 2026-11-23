from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .intvar import IntVar, ValidationError

STANDARD_BAUD_RATES: list[int] = [
    9600,
    19200,
    38400,
    57600,
    115200,
    230400,
    460800,
    921600,
    1000000,
    1843200,
]

DEFAULT_MIN_BAUD_RATE = 300
DEFAULT_MAX_BAUD_RATE = 4000000


def validate_baud_rate(rate: int | None):
    """Raises ValidationError if the baud rate is empty."""
    if rate is None:
        raise ValidationError(_("Baud rate cannot be empty."))


class BaudrateVar(IntVar):
    """
    A Var subclass for serial port baud rates, for use with a dropdown.

    ``choices`` holds the rates offered as shortcuts in the dropdown.
    Any rate within the min/max bounds is accepted as well, so users
    can enter a custom rate for their device.
    """

    display_name = _("Baud Rate")

    def __init__(
        self,
        key: str,
        label: str = _("Baud Rate"),
        description: str | None = _("Connection speed in bits per second"),
        default: int | None = 115200,
        value: int | None = None,
        min_val: int | None = None,
        max_val: int | None = None,
        choices: list[int] | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        self.choices: list[int] = (
            choices if choices is not None else list(STANDARD_BAUD_RATES)
        )
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            # Provide sensible, non-None bounds for serialization
            min_val=(DEFAULT_MIN_BAUD_RATE if min_val is None else min_val),
            max_val=(DEFAULT_MAX_BAUD_RATE if max_val is None else max_val),
            validator=validate_baud_rate,
            visible_when=visible_when,
        )

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data["choices"] = self.choices
        return data
