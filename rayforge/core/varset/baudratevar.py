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


def validate_baud_rate(rate: int | None, choices: list[int]):
    """Raises ValidationError if the baud rate is not in the choices list."""
    if rate is None:
        raise ValidationError(_("Baud rate cannot be empty."))
    if rate not in choices:
        raise ValidationError(
            _("'{rate}' is not a standard baud rate.").format(rate=rate)
        )


class BaudrateVar(IntVar):
    """A Var subclass for serial port baud rates, for use with a dropdown."""

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
            min_val=300,
            max_val=4000000,
            validator=lambda v: validate_baud_rate(v, self.choices),
            visible_when=visible_when,
        )

    def to_dict(self, include_value: bool = False) -> dict[str, Any]:
        data = super().to_dict(include_value=include_value)
        data["choices"] = self.choices
        return data
