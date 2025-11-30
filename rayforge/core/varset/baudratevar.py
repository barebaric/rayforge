from typing import Optional
from .intvar import IntVar, ValidationError
from ...machine.transport.serial import SerialTransport


def baud_rate_validator(rate: Optional[int]):
    """Raises ValidationError if the baud rate is not a standard one."""
    if rate is None:
        raise ValidationError(_("Baud rate cannot be empty."))
    if rate not in SerialTransport.list_baud_rates():
        raise ValidationError(
            _("'{rate}' is not a standard baud rate.").format(rate=rate)
        )


class BaudrateVar(IntVar):
    """A Var subclass for serial port baud rates, for use with a dropdown."""

    def __init__(
        self,
        key: str,
        label: str = _("Baud Rate"),
        description: Optional[str] = _("Connection speed in bits per second"),
        default: Optional[int] = 115200,
        value: Optional[int] = None,
        min_val: Optional[int] = None,
        max_val: Optional[int] = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            # Provide sensible, non-None bounds for serialization
            min_val=300,
            max_val=4000000,
            validator=baud_rate_validator,
        )
