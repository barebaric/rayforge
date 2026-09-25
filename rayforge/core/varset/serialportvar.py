import re
from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import ValidationError, Var

VIDPID_RE = re.compile(
    r"^(?:0[xX])?([0-9a-fA-F]{1,4}):(?:0[xX])?([0-9a-fA-F]{1,4})$"
)


def parse_vidpid(value: str | None) -> tuple[int, int] | None:
    """
    Parses a USB vendor/product ID spec such as '0403:6001'. Returns
    None when the value is not in that form (e.g. a device path).
    """
    if not value:
        return None
    match = VIDPID_RE.match(value.strip())
    if not match:
        return None
    return int(match.group(1), 16), int(match.group(2), 16)


def format_vidpid(vid: int, pid: int) -> str:
    """Formats a vendor/product ID pair in the canonical spec form."""
    return f"{vid:04x}:{pid:04x}"


def serial_port_validator(port: str | None):
    """Raises ValidationError if the serial port is not specified."""
    if not port or not port.strip():
        raise ValidationError(_("Serial port cannot be empty."))


class SerialPortVar(Var[str]):
    """A Var subclass for serial port names."""

    display_name = _("Serial Port")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
            validator=serial_port_validator,
            visible_when=visible_when,
        )
