from typing import Optional
from .var import Var, ValidationError


def serial_port_validator(port: Optional[str]):
    """Raises ValidationError if the serial port is not specified."""
    if not port:
        raise ValidationError(_("Serial port cannot be empty."))


class SerialPortVar(Var[str]):
    """A Var subclass for serial port names."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[str] = None,
        value: Optional[str] = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
            validator=serial_port_validator,
        )
