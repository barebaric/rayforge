from typing import Optional
from .var import Var, ValidationError


def port_validator(port: Optional[int]):
    """Raises ValidationError if port is not a valid network port."""
    if port is None:
        raise ValidationError(_("Port cannot be empty."))
    if not isinstance(port, int):
        raise ValidationError(_("Port must be a number."))
    if port < 1 or port > 65535:
        raise ValidationError(_("Port must be between 1 and 65535."))


class PortVar(Var[int]):
    """A Var subclass for network port numbers."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
        value: Optional[int] = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=int,
            description=description,
            default=default,
            value=value,
            validator=port_validator,
        )
