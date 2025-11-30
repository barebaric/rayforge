from typing import Optional
from .intvar import IntVar, ValidationError


def port_validator(port: Optional[int]):
    """Raises ValidationError if port is not a valid network port."""
    if port is None:
        raise ValidationError(_("Port cannot be empty."))
    if not isinstance(port, int):
        raise ValidationError(_("Port must be a number."))
    # The range check (1-65535) is handled by IntVar's validator logic
    # because we pass min_val and max_val to its constructor.


class PortVar(IntVar):
    """A Var subclass for network port numbers."""

    def __init__(
        self,
        key: str,
        label: str,
        description: Optional[str] = None,
        default: Optional[int] = None,
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
            min_val=1,
            max_val=65535,
            validator=port_validator,
        )
