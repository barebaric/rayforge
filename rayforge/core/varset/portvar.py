from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .intvar import IntVar, ValidationError


def port_validator(port: int | None):
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
        description: str | None = None,
        default: int | None = None,
        value: int | None = None,
        min_val: int | None = 1,
        max_val: int | None = 65535,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            description=description,
            default=default,
            value=value,
            min_val=min_val,
            max_val=max_val,
            validator=port_validator,
            visible_when=visible_when,
        )
