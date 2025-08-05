from typing import Optional
from .var import Var, ValidationError
from ...machine.transport.validators import is_valid_hostname_or_ip


def hostname_validator(hostname: Optional[str]):
    """Raises ValidationError if the string is not a valid hostname/IP."""
    if not hostname:
        raise ValidationError(_("Hostname or IP address cannot be empty."))
    if not is_valid_hostname_or_ip(hostname):
        raise ValidationError(_("Invalid hostname or IP address format."))


class HostnameVar(Var[str]):
    """A Var subclass for hostnames or IP addresses."""

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
            validator=hostname_validator,
        )
