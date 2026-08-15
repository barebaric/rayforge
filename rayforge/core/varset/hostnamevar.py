import ipaddress
from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import ValidationError, Var


def is_valid_hostname_or_ip(s: str) -> bool:
    if not isinstance(s, str):
        return False
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        pass
    if len(s) <= 2 or len(s) > 253 or s.endswith("."):
        return False
    labels = s.split(".")
    if (
        len(labels) == 4
        and any(label.isdigit() for label in labels)
        and not all(label.isdigit() for label in labels)
    ):
        return False
    if s.replace(".", "").isdigit():
        return False
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not all(c.isalnum() or c == "-" for c in label):
            return False
    return True


def hostname_validator(hostname: str | None):
    """Raises ValidationError if the string is not a valid hostname/IP."""
    if not hostname:
        raise ValidationError(_("Hostname or IP address cannot be empty."))
    if not is_valid_hostname_or_ip(hostname):
        raise ValidationError(_("Invalid hostname or IP address format."))


class HostnameVar(Var[str]):
    """A Var subclass for hostnames or IP addresses."""

    display_name = _("Hostname / IP")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        validator: Callable[[str | None], None] | None = hostname_validator,
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
            validator=validator,
            visible_when=visible_when,
        )
