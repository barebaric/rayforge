from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from .var import ValidationError, Var


def remote_device_validator(device: str | None):
    """Raises ValidationError if no device is selected."""
    if not device or not device.strip():
        raise ValidationError(_("Select a device on the daemon."))


class RemoteDeviceVar(Var[str]):
    """A device on a raydriver daemon, chosen from a live list.

    The value is the device reference (its serial port or host). The
    choices are populated by the adapter from the daemon identified
    by the sibling endpoint vars; the adapter also records the
    daemon-resolved protocol under a related key so capabilities can
    be derived without the driver naming a protocol.
    """

    display_name = _("Remote Device")

    def __init__(
        self,
        key: str,
        label: str,
        description: str | None = None,
        default: str | None = None,
        value: str | None = None,
        *,
        visible_when: "Callable[[dict[str, Any]], bool] | None" = None,
        required_when: "Callable[[dict[str, Any]], bool] | None" = None,
    ):
        super().__init__(
            key=key,
            label=label,
            var_type=str,
            description=description,
            default=default,
            value=value,
            validator=remote_device_validator,
            visible_when=visible_when,
            required_when=required_when,
        )
