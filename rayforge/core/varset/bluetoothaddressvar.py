from collections.abc import Callable
from gettext import gettext as _
from typing import Any

from raygeo.driver.core.bluetooth import normalize_address

from .var import ValidationError, Var


def bluetooth_address_validator(address: str | None):
    """Raises ValidationError if the address is not a Bluetooth MAC."""
    if not address or not address.strip():
        raise ValidationError(_("Bluetooth address cannot be empty."))
    try:
        normalize_address(address)
    except ValueError as e:
        raise ValidationError(
            _("Invalid Bluetooth address: '{address}'").format(address=address)
        ) from e


class BluetoothAddressVar(Var[str]):
    """A Var subclass for Bluetooth device addresses."""

    display_name = _("Bluetooth Address")

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
            validator=bluetooth_address_validator,
            visible_when=visible_when,
        )
