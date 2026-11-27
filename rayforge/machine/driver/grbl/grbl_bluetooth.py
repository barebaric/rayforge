import logging
from gettext import gettext as _
from typing import Any, cast

import raygeo.driver.grbl
from raygeo.driver.core.bluetooth import is_supported, normalize_address

from ....core.varset import (
    BluetoothAddressVar,
    IntVar,
    Var,
    VarSet,
)
from ..driver import (
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
)
from .grbl_serial import GrblSerialDriver

logger = logging.getLogger(__name__)


class GrblBluetoothDriver(GrblSerialDriver):
    """
    GRBL-compatible controller connected over Bluetooth Classic
    SPP/RFCOMM.

    The device must be paired in the operating system first.  The
    entire GRBL protocol stack runs in Rust via the ``raygeo``
    session; only the transport selection differs from the serial
    driver.
    """

    NATIVE = raygeo.driver.grbl.GrblBluetoothDriver

    label = _("GRBL (Bluetooth)")
    subtitle = _("GRBL-compatible controller over Bluetooth SPP")
    maturity = DriverMaturity.EXPERIMENTAL
    # Serial-port discovery is inherited from GrblSerialDriver but
    # makes no sense here: a Bluetooth connection has no serial port
    # and must not claim devices found on one.
    DISCOVERY = None

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        if not is_supported():
            raise DriverPrecheckError(
                _("Bluetooth is not supported on this platform.")
            )
        address = cast(str, kwargs.get("address", ""))
        if not address:
            raise DriverPrecheckError(
                _("Bluetooth address must be configured.")
            )
        try:
            normalize_address(address)
        except ValueError as e:
            raise DriverPrecheckError(
                _("Invalid Bluetooth address: '{address}'").format(
                    address=address
                )
            ) from e
        channel = kwargs.get("channel", 1)
        try:
            channel = int(channel)
        except (TypeError, ValueError) as e:
            raise DriverPrecheckError(
                _("Bluetooth channel must be a number.")
            ) from e
        if not 1 <= channel <= 30:
            raise DriverPrecheckError(
                _("Bluetooth channel must be between 1 and 30.")
            )

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                BluetoothAddressVar(
                    key="address",
                    label=_("Bluetooth device"),
                    description=_(
                        "A device paired in the operating system's "
                        "Bluetooth settings"
                    ),
                ),
                IntVar(
                    key="channel",
                    label=_("RFCOMM channel"),
                    description=_(
                        "Serial Port Profile channel; usually 1. "
                        "Windows resolves it automatically."
                    ),
                    default=1,
                    min_val=1,
                    max_val=30,
                ),
                Var(
                    key="poll_status_while_running",
                    label=_("Poll device status during jobs"),
                    description=_(
                        "Periodically query the device for position and "
                        "status while a job is running. Warning: Some "
                        "devices have trouble maintaining a stable "
                        "connection if this is used!"
                    ),
                    var_type=bool,
                    default=False,
                ),
                Var(
                    key="deadlock_detection",
                    label=_("Deadlock detection"),
                    description=_(
                        "Detect and recover from communication "
                        "deadlocks during jobs. If disabled, the driver "
                        "will simply wait for the machine to respond. "
                        "Disable if you experience false ALARM:3 errors."
                    ),
                    var_type=bool,
                    default=False,
                ),
                IntVar(
                    key="rx_buffer_size_override",
                    label=_("RX Buffer Size Override"),
                    description=_(
                        "Force a specific RX buffer size in bytes. "
                        "Set to 0 to auto-detect from the device."
                    ),
                    default=0,
                    min_val=0,
                    max_val=1024,
                ),
            ]
        )

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        address = cast(str, kwargs.get("address", ""))
        channel = int(kwargs.get("channel", 1) or 1)

        if not address:
            raise DriverSetupError(_("Bluetooth address must be configured."))
        try:
            address = normalize_address(address)
        except ValueError as e:
            raise DriverSetupError(str(e)) from e
        if not 1 <= channel <= 30:
            raise DriverSetupError(
                _("Bluetooth channel must be between 1 and 30.")
            )

        cached_rx_buffer_size = self.config.get("rx_buffer_size")
        config = {
            "address": address,
            "channel": channel,
            "poll_status_while_running": bool(
                kwargs.get("poll_status_while_running", False)
            ),
            "deadlock_detection": bool(
                kwargs.get("deadlock_detection", False)
            ),
            "rx_buffer_size_override": int(
                kwargs.get("rx_buffer_size_override", 0) or 0
            ),
            "homing_timeout": self._estimate_homing_timeout(),
        }
        if cached_rx_buffer_size is not None:
            config["cached_rx_buffer_size"] = int(cached_rx_buffer_size)
        return config
