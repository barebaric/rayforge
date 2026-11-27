"""Marlin firmware over serial, backed by the native raygeo driver.

The protocol stack runs in Rust via the ``raygeo`` package; this
binding only supplies the Rayforge-side setup, probing and the
presentation data.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import raygeo.driver.marlin
import serial
from raygeo.driver.core.discovery.spec import DiscoverySpec, SerialRecognizer
from raygeo.driver.marlin.parser import (
    extract_marlin_banner_from_output,
    is_marlin_output,
)

from ....core.varset import (
    BaudrateVar,
    SerialPortVar,
    VarSet,
)
from ...transport.serial import (
    SerialPortPermissionError,
    SerialTransport,
    resolve_serial_port,
)
from ..driver import DriverPrecheckError, DriverSetupError
from ..native import NativeDriver
from .marlin_probe import probe_marlin_device

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


class MarlinSerialDriver(NativeDriver):
    """
    Marlin firmware over serial, with the entire protocol stack
    running in Rust via the ``raygeo`` package.
    """

    NATIVE = raygeo.driver.marlin.MarlinSerialDriver

    label = _("Marlin (Serial)")
    subtitle = _("Marlin firmware via serial connection")
    supports_settings = False
    supports_probing = True
    supports_unit_detection = True
    DISCOVERY = DiscoverySpec(
        serial=SerialRecognizer(
            label=lambda: _("Marlin device"),
            firmware="marlin",
            matches=is_marlin_output,
            name=extract_marlin_banner_from_output,
        )
    )

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def boot_lines(self) -> list[str]:
        """Boot banner lines captured by the native driver."""
        if self._session is None:
            return []
        return self._session.boot_lines

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        try:
            SerialTransport.check_serial_permissions_globally()
        except SerialPortPermissionError as e:
            raise DriverPrecheckError(str(e)) from e

    @classmethod
    def get_setup_vars(cls) -> "VarSet":
        return VarSet(
            vars=[
                SerialPortVar(
                    key="port",
                    label=_("Port"),
                    description=(
                        _("Serial port or USB VID:PID (e.g. 0403:6001)")
                    ),
                ),
                BaudrateVar(
                    "baudrate",
                    choices=SerialTransport.list_baud_rates(),
                ),
            ]
        )

    def get_setting_vars(self) -> list["VarSet"]:
        return [VarSet()]

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        return await probe_marlin_device(cls, context, **kwargs)

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        port = kwargs.get("port")
        baudrate = kwargs.get("baudrate", 115200)

        if not isinstance(port, str) or not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        # The native driver retries its fixed port internally, so a
        # 'vid:pid' spec can only be resolved once, here at setup.
        try:
            port = resolve_serial_port(port)
        except serial.SerialException as e:
            raise DriverSetupError(str(e)) from e

        return {
            "port": port,
            "baudrate": int(baudrate),
        }
