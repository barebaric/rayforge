"""
GRBL Simple Serial Driver -- ping-pong protocol without buffer counting.

This driver uses the simplest possible communication strategy: send one
G-code line, wait for the ``ok`` acknowledgement, then send the next.
There is no character-counting flow control, no deadlock detection, and
no buffer management.  This makes it extremely robust at the cost of
lower throughput compared to the advanced ``GrblSerialDriver``.

The protocol itself runs in Rust via the ``raygeo`` package; this
module only supplies the Rayforge-side setup and GRBL device settings.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import raygeo.driver.grbl.simple
import serial
from raygeo.driver.grbl.parser import error_code_to_device_error

from ....core.varset import (
    BaudrateVar,
    SerialPortVar,
    Var,
    VarSet,
)
from ...transport.serial import (
    SerialPortPermissionError,
    SerialTransport,
    resolve_serial_port,
)
from ..driver import (
    DeviceError,
    DriverPrecheckError,
    DriverSetupError,
)
from ..session_state import error_from_session_state
from .grbl_probe import probe_grbl_device
from .grbl_serial import GrblSerialDriver
from .grbl_util import (
    apply_setting_to_varset,
    get_grbl_setting_varsets,
)

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


class GrblSerialSimpleDriver(GrblSerialDriver):
    """
    A minimal GRBL serial driver using ping-pong communication.

    Sends one G-code line at a time and waits for ``ok`` before
    proceeding.  No character-counting, no buffer management, no
    deadlock detection.  Ideal for users experiencing communication
    issues with the advanced driver.
    """

    NATIVE = raygeo.driver.grbl.simple.GrblSerialSimpleDriver
    DISCOVERY = None

    label = _("GRBL (USB/Serial — compatibility mode)")
    subtitle = _(
        "Only needed if the standard serial connection misbehaves "
        "(ping-pong protocol, no buffer counting)"
    )
    supports_settings = True
    supports_probing = True
    supports_unit_detection = True

    @classmethod
    def precheck(cls, **kwargs: Any) -> None:
        """Checks for systemic serial port issues before setup."""
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

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> tuple["DeviceProfile", list[str]]:
        return await probe_grbl_device(cls, context, **kwargs)

    def build_config(self, kwargs: dict[str, Any]) -> dict[str, Any]:
        port = kwargs.get("port")
        baudrate = kwargs.get("baudrate", 115200)
        if not isinstance(port, str) or not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baudrate must be configured."))

        # Note that we intentionally do not check if the serial port
        # exists, as a missing port is a common occurrence when e.g.
        # the USB cable is not plugged in, and not a sign of
        # misconfiguration.
        if port.startswith("/dev/ttyS"):
            logger.warning(
                f"Port {port} is a hardware serial port, which is "
                f"unlikely for USB-based GRBL devices."
            )

        # The Rust session retries its fixed port internally, so a
        # 'vid:pid' spec can only be resolved once, here at setup.
        try:
            port = resolve_serial_port(port)
        except serial.SerialException as e:
            raise DriverSetupError(str(e)) from e

        return {
            "port": port,
            "baudrate": int(baudrate),
            "homing_timeout": self._estimate_homing_timeout(),
        }

    def get_setting_vars(self) -> list["VarSet"]:
        return get_grbl_setting_varsets()

    async def read_settings(self) -> None:
        pairs = await self._require_session().read_settings()

        known_varsets = self.get_setting_vars()
        key_to_varset_map = {
            var_key: varset
            for varset in known_varsets
            for var_key in varset.keys()  # noqa: SIM118
        }
        unknown_vars = VarSet(
            title=_("Unknown Settings"),
            description=_(
                "Settings reported by the device not in the standard list."
            ),
        )
        for key, value_str in pairs:
            target_varset = key_to_varset_map.get(key)
            if target_varset:
                apply_setting_to_varset(target_varset, key, value_str)
            else:
                unknown_vars.add(
                    Var(
                        key=key,
                        label=f"${key}",
                        var_type=str,
                        value=value_str,
                        description=_("Unknown setting from device"),
                    )
                )
        result = known_varsets
        if len(unknown_vars) > 0:
            result.append(unknown_vars)

        num_settings = sum(len(vs) for vs in result)
        logger.info(
            f"Driver settings read with {num_settings} settings.",
            extra={"log_category": "DRIVER_EVENT"},
        )
        self.settings_read.send(self, settings=result)

    async def write_setting(self, key: str, value: Any) -> None:
        if isinstance(value, bool):
            value = 1 if value else 0
        await self._require_session().write_setting(key, str(value))

    def get_error(self, error_code: str) -> DeviceError | None:
        """Returns error details for a given GRBL error code."""
        return error_from_session_state(error_code_to_device_error(error_code))
