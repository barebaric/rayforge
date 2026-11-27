"""GRBL serial driver backed by a raygeo session.

The entire protocol stack (flow control, streaming, stall detection
and deadlock recovery) runs in Rust via the ``raygeo`` package.
This shell only supplies the Rayforge-side setup, GRBL device
settings and probing; the session interface itself is shared with
every other driver through :class:`NativeDriver`.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

import raygeo.driver.grbl
import serial
from raygeo.driver.core.discovery.spec import DiscoverySpec, SerialRecognizer
from raygeo.driver.grbl.parser import (
    error_code_to_device_error,
    extract_device_name_from_output,
    is_grbl_output,
)

from ....core.varset import (
    BaudrateVar,
    IntVar,
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
from ..native import NativeDriver
from ..session_state import error_from_session_state
from .grbl_probe import probe_grbl_device
from .grbl_util import (
    apply_setting_to_varset,
    get_grbl_setting_varsets,
)

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile

logger = logging.getLogger(__name__)


class GrblSerialDriver(NativeDriver):
    """
    The GRBL serial driver with the entire protocol stack (flow
    control, streaming, stall detection and deadlock recovery)
    running in Rust via the ``raygeo`` package.

    This shell only translates between the Rayforge ``Driver``
    interface and the Rust session: dialects remain Rayforge data
    and are passed to the session as resolved command templates.
    """

    NATIVE = raygeo.driver.grbl.GrblSerialDriver

    label = _("GRBL (USB/Serial)")
    subtitle = _("GRBL-compatible USB/serial connection (recommended)")
    supports_settings = True
    supports_probing = True
    supports_unit_detection = True
    DISCOVERY = DiscoverySpec(
        serial=SerialRecognizer(
            label=lambda: _("GRBL device"),
            firmware="grbl",
            matches=is_grbl_output,
            name=extract_device_name_from_output,
        )
    )

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
                        "Detect and recover from serial communication "
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
            raise DriverSetupError(_("Baud rate must be configured."))

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

        config = {
            "port": port,
            "baudrate": int(baudrate),
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
        cached_rx_buffer_size = self.config.get("rx_buffer_size")
        if cached_rx_buffer_size is not None:
            config["cached_rx_buffer_size"] = int(cached_rx_buffer_size)

        return config

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
