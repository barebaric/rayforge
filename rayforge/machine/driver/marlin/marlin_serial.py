"""Marlin firmware over serial, backed by a raydriver session.

The protocol stack runs in Rust via the ``raydriver`` package; this
shell only supplies the Rayforge-side setup, probing and the
capability overrides for operations Marlin does not support.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

import serial
from raydriver.core import SessionEvent
from raydriver.core.discovery.spec import DiscoverySpec, SerialRecognizer
from raydriver.marlin import MarlinSession
from raydriver.marlin.parser import (
    extract_marlin_banner_from_output,
    is_boot_message,
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
from ..driver import (
    Axis,
    DeviceStatus,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from ..session_driver import SessionDriverBase
from .marlin_probe import probe_marlin_device

if TYPE_CHECKING:
    from ....context import RayforgeContext
    from ...device.profile import DeviceProfile
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class MarlinSerialDriver(SessionDriverBase):
    """
    Marlin firmware over serial, with the entire protocol stack
    running in Rust via the ``raydriver`` package.

    This shell only translates between the Rayforge ``Driver``
    interface and the Rust session: dialects remain Rayforge data
    and are passed to the session as resolved command templates.
    """

    label = _("Marlin (Serial)")
    subtitle = _("Marlin firmware via serial connection")
    supports_settings = False
    maturity = DriverMaturity.EXPERIMENTAL
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

    def __init__(self, context: "RayforgeContext", machine: "Machine"):
        super().__init__(context, machine)
        self._boot_lines: list[str] = []

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def boot_lines(self) -> list[str]:
        """Boot banner lines received from the device, used by the
        probe for device naming."""
        return self._boot_lines

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

    def _setup_implementation(self, **kwargs: Any) -> None:
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)

        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        # The Rust session retries its fixed port internally, so a
        # 'vid:pid' spec can only be resolved once, here at setup.
        try:
            port = resolve_serial_port(port)
        except serial.SerialException as e:
            raise DriverSetupError(str(e)) from e

        config = {
            "port": port,
            "baudrate": int(baudrate),
        }
        self._session = MarlinSession(
            config=config,
            dialect=self._dialect_templates(),
            event_callback=self._on_session_event,
        )

    def _on_session_event(self, event: SessionEvent) -> None:
        if isinstance(event, SessionEvent.RawRx) and is_boot_message(
            event.line
        ):
            self._boot_lines.append(event.line)
        super()._on_session_event(event)

    def _on_connection_status_changed(
        self, status_str: str, message: str | None
    ) -> None:
        super()._on_connection_status_changed(status_str, message)
        if status_str in ("ERROR", "DISCONNECTED") and (
            self.state.status != DeviceStatus.UNKNOWN
        ):
            self.state.status = DeviceStatus.UNKNOWN
            self.state_changed.send(self, state=self.state)

    async def execute_interactive_command(self, command: str) -> list[str]:
        """Send a command and return its response lines.

        Used during probing to query M115, M211, M503 etc.  The
        terminating ``ok`` is filtered out, so callers receive the
        reply lines only.
        """
        lines = await super().execute_interactive_command(command)
        return [line for line in lines if line != "ok"]

    async def set_hold(self, hold: bool = True) -> None:
        logger.warning(
            "Marlin does not support reliable pause/resume "
            "on 8-bit boards. set_hold is a best-effort no-op."
        )

    async def jog(self, speed: int, **deltas: float) -> None:
        await super().jog(speed, **deltas)
        await self._require_session().execute_command("G90")

    async def clear_alarm(self) -> None:
        session = self._require_session()
        response = await session.execute_command(self.dialect.clear_alarm)
        has_error = any(
            line.startswith(("Error:", "error:")) for line in response
        )
        if not has_error:
            self.state.error = None
            self.state_changed.send(self, state=self.state)

    async def read_settings(self) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def read_wcs_offsets(self) -> dict[str, Pos]:
        raise NotImplementedError(
            "Reading all WCS offsets is not supported by Marlin."
        )

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Pos | None:
        raise NotImplementedError(
            "Probing is not implemented for the Marlin driver."
        )
