import asyncio
import inspect
import logging
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    cast,
)

import serial.serialutil

from ....context import RayforgeContext
from ....core.varset import (
    BaudrateVar,
    SerialPortVar,
    VarSet,
)
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import SerialTransport, TransportStatus
from ...transport.serial import SerialPortPermissionError
from ..driver import (
    Axis,
    DeviceConnectionError,
    DeviceState,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from .marlin_probe import probe_marlin_device
from .marlin_util import (
    gcode_to_p_number,
    is_boot_message,
    is_error_response,
    is_ok_response,
    m114_pos_re,
)

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class MarlinSerialDriver(Driver):
    label = _("Marlin (Serial)")
    subtitle = _("Marlin firmware via serial connection")
    supports_settings = False
    reports_granular_progress = True
    maturity = DriverMaturity.EXPERIMENTAL
    supports_probing = True

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._transport: Optional[SerialTransport] = None
        self._port: str = ""
        self._baudrate: int = 115200
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._ok_event = asyncio.Event()
        self._handshake_event = asyncio.Event()
        self._job_running = False
        self._is_cancelled = False
        self._on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None
        self._last_reported_op_index = -1
        self._response_lines: List[str] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._rx_buffer = ""
        self._boot_lines: List[str] = []

    @property
    def machine_space_wcs(self) -> str:
        return "MACHINE"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates")

    @property
    def resource_uri(self) -> Optional[str]:
        if self._port:
            return f"serial://{self._port}"
        return None

    @property
    def boot_lines(self) -> List[str]:
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
                    description=_("Serial port for the device"),
                ),
                BaudrateVar(
                    "baudrate",
                    choices=SerialTransport.list_baud_rates(),
                ),
            ]
        )

    @classmethod
    def create_encoder(cls, machine: "Machine") -> "OpsEncoder":
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    def get_setting_vars(self) -> List["VarSet"]:
        return [VarSet()]

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> Tuple["DeviceProfile", List[str]]:
        return await probe_marlin_device(cls, context, **kwargs)

    def _setup_implementation(self, **kwargs: Any) -> None:
        port = cast(str, kwargs.get("port", ""))
        baudrate = kwargs.get("baudrate", 115200)

        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baud rate must be configured."))

        self._port = port
        self._baudrate = int(baudrate)
        self._transport = SerialTransport(port, self._baudrate)
        self._transport.received.connect(self.on_serial_data_received)
        self._transport.status_changed.connect(self.on_serial_status_changed)

    def on_serial_status_changed(
        self, sender, status: TransportStatus, message: Optional[str] = None
    ):
        if status == TransportStatus.CONNECTED:
            logger.debug(
                "Suppressing transport-level CONNECTED. Driver will "
                "emit CONNECTED after handshake verification."
            )
            return
        logger.debug(
            f"Serial transport status changed: {status}, message: {message}"
        )
        self._update_connection_status(status, message)

    async def cleanup(self):
        logger.debug("Cleanup initiated.")
        self.keep_running = False
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None

        if self._connection_task:
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.warning(
                    f"Ignored exception in connection task during cleanup: {e}"
                )
            self._connection_task = None

        if self._transport:
            self._transport.received.disconnect(self.on_serial_data_received)
            self._transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
            if self._transport.is_connected:
                await self._transport.disconnect()

        await super().cleanup()
        logger.debug("Cleanup completed.")

    async def _connect_implementation(self):
        if self._connection_task and not self._connection_task.done():
            logger.warning(
                "Connect called with active connection task. Cleaning up."
            )
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass

        if not self._transport:
            logger.error(
                "Cannot connect: Transport not initialized "
                "(check port settings)."
            )
            self._update_connection_status(
                TransportStatus.ERROR, _("Port not configured")
            )
            return

        logger.debug("Connect initiated.")
        self._loop = asyncio.get_running_loop()
        self.keep_running = True
        self._is_cancelled = False
        self._job_running = False
        self._on_command_done = None
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        logger.debug("Entering _connection_loop.")
        while self.keep_running:
            logger.debug("Attempting connection...")

            try:
                transport = self._transport
                if not transport:
                    raise DriverSetupError("Transport not initialized")

                await transport.connect()
                logger.debug("Serial port opened. Waiting for Marlin start...")

                self._handshake_event.clear()
                try:
                    await asyncio.wait_for(
                        self._handshake_event.wait(), timeout=5.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "No 'start' received from Marlin. Port may "
                        "be a phantom COM port."
                    )
                    await transport.disconnect()
                    self._update_connection_status(
                        TransportStatus.ERROR,
                        _("No response from device"),
                    )
                    self._update_connection_status(TransportStatus.SLEEPING)
                    await asyncio.sleep(5)
                    continue

                logger.info("Connection established successfully.")

                await asyncio.sleep(1.0)

                self._update_connection_status(TransportStatus.CONNECTED)

                logger.debug("Connection verified. Starting M114 polling.")
                while transport.is_connected and self.keep_running:
                    if not self._job_running:
                        try:
                            await self._send_and_wait(
                                "M114", wait_for_ok=False
                            )
                        except ConnectionError as e:
                            logger.warning(
                                f"Connection lost during M114 poll: {e}"
                            )
                            break
                    await asyncio.sleep(2.0)

                    if not self.keep_running:
                        break

            except (serial.serialutil.SerialException, OSError) as e:
                logger.error(f"Connection error: {e}")
            except asyncio.CancelledError:
                logger.info("Connection loop cancelled.")
                break
            except Exception as e:
                logger.error(
                    f"Unexpected error in connection loop: {e}",
                    exc_info=True,
                )
                self._update_connection_status(TransportStatus.ERROR, str(e))
            finally:
                if self._transport and self._transport.is_connected:
                    logger.debug("Disconnecting transport in finally block")
                    await self._transport.disconnect()

            if not self.keep_running:
                break

            logger.debug("Connection lost. Reconnecting in 5s...")
            self._update_connection_status(TransportStatus.SLEEPING)
            await asyncio.sleep(5)

        logger.debug("Leaving _connection_loop.")

    async def _send_and_wait(
        self, command: str, wait_for_ok: bool = True
    ) -> List[str]:
        if not self._transport or not self._transport.is_connected:
            raise ConnectionError("Serial transport not connected")

        self._response_lines = []
        if wait_for_ok:
            self._ok_event.clear()

        cmd_str = command.strip()
        if cmd_str:
            logger.info(cmd_str, extra=self._log_extra("USER_COMMAND"))
        payload = (command + "\n").encode("utf-8")
        logger.debug(
            f"TX: {payload!r}",
            extra={
                "log_category": "RAW_IO",
                "direction": "TX",
                "data": payload,
            },
        )
        await self._transport.send(payload)

        if wait_for_ok:
            try:
                await asyncio.wait_for(self._ok_event.wait(), 30.0)
            except asyncio.TimeoutError as e:
                raise ConnectionError(
                    f"Command '{command}' not confirmed"
                ) from e

        return self._response_lines

    async def execute_interactive_command(self, command: str) -> List[str]:
        """
        Send a command and return its response lines.

        Used during probing to query M115, M211, M503 etc.
        Sets ``_job_running = True`` to suppress M114 status
        polling while the command is in flight.
        """
        if not self._transport or not self._transport.is_connected:
            raise ConnectionError("Serial transport not connected")

        was_job_running = self._job_running
        self._job_running = True
        try:
            return await self._send_and_wait(command)
        finally:
            self._job_running = was_job_running

    def _start_job(
        self,
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ):
        self._is_cancelled = False
        self._job_running = True
        self._on_command_done = on_command_done
        self._last_reported_op_index = -1

    async def _stream_gcode(
        self,
        gcode_lines: List[str],
        machine_code_to_op_map: Optional[Dict[int, int]] = None,
    ):
        logger.debug(
            f"Starting Marlin streaming job with {len(gcode_lines)} lines."
        )
        try:
            for line_idx, line in enumerate(gcode_lines):
                if self._is_cancelled:
                    logger.info("Job cancelled. Stopping G-code.")
                    break

                line = line.strip()
                if not line:
                    continue

                op_index = (
                    machine_code_to_op_map.get(line_idx)
                    if machine_code_to_op_map
                    else None
                )

                await self._send_and_wait(line)

                if self._on_command_done and op_index is not None:
                    for i in range(
                        self._last_reported_op_index + 1,
                        op_index + 1,
                    ):
                        try:
                            result = self._on_command_done(i)
                            if inspect.isawaitable(result):
                                asyncio.ensure_future(result)
                        except Exception as e:
                            logger.error(
                                "Error in on_command_done callback",
                                exc_info=e,
                            )
                    self._last_reported_op_index = op_index

        except (
            asyncio.CancelledError,
            ConnectionError,
            DeviceConnectionError,
        ) as e:
            logger.warning(f"Job interrupted: {e!r}")
            if not self._is_cancelled:
                logger.info(f"Calling cancel() due to interruption: {e!r}")
                await self.cancel()
            if isinstance(e, asyncio.CancelledError):
                raise
        finally:
            self._job_running = False
            self._on_command_done = None
            self.job_finished.send(self)
            logger.debug("G-code streaming finished.")

    async def run(
        self,
        encoded: EncodedOutput,
        doc: "Doc",
        ops: "Ops",
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        self._start_job(on_command_done)

        mapping = encoded.op_map.machine_code_to_op if encoded.op_map else None
        gcode_lines = encoded.text.splitlines()

        try:
            await self._stream_gcode(gcode_lines, mapping)
        except DeviceConnectionError as e:
            logger.warning(
                f"Job terminated due to device error: {e}. "
                "Connection remains active."
            )

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip() for line in machine_code.splitlines() if line.strip()
        ]
        if not lines:
            return
        self._start_job()
        try:
            await self._stream_gcode(lines)
        except DeviceConnectionError as e:
            logger.warning(
                f"Raw G-code terminated due to device error: {e}. "
                "Connection remains active."
            )

    async def cancel(self) -> None:
        logger.debug("Cancel command initiated.")
        job_was_running = self._job_running
        self._is_cancelled = True
        self._job_running = False
        self._on_command_done = None

        if self._transport and self._transport.is_connected:
            logger.info("Sending M410 (Quick Stop) to device.")
            try:
                payload = b"M410\n"
                logger.debug(
                    f"TX: {payload!r}",
                    extra={
                        "log_category": "RAW_IO",
                        "direction": "TX",
                        "data": payload,
                    },
                )
                await self._transport.send(payload)
            except ConnectionError as e:
                logger.warning(f"Failed to send M410: {e}")

        if not self._transport:
            raise ConnectionError("Serial transport not initialized")

        if job_was_running:
            self.job_finished.send(self)

    async def set_hold(self, hold: bool = True) -> None:
        logger.warning(
            "Marlin does not support reliable pause/resume "
            "on 8-bit boards. set_hold is a best-effort no-op."
        )

    async def home(self, axes: Optional[Axis] = None) -> None:
        dialect = self.dialect
        if axes is None:
            await self._send_and_wait(dialect.home_all)
        else:
            for axis in axes:
                cmd = dialect.home_axis.format(axis_letter=axis.name)
                await self._send_and_wait(cmd)
        self.state.error = None
        self.state_changed.send(self, state=self.state)

    async def move_to(self, pos_x, pos_y) -> None:
        dialect = self.dialect
        cmd = dialect.move_to.format(
            speed=1500, x=float(pos_x), y=float(pos_y)
        )
        await self._send_and_wait(cmd)

    async def jog(self, speed: int, **deltas: float) -> None:
        dialect = self.dialect
        parts = [dialect.jog.format(speed=speed)]

        for axis_name, distance in deltas.items():
            parts.append(f"{axis_name.upper()}{distance}")

        if len(parts) == 1:
            return

        cmd = " ".join(parts)
        await self._send_and_wait(cmd)
        await self._send_and_wait("G90")

    async def select_tool(self, tool_number: int) -> None:
        dialect = self.dialect
        cmd = dialect.tool_change.format(tool_number=tool_number)
        await self._send_and_wait(cmd)

    async def clear_alarm(self) -> None:
        dialect = self.dialect
        await self._send_and_wait(dialect.clear_alarm)
        self.state.error = None
        self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        dialect = self.dialect
        if percent <= 0:
            cmd = dialect.laser_off
        else:
            power_abs = percent * head.max_power
            cmd = dialect.laser_on.format(power=power_abs)
        await self._send_and_wait(cmd)

    async def set_focus_power(self, head: "Laser", percent: float) -> None:
        dialect = self.dialect
        if percent <= 0:
            cmd = dialect.laser_off
        else:
            power_abs = percent * head.max_power
            cmd = dialect.focus_laser_on.format(power=power_abs)
        await self._send_and_wait(cmd)

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        return True

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def read_settings(self) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def write_setting(self, key: str, value: Any) -> None:
        raise NotImplementedError(
            "Device settings not implemented for this driver"
        )

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        p_num = gcode_to_p_number(wcs_slot)
        if p_num is None:
            raise ValueError(f"Invalid WCS slot: {wcs_slot}")
        dialect = self.dialect
        cmd = dialect.set_wcs_offset.format(p_num=p_num, x=x, y=y, z=z)
        await self._send_and_wait(cmd)

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        raise NotImplementedError(
            "Reading all WCS offsets is not supported by Marlin."
        )

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        raise NotImplementedError(
            "Probing is not implemented for the Marlin driver."
        )

    def on_serial_data_received(self, sender, data: bytes):
        logger.debug(
            f"RX: {data!r}",
            extra={
                "log_category": "RAW_IO",
                "direction": "RX",
                "data": data,
            },
        )

        self._rx_buffer += data.decode("utf-8", errors="replace")
        while "\n" in self._rx_buffer:
            line, self._rx_buffer = self._rx_buffer.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self._process_line(line)

    def _process_line(self, line: str):
        if is_ok_response(line):
            logger.info("ok", extra=self._log_extra("MACHINE_RESPONSE"))
            self._ok_event.set()
            return

        if is_error_response(line):
            logger.error(
                f"Marlin error: {line}",
                extra=self._log_extra("ERROR"),
            )
            self._response_lines.append(line)
            self._ok_event.set()
            return

        if line == "start":
            logger.debug("Received 'start' handshake from Marlin.")
            self._handshake_event.set()
            return

        if is_boot_message(line):
            self._boot_lines.append(line)
            logger.info(
                line,
                extra=self._log_extra("MACHINE_EVENT"),
            )
            return

        match = m114_pos_re.search(line)
        if match:
            x = float(match.group(1))
            y = float(match.group(2))
            z = float(match.group(3))
            old_pos = self.state.machine_pos
            new_pos = (x, y, z)
            if new_pos != old_pos:
                self.state.machine_pos = new_pos
                self.state.work_pos = new_pos
                old_status = self.state.status
                if self.state.status == DeviceStatus.UNKNOWN:
                    self.state.status = DeviceStatus.IDLE
                if self.state != DeviceState():
                    if self.state.status != old_status:
                        logger.info(
                            f"Device state changed: {self.state.status.name}",
                            extra=self._log_extra("STATE_CHANGE"),
                        )
                    self.state_changed.send(self, state=self.state)
            logger.info(
                f"M114: X:{x} Y:{y} Z:{z}",
                extra=self._log_extra("STATUS_POLL"),
            )
            return

        logger.info(line, extra=self._log_extra("MACHINE_EVENT"))
        self._response_lines.append(line)

    def _update_connection_status(
        self, status: TransportStatus, message: Optional[str] = None
    ):
        log_data = f"Connection status: {status.name}"
        if message:
            log_data += f" - {message}"
        logger.info(log_data, extra=self._log_extra("MACHINE_EVENT"))
        self.connection_status_changed.send(
            self, status=status, message=message
        )
        if status in [
            TransportStatus.DISCONNECTED,
            TransportStatus.ERROR,
        ]:
            if self.state.status != DeviceStatus.UNKNOWN:
                self.state.status = DeviceStatus.UNKNOWN
                logger.info(
                    f"Device state changed: {self.state.status.name}",
                    extra=self._log_extra("STATE_CHANGE"),
                )
                self.state_changed.send(self, state=self.state)
