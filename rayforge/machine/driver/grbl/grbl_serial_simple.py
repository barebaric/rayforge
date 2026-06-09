"""
GRBL Simple Serial Driver -- ping-pong protocol without buffer counting.

This driver uses the simplest possible communication strategy: send one
G-code line, wait for the ``ok`` acknowledgement, then send the next.
There is no character-counting flow control, no deadlock detection, and
no buffer management.  This makes it extremely robust at the cost of
lower throughput compared to the advanced ``GrblSerialDriver``.
"""

import asyncio
import logging
import re
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
)

from ....context import RayforgeContext
from ....core.varset import (
    BaudrateVar,
    SerialPortVar,
    Var,
    VarSet,
)
from ....pipeline.encoder.base import EncodedOutput, OpsEncoder
from ....pipeline.encoder.gcode import GcodeEncoder
from ...transport import SerialTransport, TransportStatus
from ...transport.serial import SerialPortPermissionError
from ..driver import (
    Axis,
    DeviceConnectionError,
    DeviceStatus,
    Driver,
    DriverMaturity,
    DriverPrecheckError,
    DriverSetupError,
    Pos,
)
from .grbl_probe import probe_grbl_device
from .grbl_util import (
    alarm_code_to_device_error,
    gcode_to_p_number,
    get_grbl_setting_varsets,
    grbl_setting_re,
    parse_grbl_parser_state,
    parse_state,
    prb_re,
    strip_gcode_comments,
    wcs_re,
)

if TYPE_CHECKING:
    from raygeo.ops import Ops

    from ....core.doc import Doc
    from ...device.profile import DeviceProfile
    from ...models.laser import Laser
    from ...models.machine import Machine

logger = logging.getLogger(__name__)


class _PingPongPending:
    """Tracks a single sent command awaiting its ``ok`` response."""

    __slots__ = ("command", "event", "response_lines", "error")

    def __init__(self, command: str):
        self.command = command
        self.event = asyncio.Event()
        self.response_lines: List[str] = []
        self.error: Optional[str] = None

    def set_result(self, error: Optional[str] = None):
        self.error = error
        self.event.set()

    async def wait(self, timeout: float = 30.0) -> List[str]:
        await asyncio.wait_for(self.event.wait(), timeout=timeout)
        if self.error is not None:
            raise DeviceConnectionError(self.error)
        return self.response_lines


class GrblSerialSimpleDriver(Driver):
    """
    A minimal GRBL serial driver using ping-pong communication.

    Sends one G-code line at a time and waits for ``ok`` before
    proceeding.  No character-counting, no buffer management, no
    deadlock detection.  Ideal for users experiencing communication
    issues with the advanced driver.
    """

    label = _("GRBL (Serial Simple)")
    subtitle = _(
        "GRBL serial with simple ping-pong protocol (no buffer counting)"
    )
    supports_settings = True
    reports_granular_progress = True
    supports_probing = True
    maturity = DriverMaturity.EXPERIMENTAL

    _ok_re = re.compile(rb"ok\r*\n")
    _error_re = re.compile(rb"error:(\d+)\r*\n")
    _status_re = re.compile(rb"<([^>]+)>")
    _line_re = re.compile(rb"([^\r\n]+)\r*\n")

    def __init__(self, context: RayforgeContext, machine: "Machine"):
        super().__init__(context, machine)
        self._transport: Optional[SerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._cmd_lock = asyncio.Lock()
        self._is_cancelled = False
        self._job_running = False
        self._raw_grbl_status: DeviceStatus = DeviceStatus.UNKNOWN
        self._handshake_received = asyncio.Event()
        self._on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None
        self._pending: Optional[_PingPongPending] = None
        self._line_buffer: bytearray = bytearray()
        self._current_op_index: int = -1

    @property
    def machine_space_wcs(self) -> str:
        return "G53"

    @property
    def machine_space_wcs_display_name(self) -> str:
        return _("Machine Coordinates (G53)")

    @property
    def resource_uri(self) -> Optional[str]:
        if self._transport and self._transport.port:
            return f"serial://{self._transport.port}"
        return None

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
    def create_encoder(cls, machine: "Machine") -> OpsEncoder:
        assert machine.dialect is not None
        return GcodeEncoder(machine.dialect)

    @classmethod
    async def probe(
        cls, context: "RayforgeContext", **kwargs: Any
    ) -> Tuple["DeviceProfile", List[str]]:
        return await probe_grbl_device(cls, context, **kwargs)

    def _setup_implementation(self, **kwargs: Any) -> None:
        port = kwargs.get("port", "")
        baudrate = kwargs.get("baudrate", 115200)

        if not port:
            raise DriverSetupError(_("Port must be configured."))
        if not baudrate:
            raise DriverSetupError(_("Baudrate must be configured."))

        self._transport = SerialTransport(str(port), int(baudrate))
        self._transport.received.connect(self._on_serial_data)

    async def cleanup(self) -> None:
        self.keep_running = False
        self._is_cancelled = False
        self._job_running = False
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                pass
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._connection_task = None
        self._poll_task = None
        if self._transport:
            self._transport.received.disconnect(self._on_serial_data)
            if self._transport.is_connected:
                await self._transport.disconnect()
        await super().cleanup()

    def _log_extra(self, category: str) -> Dict[str, Optional[str]]:
        return {
            "log_category": category,
            "machine_id": self._machine.id if self._machine else None,
        }

    async def _connect_implementation(self) -> None:
        if self._connection_task and not self._connection_task.done():
            self._connection_task.cancel()
        self.keep_running = True
        self._connection_task = asyncio.ensure_future(
            self._connection_loop()
        )

    async def _connection_loop(self) -> None:
        transport = self._transport
        if not transport:
            return

        while self.keep_running:
            try:
                await transport.connect()
                self._update_connection_status(
                    TransportStatus.CONNECTED, "Connected"
                )
                self._handshake_received.clear()

                await self._send_raw(b"?\n")
                try:
                    await asyncio.wait_for(
                        self._handshake_received.wait(), timeout=3.0
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Handshake timeout. Retrying connection."
                    )
                    await transport.disconnect()
                    self._update_connection_status(
                        TransportStatus.SLEEPING,
                        "Handshake failed, retrying...",
                    )
                    await asyncio.sleep(5)
                    continue

                await self._start_polling()

                while transport.is_connected and self.keep_running:
                    await asyncio.sleep(0.5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._update_connection_status(
                    TransportStatus.ERROR, str(e)
                )

            if self.keep_running:
                await transport.disconnect()
                self._update_connection_status(
                    TransportStatus.SLEEPING, "Reconnecting..."
                )
                await asyncio.sleep(5)

    async def _start_polling(self) -> None:
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
        self._poll_task = asyncio.ensure_future(self._poll_loop())

    async def _poll_loop(self) -> None:
        transport = self._transport
        if not transport:
            return
        try:
            while transport.is_connected and self.keep_running:
                if not self._job_running:
                    async with self._cmd_lock:
                        if transport.is_connected:
                            await self._send_raw(b"?\n")
                await asyncio.sleep(1.0)
        except asyncio.CancelledError:
            pass

    def _update_connection_status(
        self, status: TransportStatus, message: str = ""
    ) -> None:
        if status == TransportStatus.CONNECTED:
            self.state.status = DeviceStatus.IDLE
        elif status in (
            TransportStatus.ERROR,
            TransportStatus.SLEEPING,
        ):
            self.state.status = DeviceStatus.UNKNOWN
        self.connection_status_changed.send(
            self, status=status, message=message
        )

    async def _send_raw(self, data: bytes) -> None:
        if self._transport and self._transport.is_connected:
            await self._transport.send(data)

    async def _ping_pong(
        self, command: str, timeout: float = 30.0
    ) -> List[str]:
        """
        Send *command* and wait for ``ok`` / ``error:`` response.
        Returns collected response lines.
        """
        if not self._transport or not self._transport.is_connected:
            raise ConnectionError("Serial transport not connected")

        pending = _PingPongPending(command)
        self._pending = pending
        payload = (command.strip() + "\n").encode("utf-8")
        logger.info(command.strip(), extra=self._log_extra("USER_COMMAND"))
        await self._transport.send(payload)

        try:
            return await pending.wait(timeout=timeout)
        except asyncio.TimeoutError:
            logger.error(
                f"Ping-pong timeout on '{command}' after {timeout}s"
            )
            raise DeviceConnectionError(
                f"Device did not respond to '{command}' within "
                f"{timeout}s"
            )
        except asyncio.CancelledError:
            raise
        finally:
            self._pending = None

    async def _ping_pong_no_wait(self, command: str) -> None:
        """Send command without waiting for response (fire and forget)."""
        if not self._transport or not self._transport.is_connected:
            raise ConnectionError("Serial transport not connected")
        payload = (command.strip() + "\n").encode("utf-8")
        logger.info(command.strip(), extra=self._log_extra("USER_COMMAND"))
        await self._transport.send(payload)

    def _on_serial_data(self, sender, data: bytes) -> None:
        """Parse incoming serial data and dispatch responses."""
        self._line_buffer.extend(data)
        self._process_line_buffer()

    def _process_line_buffer(self) -> None:
        """Process complete lines from the receive buffer."""
        while b"\n" in self._line_buffer:
            idx = self._line_buffer.index(b"\n") + 1
            raw_line = bytes(self._line_buffer[:idx])
            del self._line_buffer[:idx]

            for line in (
                raw_line.decode("utf-8", errors="replace")
                .strip()
                .splitlines()
            ):
                if not line:
                    continue
                self._handle_response_line(line)

    def _handle_response_line(self, line: str) -> None:
        """Route a single decoded response line."""
        if line == "ok":
            if self._pending:
                self._pending.set_result()
            return

        if line.startswith("error:"):
            if self._pending:
                self._pending.set_result(error=line)
            return

        if line.startswith("<") and line.endswith(">"):
            self._handle_status_report(line)
            return

        if line.startswith("ALARM:"):
            logger.warning(
                line, extra=self._log_extra("MACHINE_EVENT")
            )
            alarm_code = line.split(":")[1].strip()
            self.state.error = alarm_code_to_device_error(
                alarm_code
            )
            self.state.status = DeviceStatus.ALARM
            self.state_changed.send(self, state=self.state)
            if self._pending:
                self._pending.set_result(error=line)
            return

        if line.startswith("Grbl ") or line.startswith("grbl "):
            self._handshake_received.set()
            return

        if line.startswith("["):
            if line.startswith("[OPT:"):
                logger.debug(f"Received OPT info: {line}")
            elif line.startswith("[VER:"):
                logger.debug(f"Received version: {line}")
            if self._pending:
                self._pending.response_lines.append(line)
            return

        if self._pending:
            self._pending.response_lines.append(line)

    def _handle_status_report(self, report: str) -> None:
        """Process a GRBL status report like <Idle|MPos:0,0,0>."""
        self._handshake_received.set()
        state = parse_state(
            report, self.state, lambda message: logger.info(message)
        )

        self._raw_grbl_status = state.status

        if self._job_running and state.status == DeviceStatus.IDLE:
            state.status = DeviceStatus.RUN

        self.state.status = state.status
        if state.error is not None:
            self.state.error = state.error

        self.state_changed.send(self, state=self.state)

    def _start_job(
        self,
        on_command_done: Optional[
            Callable[[int], Union[None, Awaitable[None]]]
        ] = None,
    ) -> None:
        self._is_cancelled = False
        self._job_running = True
        self._job_exception: Optional[Exception] = None
        self._on_command_done = on_command_done
        self._current_op_index = -1
        self.state.status = DeviceStatus.RUN
        self.state_changed.send(self, state=self.state)

    async def _stream_gcode_ping_pong(
        self,
        gcode_lines: List[str],
        op_map: Optional[Dict[int, int]] = None,
    ) -> None:
        """
        Stream G-code using strict ping-pong: send one line,
        wait for ok, repeat.
        """
        total = len(gcode_lines)
        logger.debug(
            f"Starting ping-pong streaming job with {total} lines."
        )
        job_completed_successfully = False
        sent_count = 0
        try:
            for line_idx, line in enumerate(gcode_lines):
                if (
                    self._is_cancelled
                    or self.state.status == DeviceStatus.ALARM
                ):
                    logger.info("Job cancelled or alarm. Stopping.")
                    break

                stripped = strip_gcode_comments(line).strip()
                if not stripped:
                    continue

                new_op = (
                    op_map.get(line_idx) if op_map else None
                )
                if new_op is not None and new_op != self._current_op_index:
                    self._current_op_index = new_op
                    if self._on_command_done:
                        result = self._on_command_done(new_op)
                        if asyncio.iscoroutine(result):
                            await result

                await self._ping_pong(stripped)
                sent_count += 1

            if (
                sent_count == total
                and not self._is_cancelled
                and self.state.status != DeviceStatus.ALARM
            ):
                job_completed_successfully = True
        except DeviceConnectionError as e:
            logger.warning(f"Job interrupted: {e}")
        except Exception as e:
            logger.error(f"Unexpected streaming error: {e!r}", exc_info=True)
        finally:
            self._job_running = False
            self._on_command_done = None
            if job_completed_successfully:
                self.job_finished.send(self)
                logger.debug(
                    f"Ping-pong streaming finished ({sent_count}/{total})."
                )
            elif self._is_cancelled:
                logger.debug(
                    f"Ping-pong streaming cancelled at "
                    f"{sent_count}/{total}."
                )
            else:
                self.job_finished.send(self)

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
        mapping = (
            encoded.op_map.machine_code_to_op if encoded.op_map else None
        )
        gcode_lines = encoded.text.splitlines()
        try:
            await self._stream_gcode_ping_pong(gcode_lines, mapping)
        except DeviceConnectionError as e:
            logger.warning(f"Job terminated: {e}")

    async def run_raw(self, machine_code: str) -> None:
        lines = [
            line.strip()
            for line in machine_code.splitlines()
            if line.strip()
        ]
        if not lines:
            return
        self._start_job()
        try:
            await self._stream_gcode_ping_pong(lines)
        except DeviceConnectionError as e:
            logger.warning(f"Raw G-code terminated: {e}")

    async def cancel(self) -> None:
        logger.debug("Cancel command initiated.")
        job_was_running = self._job_running
        self._is_cancelled = True
        self._job_running = False
        self._on_command_done = None

        if self._transport and self._transport.is_connected:
            logger.info("Sending Soft Reset (Ctrl-X) to device.")
            await self._transport.send(b"\x18")
            self._line_buffer.clear()
            self._pending = None
            if job_was_running:
                self.job_finished.send(self)
        else:
            raise ConnectionError("Serial transport not initialized")

    async def _execute_command(self, command: str) -> List[str]:
        """Send a command using ping-pong and return response lines."""
        self._is_cancelled = False
        async with self._cmd_lock:
            return await self._ping_pong(command)

    async def execute_interactive_command(
        self, command: str
    ) -> List[str]:
        """Send a command and await its full response."""
        if not self._transport or not self._transport.is_connected:
            raise ConnectionError("Serial transport not connected")
        return await self._execute_command(command)

    async def set_hold(self, hold: bool = True) -> None:
        self._is_cancelled = False
        realtime = b"!" if hold else b"~"
        if self._transport and self._transport.is_connected:
            await self._transport.send(realtime)

    def can_home(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def home(self, axes: Optional[Axis] = None) -> None:
        dialect = self.dialect
        if axes is None:
            await self._execute_command(dialect.home_all)
        else:
            for axis in axes:
                cmd = dialect.home_axis.format(
                    axis_letter=axis.name
                )
                await self._execute_command(cmd)

        active_wcs = self._machine.active_wcs
        temp_wcs = "G55" if active_wcs == "G54" else "G54"
        await self._execute_command("G4 P0.01")
        await self._execute_command(temp_wcs)
        await self._execute_command(active_wcs)
        self.state.error = None
        self.state_changed.send(self, state=self.state)

    async def move_to(self, pos_x, pos_y) -> None:
        dialect = self.dialect
        cmd = dialect.move_to.format(
            speed=1500, x=float(pos_x), y=float(pos_y)
        )
        await self._execute_command(cmd)

    async def select_tool(self, tool_number: int) -> None:
        dialect = self.dialect
        cmd = dialect.tool_change.format(tool_number=tool_number)
        await self._execute_command(cmd)

    async def clear_alarm(self) -> None:
        dialect = self.dialect
        response = await self._execute_command(dialect.clear_alarm)
        has_error = any(
            line.startswith("error:") for line in response
        )
        if not has_error:
            self.state.error = None
            self.state_changed.send(self, state=self.state)

    async def set_power(self, head: "Laser", percent: float) -> None:
        dialect = self.dialect
        if percent <= 0:
            cmd = dialect.laser_off
        else:
            power_abs = percent * head.max_power
            cmd = dialect.laser_on.format(power=power_abs)
        await self._execute_command(cmd)

    async def set_focus_power(
        self, head: "Laser", percent: float
    ) -> None:
        dialect = self.dialect
        if percent <= 0:
            cmd = dialect.laser_off
        else:
            power_abs = percent * head.max_power
            cmd = dialect.focus_laser_on.format(power=power_abs)
        await self._execute_command(cmd)

    def can_jog(self, axis: Optional[Axis] = None) -> bool:
        return True

    async def jog(self, speed: int, **deltas: float) -> None:
        dialect = self.dialect
        cmd_parts = [dialect.jog.format(speed=speed)]
        for axis_name, distance in deltas.items():
            cmd_parts.append(f"{axis_name.upper()}{distance}")
        if len(cmd_parts) == 1:
            return
        cmd = " ".join(cmd_parts)
        await self._execute_command(cmd)

    def get_setting_vars(self) -> List["VarSet"]:
        return get_grbl_setting_varsets()

    async def read_settings(self) -> None:
        response_lines = await self.execute_interactive_command("$$")
        known_varsets = self.get_setting_vars()
        key_to_varset_map = {
            var_key: varset
            for varset in known_varsets
            for var_key in varset.keys()
        }
        unknown_vars = VarSet(
            title=_("Unknown Settings"),
            description=_(
                "Settings reported by the device not in the "
                "standard list."
            ),
        )
        for line in response_lines:
            match = grbl_setting_re.match(line)
            if match:
                key, value_str = match.groups()
                target_varset = key_to_varset_map.get(key)
                if target_varset:
                    target_varset[key] = value_str
                else:
                    unknown_vars.add(
                        Var(
                            key=key,
                            label=f"${key}",
                            var_type=str,
                            value=value_str,
                            description=_(
                                "Unknown setting from device"
                            ),
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
        cmd = f"${key}={value}"
        await self._execute_command(cmd)

    async def set_wcs_offset(
        self, wcs_slot: str, x: float, y: float, z: float
    ) -> None:
        p_num = gcode_to_p_number(wcs_slot)
        if p_num is None:
            raise ValueError(f"Invalid WCS slot: {wcs_slot}")
        dialect = self.dialect
        cmd = dialect.set_wcs_offset.format(
            p_num=p_num, x=x, y=y, z=z
        )
        await self._execute_command(cmd)

    async def read_wcs_offsets(self) -> Dict[str, Pos]:
        response_lines = await self.execute_interactive_command("$#")
        offsets = {}
        for line in response_lines:
            match = wcs_re.match(line)
            if match:
                slot, x_str, y_str, z_str = match.groups()
                z_str = z_str or "0.000"
                offsets[slot] = (
                    float(x_str),
                    float(y_str),
                    float(z_str),
                )
        self.wcs_updated.send(self, offsets=offsets)
        return offsets

    async def read_parser_state(self) -> Optional[str]:
        try:
            response_lines = await self.execute_interactive_command(
                "$G"
            )
            return parse_grbl_parser_state(response_lines)
        except DeviceConnectionError as e:
            logger.warning(f"Could not read parser state: {e}")
            return None

    async def run_probe_cycle(
        self, axis: Axis, max_travel: float, feed_rate: int
    ) -> Optional[Pos]:
        assert axis.name, "Probing requires a single, named axis."
        axis_letter = axis.name.upper()
        dialect = self.dialect
        cmd = dialect.probe_cycle.format(
            axis_letter=axis_letter,
            max_travel=max_travel,
            feed_rate=feed_rate,
        )
        self.probe_status_changed.send(
            self, message=f"Probing {axis_letter}..."
        )
        try:
            response_lines = await self.execute_interactive_command(
                cmd
            )
        except DeviceConnectionError:
            self.probe_status_changed.send(
                self, message="Probe failed: Timed out"
            )
            return None
        for line in response_lines:
            match = prb_re.match(line)
            if match:
                x_str, y_str, z_str, success = match.groups()
                if int(success) == 1:
                    pos: Pos = (
                        float(x_str),
                        float(y_str),
                        float(z_str),
                    )
                    self.probe_status_changed.send(
                        self,
                        message=f"Probe triggered at {pos}",
                    )
                    return pos
        self.probe_status_changed.send(
            self, message="Probe failed"
        )
        return None
