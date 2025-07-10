import logging
import asyncio
from typing import Optional
from ..transport import SerialTransport, TransportStatus
from ..transport.serial import SerialPort
from ..opsencoder.gcode import GcodeEncoder
from ..models.ops import Ops
from ..models.machine import Machine
from .driver import Driver
from .grbl import _parse_state

logger = logging.getLogger(__name__)


class GrblSerialDriver(Driver):
    """
    Handles GRBL based devices via Serial port
    """
    label = "GRBL Serial"
    subtitle = 'Send GRBL-compatible Gcode via serial connection'

    def __init__(self):
        super().__init__()
        self.encoder = GcodeEncoder()
        self.serial_transport: Optional[SerialTransport] = None
        self.keep_running = False
        self._connection_task: Optional[asyncio.Task] = None

    def setup(self, port: SerialPort = "", baudrate: int = 115200):
        """
        Parameters:
          - port: Serial port (e.g., "/dev/ttyUSB0" or "COM1")
          - baudrate: Baud rate (default: 115200)
        """
        assert not self.did_setup
        if not port or not baudrate:
            return  # Leave unconfigured

        super().setup()

        self.serial_transport = SerialTransport(port, baudrate)
        self.serial_transport.received.connect(self.on_serial_data_received)
        self.serial_transport.status_changed.connect(
            self.on_serial_status_changed
        )

    async def cleanup(self):
        logger.debug("GrblSerialDriver cleanup initiated.")
        self.keep_running = False
        if self._connection_task:
            logger.debug("Cancelling _connection_task...")
            self._connection_task.cancel()
            try:
                await self._connection_task
            except asyncio.CancelledError:
                logger.debug("_connection_task successfully cancelled.")
            except Exception as e:
                logger.error(f"Error waiting for _connection_task: {e}")
        if self.serial_transport:
            logger.debug("Disconnecting serial transport...")
            await self.serial_transport.disconnect()
            self.serial_transport.received.disconnect(
                self.on_serial_data_received
            )
            self.serial_transport.status_changed.disconnect(
                self.on_serial_status_changed
            )
            self.serial_transport = None
            logger.debug("Serial transport disconnected and cleared.")
        await super().cleanup()
        logger.debug("GrblSerialDriver cleanup completed.")

    async def _send_command(self, command: str):
        logger.debug(f"Sending command: {command}")
        if not self.serial_transport:
            raise ConnectionError("Serial transport not initialized")
        # GRBL commands usually need a newline
        await self.serial_transport.send((command + '\n').encode('utf-8'))

    async def connect(self):
        logger.debug("GrblSerialDriver connect initiated.")
        self.keep_running = True
        self._connection_task = asyncio.create_task(self._connection_loop())

    async def _connection_loop(self) -> None:
        logger.debug("Entering _connection_loop.")
        while self.keep_running:
            self._on_connection_status_changed(TransportStatus.CONNECTING)
            logger.debug("Attempting connection…")
            try:
                assert self.serial_transport, "Transport not initialized"
                await self.serial_transport.connect()
                # Send a status report request to get initial state
                await self._send_command('?')
            except Exception as e:
                logger.error(f"Connection error: {e}")
                self._on_connection_status_changed(
                    TransportStatus.ERROR, str(e)
                )
                if self.serial_transport:
                    await self.serial_transport.disconnect()
                logger.debug("Reconnecting in 5s…")
                await asyncio.sleep(5)
            else:
                break
        logger.debug("Leaving _connection_loop.")

    async def run(self, ops: Ops, machine: Machine) -> None:
        gcode = self.encoder.encode(ops, machine)
        # Split gcode into lines and send them one by one
        for line in gcode.splitlines():
            if line.strip():
                await self._send_command(line.strip())
                # Add a small delay or wait for 'ok' from GRBL if flow control
                # is not handled by the transport layer
                await asyncio.sleep(0.01)

    async def set_hold(self, hold: bool = True) -> None:
        if hold:
            await self._send_command('!')
        else:
            await self._send_command('~')

    async def cancel(self) -> None:
        # GRBL reset command (Ctrl-X) is usually sent as a byte, not a string
        # This might need direct serial write if _send_command adds newline
        # For now, using a soft reset command if available, or assuming
        # the transport can handle raw bytes.
        # A common way to reset GRBL is to send 0x18 (Ctrl-X)
        if self.serial_transport:
            await self.serial_transport.send(b'\x18')
        else:
            raise ConnectionError("Serial transport not initialized")

    async def home(self) -> None:
        await self._send_command('$H')

    async def move_to(self, pos_x, pos_y) -> None:
        cmd = f"$J=G90 G21 F1500 X{float(pos_x)} Y{float(pos_y)}"
        await self._send_command(cmd)

    def on_serial_data_received(self, sender, data: bytes):
        data_str = data.decode('utf-8').strip()
        for line in data_str.splitlines():
            self._log(line)
            if line.startswith('<') and line.endswith('>'):
                state = _parse_state(line[1:-1], self.state, self._log)
                if state != self.state:
                    self.state = state
                    self._on_state_changed()
            elif line == 'ok':
                # GRBL 'ok' response, indicates command received
                self._on_command_status_changed(TransportStatus.IDLE)
            elif line.startswith('error:'):
                self._on_command_status_changed(
                    TransportStatus.ERROR,
                    message=line
                )
            # Add more GRBL specific responses handling here if needed

    def on_serial_status_changed(self,
                                 sender,
                                 status: TransportStatus,
                                 message: Optional[str] = None):
        self._on_connection_status_changed(status, message)
