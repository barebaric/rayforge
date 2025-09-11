import logging
import asyncio
import os
import serial_asyncio
from typing import Optional, List
from serial.tools import list_ports
from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)


class SerialPort(str):
    """A string subclass for identifying serial ports, for UI generation."""

    pass


class SerialPortPermissionError(Exception):
    """Custom exception for systemic serial port permission issues."""

    pass


class SerialTransport(Transport):
    """
    Asynchronous serial port transport.
    """

    @staticmethod
    def list_ports() -> List[str]:
        """Lists available serial ports."""
        try:
            ports = []
            for port in list_ports.comports():
                ports.append(port.device)
            return ports
        except TypeError:
            # This TypeError can occur in a misconfigured Snap without
            # serial-port access when pyserial fails to read device properties.
            # Log the issue but return an empty list for graceful UI failure.
            logger.warning(
                "Could not list serial ports. This may be due to a "
                "permission error (e.g., Snap confinement)."
            )
            return []

    @staticmethod
    def check_serial_permissions_globally() -> None:
        """
        On POSIX systems, checks if there are visible serial ports that the
        user cannot access. This is a strong indicator that the user is not
        in the correct group (e.g., 'dialout') or, in a Snap, lacks the
        necessary permissions.

        Raises:
            SerialPortPermissionError: If systemic permission issues are
              detected.
        """
        if os.name != "posix":
            return  # This check is only for POSIX-like systems (Linux, macOS)

        try:
            all_ports = list_ports.comports()
        except TypeError as e:
            # This TypeError occurs when pyserial failed to read a
            # device property, often due to sandbox permissions.
            if 'SNAP' in os.environ:
                # We are running inside a Snap, provide a specific, actionable
                # error.
                snap_name = os.environ.get('SNAP_NAME', 'rayforge')
                msg = _(
                    "Failed to list serial ports due to a Snap confinement "
                    "error. Please grant permission by running:\n\n"
                    f"sudo snap connect {snap_name}:serial-port"
                )
                raise SerialPortPermissionError(msg) from e
            else:
                # For non-snap environments, this points to a different issue.
                msg = _(
                    "An unexpected error occurred while listing serial ports. "
                    "This may be caused by a malfunctioning USB device or "
                    "a system driver issue."
                )
                raise SerialPortPermissionError(msg) from e

        # Filter for common USB-to-serial device names
        relevant_ports = [
            p.device
            for p in all_ports
            if "ttyUSB" in p.device or "ttyACM" in p.device
        ]

        if not relevant_ports:
            return  # No relevant ports found, nothing to check

        if not any(os.access(p, os.R_OK | os.W_OK) for p in relevant_ports):
            msg = _(
                "Could not access any serial ports. On Linux, ensure "
                "your user is in the 'dialout' group."
            )
            raise SerialPortPermissionError(msg)

    @staticmethod
    def list_baud_rates() -> List[int]:
        """Returns a list of common serial baud rates."""
        return [
            9600,
            19200,
            38400,
            57600,
            115200,
            230400,
            460800,
        ]

    def __init__(self, port: str, baudrate: int):
        """
        Initialize serial transport.

        Args:
            port: Device path (e.g., '/dev/ttyUSB0')
            baudrate: Communication speed in bits per second
        """
        super().__init__()
        self.port = port
        self.baudrate = baudrate
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._running = False
        self._receive_task: Optional[asyncio.Task] = None

    @property
    def is_connected(self) -> bool:
        """Check if the transport is actively connected."""
        return self._writer is not None and self._running

    async def connect(self) -> None:
        logger.debug("Attempting to connect serial port...")
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        try:
            result = await serial_asyncio.open_serial_connection(
                url=self.port, baudrate=self.baudrate
            )
            self._reader, self._writer = result
            logger.debug("serial_asyncio.open_serial_connection returned.")
            self._running = True
            self.status_changed.send(self, status=TransportStatus.CONNECTED)
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.debug("Serial port connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect serial port: {e}")
            self.status_changed.send(
                self, status=TransportStatus.ERROR, message=str(e)
            )
            raise

    async def disconnect(self) -> None:
        """
        Gracefully terminate the serial connection and cleanup resources.
        """
        logger.debug("Attempting to disconnect serial port...")
        self.status_changed.send(self, status=TransportStatus.CLOSING)
        self._running = False

        # Cancel the receive task if it exists
        if self._receive_task:
            logger.debug("Cancelling receive task...")
            self._receive_task.cancel()
            try:
                await asyncio.wait_for(self._receive_task, timeout=2.0)
                logger.debug("Receive task awaited successfully.")
            except asyncio.CancelledError:
                logger.debug("Receive task cancelled successfully.")
            except asyncio.TimeoutError:
                logger.warning("Timeout waiting for receive task to cancel.")
            except Exception as e:
                logger.error(f"Error cancelling receive task: {e}")
            self._receive_task = None
        else:
            logger.debug("No receive task to cancel.")

        # Close the writer without waiting
        if self._writer:
            logger.debug("Closing writer...")
            self._writer.close()
            self._writer = None

        # Clear reader reference (optional, for safety)
        if self._reader:
            logger.debug("Clearing reader reference.")
            self._reader = None

        # Signal disconnection and log completion
        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
        logger.debug("Serial port disconnected.")

    async def send(self, data: bytes) -> None:
        """
        Write data to serial port.
        """
        if not self._writer:
            raise ConnectionError("Serial port not open")
        logger.debug(f"Sending data: {data!r}")
        self._writer.write(data)
        await self._writer.drain()

    async def _receive_loop(self) -> None:
        """
        Continuous data reception loop.
        """
        logger.debug("Entering _receive_loop.")
        while self._running and self._reader:
            try:
                data = await self._reader.read(100)
                if data:
                    logger.debug(f"Received data: {data!r}")
                    self.received.send(self, data=data)
                else:
                    logger.error("Received empty data, connection closed.")
                    break  # Exit loop if connection is closed
            except asyncio.CancelledError:
                logger.debug("_receive_loop cancelled.")
                break
            except Exception as e:
                logger.error(f"Error in _receive_loop: {e}")
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
                break
        logger.debug("Exiting _receive_loop.")
