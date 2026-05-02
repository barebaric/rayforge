import glob
import logging
import asyncio
import os
import serial
import threading
from typing import Optional, List
from serial.tools import list_ports
from gettext import gettext as _
from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)


class SerialPort(str):
    """A string subclass for identifying serial ports, for UI generation."""

    pass


class SerialPortPermissionError(Exception):
    """Custom exception for systemic serial port permission issues."""

    pass


def safe_list_ports_linux() -> List[str]:
    """
    A non-crashing implementation of list_ports for sandboxed Linux envs.

    pyserial's default list_ports.comports() tries to access /dev/ttyS*
    ports, which is forbidden by the snap sandbox. This leads to a
    permission error that causes a TypeError in the pyserial code.

    This function avoids that by only looking for common USB-to-serial
    device patterns that are permitted by the serial-port interface.
    """
    ports = []
    # Use glob to find all devices matching the common patterns
    for pattern in [
        "/dev/ttyUSB*",
        "/dev/ttyACM*",
        "/dev/serial/by-id/*",
        "/dev/serial/by-path/*",
    ]:
        try:
            ports.extend(glob.glob(pattern))
        except Exception as e:
            logger.warning(
                f"Error scanning for serial ports. Pattern '{pattern}': {e}"
            )
    return sorted(ports)


class SerialTransport(Transport):
    """
    Asynchronous serial port transport.
    """

    @staticmethod
    def list_ports() -> List[str]:
        """Lists available serial ports."""
        # If we're on Linux (posix) and running in a Snap, use our
        # safe scanner, as list_ports.comports() fails with permission errors.
        if os.name == "posix" and "SNAP" in os.environ:
            return safe_list_ports_linux()

        # On other systems or outside a Snap, the default is fine.
        try:
            return sorted([p.device for p in list_ports.comports()])
        except Exception as e:
            # Fallback for any other unexpected errors
            logger.error(f"Failed to list serial ports with pyserial: {e}")
            return []

    @staticmethod
    def list_usb_ports() -> List[str]:
        """Like list_ports, but only returns USB serial ports."""

        all_ports = SerialTransport.list_ports()
        if os.name != "posix":
            # On non-POSIX systems, we can't reliably filter, so return all.
            return all_ports

        return [p for p in all_ports if "ttyUSB" in p or "ttyACM" in p]

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

        # Retrieve a list of all relevant serial ports.
        all_ports = SerialTransport.list_usb_ports()
        snap_name = os.environ.get("SNAP_NAME", "rayforge")

        # First, handle the case where no ports are found and
        # provide environment-specific guidance if applicable.
        if not all_ports and "SNAP" in os.environ:
            msg = _(
                "Failed to list serial ports due to a Snap confinement!"
                " Please ensure the device is connected via USB and run:"
                "\n\n"
                "sudo snap set system experimental.hotplug=true\n"
                "sudo snap connect {snap_name}:serial-port"
            ).format(snap_name=snap_name)
            raise SerialPortPermissionError(msg)

        elif not all_ports:
            msg = "No USB serial ports found."
            raise SerialPortPermissionError(msg)

        # Next, check if any of the found ports are accessible.
        if any(os.access(p, os.R_OK | os.W_OK) for p in all_ports):
            return  # At least one port is accessible; no systemic issue.

        if "SNAP" in os.environ:
            msg = _(
                "Serial ports found, but none are accessible. Please ensure"
                " your Snap has the 'serial-port' interface connected by"
                " running:\n\n"
                "sudo snap set system experimental.hotplug=true\n"
                "sudo snap connect {snap_name}:serial-port"
            ).format(snap_name=snap_name)
            raise SerialPortPermissionError(msg)
        else:
            msg = (
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
            921600,
            1000000,
            1843200,
        ]

    _READ_TIMEOUT = 0.5

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
        self._serial: Optional[serial.Serial] = None
        self._running = False
        self._stop_event = threading.Event()
        self._reader_thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    @property
    def is_connected(self) -> bool:
        """Check if the transport is actively connected."""
        return self._serial is not None and self._running

    async def connect(self) -> None:
        logger.debug("Attempting to connect serial port...")
        self.status_changed.send(self, status=TransportStatus.CONNECTING)
        try:
            self._serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                timeout=self._READ_TIMEOUT,
                exclusive=True,
            )
            logger.debug("serial.Serial opened successfully.")
            self._running = True
            self._loop = asyncio.get_running_loop()
            self._stop_event.clear()
            self.status_changed.send(self, status=TransportStatus.CONNECTED)
            self._reader_thread = threading.Thread(
                target=self._reader_thread_func,
                name="serial-reader",
                daemon=True,
            )
            self._reader_thread.start()
            logger.debug("Serial port connected successfully.")
        except Exception as e:
            logger.error(f"Failed to connect serial port: {e}")
            self._serial = None
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

        self._stop_event.set()

        # Close the serial port; this will cause the blocking read()
        # in the reader thread to raise SerialException or return b"".
        if self._serial:
            try:
                self._serial.close()
            except Exception as e:
                logger.warning(f"Error closing serial port: {e}")

        # Wait for the reader thread to finish.
        if self._reader_thread and self._reader_thread.is_alive():
            logger.debug("Waiting for reader thread to finish...")
            self._reader_thread.join(timeout=2.0)
            if self._reader_thread.is_alive():
                logger.warning("Reader thread did not stop in time.")
        self._reader_thread = None
        self._serial = None
        self._loop = None

        self.status_changed.send(self, status=TransportStatus.DISCONNECTED)
        logger.debug("Serial port disconnected.")

    async def send(self, data: bytes) -> None:
        """
        Write data to serial port and flush to ensure physical
        transmission.

        Without flush, data may sit in the kernel TTY buffer
        indefinitely.  This causes GRBL to never receive commands
        while the host believes they were sent, leading to false
        deadlock detection.  When the deadlock recovery eventually
        writes more data, the entire buffered payload is flushed at
        once, overflowing the device's small RX buffer and causing
        error responses.
        """
        if not self._serial:
            raise ConnectionError("Serial port not open")
        logger.debug(f"Sending data: {data!r}")
        try:
            self._serial.write(data)
            self._serial.flush()
        except (serial.SerialException, OSError) as e:
            # Wrap low-level serial errors as ConnectionError so drivers
            # can handle them gracefully
            raise ConnectionError(
                f"Failed to write to serial port: {e}"
            ) from e

    async def purge(self) -> None:
        """
        Clear any buffered data in the serial transport.

        Discards any pending data in the receive buffer to resync
        communications. Does not affect the connection state.
        """
        if not self._serial:
            return

        try:
            self._serial.reset_input_buffer()
            logger.debug("Input buffer purged.")
        except Exception as e:
            logger.warning(f"Error during purge: {e}")

    def _dispatch_received(self, data: bytes) -> None:
        """Emit received signal on the event loop thread."""
        self.received.send(self, data=data)

    def _dispatch_error(self, message: str) -> None:
        """Emit error status on the event loop thread."""
        self.status_changed.send(
            self, status=TransportStatus.ERROR, message=message
        )

    def _reader_thread_func(self) -> None:
        """
        Dedicated reader thread that continuously reads from the serial
        port and dispatches received data to the event loop.

        Uses a blocking read with timeout so the thread can check the
        stop event periodically. This ensures the OS-level serial buffer
        is always being drained, preventing data loss on platforms (like
        Windows) where the default COM input buffer is small (4096 bytes).
        """
        assert self._serial is not None
        ser = self._serial
        logger.debug("Reader thread started.")
        while not self._stop_event.is_set():
            try:
                data = ser.read(1024)
            except serial.SerialException as e:
                if self._stop_event.is_set():
                    break
                msg = str(e)
                if (
                    "device reports readiness to read but returned no data"
                    in msg
                ):
                    logger.warning(
                        f"Serial connection lost (device disconnected?): {e}"
                    )
                else:
                    logger.error(f"Serial error in reader thread: {e}")
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(self._dispatch_error, msg)
                break
            except OSError as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"OS error in reader thread: {e}")
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(
                        self._dispatch_error, str(e)
                    )
                break
            except Exception as e:
                if self._stop_event.is_set():
                    break
                logger.error(f"Unexpected error in reader thread: {e}")
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(
                        self._dispatch_error, str(e)
                    )
                break

            if not data:
                continue

            logger.debug(f"Received data: {data!r}")
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._dispatch_received, data)

        logger.debug("Reader thread exiting.")
