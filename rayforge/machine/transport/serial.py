import asyncio
import glob
import logging
import os
import re
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from gettext import gettext as _

import serial
from serial.tools import list_ports

from .transport import Transport, TransportStatus

logger = logging.getLogger(__name__)

USB_PORT_MARKERS = ("ttyUSB", "ttyACM")


class SerialPort(str):
    """A string subclass for identifying serial ports, for UI generation."""


class SerialPortPermissionError(Exception):
    """Custom exception for systemic serial port permission issues."""


@dataclass(frozen=True)
class SerialPortInfo:
    """A serial port path together with an optional human-readable
    description."""

    device: str
    description: str | None = None


def is_usb_serial_port(port: str) -> bool:
    """
    Heuristically determines if a port path refers to a USB serial
    device. On non-POSIX systems (e.g. Windows), every port is treated
    as potentially USB.
    """
    if os.name != "posix":
        return True
    return any(marker in port for marker in USB_PORT_MARKERS)


def natural_key(s: str) -> list[int | str]:
    return [
        int(t) if t.isdigit() else t.lower() for t in re.split("([0-9]+)", s)
    ]


def _port_sort_key(port: str) -> tuple[int, list[int | str]]:
    return (not is_usb_serial_port(port), natural_key(port))


def sort_ports(ports: Iterable[str]) -> list[str]:
    """
    Sorts ports so USB serial adapters come first, followed by all
    other ports. Each group is ordered naturally, e.g.
    ttyUSB2 before ttyUSB10.
    """
    return sorted(ports, key=_port_sort_key)


_BY_ID_SUFFIX_RE = re.compile(r"-if\d+(-port\d+)?$")


def _describe_by_id_link(path: str) -> str | None:
    """
    Extracts a device description from a /dev/serial/by-id symlink
    name, which embeds vendor and product strings, e.g.
    'usb-FTDI_FT232R_USB_UART_AH03K1A0-if00-port0'.
    """
    name = os.path.basename(path)
    if not name.startswith("usb-"):
        return None
    name = name[len("usb-") :]
    name = _BY_ID_SUFFIX_RE.sub("", name)
    name = name.replace("_", " ").strip()
    return name or None


def _collect_by_id_descriptions(paths: list[str]) -> dict[str, str]:
    """
    Maps port paths to descriptions derived from /dev/serial/by-id
    symlinks. Both the link path itself and the resolved device path
    are mapped, so ttyUSB devices get described too.
    """
    descriptions: dict[str, str] = {}
    for path in paths:
        if "/by-id/" not in path:
            continue
        desc = _describe_by_id_link(path)
        if not desc:
            continue
        try:
            resolved = os.path.realpath(path)
        except OSError:
            continue
        descriptions.setdefault(path, desc)
        descriptions.setdefault(resolved, desc)
    return descriptions


def safe_list_ports_linux() -> list[str]:
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
        except OSError as e:
            logger.warning(
                f"Error scanning for serial ports. Pattern '{pattern}': {e}"
            )
    return sorted(ports)


class SerialTransport(Transport):
    """
    Asynchronous serial port transport.
    """

    @staticmethod
    def list_ports() -> list[str]:
        """Lists available serial ports, USB adapters first."""
        # If we're on Linux (posix) and running in a Snap, use our
        # safe scanner, as list_ports.comports() fails with permission errors.
        if os.name == "posix" and "SNAP" in os.environ:
            return sort_ports(safe_list_ports_linux())

        # On other systems or outside a Snap, the default is fine.
        try:
            return sort_ports([p.device for p in list_ports.comports()])
        except (OSError, serial.SerialException, TypeError) as e:
            # Fallback for any other unexpected errors
            logger.error(f"Failed to list serial ports with pyserial: {e}")
            return []

    @staticmethod
    def list_port_info() -> list[SerialPortInfo]:
        """
        Lists available serial ports together with a human-readable
        description when the platform provides one. USB adapters are
        listed first.
        """
        if os.name == "posix" and "SNAP" in os.environ:
            paths = safe_list_ports_linux()
            descriptions = _collect_by_id_descriptions(paths)
            infos = [
                SerialPortInfo(path, descriptions.get(path)) for path in paths
            ]
            infos.sort(key=lambda info: _port_sort_key(info.device))
            return infos

        try:
            infos = []
            for port in list_ports.comports():
                desc = port.description
                if not desc or desc == "n/a":
                    desc = None
                infos.append(SerialPortInfo(port.device, desc))
            infos.sort(key=lambda info: _port_sort_key(info.device))
            return infos
        except (OSError, serial.SerialException, TypeError) as e:
            logger.error(f"Failed to list serial ports with pyserial: {e}")
            return []

    @staticmethod
    def list_usb_ports() -> list[str]:
        """Like list_ports, but only returns USB serial ports."""

        all_ports = SerialTransport.list_ports()
        if os.name != "posix":
            # On non-POSIX systems, we can't reliably filter, so return all.
            return all_ports

        return [p for p in all_ports if is_usb_serial_port(p)]

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
    def list_baud_rates() -> list[int]:
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

    # Non-blocking read to prevent holding OS driver locks.
    _READ_TIMEOUT = 0

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
        self._serial: serial.Serial | None = None
        self._running = False
        self._stop_event = threading.Event()
        self._reader_thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

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
            except serial.SerialException as e:
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
        assert self._loop is not None
        logger.debug(f"Sending data: {data!r}")

        try:
            # Offloading to an executor prevents blocking C-level calls
            # from tying up the asyncio event loop thread, which can cause
            # deferred OS/kernel execution of the actual transmission.
            await self._loop.run_in_executor(None, self._sync_send, data)
        except (serial.SerialException, OSError) as e:
            # Wrap low-level serial errors as ConnectionError so drivers
            # can handle them gracefully
            raise ConnectionError(
                f"Failed to write to serial port: {e}"
            ) from e

    def _sync_send(self, data: bytes) -> None:
        """Synchronous wrapper for blocking write and flush operations."""
        if self._serial:
            self._serial.write(data)
            self._serial.flush()

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
        except serial.SerialException as e:
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

        Uses a non-blocking read to allow the OS to manage the hardware
        transmit locks freely, paired with a tiny CPU yield to prevent
        busy-looping.
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
            except Exception as e:  # noqa: BLE001 - reader thread boundary
                if self._stop_event.is_set():
                    break
                logger.error(f"Unexpected error in reader thread: {e}")
                if self._loop and not self._loop.is_closed():
                    self._loop.call_soon_threadsafe(
                        self._dispatch_error, str(e)
                    )
                break

            if not data:
                # With timeout=0, this microscopic sleep prevents a 100% CPU
                # busy-loop. Crucially, it ensures the Python thread spends
                # almost all of its time OUTSIDE the kernel, keeping the OS
                # serial lock free so that concurrent writes from the asyncio
                # thread can physically transmit immediately.
                time.sleep(0.005)
                continue

            logger.debug(f"Received data: {data!r}")
            if self._loop and not self._loop.is_closed():
                self._loop.call_soon_threadsafe(self._dispatch_received, data)

        logger.debug("Reader thread exiting.")
