import glob
import logging
import asyncio
import os
import serial
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
        try:
            self._writer.write(data)
            await self._writer.drain()
        except (serial.SerialException, OSError) as e:
            # Wrap low-level serial errors as ConnectionError so drivers
            # can handle them gracefully (e.g., breaking a poll loop).
            raise ConnectionError(
                f"Failed to write to serial port: {e}"
            ) from e

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
            except serial.SerialException as e:
                # Handle common Linux disconnect error gracefully
                msg = str(e)
                if (
                    "device reports readiness to read but returned no data"
                    in msg
                ):
                    logger.warning(
                        f"Serial connection lost (device disconnected?): {e}"
                    )
                else:
                    logger.error(f"Serial error in _receive_loop: {e}")

                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=msg
                )
                break
            except Exception as e:
                logger.error(f"Error in _receive_loop: {e}")
                self.status_changed.send(
                    self, status=TransportStatus.ERROR, message=str(e)
                )
                break
        logger.debug("Exiting _receive_loop.")
