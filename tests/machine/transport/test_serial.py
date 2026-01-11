import asyncio
import pytest
from unittest.mock import AsyncMock

from rayforge.machine.transport.serial import (
    SerialPort,
    SerialPortPermissionError,
    SerialTransport,
    safe_list_ports_linux,
)
from rayforge.machine.transport import TransportStatus


class SignalTracker:
    """A helper to track calls to a blinker Signal."""

    def __init__(self, signal):
        self.calls = []
        signal.connect(self._callback)

    def _callback(self, sender, **kwargs):
        self.calls.append({"sender": sender, "kwargs": kwargs})

    def clear(self):
        self.calls = []

    @property
    def last_data(self) -> bytes:
        """Returns the data from the last 'received' signal call."""
        if not self.calls:
            return b""
        return self.calls[-1]["kwargs"].get("data", b"")


def test_serial_port_subclass():
    """Test that SerialPort is a string subclass."""
    port = SerialPort("/dev/ttyUSB0")
    assert isinstance(port, str)
    assert port == "/dev/ttyUSB0"


def test_safe_list_ports_linux(monkeypatch):
    """Test the safe_list_ports_linux globbing logic."""

    def mock_glob(pattern):
        if pattern == "/dev/ttyUSB*":
            return ["/dev/ttyUSB1", "/dev/ttyUSB0"]
        if pattern == "/dev/ttyACM*":
            return ["/dev/ttyACM0"]
        if pattern == "/dev/serial/by-id/*":
            return [
                "/dev/serial/by-id/"
                "usb-FTDI_FT232R_USB_UART_AH03K1A0-if00-port0"
            ]
        if pattern == "/dev/serial/by-path/*":
            return ["/dev/serial/by-path/pci-0000:00:14.0-usb-0:1:1.0-port0"]
        return []

    monkeypatch.setattr(
        "rayforge.machine.transport.serial.glob.glob", mock_glob
    )
    ports = safe_list_ports_linux()
    expected_ports = [
        "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AH03K1A0-if00-port0",
        "/dev/serial/by-path/pci-0000:00:14.0-usb-0:1:1.0-port0",
        "/dev/ttyACM0",
        "/dev/ttyUSB0",
        "/dev/ttyUSB1",
    ]
    assert ports == expected_ports


def test_safe_list_ports_linux_symlinks_only(monkeypatch):
    """Test that symlinks are included when no standard ports exist."""

    def mock_glob(pattern):
        if pattern in ["/dev/ttyUSB*", "/dev/ttyACM*"]:
            return []
        if pattern == "/dev/serial/by-id/*":
            return [
                "/dev/serial/by-id/"
                "usb-FTDI_FT232R_USB_UART_AH03K1A0-if00-port0"
            ]
        if pattern == "/dev/serial/by-path/*":
            return []
        return []

    monkeypatch.setattr(
        "rayforge.machine.transport.serial.glob.glob", mock_glob
    )
    ports = safe_list_ports_linux()
    expected_ports = [
        "/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AH03K1A0-if00-port0"
    ]
    assert ports == expected_ports


def test_list_usb_ports_filtering(mocker):
    """Test the USB port filtering logic."""
    mock_ports = [
        "/dev/ttyS0",
        "/dev/ttyUSB0",
        "/dev/ttyACM1",
        "COM3",
        "/dev/cu.usbmodem123",
    ]
    mocker.patch.object(SerialTransport, "list_ports", return_value=mock_ports)

    mocker.patch("os.name", "posix")
    usb_ports = SerialTransport.list_usb_ports()
    assert usb_ports == ["/dev/ttyUSB0", "/dev/ttyACM1"]

    mocker.patch("os.name", "nt")
    all_ports = SerialTransport.list_usb_ports()
    assert all_ports == mock_ports


class TestSerialPermissions:
    """Tests for the check_serial_permissions_globally static method."""

    def test_non_posix_system(self, mocker):
        mocker.patch("os.name", "nt")
        SerialTransport.check_serial_permissions_globally()

    def test_no_ports_found_non_snap(self, mocker):
        mocker.patch("os.name", "posix")
        mocker.patch("os.environ", {})
        mocker.patch.object(SerialTransport, "list_usb_ports", return_value=[])
        with pytest.raises(
            SerialPortPermissionError, match="No USB serial ports found."
        ):
            SerialTransport.check_serial_permissions_globally()

    def test_no_ports_found_snap(self, mocker):
        mocker.patch("os.name", "posix")
        mocker.patch(
            "os.environ", {"SNAP": "/snap/foo/123", "SNAP_NAME": "my-app"}
        )
        mocker.patch.object(SerialTransport, "list_usb_ports", return_value=[])
        with pytest.raises(
            SerialPortPermissionError, match="my-app:serial-port"
        ):
            SerialTransport.check_serial_permissions_globally()

    def test_ports_found_but_inaccessible_non_snap(self, mocker):
        mocker.patch("os.name", "posix")
        mocker.patch("os.environ", {})
        mocker.patch.object(
            SerialTransport, "list_usb_ports", return_value=["/dev/ttyUSB0"]
        )
        mocker.patch("os.access", return_value=False)
        with pytest.raises(SerialPortPermissionError, match="dialout"):
            SerialTransport.check_serial_permissions_globally()

    def test_ports_found_but_inaccessible_snap(self, mocker):
        mocker.patch("os.name", "posix")
        mocker.patch(
            "os.environ", {"SNAP": "/snap/foo/123", "SNAP_NAME": "my-app"}
        )
        mocker.patch.object(
            SerialTransport, "list_usb_ports", return_value=["/dev/ttyUSB0"]
        )
        mocker.patch("os.access", return_value=False)
        with pytest.raises(
            SerialPortPermissionError, match="serial-port' interface connected"
        ):
            SerialTransport.check_serial_permissions_globally()

    def test_at_least_one_port_accessible(self, mocker):
        mocker.patch("os.name", "posix")
        mocker.patch.object(
            SerialTransport,
            "list_usb_ports",
            return_value=["/dev/ttyUSB0", "/dev/ttyUSB1"],
        )
        mocker.patch("os.access", side_effect=[False, True])
        SerialTransport.check_serial_permissions_globally()


@pytest.fixture
def mock_serial_connection(mocker):
    """Mocks serial_asyncio.open_serial_connection."""
    mock_reader = AsyncMock(spec=asyncio.StreamReader)
    mock_writer = AsyncMock(spec=asyncio.StreamWriter)

    # Configure the reader to be pausable
    read_event = asyncio.Event()
    read_data = b""

    async def mock_read(n):
        nonlocal read_data
        await read_event.wait()
        data_to_return = read_data
        read_data = b""
        read_event.clear()
        return data_to_return

    mock_reader.read.side_effect = mock_read

    def feed_data(data: bytes):
        nonlocal read_data
        read_data = data
        read_event.set()

    mock_reader.feed_data = feed_data

    # Configure writer's drain to be awaitable
    mock_writer.drain = AsyncMock()

    mock_open = mocker.patch(
        "rayforge.machine.transport"
        ".serial.serial_asyncio.open_serial_connection",
        return_value=(mock_reader, mock_writer),
    )
    return mock_open, mock_reader, mock_writer


class TestSerialTransportIntegration:
    """
    Tests the logic of SerialTransport by mocking the serial_asyncio boundary.
    This provides fast, deterministic, and platform-independent testing.
    """

    @pytest.mark.asyncio
    async def test_connect_disconnect_cycle(self, mock_serial_connection):
        """Test the connection and disconnection lifecycle and signals."""
        mock_open, _, mock_writer = mock_serial_connection
        transport = SerialTransport(port="/dev/mock", baudrate=9600)
        status_tracker = SignalTracker(transport.status_changed)

        assert not transport.is_connected

        await transport.connect()
        assert transport.is_connected
        mock_open.assert_called_once_with(url="/dev/mock", baudrate=9600)

        await transport.disconnect()
        assert not transport.is_connected
        mock_writer.close.assert_called_once()

        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.CONNECTED,
            TransportStatus.CLOSING,
            TransportStatus.DISCONNECTED,
        ]

    @pytest.mark.asyncio
    async def test_connection_failure(self, mocker):
        """Test that connection failures are handled gracefully."""
        mocker.patch(
            "rayforge.machine.transport"
            ".serial.serial_asyncio.open_serial_connection",
            side_effect=IOError("Connection failed"),
        )
        transport = SerialTransport(port="/dev/fail", baudrate=9600)
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(IOError):
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [TransportStatus.CONNECTING, TransportStatus.ERROR]
        error_call = status_tracker.calls[1]
        assert "message" in error_call["kwargs"]
        assert "Connection failed" in error_call["kwargs"]["message"]

    @pytest.mark.asyncio
    async def test_send_and_receive(self, mock_serial_connection):
        """Test sending data and simulating a reception."""
        _, mock_reader, mock_writer = mock_serial_connection
        transport = SerialTransport(port="/dev/mock", baudrate=115200)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()
            assert transport.is_connected

            # Test sending
            test_message_send = b"hello from transport"
            await transport.send(test_message_send)
            mock_writer.write.assert_called_once_with(test_message_send)
            mock_writer.drain.assert_awaited_once()

            # Test receiving
            test_message_recv = b"hello from device"
            mock_reader.feed_data(test_message_recv)
            await asyncio.sleep(0)  # Yield to the event loop

            assert len(received_tracker.calls) == 1
            assert received_tracker.last_data == test_message_recv

        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_send_on_disconnected_transport(self):
        """
        Test that sending data on a disconnected transport raises an error.
        """
        transport = SerialTransport(port="/dev/mock", baudrate=9600)
        with pytest.raises(ConnectionError, match="Serial port not open"):
            await transport.send(b"test")
