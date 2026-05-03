import asyncio
import queue

import pytest

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


class MockSerial:
    """A mock pyserial Serial object for testing the threaded reader."""

    def __init__(self, *args, **kwargs):
        self.port = kwargs.get("port", "")
        self.baudrate = kwargs.get("baudrate", 9600)
        self.timeout = kwargs.get("timeout", 0.3)
        self._read_queue = queue.Queue()
        self._closed = False
        self._written = []

    def read(self, size=1024):
        if self._closed:
            raise OSError("Port is closed")
        try:
            return self._read_queue.get(timeout=self.timeout)
        except queue.Empty:
            return b""

    def write(self, data):
        if self._closed:
            raise OSError("Port is closed")
        self._written.append(data)
        return len(data)

    def close(self):
        self._closed = True

    def flush(self):
        pass

    def reset_input_buffer(self):
        while True:
            try:
                self._read_queue.get_nowait()
            except queue.Empty:
                break

    def feed_data(self, data: bytes):
        """Simulate incoming data from the device."""
        self._read_queue.put(data)


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
def mock_serial(mocker):
    """Mocks serial.Serial and returns the mock instance for inspection."""
    mock_instance = MockSerial()
    mock_cls = mocker.patch(
        "rayforge.machine.transport.serial.serial.Serial",
        return_value=mock_instance,
    )
    return mock_cls, mock_instance


class TestSerialTransportIntegration:
    """
    Tests the logic of SerialTransport by mocking pyserial.Serial.
    This provides fast, deterministic, and platform-independent testing.
    """

    @pytest.mark.asyncio
    async def test_connect_disconnect_cycle(self, mock_serial):
        """Test the connection and disconnection lifecycle and signals."""
        mock_cls, mock_ser = mock_serial
        transport = SerialTransport(port="/dev/mock", baudrate=9600)
        status_tracker = SignalTracker(transport.status_changed)

        assert not transport.is_connected

        await transport.connect()
        assert transport.is_connected
        mock_cls.assert_called_once_with(
            port="/dev/mock",
            baudrate=9600,
            timeout=0.3,
            exclusive=True,
        )

        await transport.disconnect()
        assert not transport.is_connected
        assert mock_ser._closed

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
            "rayforge.machine.transport.serial.serial.Serial",
            side_effect=IOError("Connection failed"),
        )
        transport = SerialTransport(port="/dev/fail", baudrate=9600)
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(IOError):
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.ERROR,
        ]
        error_call = status_tracker.calls[1]
        assert "message" in error_call["kwargs"]
        assert "Connection failed" in error_call["kwargs"]["message"]

    @pytest.mark.asyncio
    async def test_send_and_receive(self, mock_serial):
        """Test sending data and simulating a reception."""
        _, mock_ser = mock_serial
        transport = SerialTransport(port="/dev/mock", baudrate=115200)
        received_tracker = SignalTracker(transport.received)

        try:
            await transport.connect()
            assert transport.is_connected

            # Test sending
            test_message_send = b"hello from transport"
            await transport.send(test_message_send)
            assert mock_ser._written == [test_message_send]

            # Test receiving
            test_message_recv = b"hello from device"
            mock_ser.feed_data(test_message_recv)

            for _ in range(20):
                await asyncio.sleep(0.05)
                if received_tracker.calls:
                    break

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

    @pytest.mark.asyncio
    async def test_purge_on_connected_transport(self, mock_serial):
        """
        Test that purge clears buffered data from the reader.
        """
        _, mock_ser = mock_serial
        transport = SerialTransport(port="/dev/mock", baudrate=115200)

        try:
            await transport.connect()
            assert transport.is_connected

            # Feed some data to the reader
            mock_ser.feed_data(b"buffered data")
            # Purge should read and discard the buffered data
            await transport.purge()

            # Verify that read was called to consume data
            assert mock_ser._read_queue.empty()
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_purge_on_disconnected_transport(self):
        """
        Test that purge on a disconnected transport is a no-op.
        """
        transport = SerialTransport(port="/dev/mock", baudrate=9600)
        assert not transport.is_connected

        # Should not raise any errors
        await transport.purge()
