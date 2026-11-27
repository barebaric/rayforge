import asyncio
import os
import queue

import pytest
import serial
from raygeo.driver.core.discovery.ports import (
    SerialPortInfo,
    is_usb_serial_port,
    sort_ports,
)

from rayforge.machine.transport import TransportStatus
from rayforge.machine.transport.serial import (
    SerialPort,
    SerialPortPermissionError,
    SerialTransport,
    resolve_serial_port,
)


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
        self.timeout = kwargs.get("timeout", 0)
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
            timeout=0,
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
    async def test_connect_resolves_vidpid_spec(self, mock_serial, mocker):
        """
        A 'vid:pid' port is resolved to a device path on every connect,
        so reconnects follow the device when the OS reassigns paths.
        """
        mock_cls, _mock_ser = mock_serial
        infos = [
            SerialPortInfo("/dev/ttyUSB0", "CH340", vid=0x1A86, pid=0x7523),
            SerialPortInfo("/dev/ttyUSB1", "FTDI", vid=0x0403, pid=0x6001),
        ]
        mocker.patch.object(
            SerialTransport, "list_port_info", return_value=infos
        )
        transport = SerialTransport(port="0403:6001", baudrate=115200)

        await transport.connect()
        try:
            mock_cls.assert_called_once_with(
                port="/dev/ttyUSB1",
                baudrate=115200,
                timeout=0,
                exclusive=True,
            )
        finally:
            await transport.disconnect()

    @pytest.mark.asyncio
    async def test_connect_vidpid_spec_without_match(self, mocker):
        """
        A 'vid:pid' port with no matching device fails the connect
        attempt with a descriptive error, like a missing path does.
        """
        mocker.patch.object(SerialTransport, "list_port_info", return_value=[])
        transport = SerialTransport(port="0403:6001", baudrate=115200)
        status_tracker = SignalTracker(transport.status_changed)

        with pytest.raises(serial.SerialException, match="0403:6001"):
            await transport.connect()

        assert not transport.is_connected
        statuses = [call["kwargs"]["status"] for call in status_tracker.calls]
        assert statuses == [
            TransportStatus.CONNECTING,
            TransportStatus.ERROR,
        ]

    @pytest.mark.asyncio
    async def test_connection_failure(self, mocker):
        """Test that connection failures are handled gracefully."""
        mocker.patch(
            "rayforge.machine.transport.serial.serial.Serial",
            side_effect=OSError("Connection failed"),
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


def test_resolve_serial_port_passes_through_device_paths():
    """Values that are not 'vid:pid' specs are returned unchanged."""
    assert resolve_serial_port("/dev/ttyUSB0") == "/dev/ttyUSB0"
    assert resolve_serial_port("/dev/serial/by-id/usb-foo") == (
        "/dev/serial/by-id/usb-foo"
    )
    assert resolve_serial_port("COM3") == "COM3"


def test_resolve_serial_port_by_vidpid(mocker):
    """A 'vid:pid' spec resolves to the matching device path."""
    infos = [
        SerialPortInfo("/dev/ttyUSB0", "CH340", vid=0x1A86, pid=0x7523),
        SerialPortInfo("/dev/ttyUSB1", "FTDI", vid=0x0403, pid=0x6001),
    ]
    mocker.patch.object(SerialTransport, "list_port_info", return_value=infos)
    assert resolve_serial_port("0403:6001") == "/dev/ttyUSB1"
    assert resolve_serial_port("0403:6001".upper()) == "/dev/ttyUSB1"


def test_resolve_serial_port_by_vidpid_not_found(mocker):
    """A 'vid:pid' spec without a matching device raises clearly."""
    mocker.patch.object(SerialTransport, "list_port_info", return_value=[])
    with pytest.raises(serial.SerialException, match="0403:6001"):
        resolve_serial_port("0403:6001")


@pytest.mark.skipif(os.name != "posix", reason="POSIX heuristic")
def test_is_usb_serial_port_posix():
    """Test the USB port detection heuristic."""
    assert is_usb_serial_port("/dev/ttyUSB0")
    assert is_usb_serial_port("/dev/ttyACM0")
    assert not is_usb_serial_port("/dev/ttyS0")


@pytest.mark.skipif(os.name == "posix", reason="non-POSIX treats all as USB")
def test_is_usb_serial_port_windows():
    assert is_usb_serial_port("/dev/ttyS0")


@pytest.mark.skipif(os.name != "posix", reason="POSIX sort order")
def test_sort_ports_usb_first():
    """
    USB adapters must come first; each group ordered naturally,
    so ttyUSB2 sorts before ttyUSB10.
    """
    ports = [
        "/dev/ttyS3",
        "/dev/ttyUSB10",
        "/dev/ttyACM0",
        "/dev/ttyUSB2",
        "/dev/ttyS0",
    ]
    assert sort_ports(ports) == [
        "/dev/ttyACM0",
        "/dev/ttyUSB2",
        "/dev/ttyUSB10",
        "/dev/ttyS0",
        "/dev/ttyS3",
    ]
