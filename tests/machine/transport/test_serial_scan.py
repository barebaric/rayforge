import asyncio
import threading
from typing import ClassVar

import pytest
import serial as pyserial

from rayforge.machine.transport.serial import SerialPortInfo
from rayforge.machine.transport.serial_scan import (
    looks_like_device_output,
    scan_serial_ports,
)


class FakeSerial:
    """A fake pyserial.Serial for scanning tests."""

    attempts: ClassVar[list[tuple[str, int]]] = []

    def __init__(
        self,
        port=None,
        baudrate=9600,
        timeout=0,
        responses=None,
        busy=(),
    ):
        FakeSerial.attempts.append((str(port), int(baudrate)))
        if port in busy:
            raise pyserial.SerialException("Device or resource busy")
        self.port = port
        self.baudrate = baudrate
        self._responses = responses or {}
        response = self._responses.get(port, b"")
        # A port may answer differently per baud rate.
        if isinstance(response, dict):
            response = response.get(int(baudrate), b"")
        self._pending = bytearray(response)

    def read(self, size=1):
        if not self._pending:
            return b""
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, data):
        return len(data)

    def flush(self):
        pass

    def close(self):
        pass


@pytest.fixture(autouse=True)
def fast_timeouts(monkeypatch):
    from rayforge.machine.transport import serial_scan

    monkeypatch.setattr(serial_scan, "_BANNER_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_NUDGE_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_DRAIN_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_QUIET_GAP", 0.01)
    monkeypatch.setattr(serial_scan, "_READ_CHUNK", 0.01)
    FakeSerial.attempts.clear()


def _patch_serial(monkeypatch, responses=None, busy=()):
    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        lambda **kw: FakeSerial(responses=responses, busy=busy, **kw),
    )


def test_looks_like_device_output():
    assert looks_like_device_output(b"Grbl 1.1f ['$' for help]\r\n")
    assert looks_like_device_output(b"start\r\necho:busy: processing")
    assert not looks_like_device_output(b"")
    assert not looks_like_device_output(b"\x00\xff\x0f\xd3\xa1")
    assert not looks_like_device_output(b"\n\n\r\n")


@pytest.mark.asyncio
async def test_scan_observes_responding_port(monkeypatch):
    responses = {"/dev/ttyUSB0": b"Grbl 1.1f ['$' for help]\r\n"}
    _patch_serial(monkeypatch, responses)
    observations = await scan_serial_ports(ports=["/dev/ttyUSB0"])
    assert len(observations) == 1
    obs = observations[0]
    assert obs.port == "/dev/ttyUSB0"
    assert obs.baud_rate == 115200
    assert obs.data == b"Grbl 1.1f ['$' for help]\r\n"


@pytest.mark.asyncio
async def test_scan_opens_port_once_per_scan(monkeypatch):
    """A responding port is accepted at the first baud rate that
    yields device-like output; later rates are not tried."""
    responses = {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"}
    _patch_serial(monkeypatch, responses)
    await scan_serial_ports(ports=["/dev/ttyUSB0"])
    assert FakeSerial.attempts == [("/dev/ttyUSB0", 115200)]


@pytest.mark.asyncio
async def test_scan_tries_baud_rates_in_order(monkeypatch):
    _patch_serial(monkeypatch, {})
    await scan_serial_ports(ports=["/dev/ttyA"], baud_rates=[115200, 9600])
    assert FakeSerial.attempts == [("/dev/ttyA", 115200), ("/dev/ttyA", 9600)]


@pytest.mark.asyncio
async def test_scan_skips_port_with_only_garbage(monkeypatch):
    """Framing garbage at one baud rate must not end the probe:
    the scan moves on to the next rate."""
    responses = {
        "/dev/ttyA": {
            115200: b"\x00\xff\x0f\xd3\xa1\x7f",
            9600: b"Grbl 1.1f\r\n",
        }
    }
    _patch_serial(monkeypatch, responses)
    observations = await scan_serial_ports(
        ports=["/dev/ttyA"], baud_rates=[115200, 9600]
    )
    assert [(o.port, o.baud_rate) for o in observations] == [
        ("/dev/ttyA", 9600)
    ]


@pytest.mark.asyncio
async def test_scan_collects_nudge_responses_for_silent_devices(
    monkeypatch,
):
    """A device without a power-on banner is provoked into talking:
    every nudge is answered, all responses are captured together
    (a bare ``ok`` ack alone is not enough to identify firmware)."""

    class NudgeDevice:
        writes: ClassVar[list[bytes]] = []

        def __init__(self, port=None, baudrate=9600, timeout=0):
            self._replies = iter(
                [b"ok\r\n", b"<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n"]
            )
            self._pending = bytearray()

        def read(self, size=1):
            if not self._pending:
                return b""
            chunk = bytes(self._pending[:size])
            del self._pending[:size]
            return chunk

        def write(self, data):
            NudgeDevice.writes.append(bytes(data))
            try:
                self._pending.extend(next(self._replies))
            except StopIteration:
                pass
            return len(data)

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial", NudgeDevice
    )
    NudgeDevice.writes.clear()

    observations = await scan_serial_ports(ports=["/dev/ttyUSB0"])
    assert NudgeDevice.writes == [b"\n", b"?"]
    assert len(observations) == 1
    assert observations[0].data == (
        b"ok\r\n<Idle|MPos:0.000,0.000,0.000|FS:0,0>\r\n"
    )


@pytest.mark.asyncio
async def test_scan_skips_busy_port(monkeypatch):
    responses = {"/dev/ttyUSB1": b"Grbl 1.1f\r\n"}
    _patch_serial(monkeypatch, responses, busy=("/dev/ttyUSB0",))
    observations = await scan_serial_ports(
        ports=["/dev/ttyUSB0", "/dev/ttyUSB1"], baud_rates=[115200]
    )
    assert [o.port for o in observations] == ["/dev/ttyUSB1"]


@pytest.mark.asyncio
async def test_scan_excludes_requested_ports(monkeypatch):
    responses = {
        "/dev/ttyUSB0": b"Grbl 1.1f\r\n",
        "/dev/ttyUSB1": b"Grbl 1.1f\r\n",
    }
    _patch_serial(monkeypatch, responses)
    observations = await scan_serial_ports(
        ports=list(responses), exclude_ports={"/dev/ttyUSB0"}
    )
    assert [o.port for o in observations] == ["/dev/ttyUSB1"]
    assert FakeSerial.attempts == [("/dev/ttyUSB1", 115200)]


@pytest.mark.asyncio
async def test_scan_uses_port_metadata(monkeypatch):
    responses = {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"}
    _patch_serial(monkeypatch, responses)
    port_info = [
        SerialPortInfo(
            device="/dev/ttyUSB0",
            description="Ortur Laser Master",
            manufacturer="Ortur",
            vid=0x1A86,
            pid=0x7523,
        )
    ]
    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.SerialTransport"
        ".list_port_info",
        lambda: port_info,
    )
    observations = await scan_serial_ports(ports=["/dev/ttyUSB0"])
    info = observations[0].info
    assert info is not None
    assert info.description == "Ortur Laser Master"
    assert info.vid == 0x1A86
    assert info.pid == 0x7523


@pytest.mark.asyncio
async def test_concurrent_scans_serialize_on_port(monkeypatch):
    """A second scan cannot open a port while another scan is still
    probing it (pyserial opens exclusively)."""
    opened = threading.Event()
    release = threading.Event()

    class BlockingSerial:
        attempts: ClassVar[list[str]] = []

        def __init__(self, port=None, baudrate=9600, timeout=0):
            BlockingSerial.attempts.append(str(port))
            self._first = len(BlockingSerial.attempts) == 1

        def read(self, size=1):
            if self._first:
                opened.set()
                release.wait(timeout=2.0)
            return b""

        def write(self, data):
            return len(data)

        def flush(self):
            pass

        def close(self):
            pass

    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        BlockingSerial,
    )

    loop = asyncio.get_running_loop()
    scan1 = asyncio.create_task(
        scan_serial_ports(ports=["/dev/ttyUSB0"], baud_rates=[115200])
    )
    await loop.run_in_executor(None, opened.wait, 2.0)

    scan2 = asyncio.create_task(
        scan_serial_ports(ports=["/dev/ttyUSB0"], baud_rates=[115200])
    )
    await asyncio.sleep(0.1)
    assert BlockingSerial.attempts == ["/dev/ttyUSB0"]

    release.set()
    await asyncio.gather(scan1, scan2)
    assert BlockingSerial.attempts == ["/dev/ttyUSB0", "/dev/ttyUSB0"]
