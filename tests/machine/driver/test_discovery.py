import asyncio
from typing import ClassVar

import pytest

from rayforge.machine.driver.discovery import (
    GENERIC_TOKENS,
    DeviceIdentity,
    DeviceRecognizer,
    DiscoveredDevice,
    build_identity,
    extract_banner,
    find_all_devices,
    normalize_tokens,
)
from rayforge.machine.transport.serial import SerialPortInfo


def _grbl_matcher(data: bytes) -> bool:
    # Mirrors the real is_grbl_output(): banner or realtime status
    # report. A bare "ok" ack must NOT match (Marlin acks alike).
    return b"Grbl" in data or b"<Idle" in data


def _marlin_matcher(data: bytes) -> bool:
    return b"start" in data or b"echo:" in data


class FakeGrblDriver:
    DISCOVERY = DeviceRecognizer(
        label=lambda: "GRBL device",
        matches=_grbl_matcher,
        firmware="grbl",
    )


class FakeMarlinDriver:
    DISCOVERY = DeviceRecognizer(
        label=lambda: "Marlin device",
        matches=_marlin_matcher,
        firmware="marlin",
    )


class FakeSerial:
    """A fake pyserial.Serial for discovery tests."""

    attempts: ClassVar[list[tuple[str, int]]] = []

    def __init__(self, port=None, baudrate=9600, timeout=0, responses=None):
        FakeSerial.attempts.append((str(port), int(baudrate)))
        self.port = port
        self.baudrate = baudrate
        self._pending = bytearray((responses or {}).get(port, b""))

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


class NudgeAnsweringSerial:
    """
    A device without a power-on banner that acks every nudge like
    GRBL: ``ok`` for a newline, a status report for ``?``.
    """

    def __init__(self, port=None, baudrate=9600, timeout=0):
        self._replies = iter([b"ok\r\n", b"<Idle|MPos:0.0,0.0,0.0>\r\n"])
        self._pending = bytearray()

    def read(self, size=1):
        if not self._pending:
            return b""
        chunk = bytes(self._pending[:size])
        del self._pending[:size]
        return chunk

    def write(self, data):
        try:
            self._pending.extend(next(self._replies))
        except StopIteration:
            pass
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


def _patch_serial(monkeypatch, responses):
    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        lambda **kw: FakeSerial(responses=responses, **kw),
    )


@pytest.mark.asyncio
async def test_one_scan_serves_all_drivers(monkeypatch):
    """Adding discovery-capable drivers never multiplies port opens:
    both recognizers evaluate a single scan."""
    responses = {
        "/dev/ttyUSB0": b"Grbl 1.1f ['$' for help]\r\n",
        "/dev/ttyACM0": b"start\r\necho:Marlin initialized\r\n",
    }
    _patch_serial(monkeypatch, responses)
    devices = await find_all_devices(
        [FakeGrblDriver, FakeMarlinDriver],
        ports=["/dev/ttyUSB0", "/dev/ttyACM0"],
    )
    by_driver = {d.driver_name: d for d in devices}
    assert set(by_driver) == {"FakeGrblDriver", "FakeMarlinDriver"}

    grbl = by_driver["FakeGrblDriver"]
    assert grbl.params == {"port": "/dev/ttyUSB0", "baudrate": 115200}
    assert grbl.label == "GRBL device"
    assert grbl.detail == "/dev/ttyUSB0 at 115200 baud"
    assert grbl.identity.firmware == "grbl"
    assert grbl.identity.banner == "Grbl 1.1f ['$' for help]"

    marlin = by_driver["FakeMarlinDriver"]
    assert marlin.params["port"] == "/dev/ttyACM0"
    assert marlin.identity.firmware == "marlin"

    assert sorted(FakeSerial.attempts) == [
        ("/dev/ttyACM0", 115200),
        ("/dev/ttyUSB0", 115200),
    ]


@pytest.mark.asyncio
async def test_bannerless_grbl_is_recognized(monkeypatch):
    """A GRBL device that never emits a boot banner (no DTR reset)
    still identifies itself through its nudge responses alone."""
    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        NudgeAnsweringSerial,
    )
    devices = await find_all_devices(
        [FakeGrblDriver, FakeMarlinDriver], ports=["/dev/ttyUSB0"]
    )
    assert [d.driver_name for d in devices] == ["FakeGrblDriver"]
    assert devices[0].identity.banner == "ok"


# Real-world capture from a Sculpfun iCube: a Grbl fork whose boot
# output never mentions Grbl — only $I-style build-info lines.
SCULPFUN_ICUBE_BANNER = (
    b"[VER:1.0.15,20240923:]\r\n"
    b"[OPT:VMP,31,511]\r\n"
    b"[MSG:mechine:Sculpfun iCube]\r\n"
    b"[MSG:Mode=BT]\r\n"
    b"[BT_VER:8.1.2,FSC-BT836B]\r\n"
    b"Connection status: CONNECTED \r\n"
)


@pytest.mark.asyncio
async def test_sculpfun_banner_end_to_end(monkeypatch):
    """The real GrblSerialDriver recognizer claims build-info-only
    banners and extracts the machine name for identity/matching."""
    from rayforge.machine.driver.grbl import GrblSerialDriver

    _patch_serial(monkeypatch, {"/dev/ttyUSB0": SCULPFUN_ICUBE_BANNER})
    devices = await find_all_devices(
        [GrblSerialDriver], ports=["/dev/ttyUSB0"]
    )
    assert [d.driver_name for d in devices] == ["GrblSerialDriver"]
    device = devices[0]
    assert device.identity.firmware == "grbl"
    assert device.identity.banner == "Sculpfun iCube"
    assert "sculpfun" in device.identity.tokens
    assert "icube" in device.identity.tokens


@pytest.mark.asyncio
async def test_unrecognized_output_is_not_reported(monkeypatch):
    _patch_serial(monkeypatch, {"/dev/ttyUSB0": b"SomeOtherFW v2\r\n"})
    devices = await find_all_devices(
        [FakeGrblDriver, FakeMarlinDriver], ports=["/dev/ttyUSB0"]
    )
    assert devices == []


@pytest.mark.asyncio
async def test_drivers_without_recognizer_are_ignored(monkeypatch):
    _patch_serial(monkeypatch, {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"})

    class UndiscoverableDriver:
        pass

    devices = await find_all_devices(
        [UndiscoverableDriver], ports=["/dev/ttyUSB0"]
    )
    assert devices == []
    # No recognizer at all means no scan is even attempted.
    assert FakeSerial.attempts == []


@pytest.mark.asyncio
async def test_broken_recognizer_does_not_break_discovery(monkeypatch):
    _patch_serial(monkeypatch, {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"})

    def boom(data: bytes) -> bool:
        raise OSError("boom")

    class BrokenDriver:
        DISCOVERY = DeviceRecognizer(
            label=lambda: "Broken",
            matches=boom,
        )

    devices = await find_all_devices(
        [BrokenDriver, FakeGrblDriver], ports=["/dev/ttyUSB0"]
    )
    assert [d.driver_name for d in devices] == ["FakeGrblDriver"]


@pytest.mark.asyncio
async def test_scan_timeout_returns_empty(monkeypatch):
    from rayforge.machine.driver import discovery

    async def slow_scan(**kwargs):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr(discovery, "scan_serial_ports", slow_scan)
    monkeypatch.setattr(discovery, "_SCAN_TIMEOUT", 0.05)
    devices = await find_all_devices([FakeGrblDriver], ports=["/dev/x"])
    assert devices == []


@pytest.mark.asyncio
async def test_scan_failure_returns_empty(monkeypatch):
    from rayforge.machine.driver import discovery

    async def broken_scan(**kwargs):
        raise OSError("boom")

    monkeypatch.setattr(discovery, "scan_serial_ports", broken_scan)
    devices = await find_all_devices([FakeGrblDriver], ports=["/dev/x"])
    assert devices == []


@pytest.mark.asyncio
async def test_builtin_discovery_drivers():
    from rayforge.machine.driver import (
        GrblSerialDriver,
        GrblTelnetDriver,
        MarlinSerialDriver,
        drivers,
    )

    discoverable = [d for d in drivers if d.DISCOVERY is not None]
    assert {d.__name__ for d in discoverable} == {
        "GrblSerialDriver",
        "MarlinSerialDriver",
    }
    assert GrblTelnetDriver.DISCOVERY is None
    assert GrblSerialDriver.DISCOVERY is not None
    assert GrblSerialDriver.DISCOVERY.firmware == "grbl"
    assert MarlinSerialDriver.DISCOVERY is not None
    assert MarlinSerialDriver.DISCOVERY.firmware == "marlin"


def test_extract_banner():
    assert extract_banner(b"\r\nGrbl 1.1f\r\nok\r\n") == "Grbl 1.1f"
    assert extract_banner(b"") is None
    assert extract_banner(b"\n\n") is None


def test_normalize_tokens():
    tokens = normalize_tokens("Ortur Laser_Master", "CH340", None)
    # Generic chip names stay in the token set; they are filtered at
    # match time via GENERIC_TOKENS.
    assert tokens == {"ortur", "laser", "master", "ch340"}
    assert normalize_tokens(None, "") == frozenset()
    assert "ch340" in GENERIC_TOKENS


def test_build_identity():
    info = SerialPortInfo("/dev/ttyUSB0", "USB Serial", vid=1, pid=2)
    identity = build_identity("grbl", b"Grbl 1.1f\r\n", info)
    assert identity.firmware == "grbl"
    assert identity.banner == "Grbl 1.1f"
    assert identity.usb_vid == 1
    assert identity.usb_pid == 2
    assert "usb" in identity.tokens
    assert "serial" in identity.tokens


def test_discovered_device_key():
    device = DiscoveredDevice(
        driver_name="GrblSerialDriver",
        params={"port": "/dev/ttyUSB0", "baudrate": 115200},
        label="GRBL device",
        detail="/dev/ttyUSB0 at 115200 baud",
    )
    assert device.key == "GrblSerialDriver:/dev/ttyUSB0"
    assert DeviceIdentity() == DeviceIdentity()
