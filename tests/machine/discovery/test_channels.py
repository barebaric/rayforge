import asyncio
import time
from typing import ClassVar

import pytest

from rayforge.machine.discovery import (
    GENERIC_TOKENS,
    DiscoverySpec,
    MdnsRecognizer,
    SerialRecognizer,
    build_identity,
    find_all_devices,
    find_network_devices,
    mdns_channel,
    normalize_tokens,
    serial_channel,
)
from rayforge.machine.driver import (
    GrblNetworkDriver,
    GrblSerialDriver,
    GrblTelnetDriver,
    MarlinSerialDriver,
    OctoPrintDriver,
    drivers,
)
from rayforge.machine.driver.grbl.grbl_fingerprint import (
    fingerprint_grbl_http,
)
from rayforge.machine.transport import serial_scan
from rayforge.machine.transport.mdns_scan import MDNSService
from rayforge.machine.transport.serial import SerialPortInfo


def _grbl_matcher(data: bytes) -> bool:
    # Mirrors the real is_grbl_output(): banner or realtime status
    # report. A bare "ok" ack must NOT match (Marlin acks alike).
    return b"Grbl" in data or b"<Idle" in data


def _grbl_name(data: bytes) -> str | None:
    # Mirrors the real extract_device_name_from_output(): the first
    # informative line, skipping bare "ok" acks.
    for raw in data.decode("ascii", errors="replace").splitlines():
        line = raw.strip()
        if line and line != "ok":
            return line[:80]
    return None


def _marlin_matcher(data: bytes) -> bool:
    return b"start" in data or b"echo:" in data


def _marlin_name(data: bytes) -> str | None:
    # Mirrors the real extract_marlin_banner_from_output(): the first
    # boot message line, skipping "ok" acks.
    for raw in data.decode("ascii", errors="replace").splitlines():
        line = raw.strip()
        if not line or line == "ok":
            continue
        if line.startswith(("start", "Marlin")) or "echo:" in line:
            return line[:80]
    return None


class FakeGrblDriver:
    DISCOVERY = DiscoverySpec(
        serial=SerialRecognizer(
            label=lambda: "GRBL device",
            matches=_grbl_matcher,
            name=_grbl_name,
            firmware="grbl",
        )
    )


class FakeMarlinDriver:
    DISCOVERY = DiscoverySpec(
        serial=SerialRecognizer(
            label=lambda: "Marlin device",
            matches=_marlin_matcher,
            name=_marlin_name,
            firmware="marlin",
        )
    )


class FakeOctoPrintDriver:
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(services=("_octoprint._tcp.local.",))
    )
    label = "OctoPrint"


class FakeOctoPrintWithTxtDriver:
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(
            services=("_octoprint._tcp.local.",),
            txt_map={"path": "path"},
        )
    )
    label = "OctoPrint"


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
    monkeypatch.setattr(serial_scan, "_BANNER_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_NUDGE_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_DRAIN_TIMEOUT", 0.05)
    monkeypatch.setattr(serial_scan, "_QUIET_GAP", 0.01)
    monkeypatch.setattr(serial_scan, "_READ_CHUNK", 0.01)
    FakeSerial.attempts.clear()


@pytest.fixture(autouse=True)
def no_mdns(monkeypatch):
    """Tests opt into mDNS explicitly; by default no browse happens
    (find_all_devices also scans the network)."""

    async def _no_browse(service_types):
        return []

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", _no_browse)


def _patch_serial(monkeypatch, responses):
    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        lambda **kw: FakeSerial(responses=responses, **kw),
    )


# ----- serial channel ---------------------------------------------------


def test_normalize_tokens():
    tokens = normalize_tokens("Ortur Laser_Master", "CH340", None)
    # Generic chip names stay in the token set; they are filtered at
    # match time via GENERIC_TOKENS.
    assert tokens == {"ortur", "laser", "master", "ch340"}
    assert normalize_tokens(None, "") == frozenset()
    assert "ch340" in GENERIC_TOKENS


def test_build_identity():
    info = SerialPortInfo("/dev/ttyUSB0", "USB Serial", vid=1, pid=2)
    identity = build_identity("grbl", info)
    assert identity.firmware == "grbl"
    assert identity.banner is None
    assert identity.usb_vid == 1
    assert identity.usb_pid == 2
    assert "usb" in identity.tokens
    assert "serial" in identity.tokens


def test_build_identity_uses_device_name_as_banner():
    info = SerialPortInfo("/dev/ttyUSB0", "USB Serial", vid=1, pid=2)
    identity = build_identity("grbl", info, "Grbl 1.1f")
    assert identity.banner == "Grbl 1.1f"
    assert "grbl" in identity.tokens


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
    assert marlin.identity.banner == "start"

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
    # The bare "ok" ack is skipped in favour of the status report,
    # which is a more informative banner for a bannerless device.
    assert devices[0].identity.banner == "<Idle|MPos:0.0,0.0,0.0>"


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
        DISCOVERY = None

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
        DISCOVERY = DiscoverySpec(
            serial=SerialRecognizer(
                label=lambda: "Broken",
                matches=boom,
            )
        )

    devices = await find_all_devices(
        [BrokenDriver, FakeGrblDriver], ports=["/dev/ttyUSB0"]
    )
    assert [d.driver_name for d in devices] == ["FakeGrblDriver"]


@pytest.mark.asyncio
async def test_silent_ports_do_not_hide_responsive_ones(monkeypatch):
    """Silent adapters used to burn the whole scan budget, and hitting
    that budget discarded every observation. Now ports that answered
    before the deadline are still reported, so a responsive machine
    later in sort order is found alongside dumb adapters."""

    class Serial(FakeSerial):
        def read(self, size=1):
            if "SLOW" in str(self.port) and not self._pending:
                time.sleep(0.3)
            return super().read(size)

    monkeypatch.setattr(
        "rayforge.machine.transport.serial_scan.serial.Serial",
        lambda **kw: Serial(
            responses={"/dev/ttyUSB0": b"Grbl 1.1f\r\n"}, **kw
        ),
    )
    monkeypatch.setattr(serial_channel, "SERIAL_SCAN_TIMEOUT", 0.5)

    devices = await find_all_devices(
        [FakeGrblDriver],
        ports=["/dev/ttySLOW1", "/dev/ttyUSB0", "/dev/ttySLOW2"],
    )
    assert [d.params["port"] for d in devices] == ["/dev/ttyUSB0"]


@pytest.mark.asyncio
async def test_scan_failure_returns_empty(monkeypatch):
    async def broken_scan(**kwargs):
        raise OSError("boom")

    monkeypatch.setattr(serial_channel, "scan_serial_ports", broken_scan)
    devices = await find_all_devices([FakeGrblDriver], ports=["/dev/x"])
    assert devices == []


def test_builtin_discovery_drivers():
    discoverable = [d for d in drivers if d.DISCOVERY is not None]
    assert {d.__name__ for d in discoverable} == {
        "GrblSerialDriver",
        "GrblNetworkDriver",
        "MarlinSerialDriver",
        "OctoPrintDriver",
    }
    assert GrblTelnetDriver.DISCOVERY is None
    grbl_spec = GrblSerialDriver.DISCOVERY
    assert grbl_spec is not None
    assert grbl_spec.serial is not None
    assert grbl_spec.serial.firmware == "grbl"
    assert grbl_spec.serial.name is not None
    marlin_spec = MarlinSerialDriver.DISCOVERY
    assert marlin_spec is not None
    assert marlin_spec.serial is not None
    assert marlin_spec.serial.firmware == "marlin"
    assert marlin_spec.serial.name is not None


# ----- network channel --------------------------------------------------


def _octoprint_service(**overrides):
    defaults = {
        "service_type": "_octoprint._tcp",
        "name": "OctoPrint on octopi",
        "host": "192.168.1.42",
        "port": 80,
        "server": "octopi.local",
        "txt": {},
    }
    defaults.update(overrides)
    return MDNSService(**defaults)


@pytest.mark.asyncio
async def test_network_devices_from_mdns(monkeypatch):
    async def fake_scan(service_types):
        assert service_types == {"_octoprint._tcp"}
        return [_octoprint_service()]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([FakeOctoPrintDriver])
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == "FakeOctoPrintDriver"
    assert device.params == {"host": "192.168.1.42", "port": 80}
    assert device.label == "OctoPrint"
    assert device.detail == "192.168.1.42:80 (octopi.local)"
    assert device.key == "FakeOctoPrintDriver:192.168.1.42:80"
    assert device.identity.banner == "OctoPrint on octopi"
    assert "octoprint" in device.identity.tokens
    assert "octopi" in device.identity.tokens


@pytest.mark.asyncio
async def test_mdns_txt_forwarded_into_params(monkeypatch):
    """A TXT key declared in the recognizer's txt_map is forwarded
    into the discovered device's params under the mapped arg key."""
    service = _octoprint_service(
        txt={"path": "/octoprint", "version": "1.10.3", "api": "0.1"}
    )

    async def fake_scan(service_types):
        return [service]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([FakeOctoPrintWithTxtDriver])
    assert len(devices) == 1
    device = devices[0]
    assert device.params == {
        "host": "192.168.1.42",
        "port": 80,
        "path": "/octoprint",
    }


@pytest.mark.asyncio
async def test_mdns_txt_empty_value_not_forwarded(monkeypatch):
    """A TXT key whose value is empty is not forwarded, so the
    driver's own default applies."""
    service = _octoprint_service(txt={"path": ""})

    async def fake_scan(service_types):
        return [service]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([FakeOctoPrintWithTxtDriver])
    assert len(devices) == 1
    assert devices[0].params == {"host": "192.168.1.42", "port": 80}


@pytest.mark.asyncio
async def test_mdns_txt_vendor_model_enrich_identity(monkeypatch):
    """vendor and model TXT keys strengthen the identity token set
    used for profile matching; version becomes the banner when the
    service announced no instance name."""
    service = _octoprint_service(
        name="",
        server="octopi.local",
        txt={
            "vendor": "Prusa Research",
            "model": "MK4",
            "version": "OctoPrint 1.10.3",
        },
    )

    async def fake_scan(service_types):
        return [service]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([FakeOctoPrintWithTxtDriver])
    assert len(devices) == 1
    device = devices[0]
    # version is the banner because no instance name was announced
    assert device.identity.banner == "OctoPrint 1.10.3"
    assert "prusa" in device.identity.tokens
    assert "mk4" in device.identity.tokens


@pytest.mark.asyncio
async def test_mdns_txt_instance_name_preferred_as_banner(monkeypatch):
    """When the service announces an instance name it is the banner,
    even if a version TXT is also present."""
    service = _octoprint_service(
        name="OctoPrint on octopi",
        txt={"version": "1.10.3"},
    )

    async def fake_scan(service_types):
        return [service]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([FakeOctoPrintWithTxtDriver])
    assert devices[0].identity.banner == "OctoPrint on octopi"


@pytest.mark.asyncio
async def test_esp3d_service_maps_to_grbl_network_driver(monkeypatch):
    """ESP3D v3's dedicated _esp3d._tcp announcements map to the
    GrblNetworkDriver with connectable params (the driver's own
    defaults cover ws_port and protocol)."""

    async def fake_scan(service_types):
        # GrblNetworkDriver fingerprints, so the generic _http._tcp
        # is browsed alongside the declared service types.
        assert service_types == {
            "_esp3d._tcp",
            "_octoprint._tcp",
            "_http._tcp",
        }
        return [
            MDNSService(
                service_type="_esp3d._tcp",
                name="ESP3D",
                host="192.168.1.60",
                port=80,
                server="esp3d.local",
                txt={"firmware": "ESP3D", "version": "3.0"},
            )
        ]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices(
        [FakeOctoPrintDriver, GrblNetworkDriver]
    )
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == "GrblNetworkDriver"
    assert device.params == {"host": "192.168.1.60", "port": 80}
    assert device.label == "GRBL (Network)"
    assert device.detail == "192.168.1.60:80 (esp3d.local)"
    assert device.identity.firmware == "grblnetwork"
    assert device.identity.banner == "ESP3D"
    # The version TXT feeds identity tokens for profile matching.
    assert "esp3d" in device.identity.tokens


@pytest.mark.asyncio
async def test_mdns_declaration_without_local_suffix(monkeypatch):
    """Service type declarations and browse results are normalized
    on both sides, so "_octoprint._tcp" finds "_octoprint._tcp.local."
    announcements."""

    class BareDeclarationDriver:
        DISCOVERY = DiscoverySpec(
            mdns=MdnsRecognizer(services=("_octoprint._tcp",))
        )
        label = "Bare"

    async def fake_scan(service_types):
        assert service_types == {"_octoprint._tcp"}
        return [_octoprint_service()]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([BareDeclarationDriver])
    assert [d.driver_name for d in devices] == ["BareDeclarationDriver"]


@pytest.mark.asyncio
async def test_find_all_devices_merges_serial_and_network(monkeypatch):
    _patch_serial(monkeypatch, {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"})

    async def fake_mdns(service_types):
        return [_octoprint_service()]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_mdns)
    devices = await find_all_devices(
        [FakeGrblDriver, FakeOctoPrintDriver], ports=["/dev/ttyUSB0"]
    )
    assert {d.driver_name for d in devices} == {
        "FakeGrblDriver",
        "FakeOctoPrintDriver",
    }


@pytest.mark.asyncio
async def test_no_mdns_declarations_skips_network_scan(monkeypatch):
    calls = []

    async def fake_mdns(service_types):
        calls.append(service_types)
        return []

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_mdns)
    _patch_serial(monkeypatch, {"/dev/ttyUSB0": b"Grbl 1.1f\r\n"})
    devices = await find_all_devices(
        [FakeGrblDriver, FakeMarlinDriver], ports=["/dev/ttyUSB0"]
    )
    assert [d.driver_name for d in devices] == ["FakeGrblDriver"]
    assert calls == []


@pytest.mark.asyncio
async def test_network_scan_timeout_returns_empty(monkeypatch):
    async def slow_scan(service_types):
        await asyncio.sleep(1.0)
        return []

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", slow_scan)
    monkeypatch.setattr(mdns_channel, "NETWORK_SCAN_TIMEOUT", 0.05)
    assert await find_network_devices([FakeOctoPrintDriver]) == []


@pytest.mark.asyncio
async def test_network_scan_failure_returns_empty(monkeypatch):
    async def broken_scan(service_types):
        raise OSError("boom")

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", broken_scan)
    assert await find_network_devices([FakeOctoPrintDriver]) == []


def test_builtin_mdns_declarations():
    declared = {}
    for cls in drivers:
        spec = cls.DISCOVERY
        if spec is not None and spec.mdns is not None:
            declared[cls.__name__] = spec.mdns.services
    assert declared == {
        "OctoPrintDriver": ("_octoprint._tcp.local.",),
        "GrblNetworkDriver": ("_esp3d._tcp.local.",),
    }
    octo_spec = OctoPrintDriver.DISCOVERY
    assert octo_spec is not None and octo_spec.mdns is not None
    assert octo_spec.mdns.txt_map == {"path": "path"}
    grbl_net_spec = GrblNetworkDriver.DISCOVERY
    assert grbl_net_spec is not None and grbl_net_spec.mdns is not None
    assert grbl_net_spec.mdns.fingerprint is fingerprint_grbl_http


# ----- fingerprint pass -------------------------------------------------


def _fingerprint_driver(fingerprint, name="FingerprintDriver"):
    """A fresh driver class claiming _esp3d._tcp by declaration and
    _http._tcp candidates only via its fingerprint probe."""

    class FingerprintDriver:
        DISCOVERY = DiscoverySpec(
            mdns=MdnsRecognizer(
                services=("_esp3d._tcp.local.",),
                fingerprint=fingerprint,
            )
        )
        label = "GRBL (Network)"

    FingerprintDriver.__name__ = name
    return FingerprintDriver


def _http_service(host="192.168.1.70", port=80, **txt):
    return MDNSService(
        service_type="_http._tcp",
        name="FluidNC",
        host=host,
        port=port,
        server="fluidnc.local",
        txt=dict(txt),
    )


@pytest.mark.asyncio
async def test_http_candidate_fingerprinted_and_claimed(monkeypatch):
    """A FluidNC announcing only generic _http._tcp is confirmed by
    the fingerprint probe and claimed by the driver."""

    async def ok_fingerprint(host, port, timeout=1.5):
        return _http_service()

    async def fake_scan(service_types):
        assert service_types == {"_esp3d._tcp", "_http._tcp"}
        return [_http_service()]

    driver = _fingerprint_driver(ok_fingerprint)
    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([driver])
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == "FingerprintDriver"
    assert device.params == {"host": "192.168.1.70", "port": 80}
    assert device.label == "GRBL (Network)"


@pytest.mark.asyncio
async def test_non_matching_http_candidate_ignored(monkeypatch):
    async def no_match(host, port, timeout=1.5):
        return None

    async def fake_scan(service_types):
        return [_http_service()]

    driver = _fingerprint_driver(no_match)
    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    assert await find_network_devices([driver]) == []


@pytest.mark.asyncio
async def test_declared_service_wins_over_fingerprint(monkeypatch):
    """A host already claimed via its declared service type must not
    be fingerprinted again."""
    calls = []

    async def spy_fingerprint(host, port, timeout=1.5):
        calls.append((host, port))
        return _http_service(host=host)

    async def fake_scan(service_types):
        return [
            MDNSService(
                service_type="_esp3d._tcp",
                name="ESP3D",
                host="192.168.1.60",
                port=80,
                server="esp3d.local",
                txt={},
            ),
            _http_service(host="192.168.1.60"),
        ]

    driver = _fingerprint_driver(spy_fingerprint)
    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([driver])
    assert len(devices) == 1
    assert devices[0].params["host"] == "192.168.1.60"
    assert devices[0].identity.banner == "ESP3D"
    assert calls == []


@pytest.mark.asyncio
async def test_failing_fingerprint_is_isolated(monkeypatch):
    """One candidate failing (or refusing) to fingerprint does not
    prevent other candidates from being claimed."""

    async def match_only_second(host, port, timeout=1.5):
        if host == "192.168.1.71":
            return _http_service(host=host)
        return None

    async def fake_scan(service_types):
        return [_http_service(), _http_service(host="192.168.1.71")]

    driver = _fingerprint_driver(match_only_second)
    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([driver])
    assert [d.params["host"] for d in devices] == ["192.168.1.71"]


@pytest.mark.asyncio
async def test_first_declared_driver_claims_host(monkeypatch):
    """When several fingerprinting drivers match the same host, the
    first one in declaration order claims it; every host yields at
    most one device."""

    async def always_match(host, port, timeout=1.5):
        return _http_service(host=host)

    first = _fingerprint_driver(always_match, name="FirstDriver")
    second = _fingerprint_driver(always_match, name="SecondDriver")

    async def fake_scan(service_types):
        return [_http_service(), _http_service(host="192.168.1.71")]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    devices = await find_network_devices([first, second])
    assert [d.params["host"] for d in devices] == [
        "192.168.1.70",
        "192.168.1.71",
    ]
    assert all(d.driver_name == "FirstDriver" for d in devices)


@pytest.mark.asyncio
async def test_no_candidates_skips_fingerprint_pass(monkeypatch):
    """Without _http._tcp announcements no probe runs; declared-service
    browsing works unchanged."""

    async def spy_fingerprint(host, port, timeout=1.5):
        raise AssertionError("must not be called")

    async def fake_scan(service_types):
        assert service_types == {"_esp3d._tcp"}
        return []

    driver = _fingerprint_driver(spy_fingerprint)
    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    assert await find_network_devices([driver]) == []
