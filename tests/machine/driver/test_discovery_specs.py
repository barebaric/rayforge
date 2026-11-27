"""Discovery declarations of the built-in drivers.

The channels themselves are tested in raygeo; these tests pin the
built-in drivers' DiscoverySpec declarations and end-to-end recognition
of real firmware banners.
"""

import asyncio
import os
import sys
import threading

try:
    import tty
except ImportError:  # Windows has no termios/tty
    tty = None

import pytest
from raygeo.driver.core.discovery import mdns_channel
from raygeo.driver.core.discovery.engine import find_all_devices
from raygeo.driver.core.discovery.fingerprints import (
    HttpFingerprint,
    TcpFingerprint,
)
from raygeo.driver.core.discovery.mdns_channel import find_network_devices
from raygeo.driver.core.discovery.mdns_scan import MDNSService
from raygeo.driver.core.discovery.spec import DiscoverySpec, MdnsRecognizer

from rayforge.machine.driver import (
    GrblNetworkDriver,
    GrblSerialDriver,
    GrblTelnetDriver,
    MarlinSerialDriver,
    OctoPrintDriver,
    drivers,
)

SCULPFUN_ICUBE_BANNER = (
    b"[VER:1.0.15,20240923:]\r\n"
    b"[OPT:VMP,31,511]\r\n"
    b"[MSG:mechine:Sculpfun iCube]\r\n"
    b"[MSG:Mode=BT]\r\n"
    b"[BT_VER:8.1.2,FSC-BT836B]\r\n"
    b"Connection status: CONNECTED \r\n"
)


class FakeOctoPrintDriver:
    DISCOVERY = DiscoverySpec(
        mdns=MdnsRecognizer(services=("_octoprint._tcp.local.",))
    )
    label = "OctoPrint"


requires_pty = pytest.mark.skipif(
    not hasattr(os, "openpty") or tty is None or sys.platform == "darwin",
    reason="requires a POSIX pty; macOS ptys do not deliver master "
    "writes to a serialport reader on CI runners",
)


class PtyDevice:
    """A pty-backed fake serial device: the capture probe runs in
    Rust, so the device side is a real port. *banner* is written
    before the scan opens the port; *responder* answers the nudges."""

    def __init__(self, banner: bytes = b"", responder=None):
        self.master, self.slave = os.openpty()
        # Raw mode: no echo of the scanner's nudges and no CR/NL
        # translation, so captured bytes match what the device sent.
        if tty is not None:
            tty.setraw(self.slave)
        self.port = os.ttyname(self.slave)
        self._stop = threading.Event()
        self._thread = None
        if banner:
            os.write(self.master, banner)
        if responder is not None:
            self._thread = threading.Thread(
                target=self._serve, args=(responder,), daemon=True
            )
            self._thread.start()

    def _serve(self, responder):
        while not self._stop.is_set():
            try:
                data = os.read(self.master, 256)
            except OSError:
                return
            if not data:
                return
            reply = responder(data)
            if reply:
                os.write(self.master, reply)

    def close(self):
        self._stop.set()
        # Closing the slave first wakes a responder blocked on the
        # master; on macOS close(master) blocks while the slave is open.
        os.close(self.slave)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        os.close(self.master)


def _grbl_nudge_responder(data: bytes) -> bytes:
    """A bannerless GRBL device: ``ok`` for a newline, a status
    report for ``?``."""
    return {
        b"\n": b"ok\r\n",
        b"?": b"<Idle|MPos:0.0,0.0,0.0>\r\n",
    }.get(data, b"ok\r\n")


@pytest.fixture(autouse=True)
def no_mdns(monkeypatch):
    """Tests opt into mDNS explicitly; by default no browse happens
    (find_all_devices also scans the network)."""

    async def _no_browse(service_types):
        return []

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", _no_browse)


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
@requires_pty
async def test_sculpfun_banner_end_to_end():
    """The real GrblSerialDriver recognizer claims build-info-only
    banners and extracts the machine name for identity/matching."""
    port = PtyDevice(banner=SCULPFUN_ICUBE_BANNER)
    try:
        devices = await find_all_devices([GrblSerialDriver], ports=[port.port])
    finally:
        port.close()
    assert [d.driver_name for d in devices] == ["GrblSerialDriver"]
    device = devices[0]
    assert device.identity.firmware == "grbl"
    assert device.identity.banner == "Sculpfun iCube"
    assert "sculpfun" in device.identity.tokens
    assert "icube" in device.identity.tokens


def test_builtin_discovery_drivers():
    discoverable = [d for d in drivers if d.DISCOVERY is not None]
    assert {d.__name__ for d in discoverable} == {
        "GrblSerialDriver",
        "GrblNetworkDriver",
        "GrblTelnetDriver",
        "MarlinSerialDriver",
        "OctoPrintDriver",
    }
    # Telnet devices are found by the mDNS and subnet channels, not
    # the serial one: a telnet connection has no serial port to claim.
    telnet_spec = GrblTelnetDriver.DISCOVERY
    assert telnet_spec is not None
    assert telnet_spec.serial is None
    assert telnet_spec.mdns is not None
    assert telnet_spec.mdns.services == ()
    assert telnet_spec.mdns.fingerprint_services == ("_telnet._tcp",)
    fingerprint = telnet_spec.mdns.fingerprint
    assert isinstance(fingerprint, TcpFingerprint)
    assert fingerprint.probe_command == "$I\n"
    assert fingerprint.match_prefix == "[VER:"
    assert telnet_spec.tcp is not None
    assert telnet_spec.tcp.port == 23
    assert telnet_spec.tcp.probe_command == "$I\n"
    assert telnet_spec.tcp.match_prefix == "[VER:"
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
    assert device.label == "FluidNC / ESP3D (Web API)"
    assert device.detail == "192.168.1.60:80 (esp3d.local)"
    assert device.identity.firmware == "grblnetwork"
    assert device.identity.banner == "ESP3D"
    # The version TXT feeds identity tokens for profile matching.
    assert "esp3d" in device.identity.tokens


def test_builtin_mdns_declarations():
    declared = {}
    for cls in drivers:
        spec = cls.DISCOVERY
        if spec is not None and spec.mdns is not None:
            declared[cls.__name__] = spec.mdns.services
    assert declared == {
        "OctoPrintDriver": ("_octoprint._tcp.local.",),
        "GrblNetworkDriver": ("_esp3d._tcp.local.",),
        "GrblTelnetDriver": (),
    }
    octo_spec = OctoPrintDriver.DISCOVERY
    assert octo_spec is not None and octo_spec.mdns is not None
    assert octo_spec.mdns.txt_map == {"path": "path"}
    grbl_net_spec = GrblNetworkDriver.DISCOVERY
    assert grbl_net_spec is not None and grbl_net_spec.mdns is not None
    fingerprint = grbl_net_spec.mdns.fingerprint
    assert isinstance(fingerprint, HttpFingerprint)
    assert fingerprint.path == "/command?plain=%5BESP800%5D&PAGEID="
    assert fingerprint.match_regex == r"\b(fluidnc|esp3d|grblhal|grbl)\b"
    assert grbl_net_spec.mdns.fingerprint_services == ("_http._tcp",)
    telnet_spec = GrblTelnetDriver.DISCOVERY
    assert telnet_spec is not None and telnet_spec.mdns is not None
    assert isinstance(telnet_spec.mdns.fingerprint, TcpFingerprint)
    assert telnet_spec.mdns.fingerprint_services == ("_telnet._tcp",)


@pytest.mark.asyncio
async def test_telnet_mdns_candidate_is_claimed(monkeypatch):
    """A board announcing _telnet._tcp is confirmed by the driver's
    shared TCP fingerprint and claimed with SRV host/port params."""

    async def handler(reader, writer):
        try:
            await reader.readline()
            writer.write(b"[VER:1.1h.20240101:]\r\n[OPT:V,15,128]\r\n")
            await writer.drain()
            await reader.read()
        finally:
            writer.close()

    server = await asyncio.start_server(handler, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    seen = []

    async def fake_scan(service_types):
        seen.append(set(service_types))
        return [
            MDNSService(
                service_type="_telnet._tcp",
                name="grblHAL",
                host="127.0.0.1",
                port=port,
                server="grblhal.local",
                txt={},
            )
        ]

    monkeypatch.setattr(mdns_channel, "scan_mdns_services", fake_scan)
    try:
        devices = await find_network_devices([GrblTelnetDriver])
    finally:
        server.close()
        await server.wait_closed()
    assert seen == [{"_telnet._tcp"}]
    assert len(devices) == 1
    device = devices[0]
    assert device.driver_name == "GrblTelnetDriver"
    assert device.params == {"host": "127.0.0.1", "port": port}
    assert device.label == "grblHAL (Telnet)"
    assert device.identity.banner == "grblHAL"
