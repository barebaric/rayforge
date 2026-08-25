"""Tests for mDNS network service scanning."""

import asyncio
from typing import ClassVar, cast

import pytest
from zeroconf import ServiceListener, Zeroconf

from rayforge.machine.transport.mdns_scan import (
    MDNSService,
    _build_service,
    _ServiceCollector,
    normalize_service_type,
    scan_mdns_services,
)

_OCTOPRINT_TYPE = "_octoprint._tcp.local."
_OCTOPRINT_NAME = f"OctoPrint on octopi.{_OCTOPRINT_TYPE}"


class FakeAsyncServiceInfo:
    """Stands in for zeroconf's AsyncServiceInfo."""

    resolves = True

    def __init__(self, service_type: str, name: str):
        self.type = service_type
        self.name = name
        self.addresses = ["192.168.1.42", "fe80::1a2b"]
        self.port = 80
        self.server = "octopi.local."
        self.properties = {"path": b"/", "version": b"1.9.3"}
        self.resolves = type(self).resolves

    async def async_request(self, zeroconf, timeout):
        return self.resolves

    def parsed_addresses(self):
        return list(self.addresses)


@pytest.fixture
def fake_service_info(monkeypatch):
    monkeypatch.setattr(
        "rayforge.machine.transport.mdns_scan.AsyncServiceInfo",
        FakeAsyncServiceInfo,
    )
    return FakeAsyncServiceInfo


class FakeAsyncZeroconf:
    """Stands in for AsyncZeroconf; records close() calls."""

    instances: ClassVar[list["FakeAsyncZeroconf"]] = []
    raises_on_init: ClassVar[bool] = False

    def __init__(self):
        if FakeAsyncZeroconf.raises_on_init:
            raise OSError("no network")
        self.zeroconf = object()
        self.closed = False
        FakeAsyncZeroconf.instances.append(self)

    async def async_close(self):
        self.closed = True


class FakeBrowser:
    """Announces a known service as soon as it is created."""

    raises_on_init: ClassVar[bool] = False

    def __init__(
        self,
        zeroconf,
        service_types,
        listener: ServiceListener | None = None,
    ):
        if FakeBrowser.raises_on_init:
            raise OSError("boom")
        if listener is not None:
            for service_type in service_types:
                listener.add_service(zeroconf, service_type, _OCTOPRINT_NAME)


@pytest.fixture(autouse=True)
def _fake_zeroconf(monkeypatch):
    FakeAsyncZeroconf.instances = []
    FakeAsyncZeroconf.raises_on_init = False
    FakeBrowser.raises_on_init = False
    monkeypatch.setattr(
        "rayforge.machine.transport.mdns_scan.AsyncZeroconf",
        FakeAsyncZeroconf,
    )
    monkeypatch.setattr(
        "rayforge.machine.transport.mdns_scan.AsyncServiceBrowser",
        FakeBrowser,
    )


def test_normalize_service_type():
    assert (
        normalize_service_type("_octoprint._tcp.local.") == "_octoprint._tcp"
    )
    assert normalize_service_type("_octoprint._tcp") == "_octoprint._tcp"
    assert normalize_service_type("_ruida._udp.local") == "_ruida._udp"
    assert normalize_service_type("local.") == ""
    assert normalize_service_type("") == ""


def test_build_service_prefers_ipv4():
    service = _build_service(
        FakeAsyncServiceInfo(_OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    )
    assert isinstance(service, MDNSService)
    assert service.host == "192.168.1.42"
    assert service.port == 80
    assert service.server == "octopi.local"
    assert service.service_type == "_octoprint._tcp"
    assert service.name == "OctoPrint on octopi"
    assert service.txt == {"path": "/", "version": "1.9.3"}


def test_build_service_falls_back_to_ipv6():
    info = FakeAsyncServiceInfo(_OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    info.addresses = ["fe80::1a2b"]
    service = _build_service(info)
    assert service is not None
    assert service.host == "fe80::1a2b"


def test_build_service_without_address_returns_none():
    info = FakeAsyncServiceInfo(_OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    info.addresses = []
    assert _build_service(info) is None


@pytest.mark.asyncio
async def test_collector_resolves_and_stores_service(fake_service_info):
    collector = _ServiceCollector()
    zc = cast(Zeroconf, object())
    collector.add_service(zc, _OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    await collector.wait_for_pending(timeout=1.0)

    assert list(collector.services) == [_OCTOPRINT_NAME]
    service = collector.services[_OCTOPRINT_NAME]
    assert service.host == "192.168.1.42"
    assert service.service_type == "_octoprint._tcp"


@pytest.mark.asyncio
async def test_collector_drops_unresolved_service(fake_service_info):
    collector = _ServiceCollector()
    zc = cast(Zeroconf, object())
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(FakeAsyncServiceInfo, "resolves", False)
        collector.add_service(zc, _OCTOPRINT_TYPE, _OCTOPRINT_NAME)
        await collector.wait_for_pending(timeout=1.0)
    assert collector.services == {}

    # A later update may still resolve it.
    collector.update_service(zc, _OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    await collector.wait_for_pending(timeout=1.0)
    assert list(collector.services) == [_OCTOPRINT_NAME]


@pytest.mark.asyncio
async def test_collector_removes_removed_service(fake_service_info):
    collector = _ServiceCollector()
    zc = cast(Zeroconf, object())
    collector.add_service(zc, _OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    await collector.wait_for_pending(timeout=1.0)
    collector.remove_service(zc, _OCTOPRINT_TYPE, _OCTOPRINT_NAME)
    assert collector.services == {}


@pytest.mark.asyncio
async def test_scan_returns_resolved_services(fake_service_info):
    services = await scan_mdns_services(["_octoprint._tcp.local."])
    assert len(services) == 1
    assert services[0].host == "192.168.1.42"
    assert services[0].port == 80
    # The scanner is fully closed after every scan.
    assert FakeAsyncZeroconf.instances[0].closed is True


@pytest.mark.asyncio
async def test_scan_without_service_types_is_noop():
    assert await scan_mdns_services([]) == []
    assert await scan_mdns_services(["local."]) == []
    assert FakeAsyncZeroconf.instances == []


@pytest.mark.asyncio
async def test_scan_survives_zeroconf_failure(monkeypatch):
    monkeypatch.setattr(FakeAsyncZeroconf, "raises_on_init", True)
    assert await scan_mdns_services(["_octoprint._tcp"]) == []


@pytest.mark.asyncio
async def test_scan_survives_browse_failure(monkeypatch):
    monkeypatch.setattr(FakeBrowser, "raises_on_init", True)
    assert await scan_mdns_services(["_octoprint._tcp"]) == []
    assert FakeAsyncZeroconf.instances[0].closed is True


@pytest.mark.asyncio
async def test_scan_cancellable_from_caller(fake_service_info):
    """A caller-side timeout cancels the await without blocking the
    caller's event loop (the browse itself runs in a worker thread)."""
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(
            scan_mdns_services(["_octoprint._tcp.local."], browse_window=5.0),
            timeout=0.05,
        )


@pytest.mark.asyncio
async def test_scan_does_not_block_caller_loop(fake_service_info):
    """The browse runs off the caller's event loop, so other asyncio
    work keeps making progress while the scan is in flight. This is
    the regression guard: zeroconf must not attach to the app loop."""
    ticks = 0

    async def _ticker():
        nonlocal ticks
        for _ in range(5):
            await asyncio.sleep(0.02)
            ticks += 1

    await asyncio.gather(
        scan_mdns_services(["_octoprint._tcp.local."], browse_window=0.2),
        _ticker(),
    )
    assert ticks == 5
