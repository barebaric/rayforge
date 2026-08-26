"""mDNS network service scanning.

The network counterpart to :mod:`rayforge.machine.transport.serial_scan`:
a browse runs for a short, bounded window, every service announcement
on the watched types is resolved to an address, and the result is
captured as an :class:`MDNSService`.

Deciding which driver could talk to a discovered service is a
separate concern that lives with the drivers (see
:mod:`rayforge.machine.discovery`).

This module is GTK-free so it can be unit-tested in isolation. It
must work both with current zeroconf releases and the older ones
shipped by Linux distributions (e.g. Ubuntu 24.04 has 0.131), so it
sticks to long-stable APIs and never raises: a failing or slow
browse yields an empty result.

A scanner is created and closed per scan: nothing keeps sockets
open between two discovery runs.
"""

import asyncio
import ipaddress
import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from zeroconf import ServiceListener, Zeroconf
from zeroconf.asyncio import (
    AsyncServiceBrowser,
    AsyncServiceInfo,
    AsyncZeroconf,
)

logger = logging.getLogger(__name__)

# How long to listen for service announcements.
_BROWSE_WINDOW_S = 2.0
# Grace period for resolving services announced near the end of
# the browse window.
_RESOLVE_GRACE_S = 1.0
# Per-service resolve timeout, in milliseconds.
_RESOLVE_TIMEOUT_MS = 1500


@dataclass(frozen=True)
class MDNSService:
    """One service resolved during an mDNS browse."""

    #: Normalized service type, e.g. ``_octoprint._tcp``.
    service_type: str
    #: Instance name, e.g. ``OctoPrint on octopi``.
    name: str
    #: Address to connect to; IPv4 preferred over IPv6.
    host: str
    port: int
    #: The server's own hostname, e.g. ``octopi.local``.
    server: str = ""
    #: TXT record contents, values decoded to strings.
    txt: dict = field(default_factory=dict)


def normalize_service_type(service_type: str) -> str:
    """
    Reduces a service type to its canonical ``_name._transport``
    form: ``_octoprint._tcp.local.`` becomes ``_octoprint._tcp``.
    """
    parts = [
        part
        for part in service_type.strip(".").split(".")
        if part and part.lower() != "local"
    ]
    return ".".join(parts)


def _instance_name(name: str, service_type: str) -> str:
    """Strips the service type suffix from a full instance name."""
    stype = service_type.strip(".")
    label = name.rsplit(f".{stype}", 1)[0]
    return label.rstrip(".")


def _decode_txt(properties: dict) -> dict:
    txt: dict[str, str] = {}
    for key, value in (properties or {}).items():
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        txt[str(key)] = value
    return txt


def _address_version(address: str) -> int | None:
    """The IP version of *address*, or None if it is malformed."""
    try:
        return ipaddress.ip_address(address).version
    except ValueError:
        return None


class _ResolvedServiceInfo(Protocol):
    """The slice of zeroconf's ServiceInfo that :func:`_build_service`
    reads. A Protocol keeps the helper decoupled from the concrete
    zeroconf class so it can be unit-tested with plain fakes."""

    @property
    def type(self) -> str: ...

    @property
    def name(self) -> str: ...

    @property
    def port(self) -> int | None: ...

    @property
    def server(self) -> str | bytes | None: ...

    @property
    def properties(self) -> dict: ...

    def parsed_addresses(self) -> list[str]: ...


def _build_service(info: _ResolvedServiceInfo) -> MDNSService | None:
    """
    Converts resolved zeroconf info into an :class:`MDNSService`,
    preferring IPv4 addresses. Returns None when the service could
    not be resolved to any address.
    """
    addresses = info.parsed_addresses()
    versions = [_address_version(addr) for addr in addresses]
    ipv4 = [addr for addr, version in zip(addresses, versions) if version == 4]
    chosen = ipv4[0] if ipv4 else (addresses[0] if addresses else None)
    if chosen is None:
        return None
    server = info.server or ""
    if isinstance(server, bytes):
        server = server.decode("utf-8", errors="replace")
    return MDNSService(
        service_type=normalize_service_type(info.type),
        name=_instance_name(info.name, info.type),
        host=chosen,
        port=info.port or 0,
        server=server.rstrip("."),
        txt=_decode_txt(info.properties),
    )


class _ServiceCollector(ServiceListener):
    """
    Browse listener that resolves every announced service into an
    :class:`MDNSService`.

    zeroconf invokes the listener callbacks on its own thread;
    resolution is therefore scheduled onto the caller's event loop,
    which is captured when the collector is created.
    """

    def __init__(self) -> None:
        self.services: dict[str, MDNSService] = {}
        self._loop = asyncio.get_running_loop()
        self._pending: set[asyncio.Task] = set()

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._schedule_resolve(zc, type_, name)

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self._schedule_resolve(zc, type_, name)

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        self.services.pop(name, None)

    async def wait_for_pending(self, timeout: float) -> None:
        """Waits up to *timeout* seconds for in-flight resolves."""
        # Let call_soon_threadsafe callbacks (which create the resolve
        # tasks) execute before checking what is pending.
        await asyncio.sleep(0)
        if self._pending:
            await asyncio.wait(set(self._pending), timeout=timeout)

    def _schedule_resolve(self, zc: Zeroconf, type_: str, name: str) -> None:
        def _start() -> None:
            task = self._loop.create_task(self._resolve(zc, type_, name))
            self._pending.add(task)
            task.add_done_callback(self._pending.discard)

        try:
            self._loop.call_soon_threadsafe(_start)
        except RuntimeError:
            # Event loop already closed during shutdown.
            pass

    async def _resolve(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = AsyncServiceInfo(type_, name)
        try:
            resolved = await info.async_request(zc, _RESOLVE_TIMEOUT_MS)
        except Exception:
            logger.debug("mDNS resolve of %s failed", name, exc_info=True)
            self.services.pop(name, None)
            return
        if not resolved:
            self.services.pop(name, None)
            return
        service = _build_service(info)
        if service is None:
            self.services.pop(name, None)
            return
        self.services[name] = service


async def scan_mdns_services(
    service_types: Iterable[str],
    browse_window: float = _BROWSE_WINDOW_S,
) -> list[MDNSService]:
    """
    Browses *service_types* over mDNS for a short window and returns
    every service that could be resolved to an address. Failures
    (no network, no responder) are logged and yield an empty list
    rather than raising.

    The browse runs on a private event loop in a worker thread.
    zeroconf otherwise attaches its engine to whichever event loop
    is running when it is constructed, which would hijack the
    application's loop (e.g. the GTK main loop) and starve other
    asyncio work there — serial-port probing in particular.
    """
    normalized = {
        t for t in (normalize_service_type(s) for s in service_types) if t
    }
    if not normalized:
        return []

    try:
        return await asyncio.to_thread(
            _browse_blocking, normalized, browse_window
        )
    except Exception:
        logger.debug("mDNS browse failed", exc_info=True)
        return []


def _browse_blocking(
    service_types: set[str], browse_window: float
) -> list[MDNSService]:
    """Synchronous entry point: runs one mDNS browse on a private
    event loop and returns the resolved services."""
    return asyncio.run(_browse(service_types, browse_window))


async def _browse(
    service_types: set[str], browse_window: float
) -> list[MDNSService]:
    collector = _ServiceCollector()
    try:
        aiozc = AsyncZeroconf()
    except Exception:
        logger.debug("mDNS not available", exc_info=True)
        return []

    try:
        AsyncServiceBrowser(
            aiozc.zeroconf,
            sorted(f"{t}.local." for t in service_types),
            listener=collector,
        )
        await asyncio.sleep(browse_window)
        await collector.wait_for_pending(_RESOLVE_GRACE_S)
    finally:
        try:
            await aiozc.async_close()
        except Exception:
            logger.debug("Error closing mDNS scanner", exc_info=True)

    return list(collector.services.values())


__all__ = [
    "MDNSService",
    "normalize_service_type",
    "scan_mdns_services",
]
