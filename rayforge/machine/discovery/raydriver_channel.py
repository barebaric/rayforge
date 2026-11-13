"""Discovery channel for raydriver daemons (``_raydriver._tcp``).

A daemon is an endpoint, not a firmware: the channel browses the
service type, speaks the control protocol's welcome, and lists the
daemon's open sessions. Each session becomes a
:class:`DiscoveredDevice` with the driver class for its protocol
and the remote endpoint in its params. Token-protected daemons
cannot be enumerated without credentials; they surface as a single
daemon row that seeds the remote setup vars, and the wizard probes
once the user supplies the token.
"""

import asyncio
import logging
from gettext import gettext as _
from typing import Any

from raydriver.remote import protocol

from ..transport.mdns_scan import MDNSService, scan_mdns_services
from .types import DeviceIdentity, DiscoveredDevice

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_raydriver._tcp.local."
_SCAN_TIMEOUT = 10.0
_QUERY_TIMEOUT = 3.0

_PROTOCOL_DRIVERS = {
    "grbl_simple": "GrblSerialSimpleDriver",
    "smoothie": "SmoothieDriver",
    "marlin": "MarlinSerialDriver",
}


def _driver_name_for_session(
    protocol_name: str, config: dict[str, Any]
) -> str | None:
    if protocol_name == "grbl":
        return "GrblTelnetDriver" if config.get("host") else "GrblSerialDriver"
    return _PROTOCOL_DRIVERS.get(protocol_name)


def _driver_label(driver_name: str) -> str:
    from ..driver import get_driver_cls

    cls = get_driver_cls(driver_name)
    return cls.label if cls is not None else driver_name


async def find_raydriver_devices() -> list[DiscoveredDevice]:
    """Browse for raydriver daemons and list their sessions.

    Like every channel, this never raises: failures are logged and
    bounded by timeouts.
    """
    try:
        services = await asyncio.wait_for(
            scan_mdns_services({SERVICE_TYPE}), timeout=_SCAN_TIMEOUT
        )
    except asyncio.TimeoutError:
        logger.warning("raydriver daemon discovery timed out")
        return []
    except Exception:
        logger.exception("raydriver daemon discovery failed")
        return []
    devices: list[DiscoveredDevice] = []
    for service in services:
        devices.extend(await _query_daemon(service))
    return devices


async def _query_daemon(service: MDNSService) -> list[DiscoveredDevice]:
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(service.host, service.port),
            timeout=_QUERY_TIMEOUT,
        )
    except (OSError, asyncio.TimeoutError):
        return [_daemon_row(service, _("unreachable"))]
    try:
        welcome = await _read_message(reader)
        if welcome is None or welcome.get("type") != protocol.WELCOME:
            return []
        if welcome["payload"].get("auth_required"):
            return [_daemon_row(service, _("token required"))]
        await protocol.write_frame(
            writer, protocol.request(1, protocol.SESSION_LIST)
        )
        reply = await _read_message(reader)
        if reply is None or not reply.get("ok"):
            return [_daemon_row(service, _("token required"))]
        rows: list[DiscoveredDevice] = []
        for session in reply["payload"].get("sessions", []):
            config = session.get("config") or {}
            driver_name = _driver_name_for_session(
                session.get("protocol", ""), config
            )
            if driver_name is None:
                continue
            params = {
                "remote": True,
                "remote_host": service.host,
                "remote_port": service.port,
                **config,
            }
            rows.append(
                DiscoveredDevice(
                    driver_name=driver_name,
                    params=params,
                    label=_driver_label(driver_name),
                    detail=_("{host}:{port} via raydriver daemon").format(
                        host=service.host, port=service.port
                    ),
                    identity=DeviceIdentity(banner=service.name or None),
                )
            )
        return rows or [_daemon_row(service, _("no sessions"))]
    except (
        ConnectionError,
        OSError,
        asyncio.TimeoutError,
        protocol.ProtocolError,
    ):
        return [_daemon_row(service, _("unreachable"))]
    finally:
        writer.close()


async def _read_message(reader: asyncio.StreamReader) -> dict | None:
    """Read one frame, skipping keepalive pings."""
    while True:
        msg = await asyncio.wait_for(
            protocol.read_frame(reader), timeout=_QUERY_TIMEOUT
        )
        if msg.get("type") == protocol.PING:
            continue
        return msg


def _daemon_row(service: MDNSService, note: str) -> DiscoveredDevice:
    """A daemon-level row for endpoints that cannot be enumerated
    without credentials (or at all). Seeds the remote setup vars;
    the driver defaults to GRBL serial and the wizard probes once
    the connection is configured."""
    return DiscoveredDevice(
        driver_name="GrblSerialDriver",
        params={
            "remote": True,
            "remote_host": service.host,
            "remote_port": service.port,
        },
        label=_("raydriver daemon"),
        detail=f"{service.host}:{service.port} ({note})",
        identity=DeviceIdentity(banner=service.name or None),
    )


__all__ = ["SERVICE_TYPE", "find_raydriver_devices"]
