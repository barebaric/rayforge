"""Discovery channel for raydriver daemons (``_raydriver._tcp``).

A daemon is an endpoint, not a firmware: the channel browses the
service type, speaks the control protocol's welcome, and lists the
daemon's devices. Every device becomes a :class:`DiscoveredDevice`
using the single generic remote driver; the protocol is carried as
info (``_raydriver_protocol``), never used to pick a driver class.
Token-protected daemons cannot be enumerated without credentials;
they surface as a single daemon row that seeds the endpoint vars,
and the wizard probes once the user supplies the token.
"""

import asyncio
import logging
from gettext import gettext as _

from raydriver.remote import protocol

from ..transport.mdns_scan import MDNSService, scan_mdns_services
from .types import DeviceIdentity, DiscoveredDevice

logger = logging.getLogger(__name__)

SERVICE_TYPE = "_raydriver._tcp.local."
_SCAN_TIMEOUT = 10.0
_QUERY_TIMEOUT = 3.0

#: The one driver used for every raydriver-hosted device.
REMOTE_DRIVER = "RaydriverRemoteDriver"


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
            writer, protocol.request(1, protocol.DEVICE_LIST)
        )
        reply = await _read_message(reader)
        if reply is None or not reply.get("ok"):
            return [_daemon_row(service, _("token required"))]
        rows: list[DiscoveredDevice] = []
        for device in reply["payload"].get("devices", []):
            ref = device.get("device")
            if not ref:
                continue
            rows.append(
                DiscoveredDevice(
                    driver_name=REMOTE_DRIVER,
                    params={
                        "remote_host": service.host,
                        "remote_port": service.port,
                        "device": ref,
                        "_raydriver_protocol": device.get("protocol", ""),
                    },
                    label=device.get("label") or ref,
                    detail=_("{host}:{port} via raydriver daemon").format(
                        host=service.host, port=service.port
                    ),
                    identity=DeviceIdentity(banner=service.name or None),
                )
            )
        return rows or [_daemon_row(service, _("no devices"))]
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
    without credentials (or at all). Seeds the endpoint vars; the
    user picks a device and the wizard probes once the connection
    is configured."""
    return DiscoveredDevice(
        driver_name=REMOTE_DRIVER,
        params={
            "remote_host": service.host,
            "remote_port": service.port,
        },
        label=_("raydriver daemon"),
        detail=f"{service.host}:{service.port} ({note})",
        identity=DeviceIdentity(banner=service.name or None),
    )


__all__ = ["SERVICE_TYPE", "find_raydriver_devices"]
