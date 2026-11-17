"""Fingerprint probes for networked GRBL-family controllers.

The implementation behind the GRBL drivers' ``DISCOVERY.mdns``
fingerprints: some devices announce only a generic mDNS service type
that says nothing about their firmware. Each probe actively questions
a candidate host and reports a synthetic
:class:`~raydriver.core.discovery.mdns_scan.MDNSService` only when the
response identifies a known firmware.

The discovery engine knows nothing about GRBL — it only calls the
callables this module provides. Each probe is bounded by a timeout and
never raises: any failure yields None.
"""

import asyncio
import logging
import re

import aiohttp
from raydriver.core.discovery.mdns_scan import MDNSService

logger = logging.getLogger(__name__)

# The [ESP800] firmware-info endpoint served by ESP3D v3 / FluidNC.
_FW_INFO_URL = "/command?plain=%5BESP800%5D&PAGEID="

# Firmware names accepted as a GRBL-compatible match.
_GRBL_FW_RE = re.compile(r"\b(fluidnc|esp3d|grblhal|grbl)\b", re.IGNORECASE)

_DEFAULT_TIMEOUT_S = 1.5
_MAX_RESPONSE_BYTES = 4096

# grblHAL's build-info response line; the same prefix the telnet
# subnet recognizer matches after sending $I.
_TELNET_VER_PREFIX = "[VER:"
_TELNET_SERVICE_TYPE = "_telnet._tcp"
_TELNET_DEFAULT_TIMEOUT_S = 1.9
# One read window per phase: the unsolicited banner, then the reply to
# $I. Two windows plus the connect fit inside the channel's probe
# timeout.
_TELNET_WAIT_S = 0.9
_MAX_BANNER_LEN = 80


async def fingerprint_grbl_http(
    host: str, port: int, timeout: float = _DEFAULT_TIMEOUT_S
) -> MDNSService | None:
    """
    GETs the firmware-info endpoint at *host*:*port* and returns a
    synthetic :class:`MDNSService` when the response names known
    GRBL-family firmware, ``None`` otherwise.
    """
    url = f"http://{host}:{port}{_FW_INFO_URL}"
    try:
        text = await asyncio.wait_for(_fetch(url), timeout=timeout)
    except Exception:
        logger.debug(
            "HTTP fingerprint of %s:%s failed", host, port, exc_info=True
        )
        return None
    match = _GRBL_FW_RE.search(text)
    if not match:
        return None
    return MDNSService(
        service_type="_http._tcp",
        name=_first_line(text) or match.group(1),
        host=host,
        port=port,
    )


async def fingerprint_grbl_telnet(
    host: str, port: int, timeout: float = _TELNET_DEFAULT_TIMEOUT_S
) -> MDNSService | None:
    """
    Probes the raw telnet service at *host*:*port* for a grblHAL
    ``[VER:`` build-info line and returns a synthetic
    :class:`MDNSService` when one arrives, ``None`` otherwise.

    Some boards emit the line unsolicited on connect; the probe gives
    them a short window, then sends ``$I`` and listens again.
    """
    try:
        return await asyncio.wait_for(
            _probe_telnet(host, port), timeout=timeout
        )
    except Exception:
        logger.debug(
            "Telnet fingerprint of %s:%s failed", host, port, exc_info=True
        )
        return None


async def _probe_telnet(host: str, port: int) -> MDNSService | None:
    reader, writer = await asyncio.open_connection(host, port)
    try:
        banner = await _wait_for_build_info(reader)
        if banner is None:
            writer.write(b"$I\n")
            await writer.drain()
            banner = await _wait_for_build_info(reader)
        if banner is None:
            return None
        return MDNSService(
            service_type=_TELNET_SERVICE_TYPE,
            name=banner,
            host=host,
            port=port,
        )
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            logger.debug(
                "Closing telnet fingerprint connection failed",
                exc_info=True,
            )


async def _wait_for_build_info(
    reader: asyncio.StreamReader,
) -> str | None:
    """Reads lines for one short window and returns the first
    build-info line, if any."""
    try:
        return await asyncio.wait_for(
            _read_build_info(reader), timeout=_TELNET_WAIT_S
        )
    except (asyncio.TimeoutError, OSError):
        return None


async def _read_build_info(
    reader: asyncio.StreamReader,
) -> str | None:
    while True:
        line = await reader.readline()
        if not line:
            return None
        text = line.decode("utf-8", errors="replace").strip()
        if text.startswith(_TELNET_VER_PREFIX):
            return text[:_MAX_BANNER_LEN]


async def _fetch(url: str) -> str:
    async with (
        aiohttp.ClientSession() as session,
        session.get(url) as response,
    ):
        if response.status != 200:
            return ""
        raw = await response.content.read(_MAX_RESPONSE_BYTES)
    return raw.decode("utf-8", errors="replace")


def _first_line(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return ""


__all__ = [
    "fingerprint_grbl_http",
    "fingerprint_grbl_telnet",
]
