"""HTTP fingerprint probe for networked GRBL-family controllers.

The implementation behind
``GrblNetworkDriver.DISCOVERY.mdns.fingerprint``: some devices this
driver can talk to (FluidNC, ESP3D v3) announce only a generic mDNS
service type that says nothing about their firmware. This probe
actively questions such a candidate host over HTTP with the ESP800
firmware-info command both firmwares answer, and reports a synthetic
:class:`~rayforge.machine.transport.mdns_scan.MDNSService` when the
response identifies GRBL-family firmware.

The discovery engine knows nothing about GRBL — it only calls the
callable this module provides. Bounded by a timeout, never raises:
any failure yields None.
"""

import asyncio
import logging
import re

import aiohttp

from ...transport.mdns_scan import MDNSService

logger = logging.getLogger(__name__)

# The [ESP800] firmware-info endpoint served by ESP3D v3 / FluidNC.
_FW_INFO_URL = "/command?plain=%5BESP800%5D&PAGEID="

# Firmware names accepted as a GRBL-compatible match.
_GRBL_FW_RE = re.compile(r"\b(fluidnc|esp3d|grblhal|grbl)\b", re.IGNORECASE)

_DEFAULT_TIMEOUT_S = 1.5
_MAX_RESPONSE_BYTES = 4096


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
]
