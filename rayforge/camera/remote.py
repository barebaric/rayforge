"""Cameras advertised by a remote raydriver daemon.

A daemon never proxies video: it only lists the stream endpoints
configured on the device host. Rayforge's ordinary network camera
sources consume those URLs, so this module translates the daemon's
advertisement into the camera model's source types.
"""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from raydriver.core.discovery.daemon import (
    list_cameras as daemon_list_cameras,
)

from .models.source_type import CameraSourceType

logger = logging.getLogger(__name__)

#: Daemon camera entries must name a network source Rayforge can open.
_NETWORK_SOURCE_TYPES = frozenset(
    {
        CameraSourceType.HTTP_SNAPSHOT,
        CameraSourceType.HTTP_STREAM,
        CameraSourceType.RTSP,
    }
)


@dataclass(frozen=True)
class RemoteCamera:
    """One camera endpoint advertised by a raydriver daemon."""

    name: str
    uri: str
    source_type: CameraSourceType


def is_remote_driver_args(driver_args: Mapping[str, Any] | None) -> bool:
    """Whether *driver_args* describe a raydriver daemon endpoint."""
    return bool(driver_args and driver_args.get("remote_host"))


async def fetch_remote_cameras(
    driver_args: Mapping[str, Any] | None,
) -> list[RemoteCamera]:
    """Return the cameras a remote machine's daemon advertises.

    Returns an empty list for local machines and for unreachable or
    misconfigured daemons; discovery is best-effort and must never
    block camera setup.
    """
    if not is_remote_driver_args(driver_args):
        return []
    assert driver_args is not None
    host = str(driver_args["remote_host"])
    port = int(driver_args.get("remote_port") or 8765)
    token = driver_args.get("token") or None
    tls_ca = driver_args.get("tls_ca") or None
    tls_hostname = driver_args.get("tls_hostname") or None
    try:
        entries = await daemon_list_cameras(
            host,
            port,
            token=token,
            tls_ca=tls_ca,
            tls_hostname=tls_hostname,
        )
    except (ConnectionError, OSError, ValueError) as err:
        logger.info(
            "Could not read cameras from daemon %s:%s: %s", host, port, err
        )
        return []
    return _parse_cameras(entries)


def _parse_cameras(entries: list[Any]) -> list[RemoteCamera]:
    cameras: list[RemoteCamera] = []
    seen: set[tuple[CameraSourceType, str]] = set()
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        uri = str(entry.get("uri") or "").strip()
        source_type = _source_type(entry.get("type"))
        if not uri or source_type is None:
            continue
        key = (source_type, uri)
        if key in seen:
            continue
        seen.add(key)
        cameras.append(
            RemoteCamera(
                name=str(entry.get("name") or uri),
                uri=uri,
                source_type=source_type,
            )
        )
    return cameras


def _source_type(value: Any) -> CameraSourceType | None:
    try:
        source_type = CameraSourceType(str(value or ""))
    except ValueError:
        return None
    return source_type if source_type in _NETWORK_SOURCE_TYPES else None
