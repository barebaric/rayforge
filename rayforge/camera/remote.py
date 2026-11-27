"""Cameras advertised by a remote raydriver daemon.

A daemon never proxies video: it only lists the stream endpoints
configured on the device host. Rayforge's ordinary network camera
sources consume those URLs, so this module translates the daemon's
advertisement into the camera model's source types.
"""

import logging
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from raygeo.driver.core.discovery.daemon import (
    list_cameras as daemon_list_cameras,
)

from .models.source_type import CameraSourceType

if TYPE_CHECKING:
    from ..machine.models.machine import Machine

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
class RemoteDaemon:
    """A raydriver daemon a camera can be selected from."""

    label: str
    host: str
    port: int = 8765
    token: str | None = None
    tls_ca: str | None = None
    tls_hostname: str | None = None

    @property
    def driver_args(self) -> dict[str, Any]:
        args: dict[str, Any] = {
            "remote_host": self.host,
            "remote_port": self.port,
        }
        for key, value in (
            ("token", self.token),
            ("tls_ca", self.tls_ca),
            ("tls_hostname", self.tls_hostname),
        ):
            if value:
                args[key] = value
        return args


@dataclass(frozen=True)
class RemoteCamera:
    """One camera endpoint advertised by a raydriver daemon."""

    name: str
    uri: str
    source_type: CameraSourceType


def collect_remote_daemons(
    machines: Iterable["Machine"],
) -> list[RemoteDaemon]:
    """Remote daemons named by the configured machines, deduplicated."""
    daemons: list[RemoteDaemon] = []
    seen: set[tuple[Any, ...]] = set()
    for machine in machines:
        args = machine.driver_args or {}
        host = args.get("remote_host")
        if not host:
            continue
        port = int(args.get("remote_port") or 8765)
        name = machine.name
        daemon = RemoteDaemon(
            label=f"{name} ({host}:{port})" if name else f"{host}:{port}",
            host=str(host),
            port=port,
            token=args.get("token") or None,
            tls_ca=args.get("tls_ca") or None,
            tls_hostname=args.get("tls_hostname") or None,
        )
        key = (
            daemon.host,
            daemon.port,
            daemon.token,
            daemon.tls_ca,
            daemon.tls_hostname,
        )
        if key in seen:
            continue
        seen.add(key)
        daemons.append(daemon)
    return daemons


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
