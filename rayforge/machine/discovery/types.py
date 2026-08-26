"""Result types shared by all discovery channels.

A :class:`DiscoveredDevice` is what every channel yields; a
:class:`DeviceIdentity` bundles everything known about a device
that can be used to match it against a known device profile.

This module is GTK-free so it can be unit-tested in isolation.
"""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..device.profile import DeviceProfile


@dataclass(frozen=True)
class DeviceIdentity:
    """
    Everything known about a discovered device that can be used to
    match it against a known device profile.
    """

    firmware: str | None = None
    banner: str | None = None
    usb_vid: int | None = None
    usb_pid: int | None = None
    tokens: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class DiscoveredDevice:
    """A device found by evaluating scan output against a driver's
    discovery declaration."""

    driver_name: str
    #: Connection parameters that seed ``driver_args`` (port,
    #: baudrate, ...).
    params: dict
    #: Short display name, e.g. "GRBL device".
    label: str
    #: One-line connection detail, e.g. "/dev/ttyUSB0 at 115200 baud".
    detail: str
    identity: DeviceIdentity = field(default_factory=DeviceIdentity)
    #: Profile data collected by probing the device after discovery
    #: (firmware build info, axis extents, speeds). Attached by the
    #: discovery UI once available; ``None`` until then.
    probe_profile: "DeviceProfile | None" = None

    @property
    def key(self) -> str:
        """Stable identity for deduplication in the UI."""
        # Serial devices are identified by their port path, network
        # devices by host and TCP port (so two servers that both
        # listen on 80 stay distinguishable).
        serial_port = self.params.get("port")
        if isinstance(serial_port, str) and serial_port:
            return f"{self.driver_name}:{serial_port}"
        host = self.params.get("host")
        if isinstance(host, str) and host:
            return f"{self.driver_name}:{host}:{self.params.get('port')}"
        return f"{self.driver_name}:{serial_port!r}"

    @property
    def probe_name(self) -> str | None:
        """The machine name the device itself reported, if any."""
        if self.probe_profile is None:
            return None
        name = (self.probe_profile.meta.name or "").strip()
        if not name or name.startswith("Unknown"):
            return None
        return name


__all__ = [
    "DeviceIdentity",
    "DiscoveredDevice",
]
