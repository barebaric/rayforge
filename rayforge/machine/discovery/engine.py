"""Discovery engine.

Ties the channels together: :func:`find_all_devices` runs every
channel's scan concurrently and merges the results. Each channel
owns its scanner (transport work), its recognizer type (declared in
a driver's :class:`~rayforge.machine.discovery.spec.DiscoverySpec`)
and its timeout/error handling, so adding a channel never changes
the engine or the driver base class.
"""

import asyncio
from collections.abc import Iterable

from .mdns_channel import find_network_devices
from .serial_channel import find_serial_devices
from .types import DiscoveredDevice


async def find_all_devices(
    driver_classes: Iterable[type] | None = None,
    ports: Iterable[str] | None = None,
    baud_rates: Iterable[int] | None = None,
    exclude_ports: Iterable[str] | None = None,
) -> list[DiscoveredDevice]:
    """
    Runs every discovery channel concurrently and returns the merged,
    deduplicated results.

    *exclude_ports* keeps the serial scan away from ports whose
    devices are already held (e.g. being probed by the caller).
    Channels never raise: a failing or slow scan yields no devices
    from that channel only.
    """
    if driver_classes is None:
        from ..driver import drivers

        driver_classes = drivers

    serial_devices, network_devices = await asyncio.gather(
        find_serial_devices(driver_classes, ports, baud_rates, exclude_ports),
        find_network_devices(driver_classes),
    )
    return _dedupe(serial_devices + network_devices)


def _dedupe(devices: list[DiscoveredDevice]) -> list[DiscoveredDevice]:
    """Drops duplicate device keys, keeping the first occurrence."""
    seen: set[str] = set()
    unique: list[DiscoveredDevice] = []
    for device in devices:
        if device.key in seen:
            continue
        seen.add(device.key)
        unique.append(device)
    return unique


__all__ = [
    "find_all_devices",
]
