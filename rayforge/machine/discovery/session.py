"""GTK-free state behind a discovery UI.

A :class:`DiscoverySession` owns everything about an ongoing
discovery that is not widget work: the known devices, the ports
held by found devices, scan-result reconciliation and probe-result
enrichment. A UI page drives it (feeds scans in) and maps its
results onto rows; the reconciliation logic itself is unit-testable
without GTK.
"""

from dataclasses import replace

from .types import DiscoveredDevice


class DiscoverySession:
    """
    Accumulates discovered devices across repeated scans and reports
    the difference between scans as plain data:

    * :meth:`apply_scan` returns ``(added, removed_keys)`` — devices
      are matched by their stable :attr:`DiscoveredDevice.key`.
    * Ports of found devices are remembered as *held*; held ports are
      meant to be excluded from rescans (re-opening them would reset
      some boards) and their devices stay listed even though a rescan
      cannot see them. They leave only via :meth:`prune_absent_ports`
      when the port vanishes from the system.
    * :meth:`apply_probe` attaches a probed profile to a device and
      returns the enriched copy.
    """

    def __init__(self) -> None:
        self._devices: dict[str, DiscoveredDevice] = {}
        self._held_ports: set[str] = set()

    @property
    def held_ports(self) -> set[str]:
        """Ports of known devices; exclude from rescans."""
        return set(self._held_ports)

    @property
    def devices(self) -> list[DiscoveredDevice]:
        return list(self._devices.values())

    def get(self, key: str) -> DiscoveredDevice | None:
        return self._devices.get(key)

    def reset(self) -> None:
        """Forgets everything (e.g. when the UI re-enters discovery)."""
        self._devices.clear()
        self._held_ports.clear()

    def apply_scan(
        self, devices: list[DiscoveredDevice]
    ) -> tuple[list[DiscoveredDevice], list[str]]:
        """
        Reconciles the known devices with a fresh scan result.
        Returns the newly seen devices and the keys of devices that
        disappeared (excluding those on held ports).
        """
        added: list[DiscoveredDevice] = []
        seen: set[str] = set()
        for device in devices:
            if device.key in seen:
                continue
            seen.add(device.key)
            port = device.params.get("port")
            if isinstance(port, str) and port:
                self._held_ports.add(port)
            if device.key not in self._devices:
                self._devices[device.key] = device
                added.append(device)

        removed = [
            key
            for key in self._devices
            if key not in seen and not self._on_held_port(key)
        ]
        for key in removed:
            del self._devices[key]
        return added, removed

    def apply_probe(
        self, key: str, profile: object
    ) -> DiscoveredDevice | None:
        """
        Attaches a probed profile to the device with *key* and
        returns the enriched device, or None when the key is unknown.
        """
        device = self._devices.get(key)
        if device is None:
            return None
        enriched = replace(device, probe_profile=profile)
        self._devices[key] = enriched
        return enriched

    def prune_absent_ports(self, present_ports: set[str]) -> list[str]:
        """
        Drops devices whose serial port is no longer present on the
        system. Returns the removed keys.
        """
        removed = []
        for key in self._devices:
            port = self._port_of(key)
            if port is not None and port not in present_ports:
                removed.append(key)
        for key in removed:
            port = self._port_of(key)
            assert port is not None
            self._held_ports.discard(port)
            del self._devices[key]
        return removed

    def _on_held_port(self, key: str) -> bool:
        port = self._port_of(key)
        return port is not None and port in self._held_ports

    def _port_of(self, key: str) -> str | None:
        port = self._devices[key].params.get("port")
        if isinstance(port, str) and port:
            return port
        return None


__all__ = [
    "DiscoverySession",
]
