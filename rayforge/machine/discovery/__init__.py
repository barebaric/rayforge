"""Automatic device discovery.

Discovery is organized as independent *channels*, each pairing one
transport-level scanner with one recognizer type:

* **Serial channel** (:mod:`~rayforge.machine.discovery.
  serial_channel`) — :func:`rayforge.machine.transport.serial_scan.
  scan_serial_ports` opens every serial port once and captures what
  the device sends; each driver's
  :class:`~rayforge.machine.discovery.spec.SerialRecognizer` decides
  from the captured bytes whether its firmware produced them.

* **Network channel** (:mod:`~rayforge.machine.discovery.
  mdns_channel`) — :func:`rayforge.machine.transport.mdns_scan.
  scan_mdns_services` browses the network for the service types each
  driver's
  :class:`~rayforge.machine.discovery.spec.MdnsRecognizer`
  declares, and optionally fingerprints generic-type candidates to
  confirm their firmware.

A driver participates by declaring a single
:class:`~rayforge.machine.discovery.spec.DiscoverySpec` as its
``DISCOVERY`` class attribute, composed only of the channels it
actually speaks. Recognition is a pure function over captured scan
output; no connection is ever opened on behalf of a single driver,
so adding drivers never multiplies scan cost.

:func:`find_all_devices` (the engine) runs all channels concurrently
and merges their results. A failing or slow channel never affects
the others. :class:`DiscoverySession` offers the GTK-free scan/probe
state machine behind a discovery UI.

This package is GTK-free so it can be unit-tested in isolation.
"""

from .engine import find_all_devices
from .mdns_channel import find_network_devices
from .serial_channel import (
    CORPORATE_TOKENS,
    GENERIC_DEVICE_TOKENS,
    GENERIC_TOKENS,
    USB_CHIP_TOKENS,
    build_identity,
    normalize_tokens,
)
from .session import DiscoverySession
from .spec import DiscoverySpec, MdnsRecognizer, SerialRecognizer
from .types import DeviceIdentity, DiscoveredDevice, device_key

__all__ = [
    "CORPORATE_TOKENS",
    "GENERIC_DEVICE_TOKENS",
    "GENERIC_TOKENS",
    "USB_CHIP_TOKENS",
    "DeviceIdentity",
    "DiscoveredDevice",
    "DiscoverySession",
    "DiscoverySpec",
    "MdnsRecognizer",
    "SerialRecognizer",
    "build_identity",
    "device_key",
    "find_all_devices",
    "find_network_devices",
    "normalize_tokens",
]
