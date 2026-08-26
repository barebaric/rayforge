"""Driver-facing discovery declarations.

A driver participates in automatic device discovery by declaring a
single :class:`DiscoverySpec` as its ``DISCOVERY`` class attribute.
The spec is composed of per-channel recognizers: a driver declares
only the channels it actually speaks (a serial-only driver has no
network baggage and vice versa). New fingerprinting or protocol
support is added as fields on the recognizer that owns that channel
— never as new attributes on :class:`~rayforge.machine.driver.
driver.Driver`.
"""

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field

from ..transport.mdns_scan import MDNSService


@dataclass(frozen=True)
class SerialRecognizer:
    """
    Declarative description of what a driver's device looks like on
    a serial wire. Evaluated against captured port output by the
    serial channel.
    """

    #: Display label, called at discovery time so translations are
    #: looked up at runtime.
    label: Callable[[], str]
    #: Returns True if *data* — the bytes captured from a port —
    #: was produced by this driver's firmware.
    matches: Callable[[bytes], bool]
    #: Optional: extracts a human-readable identity hint from the
    #: captured bytes — e.g. a firmware version banner or a machine
    #: name announced by the device. Interpreting the raw bytes is
    #: the driver's responsibility (it may be text or binary); the
    #: result becomes the identity banner and feeds profile matching.
    name: Callable[[bytes], str | None] | None = None
    #: Firmware identifier used for identity/profile matching.
    firmware: str | None = None


#: Probes one mDNS candidate host/port and returns a synthetic
#: :class:`MDNSService` describing the device when it matches,
#: ``None`` otherwise. Must be bounded (timeouts) and never raise.
FingerprintFn = Callable[[str, int], Awaitable[MDNSService | None]]


@dataclass(frozen=True)
class MdnsRecognizer:
    """
    Declarative description of how a driver's devices appear on the
    network. Evaluated against resolved mDNS services by the network
    channel.
    """

    #: mDNS service types announcing devices this driver can talk
    #: to, e.g. ``("_esp3d._tcp.local.",)``.
    services: tuple[str, ...] = ()
    #: Maps mDNS TXT record keys to the ``driver_args`` / setup-var
    #: keys they are forwarded as into the discovered device's
    #: params. Only keys present in both the map and the resolved
    #: TXT record are forwarded.
    txt_map: dict[str, str] = field(default_factory=dict)
    #: Optional async probe for generic service types (e.g.
    #: ``_http._tcp``): candidates announced under a type no driver
    #: declared are fingerprinted to confirm their firmware before
    #: they are claimed.
    fingerprint: FingerprintFn | None = None


@dataclass(frozen=True)
class DiscoverySpec:
    """
    Everything a driver declares about discoverability. Declared as
    the ``DISCOVERY`` class attribute; evaluated by the discovery
    channels (see :mod:`rayforge.machine.discovery`).
    """

    serial: SerialRecognizer | None = None
    mdns: MdnsRecognizer | None = None


__all__ = [
    "DiscoverySpec",
    "FingerprintFn",
    "MdnsRecognizer",
    "SerialRecognizer",
]
