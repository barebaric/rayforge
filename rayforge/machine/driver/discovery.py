"""Automatic device discovery.

Discovery is split into two processes:

* **Scanning** (one per discovery run) —
  :func:`rayforge.machine.transport.serial_scan.scan_serial_ports`
  opens every serial port once and captures what the device sends
  as a :class:`PortObservation`. This is transport work and knows
  nothing about firmware.

* **Evaluating** (this module) — each driver that wants to be
  discoverable declares a :class:`DeviceRecognizer` as its
  ``DISCOVERY`` class attribute: what its device's output looks
  like, how to label it, which firmware it runs. Recognition is a
  pure function over the captured bytes; no port is ever opened on
  behalf of a single driver, so adding drivers never multiplies
  scan cost.

:func:`find_all_devices` ties the two together: one scan, every
recognizer, merged results. The module also defines the
:class:`DiscoveredDevice` / :class:`DeviceIdentity` result types
plus identity-token helpers used for profile matching.

This module is GTK-free so it can be unit-tested in isolation.
"""

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from gettext import gettext as _
from typing import TYPE_CHECKING

from ..transport.serial import SerialPortInfo
from ..transport.serial_scan import (
    PortObservation,
    scan_serial_ports,
)

if TYPE_CHECKING:
    from ..device.profile import DeviceProfile

logger = logging.getLogger(__name__)

# Upper bound for the whole discovery scan.
_SCAN_TIMEOUT = 20.0

# USB vendor/product strings that identify the USB-serial chip, not
# the machine built around it. Never used for profile matching.
GENERIC_TOKENS = frozenset(
    {
        "usb",
        "serial",
        "uart",
        "bridge",
        "controller",
        "device",
        "machine",
        "printer",
        "board",
        "port",
        "if00",
        "cdc",
        "acm",
        "modem",
        "ch340",
        "ch340g",
        "ch340k",
        "ch9102",
        "ch9102f",
        "cp210x",
        "cp2101",
        "cp2102",
        "cp2104",
        "cp2109",
        "ftdi",
        "ft232",
        "ft232r",
        "ft231x",
        "ft2232",
        "prolific",
        "pl2303",
        "arduino",
        "leonardo",
        "mega2560",
        "inc",
        "ltd",
        "limited",
        "corp",
        "co",
        "technology",
        "electronics",
        "semiconductor",
        "integrated",
        "circuits",
        "quantities",
        "industry",
        "industrial",
    }
)


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
    """A device found by evaluating a scan against a driver's
    ``DISCOVERY`` recognizer."""

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
        return f"{self.driver_name}:{self.params.get('port', '')}"

    @property
    def probe_name(self) -> str | None:
        """The machine name the device itself reported, if any."""
        if self.probe_profile is None:
            return None
        name = (self.probe_profile.meta.name or "").strip()
        if not name or name.startswith("Unknown"):
            return None
        return name


@dataclass(frozen=True)
class DeviceRecognizer:
    """
    Declarative description of what a driver's device looks like on
    the wire. Declared by drivers as the ``DISCOVERY`` class
    attribute; evaluated against captured scan output by
    :func:`find_all_devices`.
    """

    #: Display label, called at discovery time so translations are
    #: looked up at runtime.
    label: Callable[[], str]
    #: Returns True if *data* — the bytes captured from a port —
    #: was produced by this driver's firmware.
    matches: Callable[[bytes], bool]
    #: Optional: extracts a human-readable device name from the
    #: captured bytes (e.g. Grbl's ``[MSG:machine:...]`` line). The
    #: name becomes the identity banner and feeds profile matching.
    name: Callable[[bytes], str | None] | None = None
    #: Firmware identifier used for identity/profile matching.
    firmware: str | None = None


def extract_banner(data: bytes) -> str | None:
    """
    Returns the first non-empty line of *data* as a human-readable
    hint (e.g. the Grbl version banner), or ``None`` if there is none.
    """
    text = data.decode("ascii", errors="replace")
    for line in text.splitlines():
        line = line.strip()
        if line:
            return line[:80]
    return None


def normalize_tokens(*texts: str | None) -> frozenset[str]:
    """
    Splits arbitrary identification strings (descriptions, by-id link
    names, manufacturer strings) into lowercase word tokens suitable
    for containment checks.
    """
    words: set[str] = set()
    for text in texts:
        if not text:
            continue
        for word in re.split(r"[^0-9a-zA-Z]+", text.lower()):
            if len(word) >= 3:
                words.add(word)
    return frozenset(words)


def build_identity(
    firmware: str | None,
    data: bytes | None,
    port_info: SerialPortInfo | None,
    device_name: str | None = None,
) -> DeviceIdentity:
    """
    Assembles a :class:`DeviceIdentity` from scan artifacts. A
    *device_name* extracted from the device's own output (when the
    recognizer provides one) takes precedence over the raw banner
    and contributes matching tokens.
    """
    tokens = normalize_tokens(
        port_info.description if port_info else None,
        port_info.manufacturer if port_info else None,
        device_name,
    )
    return DeviceIdentity(
        firmware=firmware,
        banner=device_name or (extract_banner(data) if data else None),
        usb_vid=port_info.vid if port_info else None,
        usb_pid=port_info.pid if port_info else None,
        tokens=tokens,
    )


async def find_all_devices(
    driver_classes: Iterable[type] | None = None,
    ports: Iterable[str] | None = None,
    baud_rates: Iterable[int] | None = None,
    exclude_ports: Iterable[str] | None = None,
) -> list[DiscoveredDevice]:
    """
    Runs a single serial scan and lets every driver's ``DISCOVERY``
    recognizer evaluate the captured output.

    *exclude_ports* keeps the scan away from ports whose devices
    are already held (e.g. being probed by the caller).

    A failing or slow scan never raises: it is bounded by a timeout
    and exceptions are logged. A recognizer that raises is skipped
    so one broken driver cannot break discovery for the rest.
    """
    if driver_classes is None:
        from . import drivers

        driver_classes = drivers

    recognizers = _collect_recognizers(driver_classes)
    if not recognizers:
        return []

    try:
        observations = await asyncio.wait_for(
            scan_serial_ports(
                ports=ports,
                baud_rates=baud_rates,
                exclude_ports=exclude_ports,
            ),
            timeout=_SCAN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Device discovery scan timed out")
        return []
    except Exception:
        logger.exception("Device discovery scan failed")
        return []

    devices: list[DiscoveredDevice] = []
    for observation in observations:
        devices.extend(_evaluate(observation, recognizers))
    return devices


def _collect_recognizers(
    driver_classes: Iterable[type],
) -> list[tuple[type, DeviceRecognizer]]:
    """Pairs every driver that declares a valid recognizer with it."""
    recognizers = []
    for cls in driver_classes:
        recognizer = getattr(cls, "DISCOVERY", None)
        if isinstance(recognizer, DeviceRecognizer):
            recognizers.append((cls, recognizer))
    return recognizers


def _evaluate(
    observation: PortObservation,
    recognizers: list[tuple[type, DeviceRecognizer]],
) -> list[DiscoveredDevice]:
    """Runs every recognizer against one observation."""
    devices: list[DiscoveredDevice] = []
    for cls, recognizer in recognizers:
        try:
            if not recognizer.matches(observation.data):
                continue
        except Exception:
            logger.exception(
                "Discovery recognizer for %s failed", cls.__name__
            )
            continue
        devices.append(_build_device(cls, recognizer, observation))
    return devices


def _build_device(
    cls: type,
    recognizer: DeviceRecognizer,
    observation: PortObservation,
) -> DiscoveredDevice:
    device_name = (
        recognizer.name(observation.data) if recognizer.name else None
    )
    return DiscoveredDevice(
        driver_name=cls.__name__,
        params={
            "port": observation.port,
            "baudrate": observation.baud_rate,
        },
        label=recognizer.label(),
        detail=_("{port} at {baud} baud").format(
            port=observation.port, baud=observation.baud_rate
        ),
        identity=build_identity(
            recognizer.firmware,
            observation.data,
            observation.info,
            device_name,
        ),
    )


__all__ = [
    "GENERIC_TOKENS",
    "DeviceIdentity",
    "DeviceRecognizer",
    "DiscoveredDevice",
    "build_identity",
    "extract_banner",
    "find_all_devices",
    "normalize_tokens",
]
