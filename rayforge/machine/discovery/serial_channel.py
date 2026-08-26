"""Serial discovery channel.

Runs one serial scan per discovery run and evaluates the captured
port observations against every driver's
:class:`~rayforge.machine.discovery.spec.SerialRecognizer`.
Recognition is a pure function over the captured bytes; no port is
ever opened on behalf of a single driver, so adding drivers never
multiplies scan cost.
"""

import logging
import re
from collections.abc import Iterable
from gettext import gettext as _

from ..transport.serial import SerialPortInfo
from ..transport.serial_scan import (
    PortObservation,
    scan_serial_ports,
)
from .spec import DiscoverySpec, SerialRecognizer
from .types import DeviceIdentity, DiscoveredDevice

logger = logging.getLogger(__name__)

# Time budget for the serial part of a discovery scan. Ports that
# have answered before it is hit are still reported; only probes still
# in flight at that moment are dropped.
SERIAL_SCAN_TIMEOUT = 20.0

# Tokens identifying the USB-serial adapter itself: the chip, its
# driver names, and the generic words USB strings use for it. A
# device identity built only from these carries no vendor signal.
USB_CHIP_TOKENS = frozenset(
    {
        "usb",
        "serial",
        "uart",
        "bridge",
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
    }
)

# Generic product words that carry no vendor identity either.
GENERIC_DEVICE_TOKENS = frozenset(
    {
        "controller",
        "device",
        "machine",
        "printer",
        "board",
    }
)

# Corporate-suffix words that appear in manufacturer strings ("Prolific
# Technology Inc.") but must not mask a genuine brand word next to
# them. Stripped from profile vendor names before matching, so a
# vendor like "Atomstack Technology" still matches on "atomstack".
CORPORATE_TOKENS = frozenset(
    {
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

#: All tokens that never count as a vendor signal on their own.
GENERIC_TOKENS = USB_CHIP_TOKENS | GENERIC_DEVICE_TOKENS


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
    port_info: SerialPortInfo | None,
    device_name: str | None = None,
) -> DeviceIdentity:
    """
    Assembles a :class:`DeviceIdentity` from serial scan artifacts. A
    *device_name* extracted from the device's own output (when the
    recognizer provides one) is the identity banner and contributes
    matching tokens.
    """
    tokens = normalize_tokens(
        port_info.description if port_info else None,
        port_info.manufacturer if port_info else None,
        device_name,
    )
    return DeviceIdentity(
        firmware=firmware,
        banner=device_name,
        usb_vid=port_info.vid if port_info else None,
        usb_pid=port_info.pid if port_info else None,
        tokens=tokens,
    )


def collect_serial_recognizers(
    driver_classes: Iterable[type],
) -> list[tuple[type, SerialRecognizer]]:
    """Pairs every driver whose ``DISCOVERY`` spec declares a valid
    serial recognizer with it."""
    recognizers = []
    for cls in driver_classes:
        recognizer = _serial_recognizer(cls)
        if recognizer is not None:
            recognizers.append((cls, recognizer))
    return recognizers


def _serial_recognizer(cls: type) -> SerialRecognizer | None:
    # Every Driver subclass carries DISCOVERY (None by default).
    spec = cls.DISCOVERY
    if not isinstance(spec, DiscoverySpec):
        return None
    return spec.serial


async def find_serial_devices(
    driver_classes: Iterable[type],
    ports: Iterable[str] | None = None,
    baud_rates: Iterable[int] | None = None,
    exclude_ports: Iterable[str] | None = None,
) -> list[DiscoveredDevice]:
    """
    Runs one serial scan and evaluates it against every declared
    serial recognizer. Like every channel, this never raises: failures
    are logged, and the scan is bounded by *SERIAL_SCAN_TIMEOUT*, so a
    slow or silent adapter can at most cost its own probe time.
    """
    recognizers = collect_serial_recognizers(driver_classes)
    if not recognizers:
        return []

    try:
        observations = await scan_serial_ports(
            ports=ports,
            baud_rates=baud_rates,
            exclude_ports=exclude_ports,
            timeout=SERIAL_SCAN_TIMEOUT,
        )
    except Exception:
        logger.exception("Serial device discovery failed")
        return []

    devices: list[DiscoveredDevice] = []
    for observation in observations:
        devices.extend(_evaluate(observation, recognizers))
    return devices


def _evaluate(
    observation: PortObservation,
    recognizers: list[tuple[type, SerialRecognizer]],
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
    recognizer: SerialRecognizer,
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
            observation.info,
            device_name,
        ),
    )


__all__ = [
    "CORPORATE_TOKENS",
    "GENERIC_DEVICE_TOKENS",
    "GENERIC_TOKENS",
    "USB_CHIP_TOKENS",
    "build_identity",
    "collect_serial_recognizers",
    "find_serial_devices",
    "normalize_tokens",
]
