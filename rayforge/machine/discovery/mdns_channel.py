"""Network (mDNS) discovery channel.

Browses the network for every service type declared by a driver's
:class:`~rayforge.machine.discovery.spec.MdnsRecognizer`, then
maps each resolved service to a :class:`~rayforge.machine.
discovery.types.DiscoveredDevice`.

Generic service types (e.g. ``_http._tcp``) that no driver declared
are additionally offered to drivers that declare a *fingerprint*
probe: each candidate is fingerprinted to confirm its firmware
before it is claimed. Declared-service results always win over
fingerprint claims on the same host.
"""

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import replace
from gettext import gettext as _

from ..transport.mdns_scan import (
    MDNSService,
    normalize_service_type,
    scan_mdns_services,
)
from .serial_channel import normalize_tokens
from .spec import DiscoverySpec, MdnsRecognizer
from .types import DeviceIdentity, DiscoveredDevice

logger = logging.getLogger(__name__)

# Upper bound for the network part of a discovery scan; the browse
# window itself is much shorter (see transport.mdns_scan).
NETWORK_SCAN_TIMEOUT = 10.0

# Generic service type browsed solely for fingerprinting.
_HTTP_SERVICE_TYPE = "_http._tcp"

# Fingerprint pass bounds: at most this many concurrent probes, and
# at most this long per candidate (the probe itself is bounded too).
_FINGERPRINT_CONCURRENCY = 8
_FINGERPRINT_TIMEOUT = 2.0


def collect_mdns_recognizers(
    driver_classes: Iterable[type],
) -> list[tuple[type, MdnsRecognizer]]:
    """Pairs every driver whose ``DISCOVERY`` spec declares a valid
    mDNS recognizer with it. The declared service types are
    normalized to their canonical ``_name._transport`` form."""
    recognizers = []
    for cls in driver_classes:
        # Every Driver subclass carries DISCOVERY (None by default).
        spec = cls.DISCOVERY
        if not isinstance(spec, DiscoverySpec):
            continue
        if not isinstance(spec.mdns, MdnsRecognizer):
            continue
        recognizer = replace(
            spec.mdns,
            services=tuple(
                normalize_service_type(s) for s in spec.mdns.services
            ),
        )
        recognizers.append((cls, recognizer))
    return recognizers


async def find_network_devices(
    driver_classes: Iterable[type] | None = None,
) -> list[DiscoveredDevice]:
    """
    Browses the network for every declared service type plus the
    generic types needed for fingerprinting, and returns one device
    per confirmed match.

    Like every channel, this never raises: failures are logged and
    bounded by a timeout.
    """
    if driver_classes is None:
        from ..driver import drivers

        driver_classes = drivers

    declarations = collect_mdns_recognizers(driver_classes)
    if not declarations:
        return []

    service_types = {
        service_type
        for _, recognizer in declarations
        for service_type in recognizer.services
    }
    if any(r.fingerprint for _, r in declarations):
        service_types.add(_HTTP_SERVICE_TYPE)

    try:
        services = await asyncio.wait_for(
            scan_mdns_services(service_types),
            timeout=NETWORK_SCAN_TIMEOUT,
        )
    except asyncio.TimeoutError:
        logger.warning("Network device discovery timed out")
        return []
    except Exception:
        logger.exception("Network device discovery failed")
        return []

    devices, claimed_hosts, candidates = _split_services(
        services, declarations
    )
    devices.extend(
        await _run_fingerprints(candidates, declarations, claimed_hosts)
    )
    return devices


def _split_services(
    services: list[MDNSService],
    declarations: list[tuple[type, MdnsRecognizer]],
) -> tuple[list[DiscoveredDevice], set[str], list[MDNSService]]:
    """Sorts resolved services into devices claimed by a declared
    service type and unclaimed generic-type candidates."""
    devices: list[DiscoveredDevice] = []
    claimed_hosts: set[str] = set()
    candidates: list[MDNSService] = []
    for service in services:
        claimed = False
        for cls, recognizer in declarations:
            if service.service_type in recognizer.services:
                devices.append(_build_device(cls, recognizer, service))
                claimed_hosts.add(service.host)
                claimed = True
                break
        if not claimed and service.service_type == _HTTP_SERVICE_TYPE:
            candidates.append(service)
    return devices, claimed_hosts, candidates


async def _run_fingerprints(
    candidates: list[MDNSService],
    declarations: list[tuple[type, MdnsRecognizer]],
    claimed_hosts: set[str],
) -> list[DiscoveredDevice]:
    """Fingerprints unclaimed generic-service candidates, bounded in
    concurrency.

    All fingerprinting drivers probe every candidate concurrently;
    the first match in declaration order claims a host, so a host that
    several drivers recognize yields exactly one device.
    """
    probes = [
        (cls, recognizer, service)
        for cls, recognizer in declarations
        if recognizer.fingerprint
        for service in candidates
        if service.host not in claimed_hosts
    ]
    if not probes:
        return []

    semaphore = asyncio.Semaphore(_FINGERPRINT_CONCURRENCY)

    async def _claim(
        cls: type, recognizer: MdnsRecognizer, service: MDNSService
    ) -> tuple[type, MdnsRecognizer, MDNSService] | None:
        assert recognizer.fingerprint is not None
        async with semaphore:
            try:
                result = await asyncio.wait_for(
                    recognizer.fingerprint(service.host, service.port),
                    timeout=_FINGERPRINT_TIMEOUT,
                )
            except Exception:
                logger.debug(
                    "Fingerprint of %s:%s failed",
                    service.host,
                    service.port,
                    exc_info=True,
                )
                return None
        if result is None:
            return None
        return cls, recognizer, service

    results = await asyncio.gather(*(_claim(*probe) for probe in probes))
    devices: list[DiscoveredDevice] = []
    claimed: set[str] = set()
    for result in results:
        if result is None:
            continue
        cls, recognizer, service = result
        if service.host in claimed:
            continue
        claimed.add(service.host)
        devices.append(_build_device(cls, recognizer, service))
    return devices


def _build_device(
    cls: type,
    recognizer: MdnsRecognizer,
    service: MDNSService,
) -> DiscoveredDevice:
    """Assembles a :class:`DiscoveredDevice` from a resolved mDNS
    service.

    TXT record fields the recognizer declared in ``txt_map`` are
    forwarded into ``params`` (mapped to the driver-arg key the
    driver consumes). Identity-strengthening TXT fields (``vendor``,
    ``model``) feed the matching token set; ``version`` becomes the
    identity banner when the service announced no instance name of
    its own.
    """
    detail = _("{host}:{port}").format(host=service.host, port=service.port)
    if service.server:
        detail = f"{detail} ({service.server})"

    params: dict = {"host": service.host, "port": service.port}
    for txt_key, arg_key in recognizer.txt_map.items():
        value = service.txt.get(txt_key)
        if value:
            params[arg_key] = value

    banner = service.name or None
    if not banner and service.txt.get("version"):
        banner = service.txt["version"]

    tokens = normalize_tokens(
        service.name,
        service.server,
        service.txt.get("vendor"),
        service.txt.get("model"),
    )
    return DiscoveredDevice(
        driver_name=cls.__name__,
        params=params,
        label=cls.label,
        detail=detail,
        identity=DeviceIdentity(
            firmware=_driver_firmware_id(cls),
            banner=banner,
            tokens=tokens,
        ),
    )


def _driver_firmware_id(cls: type) -> str:
    """A firmware identifier derived from the driver class name
    (e.g. ``OctoPrintDriver`` → ``octoprint``)."""
    name = cls.__name__.removesuffix("Driver")
    return name.lower()


__all__ = [
    "collect_mdns_recognizers",
    "find_network_devices",
]
