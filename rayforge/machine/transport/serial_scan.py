"""Serial port scanning.

Scanning is a transport-level process, independent of any driver
or firmware: every serial port is opened exactly once per scan,
probed at a sequence of baud rates, and whatever the connected
device sends (welcome banner first, then responses to nudge
characters) is captured as a :class:`PortObservation`.

Interpreting an observation — deciding which driver could talk to
the device — is a separate concern that lives with the drivers (see
:mod:`rayforge.machine.driver.discovery`).

This module is GTK-free so it can be unit-tested in isolation. The
blocking pyserial work runs in an executor; everything is bounded by
short timeouts so a full scan stays fast.
"""

import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import serial

from .serial import SerialPortInfo, SerialTransport, sort_ports

logger = logging.getLogger(__name__)

# Baud rates tried in order. 115200 is by far the most common for
# laser controllers, so it is probed across all ports first.
BAUD_CANDIDATES = [115200, 57600, 9600]

# How long to listen for an unsolicited welcome banner before nudging.
_BANNER_TIMEOUT = 2.0
# How long to wait for a response after sending a nudge character.
_NUDGE_TIMEOUT = 1.0
# Once a device is talking, how long the line must stay quiet before
# its output burst is considered complete.
_QUIET_GAP = 0.3
# Upper bound for the final drain of trailing burst data.
_DRAIN_TIMEOUT = 1.0
_READ_CHUNK = 0.2

_NUDGES = (b"\n", b"?")


@dataclass(frozen=True)
class PortObservation:
    """Raw output captured from one port during a scan."""

    port: str
    baud_rate: int
    #: Everything the device sent while the port was open.
    data: bytes
    #: USB metadata for the port, when available.
    info: SerialPortInfo | None = None


def looks_like_device_output(data: bytes) -> bool:
    """
    Generic, firmware-agnostic test for "a device is talking on
    this port": the capture contains at least one line of
    mostly-printable ASCII. Output received at a wrong baud rate is
    typically framing garbage and fails this test.
    """
    for line in data.split(b"\n"):
        line = line.strip(b" \t\r\x00")
        if not line:
            continue
        printable = sum(0x20 <= byte <= 0x7E or byte == 0x09 for byte in line)
        if printable / len(line) >= 0.8:
            return True
    return False


# Per-port locks so concurrent scans don't fight over the same
# physical port (pyserial opens exclusively). Safe to create lazily:
# the event loop is single-threaded, so there is no check-then-set
# race.
_port_locks: dict[str, asyncio.Lock] = {}


def _get_port_lock(port: str) -> asyncio.Lock:
    lock = _port_locks.get(port)
    if lock is None:
        lock = asyncio.Lock()
        _port_locks[port] = lock
    return lock


async def scan_serial_ports(
    ports: Iterable[str] | None = None,
    baud_rates: Iterable[int] | None = None,
    exclude_ports: Iterable[str] | None = None,
) -> list[PortObservation]:
    """
    Probes every *port* at *baud_rates* and returns one observation
    per port that produced device-like output. Ports named in
    *exclude_ports* are skipped entirely (used by callers that hold
    devices on those ports). Ports that are busy, absent, or silent
    are skipped silently.
    """
    if ports is None:
        port_list = sort_ports(SerialTransport.list_usb_ports())
    else:
        port_list = list(ports)
    if exclude_ports is not None:
        excluded = set(exclude_ports)
        port_list = [p for p in port_list if p not in excluded]
    rates = list(baud_rates) if baud_rates is not None else BAUD_CANDIDATES

    info_map = {info.device: info for info in SerialTransport.list_port_info()}

    observations: list[PortObservation] = []
    for port in port_list:
        async with _get_port_lock(port):
            result = await _probe_port(port, rates)
        if result is None:
            continue
        baudrate, data = result
        observations.append(
            PortObservation(
                port=port,
                baud_rate=baudrate,
                data=data,
                info=info_map.get(port),
            )
        )
    return observations


async def _probe_port(
    port: str,
    rates: list[int],
) -> tuple[int, bytes] | None:
    """
    Attempts to get a device on *port* to talk. Tries *rates* in
    order and returns ``(baudrate, accumulated_bytes)`` for the
    first rate at which the device produces plausible output, or
    None.
    """
    loop = asyncio.get_running_loop()
    for baudrate in rates:
        result = await _try_port(port, baudrate, loop)
        if result is not None:
            return baudrate, result
    return None


async def _try_port(
    port: str,
    baudrate: int,
    loop: asyncio.AbstractEventLoop,
) -> bytes | None:
    try:
        return await loop.run_in_executor(
            None, _probe_port_blocking, port, baudrate
        )
    except OSError as e:
        logger.debug("Probe of %s failed: %s", port, e)
        return None


def _probe_port_blocking(port: str, baudrate: int) -> bytes | None:
    """
    Synchronous probe of a single port/baud combination. Opens the
    port, listens for a banner, nudges a silent device, and returns
    everything it sent if the output looks like a device.

    Recognition happens later, on the captured bytes, so the probe
    always collects as much signal as it can: every nudge is sent
    even after the device starts talking (different firmwares
    identify themselves in different responses — a bare ``ok`` ack
    is not enough to tell them apart), and a trailing drain catches
    the rest of a multi-line burst.
    """
    try:
        ser = serial.Serial(port=port, baudrate=baudrate, timeout=_READ_CHUNK)
    except (OSError, serial.SerialException) as e:
        logger.debug("Cannot open %s at %d: %s", port, baudrate, e)
        return None
    try:
        buf = bytearray()
        saw_output = _listen(ser, buf, _BANNER_TIMEOUT)

        for nudge in _NUDGES:
            try:
                ser.write(nudge)
                ser.flush()
            except (OSError, serial.SerialException) as e:
                logger.debug("Write to %s failed: %s", port, e)
                return None
            saw_output |= _listen(ser, buf, _NUDGE_TIMEOUT)

        if not saw_output or not looks_like_device_output(bytes(buf)):
            return None
        _drain(ser, buf)
        logger.debug(
            "Observed output on %s at %d baud (%d bytes)",
            port,
            baudrate,
            len(buf),
        )
        return bytes(buf)
    finally:
        try:
            ser.close()
        except (OSError, serial.SerialException):
            pass


def _listen(ser: serial.Serial, buf: bytearray, timeout: float) -> bool:
    """
    Reads from *ser* into *buf* for up to *timeout* seconds,
    returning True as soon as the data captured during this call
    looks like device output. Only new data is tested: plausible
    output from earlier phases must not cut short later ones.
    """
    start = len(buf)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            chunk = ser.read(256)
        except (OSError, serial.SerialException):
            return False
        if not chunk:
            continue
        buf.extend(chunk)
        if looks_like_device_output(bytes(buf[start:])):
            return True
    return looks_like_device_output(bytes(buf[start:]))


def _drain(ser: serial.Serial, buf: bytearray) -> None:
    """Collects trailing burst data until the line stays quiet."""
    deadline = time.monotonic() + _DRAIN_TIMEOUT
    quiet = 0.0
    last = time.monotonic()
    while time.monotonic() < deadline and quiet < _QUIET_GAP:
        try:
            chunk = ser.read(256)
        except (OSError, serial.SerialException):
            return
        now = time.monotonic()
        if chunk:
            buf.extend(chunk)
            quiet = 0.0
        else:
            quiet = now - last
        last = now


__all__ = [
    "BAUD_CANDIDATES",
    "PortObservation",
    "looks_like_device_output",
    "scan_serial_ports",
]
