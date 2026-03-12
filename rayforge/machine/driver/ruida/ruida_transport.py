"""
Layer 2 (Data Link/Transport) for Ruida protocol.

Handles framing (checksums) and swizzle encoding/decoding.
Wraps a generic transport (UDP, serial) to provide Ruida-specific encoding.
"""

import logging
from typing import Tuple

from blinker import Signal

from rayforge.machine.transport.transport import Transport, TransportStatus
from rayforge.machine.transport.udp_server import UdpServerTransport

from .ruida_codec import RuidaCodec
from .ruida_util import frame_packet, validate_packet

logger = logging.getLogger(__name__)


class RuidaTransport(Transport):
    """
    Ruida L2 transport that wraps a generic transport.

    Adds framing (checksum prefix) and swizzle encoding to all data.
    Emits decoded (unswizzled) payloads via the decoded_received signal.

    Usage:
        udp = UdpTransport("192.168.1.100", 50200)
        ruida = RuidaTransport(udp)
        ruida.decoded_received.connect(handler)
        await ruida.connect()
        await ruida.send_command(b"\\xda\\x00...")  # gets swizzled + framed
    """

    def __init__(self, transport: Transport, magic: int = 0x88):
        super().__init__()
        self._transport = transport
        self._codec = RuidaCodec(magic)

        self.decoded_received = Signal()
        self.magic_changed = Signal()

        self._transport.received.connect(self._on_raw_received)
        self._transport.status_changed.connect(self._on_status_changed)

    @property
    def magic(self) -> int:
        """Current swizzle magic key."""
        return self._codec.magic

    @magic.setter
    def magic(self, value: int) -> None:
        if self._codec.set_magic(value):
            self.magic_changed.send(self, magic=value)

    @property
    def is_connected(self) -> bool:
        return self._transport.is_connected

    async def connect(self) -> None:
        await self._transport.connect()

    async def disconnect(self) -> None:
        await self._transport.disconnect()

    async def send(self, data: bytes) -> None:
        """
        Send raw data without framing/swizzling.

        Use send_command() for normal Ruida communication.
        """
        await self._transport.send(data)

    async def send_command(self, command: bytes) -> None:
        """
        Send a Ruida command with swizzle encoding and framing.

        Args:
            command: Unswizzled command bytes
        """
        swizzled = self._codec.swizzle(command)
        framed = frame_packet(swizzled)
        await self._transport.send(framed)

    async def send_response(self, response: bytes) -> None:
        """
        Send a response (swizzled but not framed with checksum).

        For UDP responses to MeerK40t, responses are sent raw swizzled
        without the checksum prefix.
        """
        swizzled = self._codec.swizzle(response)
        await self._transport.send(swizzled)

    async def purge(self) -> None:
        await self._transport.purge()

    def _on_raw_received(self, sender, data: bytes) -> None:
        """Handle raw data from underlying transport.

        Note: Server responses are NOT framed with checksums - they are
        just swizzled bytes. Only client->server packets have checksums.
        """
        unswizzled = self._codec.unswizzle(data)

        if len(data) == 1 and unswizzled[0] in (0xCC, 0xCD, 0xCE):
            self.decoded_received.send(self, data=unswizzled)
            return

        detected = self._codec.detect_magic_from_payload(data)
        if detected is not None:
            if self._codec.set_magic(detected):
                self.magic_changed.send(self, magic=detected)
        else:
            detected = self._codec.detect_magic_from_mem_request(unswizzled)
            if detected is not None:
                if self._codec.set_magic(detected):
                    self.magic_changed.send(self, magic=detected)

        self.decoded_received.send(self, data=unswizzled)

    def _on_status_changed(
        self, sender, status: TransportStatus, message: str = ""
    ) -> None:
        self.status_changed.send(self, status=status, message=message)


class RuidaServerTransport:
    """
    Ruida L2 server transport for UDP server mode.

    Unlike RuidaTransport (which wraps a client transport), this wraps
    a UdpServerTransport and handles responses to multiple clients.

    Usage:
        udp = UdpServerTransport("0.0.0.0", 50200)
        ruida = RuidaServerTransport(udp)
        ruida.decoded_received.connect(handler)  # handler(data, addr)
        await ruida.connect()
        await ruida.send_response(data, addr)
    """

    def __init__(self, transport: UdpServerTransport, magic: int = 0x88):
        self._transport: UdpServerTransport = transport
        self._codec = RuidaCodec(magic)

        self.decoded_received = Signal()
        self.magic_changed = Signal()

        self._transport.received.connect(self._on_raw_received)
        self._transport.status_changed.connect(self._on_status_changed)

    @property
    def magic(self) -> int:
        return self._codec.magic

    @magic.setter
    def magic(self, value: int) -> None:
        if self._codec.set_magic(value):
            self.magic_changed.send(self, magic=value)

    async def connect(self) -> None:
        await self._transport.connect()

    async def disconnect(self) -> None:
        await self._transport.disconnect()

    async def send_to(self, data: bytes, addr: Tuple[str, int]) -> None:
        """
        Send raw data to a specific client.

        Use send_response() for normal Ruida responses.
        """
        await self._transport.send_to(data, addr)

    async def send_response(
        self, response: bytes, addr: Tuple[str, int]
    ) -> None:
        """
        Send a Ruida response (swizzled, no checksum prefix).

        Args:
            response: Unswizzled response bytes
            addr: Client address (host, port)
        """
        swizzled = self._codec.swizzle(response)
        await self._transport.send_to(swizzled, addr)

    async def send_command(
        self, command: bytes, addr: Tuple[str, int]
    ) -> None:
        """
        Send a framed command to a specific client.

        Args:
            command: Unswizzled command bytes
            addr: Client address (host, port)
        """
        swizzled = self._codec.swizzle(command)
        framed = frame_packet(swizzled)
        await self._transport.send_to(framed, addr)

    def _on_raw_received(
        self, sender, data: bytes, addr: Tuple[str, int]
    ) -> None:
        """Handle raw data from underlying transport."""
        is_valid, payload, recv_cksum, calc_cksum = validate_packet(data)

        if not is_valid:
            logger.warning(
                f"Checksum mismatch from {addr}: received {recv_cksum:04X}, "
                f"calculated {calc_cksum:04X}"
            )
            return

        magic_detected = None

        if len(payload) == 4:
            detected = self._codec.detect_magic_from_payload(payload)
            if detected is not None:
                magic_detected = detected

        unswizzled = self._codec.unswizzle(payload)

        if magic_detected is None:
            detected = self._codec.detect_magic_from_mem_request(unswizzled)
            if detected is not None:
                magic_detected = detected

        if magic_detected is not None:
            if self._codec.set_magic(magic_detected):
                self.magic_changed.send(self, magic=magic_detected)

        self.decoded_received.send(self, data=unswizzled, addr=addr)

    def _on_status_changed(
        self, sender, status: TransportStatus, message: str = ""
    ) -> None:
        pass
