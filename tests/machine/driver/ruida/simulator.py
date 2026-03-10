"""
Ruida Simulator - Emulates a Ruida laser controller.

This module implements the Ruida protocol and can be used with different
transports (UDP, serial/PTY). Use the runner functions for standalone
operation or integrate the RuidaSimulator class with custom transports.

Based on:
- https://edutechwiki.unige.ch/en/Ruida
- https://github.com/meerk40t/meerk40t/tree/main/meerk40t/ruida
- https://github.com/StevenIsaacs/ruida-protocol-analyzer
"""

import logging
from typing import Callable, Dict, Optional

from rayforge.machine.driver.ruida.ruida_framing import (
    PacketFramer,
    PacketStatus,
    validate_packet,
)
from rayforge.machine.driver.ruida.ruida_maps import CARD_ID_TO_MAGIC
from rayforge.machine.driver.ruida.ruida_protocol import RuidaState
from rayforge.machine.driver.ruida.ruida_server import RuidaServer
from rayforge.machine.driver.ruida.ruida_util import (
    build_swizzle_lut,
    parse_mem,
)

logger = logging.getLogger(__name__)


class RuidaSimulator:
    """
    Ruida controller simulator with transport handling.

    Handles swizzle encoding and packet framing, delegating protocol
    commands to RuidaServer.
    """

    CARD_ID = 0x65106510
    DEFAULT_BED_X = 320000
    DEFAULT_BED_Y = 220000

    def __init__(
        self,
        magic: int = 0x88,
        on_command: Optional[Callable[[str, bytes], None]] = None,
        model: str = "644XG",
    ):
        self.magic = magic
        self.model = model

        self.swizzle_lut, self.unswizzle_lut = build_swizzle_lut(magic)
        self._magic_keys = self._build_magic_keys()
        self._framer = PacketFramer()

        self._server = RuidaServer(on_command=on_command, model=model)

    @property
    def state(self) -> RuidaState:
        """Access to the server state."""
        return self._server.state

    @property
    def x(self):
        return self.state.x

    @x.setter
    def x(self, value):
        self.state.x = value

    @property
    def y(self):
        return self.state.y

    @y.setter
    def y(self, value):
        self.state.y = value

    @property
    def z(self):
        return self.state.z

    @z.setter
    def z(self, value):
        self.state.z = value

    @property
    def u(self):
        return self.state.u

    @u.setter
    def u(self, value):
        self.state.u = value

    @property
    def program_mode(self):
        return self.state.program_mode

    @program_mode.setter
    def program_mode(self, value):
        self.state.program_mode = value

    @property
    def machine_status(self):
        return self.state.machine_status

    @machine_status.setter
    def machine_status(self, value):
        self.state.machine_status = value

    @property
    def ref_point_mode(self):
        return self.state.ref_point_mode

    @property
    def _ref_point_mode(self):
        return self.state.ref_point_mode

    @property
    def _jog_active(self):
        return self.state.jog_active

    @property
    def _memory_values(self):
        return self.state.memory_values

    @property
    def filename(self):
        return self.state.filename

    @property
    def jog_speed(self):
        return self.state.jog_speed

    @jog_speed.setter
    def jog_speed(self, value):
        self.state.jog_speed = value

    @property
    def _jog_speed(self):
        return self.state.jog_speed

    @_jog_speed.setter
    def _jog_speed(self, value):
        self.state.jog_speed = value

    @property
    def jog_active(self):
        return self.state.jog_active

    @property
    def bed_x(self):
        return self.state.bed_x

    @property
    def bed_y(self):
        return self.state.bed_y

    @property
    def file_checksum(self):
        return self.state.file_checksum

    @property
    def file_checksum_accumulator(self):
        return self.state.file_checksum_accumulator

    @file_checksum_accumulator.setter
    def file_checksum_accumulator(self, value):
        self.state.file_checksum_accumulator = value

    @property
    def checksum_enabled(self):
        return self.state.checksum_enabled

    @checksum_enabled.setter
    def checksum_enabled(self, value):
        self.state.checksum_enabled = value

    def _process_single_command(self, data: bytes):
        """Delegate to server for backward compatibility with tests."""
        return self._server._process_single_command(data)

    def _process_commands(self, data: bytes):
        """Delegate to server for backward compatibility with tests."""
        return self._server.process_commands(data)

    def _mem_lookup(self, mem: int):
        """Delegate to state for backward compatibility with tests."""
        return self.state.mem_lookup(mem)

    def _build_magic_keys(self) -> Dict[bytes, int]:
        """Build lookup for magic key detection."""
        keys = {}
        for g in range(256):
            swiz, _ = build_swizzle_lut(g)
            keys[bytes([swiz[b] for b in b"\xda\x00\x05\x7e"])] = g
        return keys

    def swizzle(self, data: bytes) -> bytes:
        """Swizzle data for transmission."""
        return bytes([self.swizzle_lut[b] for b in data])

    def unswizzle(self, data: bytes) -> bytes:
        """Unswizzle received data."""
        return bytes([self.unswizzle_lut[b] for b in data])

    def handle_main_packet(self, data: bytes) -> bytes:
        """
        Handle a packet on the main data channel.

        UDP packets have a 2-byte checksum prefix, followed by swizzled data.
        Returns swizzled response ready to send.
        """
        if len(data) < 2:
            return b""

        is_valid, payload, recv_cksum, calc_cksum = validate_packet(data)

        if len(payload) == 4:
            magic = self._magic_keys.get(payload)
            if magic is not None:
                self._set_magic(magic)

        if not is_valid:
            logger.warning(
                f"Checksum mismatch: received {recv_cksum:04X}, "
                f"calculated {calc_cksum:04X}"
            )
            return self.swizzle(b"\xcf")

        unswizzled = self.unswizzle(payload)

        if len(unswizzled) >= 4 and unswizzled[0] == 0xDA:
            if unswizzled[1] == 0x00:
                mem = parse_mem(unswizzled[2:4])
                if mem == 0x057E:
                    pass
                elif mem in CARD_ID_TO_MAGIC:
                    card_id = mem
                    magic = CARD_ID_TO_MAGIC.get(card_id)
                    if magic is not None and magic != self.magic:
                        self._set_magic(magic)

        response = self._server.process_commands(unswizzled)

        if response:
            return self.swizzle(response)
        return self.swizzle(b"\xcc")

    def handle_jog_packet(self, data: bytes) -> bytes:
        """
        Handle a packet on the jog control channel.

        Jog packets are NOT swizzled.
        """
        if len(data) < 1:
            return b""

        if data[0] == 0xCC:
            return b"\xcc"

        if data[0] == 0xCE:
            return b"\xcc"

        if len(data) < 3:
            return b""

        if data[0] == 0xA5:
            self._server.process_commands(data)
            return b"\xcc"

        return b"\xcc"

    def handle_serial_data(self, data: bytes) -> bytes:
        """
        Handle streaming serial data.

        Accumulates data into buffer, extracts complete packets, and returns
        concatenated responses. Packets have 2-byte checksum prefix + payload.
        """
        self._framer.add_data(data)
        responses = b""

        for packet in self._framer.extract_packets():
            if packet.status == PacketStatus.INVALID_CHECKSUM:
                logger.warning(
                    f"Serial checksum mismatch: "
                    f"recv {packet.expected_checksum:04X} "
                    f"calc {packet.actual_checksum:04X}"
                )
                responses += self.swizzle(b"\xcf")
                continue

            payload = packet.payload

            if len(payload) == 4:
                magic = self._magic_keys.get(payload)
                if magic is not None:
                    self._set_magic(magic)

            unswizzled = self.unswizzle(payload)

            if len(unswizzled) >= 4 and unswizzled[0] == 0xDA:
                if unswizzled[1] == 0x00:
                    mem = parse_mem(unswizzled[2:4])
                    if mem == 0x057E:
                        pass
                    elif mem in CARD_ID_TO_MAGIC:
                        card_id = mem
                        magic = CARD_ID_TO_MAGIC.get(card_id)
                        if magic is not None and magic != self.magic:
                            self._set_magic(magic)

            response = self._server.process_commands(unswizzled)
            responses += (
                self.swizzle(response) if response else self.swizzle(b"\xcc")
            )

        return responses

    def _set_magic(self, magic: int) -> None:
        """Set the swizzle magic number."""
        if magic != self.magic:
            self.magic = magic
            self.swizzle_lut, self.unswizzle_lut = build_swizzle_lut(magic)
            logger.info(f"Magic set to 0x{magic:02X}")


async def run_udp_simulator(
    simulator: RuidaSimulator,
    host: str = "0.0.0.0",
    port: int = 50200,
    jog_port: int = 50207,
) -> None:
    """Run the simulator with UDP transport (async)."""
    import asyncio

    from rayforge.machine.transport.udp_server import UdpServerTransport

    main_transport = UdpServerTransport(host, port)
    jog_transport = UdpServerTransport(host, jog_port)

    async def handle_main(data: bytes, addr):
        logger.debug(f"Main packet from {addr}: {data.hex()}")
        response = simulator.handle_main_packet(data)
        logger.debug(f"Response: {response.hex() if response else None}")
        if response:
            await main_transport.send_to(response, addr)

    async def handle_jog(data: bytes, addr):
        logger.debug(f"Jog packet from {addr}: {data.hex()}")
        response = simulator.handle_jog_packet(data)
        if response:
            await jog_transport.send_to(response, addr)

    main_transport.received.connect(
        lambda self, data, addr: asyncio.create_task(handle_main(data, addr))
    )
    jog_transport.received.connect(
        lambda self, data, addr: asyncio.create_task(handle_jog(data, addr))
    )

    await main_transport.connect()
    await jog_transport.connect()

    print(f"Ruida simulator running on {host}:{port} (jog: {jog_port})")
    print("Press Ctrl+C to stop")

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await main_transport.disconnect()
        await jog_transport.disconnect()


async def run_serial_simulator(simulator: RuidaSimulator) -> None:
    """Run the simulator with serial/PTY transport (async)."""
    import asyncio

    from rayforge.machine.transport.serial_server import SerialServerTransport

    transport = SerialServerTransport()

    async def handle_data(self, data: bytes):
        response = simulator.handle_serial_data(data)
        if response:
            await transport.send(response)

    transport.received.connect(handle_data)

    await transport.connect()

    print("Ruida simulator serial mode")
    print(f"Connect to: {transport.slave_path}")
    print("Press Ctrl+C to stop")

    try:
        while True:
            await asyncio.sleep(1)
    finally:
        await transport.disconnect()


def run_simulator(
    host: str = "0.0.0.0",
    port: int = 50200,
    jog_port: int = 50207,
    magic: int = 0x88,
    serial_mode: bool = False,
) -> None:
    """Run the Ruida simulator (blocking)."""
    import asyncio

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    simulator = RuidaSimulator(magic=magic)

    if serial_mode:
        asyncio.run(run_serial_simulator(simulator))
    else:
        asyncio.run(run_udp_simulator(simulator, host, port, jog_port))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ruida laser controller simulator"
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Run in serial/PTY mode instead of UDP",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="UDP host to bind to (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50200,
        help="UDP port for main channel (default: 50200)",
    )
    parser.add_argument(
        "--jog-port",
        type=int,
        default=50207,
        help="UDP port for jog control (default: 50207)",
    )
    parser.add_argument(
        "--magic",
        type=lambda x: int(x, 0),
        default=0x88,
        help="Swizzle magic number (default: 0x88)",
    )

    args = parser.parse_args()
    run_simulator(
        host=args.host,
        port=args.port,
        jog_port=args.jog_port,
        magic=args.magic,
        serial_mode=args.serial,
    )
