"""
Ruida Simulator - Emulates a Ruida laser controller.

This module provides a high-level simulator that wraps RuidaServer (L3)
for command processing. Transport layer (L2) responsibilities like
swizzle encoding and packet framing are handled separately by
RuidaTransport / RuidaServerTransport.

Layer architecture:
- L2: RuidaTransport / RuidaServerTransport (framing + swizzle)
- L3: RuidaServer (command parsing + state management)
- L4: RuidaSimulator (this class - convenience wrapper)

Based on:
- https://edutechwiki.unige.ch/en/Ruida
- https://github.com/meerk40t/meerk40t/tree/main/meerk40t/ruida
- https://github.com/StevenIsaacs/ruida-protocol-analyzer
"""

import logging
from typing import Callable, Optional

from blinker import Signal

from .ruida_server import RuidaServer

logger = logging.getLogger(__name__)


class RuidaSimulator:
    """
    Ruida controller simulator.

    This is a thin wrapper around RuidaServer (L3) that provides a
    convenient interface. All transport-layer concerns (swizzle,
    framing) are handled by the caller using RuidaServerTransport.

    For the main data channel:
        - Receive already-decoded (unswizzled) commands
        - Call process_commands() to handle them
        - Send unswizzled responses back via transport

    For the jog channel:
        - Jog packets are NOT swizzled
        - Call handle_jog_packet() to handle raw jog data
    """

    CARD_ID = 0x65106510
    DEFAULT_BED_X = 320000
    DEFAULT_BED_Y = 220000

    def __init__(
        self,
        on_command: Optional[Callable[[str, bytes], None]] = None,
        model: str = "644XG",
    ):
        self._server = RuidaServer(on_command=on_command, model=model)

        self.command_received = Signal()
        self.response_ready = Signal()

    @property
    def state(self):
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

    def process_commands(self, data: bytes) -> bytes:
        """
        Process unswizzled commands and return unswizzled response.

        This is the main entry point for the main data channel.
        The caller is responsible for swizzle/framing via RuidaServerTransport.

        Args:
            data: Unswizzled command bytes

        Returns:
            Unswizzled response bytes (may be empty for ACK-only)
        """
        return self._server.process_commands(data)

    def _process_single_command(self, data: bytes):
        """Delegate to server for backward compatibility with tests."""
        return self._server._process_single_command(data)

    def _mem_lookup(self, mem: int):
        """Delegate to state for backward compatibility with tests."""
        return self.state.mem_lookup(mem)

    def handle_jog_packet(self, data: bytes) -> bytes:
        """
        Handle a packet on the jog control channel.

        Jog packets are NOT swizzled. This method processes raw jog data
        and returns the raw response.

        Args:
            data: Raw jog packet bytes (not swizzled)

        Returns:
            Raw response bytes (not swizzled)
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


async def run_udp_simulator(
    simulator: RuidaSimulator,
    host: str = "0.0.0.0",
    port: int = 50200,
    jog_port: int = 50207,
    magic: int = 0x88,
) -> None:
    """
    Run the simulator with UDP transport (async).

    Uses proper layering:
    - L1: UdpServerTransport (raw UDP)
    - L2: RuidaServerTransport (framing + swizzle)
    - L3: RuidaServer via simulator.process_commands()
    - L4: RuidaSimulator
    """
    import asyncio

    from rayforge.machine.transport.udp_server import UdpServerTransport
    from .ruida_transport import RuidaServerTransport

    main_udp = UdpServerTransport(host, port)
    main_transport = RuidaServerTransport(main_udp, magic=magic)
    jog_transport = UdpServerTransport(host, jog_port)

    async def handle_main_decoded(sender, data: bytes, addr):
        logger.debug(f"Main decoded from {addr}: {data.hex()}")
        response = simulator.process_commands(data)
        if response == b"\xcc" or not response:
            logger.debug("Response: cc (ack)")
            await main_transport.send_response(b"\xcc", addr)
        else:
            logger.debug(f"Response: cc + {response.hex()}")
            await main_transport.send_response(b"\xcc", addr)
            await main_transport.send_response(response, addr)

    async def handle_jog(sender, data: bytes, addr):
        logger.debug(f"Jog packet from {addr}: {data.hex()}")
        response = simulator.handle_jog_packet(data)
        if response:
            await jog_transport.send_to(response, addr)

    main_transport.decoded_received.connect(
        lambda self, data, addr: asyncio.create_task(
            handle_main_decoded(self, data, addr)
        )
    )
    jog_transport.received.connect(
        lambda self, data, addr: asyncio.create_task(
            handle_jog(self, data, addr)
        )
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


def run_simulator(
    host: str = "0.0.0.0",
    port: int = 50200,
    jog_port: int = 50207,
    magic: int = 0x88,
) -> None:
    """Run the Ruida simulator (blocking)."""
    import asyncio

    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    simulator = RuidaSimulator()
    asyncio.run(run_udp_simulator(simulator, host, port, jog_port, magic))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Ruida laser controller simulator"
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
    )
