# flake8: noqa: E402
"""
Ruida Simulator with UI - A graphical application for the Ruida simulator.

Displays a WorldSurface with a laser dot that tracks the simulator's position.
"""

import argparse
import logging
import socket
from typing import Optional, Tuple

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from rayforge.ui_gtk.canvas.worldsurface import WorldSurface
from rayforge.ui_gtk.canvas2d.elements.dot import DotElement
from rayforge.machine.driver.ruida.ruida_simulator import RuidaSimulator
from rayforge.machine.driver.ruida.ruida_transport import RuidaCodec
from rayforge.machine.driver.ruida.ruida_framing import validate_packet

logger = logging.getLogger(__name__)


class SimpleUdpServer:
    def __init__(self, host: str, port: int, handler):
        self.host = host
        self.port = port
        self.handler = handler
        self._socket: Optional[socket.socket] = None
        self._timeout_id = 0

    def start(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind((self.host, self.port))
        self._socket.setblocking(False)
        self._timeout_id = GLib.timeout_add(10, self._poll)
        logger.info(f"UDP server listening on {self.host}:{self.port}")

    def stop(self):
        if self._timeout_id:
            GLib.source_remove(self._timeout_id)
            self._timeout_id = 0
        if self._socket:
            self._socket.close()
            self._socket = None

    def send_to(self, data: bytes, addr: Tuple[str, int]):
        if self._socket:
            self._socket.sendto(data, addr)

    def _poll(self):
        if not self._socket:
            return False
        try:
            data, addr = self._socket.recvfrom(2048)
            response = self.handler(data, addr)
            if response:
                if isinstance(response, list):
                    for pkt in response:
                        self.send_to(pkt, addr)
                else:
                    self.send_to(response, addr)
        except BlockingIOError:
            pass
        except Exception as e:
            logger.error(f"Error in UDP poll: {e}")
        return True


class SimulatorWindow(Gtk.ApplicationWindow):
    def __init__(
        self,
        app: "SimulatorApp",
        simulator: RuidaSimulator,
        codec: RuidaCodec,
    ):
        super().__init__(application=app)
        self.simulator = simulator
        self.codec = codec
        self.app = app
        self._update_timeout_id = 0

        self.bed_x_mm = simulator.bed_x / 1000.0
        self.bed_y_mm = simulator.bed_y / 1000.0

        self.set_size_request(int(self.bed_x_mm * 4), int(self.bed_y_mm * 4))
        self.set_title(
            f"Ruida Simulator ({self.bed_x_mm:.0f}x{self.bed_y_mm:.0f}mm)"
        )

        self.surface = self._create_surface()
        self.set_child(self.surface)

        self._laser_dot_pos_mm = (0.0, 0.0)
        self._update_timeout_id = GLib.timeout_add(
            50, self._update_laser_position
        )

        self._main_server = SimpleUdpServer(
            app.host, app.port, self._handle_main_packet
        )
        self._jog_server = SimpleUdpServer(
            app.host, app.jog_port, self._handle_jog_packet
        )
        self._main_server.start()
        self._jog_server.start()

        print(f"Ruida simulator running on {app.host}:{app.port}")

    def _create_surface(self) -> WorldSurface:
        surface = WorldSurface(
            width_mm=self.bed_x_mm,
            height_mm=self.bed_y_mm,
            show_grid=True,
            show_axis=True,
            y_axis_down=True,
        )

        self._laser_dot = DotElement(0, 0, 1.0)
        self._laser_dot.set_visible(True)
        surface.root.add(self._laser_dot)

        return surface

    def _set_laser_dot_position(self, x_mm: float, y_mm: float) -> None:
        self._laser_dot_pos_mm = x_mm, y_mm

        dot_w_mm = self._laser_dot.width
        dot_h_mm = self._laser_dot.height

        self._laser_dot.set_pos(x_mm - dot_w_mm / 2, y_mm - dot_h_mm / 2)
        self.surface.queue_draw()

    def _update_laser_position(self) -> bool:
        jog = self.simulator.jog_active
        self.simulator.x += jog.get("x", 0)
        self.simulator.y += jog.get("y", 0)

        sim_x_um = self.simulator.x
        sim_y_um = self.simulator.y

        display_x_mm = sim_x_um / 1000.0
        display_y_mm = self.bed_y_mm - sim_y_um / 1000.0

        self._set_laser_dot_position(display_x_mm, display_y_mm)
        return True

    def _handle_main_packet(self, data: bytes, addr: Tuple[str, int]):
        logger.info(f"Main packet from {addr}: {data.hex()}")

        is_valid, payload, recv_cksum, calc_cksum = validate_packet(data)
        if not is_valid:
            logger.warning(
                f"Checksum mismatch: received {recv_cksum:04X}, "
                f"calculated {calc_cksum:04X}"
            )
            return [self.codec.swizzle(b"\xcc")]

        detected = self.codec.detect_magic_from_payload(payload)
        if detected is not None:
            self.codec.set_magic(detected)

        unswizzled = self.codec.unswizzle(payload)

        detected = self.codec.detect_magic_from_mem_request(unswizzled)
        if detected is not None:
            self.codec.set_magic(detected)

        logger.info(f"Decoded command: {unswizzled.hex()}")
        if unswizzled and unswizzled[0] == 0xD9:
            logger.info("*** D9 RAPID MOVE COMMAND RECEIVED ***")
        response = self.simulator.process_commands(unswizzled)
        logger.info(
            f"Raw response: {response.hex() if response else '(empty)'}"
        )

        responses = []
        if response == b"\xcc" or not response:
            responses.append(self.codec.swizzle(b"\xcc"))
        else:
            responses.append(self.codec.swizzle(b"\xcc"))
            responses.append(self.codec.swizzle(response))

        for i, pkt in enumerate(responses):
            logger.info(f"Response[{i}]: {pkt.hex()}")

        x_mm = self.simulator.x / 1000.0
        y_mm = self.simulator.y / 1000.0
        logger.info(f"Simulator position: x={x_mm:.3f}mm y={y_mm:.3f}mm")

        return responses

    def _handle_jog_packet(self, data: bytes, addr: Tuple[str, int]):
        logger.info(f"JOG packet from {addr}: {data.hex()}")
        response = self.simulator.handle_jog_packet(data)
        logger.info(
            f"JOG response: {response.hex() if response else '(empty)'}"
        )
        return response

    def do_close_request(self) -> bool:
        if self._update_timeout_id:
            GLib.source_remove(self._update_timeout_id)
            self._update_timeout_id = 0
        self._main_server.stop()
        self._jog_server.stop()
        return False


class SimulatorApp(Gtk.Application):
    def __init__(
        self,
        simulator: RuidaSimulator,
        codec: RuidaCodec,
        host: str = "0.0.0.0",
        port: int = 50200,
        jog_port: int = 50207,
    ):
        super().__init__(application_id="com.rayforge.RuidaSimulator")
        self.simulator = simulator
        self.codec = codec
        self.host = host
        self.port = port
        self.jog_port = jog_port
        self._window: Optional[SimulatorWindow] = None

    def do_activate(self):
        self._window = SimulatorWindow(self, self.simulator, self.codec)
        self._window.present()


def run_app(
    host: str = "0.0.0.0",
    port: int = 50200,
    jog_port: int = 50207,
    magic: int = 0x88,
):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    simulator = RuidaSimulator()
    codec = RuidaCodec(magic)
    app = SimulatorApp(
        simulator=simulator,
        codec=codec,
        host=host,
        port=port,
        jog_port=jog_port,
    )
    app.run(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ruida laser controller simulator with UI"
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
    run_app(
        host=args.host,
        port=args.port,
        jog_port=args.jog_port,
        magic=args.magic,
    )
