# flake8: noqa: E402
"""
Ruida Client App - A GUI client for controlling a Ruida laser controller/simulator.

Sends movement commands to the simulator and displays the current position.
"""

import argparse
import logging
import socket
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import Gtk

from rayforge.machine.driver.ruida.ruida_client import RuidaClient
from rayforge.machine.driver.ruida.ruida_util import (
    UM_PER_MM,
    build_swizzle_lut,
    calculate_checksum,
)

logger = logging.getLogger(__name__)


class RuidaUdpClient:
    def __init__(self, host: str, port: int, magic: int = 0x88):
        self.host = host
        self.port = port
        self.magic = magic
        self.swizzle_lut, self.unswizzle_lut = build_swizzle_lut(magic)
        self.client = RuidaClient()
        self._socket: Optional[socket.socket] = None

    def connect(self):
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(1.0)

    def disconnect(self):
        if self._socket:
            self._socket.close()
            self._socket = None

    def _swizzle(self, data: bytes) -> bytes:
        return bytes([self.swizzle_lut[b] for b in data])

    def _unswizzle(self, data: bytes) -> bytes:
        return bytes([self.unswizzle_lut[b] for b in data])

    def send_command(self, cmd: bytes) -> bool:
        if not self._socket:
            return False

        swizzled = self._swizzle(cmd)
        checksum = calculate_checksum(swizzled)
        packet = bytes([checksum >> 8, checksum & 0xFF]) + swizzled

        logger.debug(f"Sending packet: {packet.hex()}")
        try:
            self._socket.sendto(packet, (self.host, self.port))
            response, addr = self._socket.recvfrom(1024)
            unswizzled = self._unswizzle(response)
            logger.debug(
                f"Received response: {response.hex()} -> {unswizzled.hex()}"
            )
            return len(unswizzled) > 0 and unswizzled[0] == 0xCC
        except socket.timeout:
            logger.warning("Command timed out")
            return False
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False


class ClientWindow(Gtk.ApplicationWindow):
    def __init__(self, app: "ClientApp"):
        super().__init__(application=app)
        self.app = app
        self.client = app.client

        self.set_default_size(400, 500)
        self.set_title("Ruida Client")

        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        main_box.set_margin_top(12)
        main_box.set_margin_bottom(12)
        main_box.set_margin_start(12)
        main_box.set_margin_end(12)
        self.set_child(main_box)

        connection_frame = Gtk.Frame(label="Connection")
        main_box.append(connection_frame)

        conn_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        conn_box.set_margin_top(6)
        conn_box.set_margin_bottom(6)
        conn_box.set_margin_start(6)
        conn_box.set_margin_end(6)
        connection_frame.set_child(conn_box)

        self.status_label = Gtk.Label(label="● Disconnected")
        conn_box.append(self.status_label)

        self.connect_btn = Gtk.Button(label="Connect")
        self.connect_btn.connect("clicked", self._on_connect_clicked)
        conn_box.append(self.connect_btn)

        pos_frame = Gtk.Frame(label="Position (mm)")
        main_box.append(pos_frame)

        pos_grid = Gtk.Grid()
        pos_grid.set_margin_top(6)
        pos_grid.set_margin_bottom(6)
        pos_grid.set_margin_start(6)
        pos_grid.set_margin_end(6)
        pos_grid.set_column_spacing(12)
        pos_grid.set_row_spacing(6)
        pos_frame.set_child(pos_grid)

        pos_grid.attach(Gtk.Label(label="X:"), 0, 0, 1, 1)
        self.x_entry = Gtk.Entry()
        self.x_entry.set_text("0")
        self.x_entry.set_hexpand(True)
        pos_grid.attach(self.x_entry, 1, 0, 1, 1)

        pos_grid.attach(Gtk.Label(label="Y:"), 0, 1, 1, 1)
        self.y_entry = Gtk.Entry()
        self.y_entry.set_text("0")
        self.y_entry.set_hexpand(True)
        pos_grid.attach(self.y_entry, 1, 1, 1, 1)

        move_frame = Gtk.Frame(label="Jog Controls")
        main_box.append(move_frame)

        jog_grid = Gtk.Grid()
        jog_grid.set_margin_top(6)
        jog_grid.set_margin_bottom(6)
        jog_grid.set_margin_start(6)
        jog_grid.set_margin_end(6)
        jog_grid.set_column_spacing(6)
        jog_grid.set_row_spacing(6)
        jog_grid.set_halign(Gtk.Align.CENTER)
        move_frame.set_child(jog_grid)

        btn_up = Gtk.Button(label="▲ Y+")
        btn_up.set_size_request(60, 40)
        btn_up.connect("clicked", self._on_jog, "y", 1)
        jog_grid.attach(btn_up, 1, 0, 1, 1)

        btn_left = Gtk.Button(label="◀ X-")
        btn_left.set_size_request(60, 40)
        btn_left.connect("clicked", self._on_jog, "x", -1)
        jog_grid.attach(btn_left, 0, 1, 1, 1)

        btn_home = Gtk.Button(label="⌂ Home")
        btn_home.set_size_request(60, 40)
        btn_home.connect("clicked", self._on_home)
        jog_grid.attach(btn_home, 1, 1, 1, 1)

        btn_right = Gtk.Button(label="▶ X+")
        btn_right.set_size_request(60, 40)
        btn_right.connect("clicked", self._on_jog, "x", 1)
        jog_grid.attach(btn_right, 2, 1, 1, 1)

        btn_down = Gtk.Button(label="▼ Y-")
        btn_down.set_size_request(60, 40)
        btn_down.connect("clicked", self._on_jog, "y", -1)
        jog_grid.attach(btn_down, 1, 2, 1, 1)

        step_frame = Gtk.Frame(label="Step Size (mm)")
        main_box.append(step_frame)

        step_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        step_box.set_margin_top(6)
        step_box.set_margin_bottom(6)
        step_box.set_margin_start(6)
        step_box.set_margin_end(6)
        step_frame.set_child(step_box)

        self.step_buttons = []
        for step in [1, 5]:
            btn = Gtk.ToggleButton(label=str(step))
            btn.connect("toggled", self._on_step_toggled, step)
            step_box.append(btn)
            self.step_buttons.append((btn, step))
            if step == 5:
                btn.set_active(True)

        self.step_size = 5.0

        abs_frame = Gtk.Frame(label="Absolute Move")
        main_box.append(abs_frame)

        abs_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        abs_box.set_margin_top(6)
        abs_box.set_margin_bottom(6)
        abs_box.set_margin_start(6)
        abs_box.set_margin_end(6)
        abs_frame.set_child(abs_box)

        btn_move_abs = Gtk.Button(label="Move to Position")
        btn_move_abs.connect("clicked", self._on_move_abs)
        btn_move_abs.set_hexpand(True)
        abs_box.append(btn_move_abs)

    def _on_connect_clicked(self, btn):
        if self.client._socket:
            self.client.disconnect()
            self.status_label.set_text("● Disconnected")
            self.connect_btn.set_label("Connect")
        else:
            try:
                self.client.connect()
                self.status_label.set_text(
                    f"● Connected to {self.client.host}:{self.client.port}"
                )
                self.connect_btn.set_label("Disconnect")
            except Exception as e:
                self.status_label.set_text(f"● Error: {e}")

    def _on_step_toggled(self, btn, step):
        if btn.get_active():
            self.step_size = step
            for other_btn, _ in self.step_buttons:
                if other_btn is not btn:
                    other_btn.set_active(False)

    def _on_jog(self, btn, axis: str, direction: int):
        if not self.client._socket:
            return

        step_um = int(self.step_size * UM_PER_MM)

        if axis == "x":
            cmd = self.client.client.build_move_rel(-direction * step_um, 0)
        else:
            cmd = self.client.client.build_move_rel(0, -direction * step_um)

        logger.info(
            f"Jog {axis} direction={direction} step_um={step_um} cmd={cmd.hex()}"
        )
        if self.client.send_command(cmd):
            try:
                x = float(self.x_entry.get_text())
                y = float(self.y_entry.get_text())
                if axis == "x":
                    x += direction * self.step_size
                else:
                    y += direction * self.step_size
                self.x_entry.set_text(str(x))
                self.y_entry.set_text(str(y))
            except ValueError:
                pass

    def _on_home(self, btn):
        if not self.client._socket:
            return

        cmd = self.client.client.build_home_xy()
        logger.info(f"Home command: {cmd.hex()}")
        self.client.send_command(cmd)
        self.x_entry.set_text("0")
        self.y_entry.set_text("0")

    def _on_move_abs(self, btn):
        if not self.client._socket:
            return

        try:
            x_mm = float(self.x_entry.get_text())
            y_mm = float(self.y_entry.get_text())
        except ValueError:
            return

        x_um = int(x_mm * UM_PER_MM)
        y_um = int(y_mm * UM_PER_MM)

        cmd = self.client.client.build_move_abs(x_um, y_um)
        logger.info(f"Move abs: x={x_mm}mm y={y_mm}mm cmd={cmd.hex()}")
        self.client.send_command(cmd)


class ClientApp(Gtk.Application):
    def __init__(self, host: str, port: int, magic: int):
        super().__init__(application_id="com.rayforge.RuidaClient")
        self.client = RuidaUdpClient(host, port, magic)
        self._window: Optional[ClientWindow] = None

    def do_activate(self):
        self._window = ClientWindow(self)
        self._window.present()


def run_app(host: str = "localhost", port: int = 50200, magic: int = 0x88):
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    app = ClientApp(host, port, magic)
    app.run(None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ruida laser controller client"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host to connect to (default: localhost)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=50200,
        help="UDP port for main channel (default: 50200)",
    )
    parser.add_argument(
        "--magic",
        type=lambda x: int(x, 0),
        default=0x88,
        help="Swizzle magic number (default: 0x88)",
    )

    args = parser.parse_args()
    run_app(host=args.host, port=args.port, magic=args.magic)
