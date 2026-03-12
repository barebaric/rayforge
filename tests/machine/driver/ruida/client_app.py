# flake8: noqa: E402
"""
Ruida Client App - A GUI client for controlling a Ruida laser controller/simulator.

Sends movement commands to the simulator and displays the current position.
"""

import argparse
import asyncio
import logging
from typing import Optional

import gi

gi.require_version("Gtk", "4.0")
from gi.repository import GLib, Gtk

from rayforge.machine.driver.ruida.ruida_client import RuidaClient
from rayforge.machine.driver.ruida.ruida_transport import RuidaTransport
from rayforge.machine.driver.ruida.ruida_util import UM_PER_MM
from rayforge.machine.transport.udp import UdpTransport

logger = logging.getLogger(__name__)


class RuidaUdpClient:
    """
    Synchronous wrapper for RuidaClient using the layered architecture.

    Uses:
    - UdpTransport (L3/4): Raw UDP communication
    - RuidaTransport (L2): Swizzle + framing
    - RuidaClient (L5/6): Protocol commands
    """

    def __init__(self, host: str, port: int, magic: int = 0x88):
        self.host = host
        self.port = port
        self.magic = magic
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._udp: Optional[UdpTransport] = None
        self._transport: Optional[RuidaTransport] = None
        self.client: Optional[RuidaClient] = None

    @property
    def is_connected(self) -> bool:
        return self._transport is not None and self._transport.is_connected

    def _run_async(self, coro):
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
        return self._loop.run_until_complete(coro)

    def connect(self):
        self._udp = UdpTransport(self.host, self.port)
        self._transport = RuidaTransport(self._udp, self.magic)
        self.client = RuidaClient(self._transport)
        self._run_async(self.client.connect())

    def disconnect(self):
        if self.client:
            self._run_async(self.client.disconnect())
        self.client = None
        self._transport = None
        self._udp = None

    def send_command(self, cmd: bytes) -> bool:
        if not self.client:
            return False
        try:
            self._run_async(self.client.send_command(cmd))
            return True
        except Exception as e:
            logger.error(f"Error sending command: {e}")
            return False


class ClientWindow(Gtk.ApplicationWindow):
    def __init__(self, app: "ClientApp"):
        super().__init__(application=app)
        self.app = app
        self.client = app.client
        self._pos_x_um = 0
        self._pos_y_um = 0

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

        conn_vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        conn_vbox.set_margin_top(6)
        conn_vbox.set_margin_bottom(6)
        conn_vbox.set_margin_start(6)
        conn_vbox.set_margin_end(6)
        connection_frame.set_child(conn_vbox)

        ip_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        ip_row.append(Gtk.Label(label="IP:"))
        self.ip_entry = Gtk.Entry()
        self.ip_entry.set_text(self.client.host)
        self.ip_entry.set_hexpand(True)
        ip_row.append(self.ip_entry)
        conn_vbox.append(ip_row)

        btn_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.status_label = Gtk.Label(label="● Disconnected")
        self.status_label.set_hexpand(True)
        btn_row.append(self.status_label)

        self.connect_btn = Gtk.Button(label="Connect")
        self.connect_btn.connect("clicked", self._on_connect_clicked)
        btn_row.append(self.connect_btn)
        conn_vbox.append(btn_row)

        GLib.idle_add(self._auto_connect)

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
        for step in [1, 5, 10]:
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
        if self.client.is_connected:
            self.client.disconnect()
            self.status_label.set_text("● Disconnected")
            self.connect_btn.set_label("Connect")
        else:
            self._do_connect()

    def _do_connect(self):
        self.client.host = self.ip_entry.get_text().strip()
        try:
            self.client.connect()
            self.status_label.set_text(
                f"● Connected to {self.client.host}:{self.client.port}"
            )
            self.connect_btn.set_label("Disconnect")
        except Exception as e:
            self.status_label.set_text(f"● Error: {e}")

    def _auto_connect(self):
        self._do_connect()
        return False

    def _on_step_toggled(self, btn, step):
        if btn.get_active():
            self.step_size = step
            for other_btn, _ in self.step_buttons:
                if other_btn is not btn:
                    other_btn.set_active(False)

    def _on_jog(self, btn, axis: str, direction: int):
        if not self.client.is_connected:
            return
        assert self.client.client

        step_um = int(self.step_size * UM_PER_MM)
        MAX_REL_MOVE = 8000

        if step_um <= MAX_REL_MOVE:
            if axis == "x":
                self._pos_x_um += direction * step_um
                dx, dy = direction * step_um, 0
            else:
                self._pos_y_um += -direction * step_um
                dx, dy = 0, -direction * step_um
            self.client._run_async(self.client.client.move_rel(dx, dy))
        else:
            if axis == "x":
                self._pos_x_um += direction * step_um
            else:
                self._pos_y_um += -direction * step_um
            self.client._run_async(
                self.client.client.rapid_move_xy(
                    self._pos_x_um, self._pos_y_um
                )
            )

        logger.info(f"Jog {axis} direction={direction} step_um={step_um}")
        self.x_entry.set_text(str(self._pos_x_um / 1000.0))
        self.y_entry.set_text(str(self._pos_y_um / 1000.0))

    def _on_home(self, btn):
        if not self.client.is_connected:
            return
        assert self.client.client

        self._pos_x_um = 0
        self._pos_y_um = 0
        self.client._run_async(self.client.client.home_xy())
        self.x_entry.set_text("0")
        self.y_entry.set_text("0")

    def _on_move_abs(self, btn):
        if not self.client.is_connected:
            return
        assert self.client.client

        try:
            x_mm = float(self.x_entry.get_text())
            y_mm = float(self.y_entry.get_text())
        except ValueError:
            return

        x_um = int(x_mm * UM_PER_MM)
        y_um = int(y_mm * UM_PER_MM)

        self._pos_x_um = x_um
        self._pos_y_um = y_um
        self.client._run_async(self.client.client.move_abs(x_um, y_um))


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
