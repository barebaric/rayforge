import logging
from typing import Optional, Tuple
from blinker import Signal
from gi.repository import Gtk
from ..icons import get_icon
from ..shared.gtk import apply_css

logger = logging.getLogger(__name__)

css = """
.point-bubble {
    background-color: @window_bg_color;
    border: 1px solid @borders;
    border-radius: 8px;
    padding: 10px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.3);
}
.active-point-bubble {
}
.point-bubble-heading {
    font-weight: bold;
}
.point-bubble .dim-label {
    opacity: 0.7;
    font-size: 0.85em;
}
"""


class PointBubbleWidget(Gtk.Box):
    def __init__(self, point_index: int, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL, spacing=8, **kwargs
        )
        self.point_index = point_index
        self.image_x: Optional[float] = None
        self.image_y: Optional[float] = None

        apply_css(css)
        self.add_css_class("point-bubble")

        # Define blinker signals
        self.value_changed = Signal()
        self.delete_requested = Signal()
        self.focus_requested = Signal()
        self.nudge_requested = Signal()  # Sends: sender, dx, dy

        # --- Header Row (Title & Delete) ---
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self.title_label = Gtk.Label(
            label=_("Point {n}").format(
                n=point_index + 1))
        self.title_label.add_css_class("point-bubble-heading")
        self.title_label.set_hexpand(True)
        self.title_label.set_halign(Gtk.Align.START)
        header_box.append(self.title_label)

        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.add_css_class("flat")
        self.delete_button.set_valign(Gtk.Align.CENTER)
        self.delete_button.set_tooltip_text(_("Delete this point"))
        self.delete_button.connect("clicked", self.on_delete_clicked)
        header_box.append(self.delete_button)
        self.append(header_box)

        # --- Coordinates Row ---
        coords_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=12)

        # World X
        x_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        x_box.append(Gtk.Label(label="X:"))
        adjustment_x = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0)
        self.world_x_spin = Gtk.SpinButton.new(adjustment_x, 0.1, 2)
        self.world_x_spin.set_valign(Gtk.Align.CENTER)
        self.world_x_spin.set_width_chars(6)
        x_box.append(self.world_x_spin)
        coords_box.append(x_box)

        # World Y
        y_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        y_box.append(Gtk.Label(label="Y:"))
        adjustment_y = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0)
        self.world_y_spin = Gtk.SpinButton.new(adjustment_y, 0.1, 2)
        self.world_y_spin.set_valign(Gtk.Align.CENTER)
        self.world_y_spin.set_width_chars(6)
        y_box.append(self.world_y_spin)
        coords_box.append(y_box)

        self.append(coords_box)

        # Connect SpinButtons
        self.world_x_spin.connect("value-changed", self.on_value_changed)
        self.world_y_spin.connect("value-changed", self.on_value_changed)

        # --- Image Nudge Row ---
        nudge_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=2)
        nudge_box.set_halign(Gtk.Align.CENTER)

        nudge_label = Gtk.Label(label=_("Nudge Pixel:"))
        nudge_label.add_css_class("dim-label")
        nudge_label.set_margin_end(8)
        nudge_box.append(nudge_label)

        btn_left = Gtk.Button(child=get_icon("go-previous-symbolic"))
        btn_up = Gtk.Button(child=get_icon("go-up-symbolic"))
        btn_down = Gtk.Button(child=get_icon("go-down-symbolic"))
        btn_right = Gtk.Button(child=get_icon("go-next-symbolic"))

        for btn in (btn_left, btn_up, btn_down, btn_right):
            btn.add_css_class("flat")
            btn.add_css_class("circular")

        # Arrow buttons emit a 0.5 sub-pixel nudge to the image coordinate
        btn_left.connect(
            "clicked",
            lambda _: self.nudge_requested.send(
                self,
                dx=-0.5,
                dy=0))
        btn_right.connect(
            "clicked",
            lambda _: self.nudge_requested.send(
                self,
                dx=0.5,
                dy=0))
        btn_up.connect(
            "clicked",
            lambda _: self.nudge_requested.send(
                self,
                dx=0,
                dy=-0.5))
        btn_down.connect(
            "clicked",
            lambda _: self.nudge_requested.send(
                self,
                dx=0,
                dy=0.5))

        nudge_box.append(btn_left)
        nudge_box.append(btn_up)
        nudge_box.append(btn_down)
        nudge_box.append(btn_right)

        self.append(nudge_box)

        # --- Event Controllers ---
        key_controller_x = Gtk.EventControllerKey.new()
        key_controller_x.connect("key-released", self.on_key_released)
        self.world_x_spin.add_controller(key_controller_x)

        key_controller_y = Gtk.EventControllerKey.new()
        key_controller_y.connect("key-released", self.on_key_released)
        self.world_y_spin.add_controller(key_controller_y)

        focus_controller_x = Gtk.EventControllerFocus()
        focus_controller_x.connect(
            "enter", self.on_spin_focus, self.world_x_spin)
        self.world_x_spin.add_controller(focus_controller_x)

        focus_controller_y = Gtk.EventControllerFocus()
        focus_controller_y.connect(
            "enter", self.on_spin_focus, self.world_y_spin)
        self.world_y_spin.add_controller(focus_controller_y)

    def set_point_index(self, index: int):
        self.point_index = index
        self.title_label.set_label(_("Point {n}").format(n=index + 1))

    def on_key_released(self, controller, keyval, keycode, state):
        self.on_value_changed(controller.get_widget())

    def on_spin_focus(self, controller, widget):
        self.focus_requested.send(self, widget=widget)

    def on_value_changed(self, widget):
        self.value_changed.send(self)

    def on_delete_clicked(self, button):
        self.delete_requested.send(self)

    def set_image_coords(self, x: float, y: float):
        self.image_x = x
        self.image_y = y

    def get_image_coords(self) -> Optional[Tuple[float, float]]:
        if self.image_x is not None and self.image_y is not None:
            return (self.image_x, self.image_y)
        return None

    def get_world_coords(self) -> Tuple[float, float]:
        try:
            x = float(self.world_x_spin.get_text())
        except ValueError:
            x = self.world_x_spin.get_value()
        try:
            y = float(self.world_y_spin.get_text())
        except ValueError:
            y = self.world_y_spin.get_value()
        return x, y

    def set_world_coords(self, x: float, y: float):
        self.world_x_spin.set_value(x)
        self.world_y_spin.set_value(y)

    def clear_focus(self):
        if self.world_x_spin.has_focus() or self.world_y_spin.has_focus():
            window = self.world_x_spin.get_ancestor(Gtk.Window)
            if isinstance(window, Gtk.Window):
                window.set_focus(None)

    def set_active(self, active: bool):
        self.set_visible(active)
        if active:
            self.add_css_class("active-point-bubble")
        else:
            self.remove_css_class("active-point-bubble")
