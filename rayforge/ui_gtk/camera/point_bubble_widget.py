import logging
from typing import Optional, Tuple
from blinker import Signal
from gi.repository import Gtk
from ..icons import get_icon
from ..shared.gtk import apply_css


logger = logging.getLogger(__name__)


css = """
.point-bubble {
    background-color: @popover_bg_color;
    border: 1px solid @borders;
    border-radius: 5px;
    padding: 5px;
}
.active-point-bubble {
}
"""


class PointBubbleWidget(Gtk.Box):
    def __init__(self, point_index: int, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, **kwargs
        )
        self.point_index = point_index
        self.image_x: Optional[float] = None
        self.image_y: Optional[float] = None

        # Add base CSS class for styling
        apply_css(css)
        self.add_css_class("point-bubble")

        # Define blinker signals
        self.value_changed = Signal()
        self.delete_requested = Signal()
        self.focus_requested = Signal()

        # SpinButtons for World X and Y
        adjustment_x = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0
        )
        self.world_x_spin = Gtk.SpinButton.new(adjustment_x, 0.1, 2)
        self.world_x_spin.set_valign(Gtk.Align.CENTER)
        adjustment_y = Gtk.Adjustment.new(
            0.0, -10000.0, 10000.0, 0.1, 1.0, 0.0
        )
        self.world_y_spin = Gtk.SpinButton.new(adjustment_y, 0.1, 2)
        self.world_y_spin.set_valign(Gtk.Align.CENTER)

        self.world_x_spin.connect("value-changed", self.on_value_changed)
        self.world_y_spin.connect("value-changed", self.on_value_changed)

        self.append(self.world_x_spin)
        self.append(self.world_y_spin)

        # Add delete button
        self.delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        self.delete_button.set_valign(Gtk.Align.CENTER)
        self.delete_button.set_tooltip_text(_("Delete this point"))
        self.append(self.delete_button)
        self.delete_button.connect("clicked", self.on_delete_clicked)

        # Connect key release events to trigger value changed
        key_controller_x = Gtk.EventControllerKey.new()
        key_controller_x.connect("key-released", self.on_key_released)
        self.world_x_spin.add_controller(key_controller_x)

        key_controller_y = Gtk.EventControllerKey.new()
        key_controller_y.connect("key-released", self.on_key_released)
        self.world_y_spin.add_controller(key_controller_y)

        # Connect focus events to emit signal
        focus_controller_x = Gtk.EventControllerFocus()
        focus_controller_x.connect(
            "enter", self.on_spin_focus, self.world_x_spin
        )
        self.world_x_spin.add_controller(focus_controller_x)

        focus_controller_y = Gtk.EventControllerFocus()
        focus_controller_y.connect(
            "enter", self.on_spin_focus, self.world_y_spin
        )
        self.world_y_spin.add_controller(focus_controller_y)

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
        self.set_visible(active)  # Control visibility
        if active:
            self.add_css_class("active-point-bubble")
        else:
            self.remove_css_class("active-point-bubble")
