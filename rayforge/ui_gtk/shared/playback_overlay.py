from gettext import gettext as _
import logging
from typing import Optional

from gi.repository import GLib, Gtk

from ..icons import get_icon
from .gtk import apply_css

logger = logging.getLogger(__name__)

SPEED_OPTIONS = [1, 2, 4, 8, 16]

css = """
.playback-overlay {
    background-color: alpha(@theme_bg_color, 0.75);
    border-radius: 6px;
    padding: 3px 6px;
}
.playback-overlay scale {
    min-width: 250px;
}
.speed-button {
    min-width: 36px;
    padding: 2px 6px;
    font-size: small;
}
"""


class PlaybackOverlay(Gtk.Box):
    """
    Playback controls (play/pause button + slider + speed button)
    overlaid on the 3D canvas. Slider drives OpPlayer.seek(); play
    button starts a 24 fps timer that advances the playhead.
    """

    def __init__(self, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
            **kwargs,
        )
        apply_css(css)
        self.add_css_class("playback-overlay")
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.END)
        self.set_margin_bottom(6)

        self._play_icon = get_icon("play-arrow-symbolic")
        self._pause_icon = get_icon("pause-symbolic")

        self._play_button = Gtk.Button()
        self._play_button.set_child(self._play_icon)
        self._play_button.set_tooltip_text(_("Play simulation"))
        self._play_button.set_sensitive(False)
        self._play_button.connect("clicked", self._on_play_clicked)
        self.append(self._play_button)

        self._slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, 0, 1, 1
        )
        self._slider.set_draw_value(False)
        self._slider.set_hexpand(True)
        self._slider.set_size_request(300, -1)
        self._slider.set_sensitive(False)
        self._slider.connect("value-changed", self._on_slider_changed)
        self.append(self._slider)

        self._speed_index = 0
        self._speed_button = Gtk.Button(label=f"{SPEED_OPTIONS[0]}x")
        self._speed_button.add_css_class("speed-button")
        self._speed_button.set_tooltip_text(_("Playback speed"))
        self._speed_button.connect("clicked", self._on_speed_clicked)
        self.append(self._speed_button)

        self._playing = False
        self._timer_id: Optional[int] = None
        self._canvas = None

        self.connect("destroy", self._on_destroy)

    def _on_destroy(self, widget):
        self._stop_playback()
        self._canvas = None

    def set_canvas(self, canvas):
        """Connect this overlay to a Canvas3D instance."""
        self._canvas = canvas

    def update_ops_range(self, command_count: int):
        """Update slider range for the given number of commands."""
        if command_count > 0:
            self._slider.set_range(0, command_count - 1)
            self._slider.set_value(0)
            self._slider.set_sensitive(True)
            self._play_button.set_sensitive(True)
        else:
            self._slider.set_range(0, 1)
            self._slider.set_value(0)
            self._slider.set_sensitive(False)
            self._play_button.set_sensitive(False)

    def get_slider_index(self) -> int:
        return int(self._slider.get_value())

    def _on_slider_changed(self, slider):
        if self._canvas and self._canvas._op_player:
            index = int(slider.get_value())
            logger.debug(
                f"[PLAYBACK] slider_changed: value={index}, "
                f"current_idx={self._canvas._op_player.current_index}"
            )
            if self._canvas._op_player.current_index != index:
                self._canvas._op_player.seek(index)
                self._canvas.queue_render()

    def _on_play_clicked(self, button):
        if self._playing:
            self._stop_playback()
        else:
            self._start_playback()

    def _start_playback(self):
        if not self._canvas or not self._canvas._op_player:
            return
        ops = self._canvas._op_player.ops
        current = int(self._slider.get_value())
        max_idx = len(ops) - 1
        if max_idx >= 0 and current >= max_idx:
            self._slider.set_value(0)
        self._playing = True
        self._play_button.set_child(self._pause_icon)
        self._play_button.set_tooltip_text(_("Pause simulation"))
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
        self._timer_id = GLib.timeout_add(42, self._on_tick)

    def _stop_playback(self):
        self._playing = False
        self._play_button.set_child(self._play_icon)
        self._play_button.set_tooltip_text(_("Play simulation"))
        if self._timer_id is not None:
            GLib.source_remove(self._timer_id)
            self._timer_id = None

    def _on_tick(self) -> bool:
        if not self._playing:
            return False
        if not self._canvas or not self._canvas._op_player:
            self._stop_playback()
            return False
        if not self._canvas.get_realized():
            self._stop_playback()
            return False

        ops = self._canvas._op_player.ops
        current = int(self._slider.get_value())
        max_idx = len(ops) - 1
        speed = SPEED_OPTIONS[self._speed_index]
        next_val = current + speed

        if next_val >= max_idx:
            self._slider.set_value(max_idx)
            self._stop_playback()
            return False

        self._slider.set_value(next_val)
        return True

    def _on_speed_clicked(self, button):
        self._speed_index = (self._speed_index + 1) % len(SPEED_OPTIONS)
        button.set_label(f"{SPEED_OPTIONS[self._speed_index]}x")
