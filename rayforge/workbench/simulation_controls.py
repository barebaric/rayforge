"""Preview playback controls overlay."""

import json
from typing import Optional
import numpy as np
from gi.repository import Gtk, GLib
from blinker import Signal
from ..core.ops import ScanLinePowerCommand
from ..pipeline.encoder.gcode import MachineCodeOpMap
from ..icons import get_icon
from ..shared.util.gtk import apply_css
from ..shared.ui.formatter import format_value


class PreviewControls(Gtk.Box):
    """
    Control panel for preview playback with play/pause, slider, and
    progress display. Designed to overlay on top of the canvas.
    """

    def __init__(
        self,
        simulation_overlay,
        target_duration_sec: float = 5.0,
        **kwargs,
    ):
        super().__init__(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=0,
            **kwargs,
        )
        self.simulation_overlay = simulation_overlay
        self.step_changed = Signal()
        self.close_requested = Signal()
        self.op_map: Optional[MachineCodeOpMap] = None
        self.num_gcode_lines = 0

        self.playing = False
        self.playback_timeout_id = None
        self.loop_enabled = False
        # How many steps to advance per frame
        self.step_increment = 1.0
        # Target playback duration
        self.target_duration_sec = target_duration_sec

        # Add CSS class for styling
        self.add_css_class("preview-controls")
        self.set_valign(Gtk.Align.END)
        self.set_halign(Gtk.Align.CENTER)
        self.set_margin_bottom(20)
        self.set_margin_start(20)
        self.set_margin_end(20)

        # Create a styled container box
        self.add_css_class("card")

        # Apply CSS for close button styling
        apply_css("""
            .preview-controls .close-button {
                min-width: 24px;
                min-height: 24px;
                padding: 0;
                margin: 0;
            }

            .preview-controls .close-button:hover {
                background-color: rgba(255, 255, 255, 0.1);
            }
        """)

        # Create a top-level box to contain the close button and main content
        self.top_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.append(self.top_box)

        # Create a header box for the close button
        header_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        header_box.set_halign(Gtk.Align.END)

        # Create the close button
        self.close_button = Gtk.Button(child=get_icon("close-symbolic"))
        self.close_button.set_tooltip_text("Close")
        self.close_button.add_css_class("flat")
        self.close_button.add_css_class("close-button")
        self.close_button.set_margin_top(4)
        self.close_button.set_margin_end(4)
        self.close_button.connect("clicked", self._on_close_clicked)

        header_box.append(self.close_button)
        self.top_box.append(header_box)

        # Create a container for the main content
        self.content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.top_box.append(self.content_box)

        # Slider for scrubbing
        self.slider = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL,
            0,
            100,
            1,
        )
        self.slider.set_draw_value(False)
        self.slider.set_hexpand(True)
        self.slider.set_size_request(600, -1)
        self.slider.set_margin_top(6)
        self.slider.set_margin_start(12)
        self.slider.set_margin_end(12)
        self.slider.connect("value-changed", self._on_slider_changed)
        self.content_box.append(self.slider)

        # Button box
        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
        )
        button_box.set_halign(Gtk.Align.CENTER)
        self.content_box.append(button_box)

        # Go to Start button
        self.go_to_start_button = Gtk.Button()
        self.go_to_start_button.set_icon_name("media-skip-backward-symbolic")
        self.go_to_start_button.set_tooltip_text("Go to Start")
        self.go_to_start_button.connect(
            "clicked", self._on_go_to_start_clicked
        )
        button_box.append(self.go_to_start_button)

        # Step Back button
        self.step_back_button = Gtk.Button()
        self.step_back_button.set_icon_name("media-seek-backward-symbolic")
        self.step_back_button.set_tooltip_text("Step Back")
        self.step_back_button.connect("clicked", self._on_step_back_clicked)
        button_box.append(self.step_back_button)

        # Play/Pause button
        self.play_button = Gtk.Button()
        self.play_button.set_icon_name("media-playback-start-symbolic")
        self.play_button.set_tooltip_text("Play/Pause")
        self.play_button.connect("clicked", self._on_play_pause_clicked)
        button_box.append(self.play_button)

        # Step Forward button
        self.step_forward_button = Gtk.Button()
        self.step_forward_button.set_icon_name("media-seek-forward-symbolic")
        self.step_forward_button.set_tooltip_text("Step Forward")
        self.step_forward_button.connect(
            "clicked", self._on_step_forward_clicked
        )
        button_box.append(self.step_forward_button)

        # Go to End button
        self.go_to_end_button = Gtk.Button()
        self.go_to_end_button.set_icon_name("media-skip-forward-symbolic")
        self.go_to_end_button.set_tooltip_text("Go to End")
        self.go_to_end_button.connect("clicked", self._on_go_to_end_clicked)
        button_box.append(self.go_to_end_button)

        # Status label with step count, speed and power
        self.status_label = Gtk.Label()
        self.status_label.set_margin_top(6)
        self.status_label.set_margin_bottom(6)
        self.content_box.append(self.status_label)

        # Initialize slider range and labels
        self._update_slider_range()
        self._update_status_label()

    def set_playback_position(self, line_number: int):
        """Programmatically sets the slider and simulation to a G-code line."""
        # Setting the slider's value will automatically trigger the
        # '_on_slider_changed' signal, which updates the simulation.
        self.slider.set_value(line_number)

    def set_playback_source(
        self,
        machine_code_bytes: Optional[np.ndarray],
        op_map_bytes: Optional[np.ndarray],
    ):
        """Sets the source data for driving the playback timeline."""
        if machine_code_bytes is not None:
            gcode_str = machine_code_bytes.tobytes().decode("utf-8")
            self.num_gcode_lines = gcode_str.count("\n")
        else:
            self.num_gcode_lines = 0

        if op_map_bytes is not None:
            map_str = op_map_bytes.tobytes().decode("utf-8")
            map_dict = json.loads(map_str)
            self.op_map = MachineCodeOpMap(
                op_to_machine_code={
                    int(k): v
                    for k, v in map_dict["op_to_machine_code"].items()
                },
                machine_code_to_op={
                    int(k): v
                    for k, v in map_dict["machine_code_to_op"].items()
                },
            )
        else:
            self.op_map = None
        self.reset()

    def _update_slider_range(self):
        """Updates the slider range based on the number of G-code lines."""
        if self.num_gcode_lines > 0:
            self.slider.set_range(0, self.num_gcode_lines - 1)
            self.slider.set_value(0)
        else:
            self.slider.set_range(0, 0)
            self.slider.set_value(0)
        self._update_status_label()

    def _update_status_label(self):
        """Updates the status label with step count, speed and power."""
        gcode_line_idx = int(self.slider.get_value())
        total = self.num_gcode_lines

        if self.op_map and gcode_line_idx in self.op_map.machine_code_to_op:
            op_index = self.op_map.machine_code_to_op[gcode_line_idx]
            timeline = self.simulation_overlay.timeline
            if timeline.steps and 0 <= op_index < len(timeline.steps):
                cmd, state, __ = timeline.steps[op_index]
                speed = state.cut_speed or 0.0
                power_percent = (state.power or 0.0) * 100.0

                if cmd.is_travel_command():
                    power_display = "0%"
                elif isinstance(cmd, ScanLinePowerCommand):
                    power_display = f"~{power_percent:.0f}% (Raster)"
                else:
                    power_display = f"{power_percent:.0f}%"

                speed_display = format_value(speed, "speed")
                self.status_label.set_markup(
                    (
                        f"<small>{_('Line')}: {gcode_line_idx + 1}/{total} | "
                        f"{_('Speed')}: {speed_display} | "
                        f"{_('Power')}: {power_display}</small>"
                    )
                )
                return

        self.status_label.set_markup(
            f"<small>{_('Line')}: {gcode_line_idx + 1}/{total} | "
            f"{_('Speed')}: - | {_('Power')}: -</small>"
        )

    def _on_slider_changed(self, slider):
        """Handles slider value changes."""
        gcode_line_idx = int(slider.get_value())
        timeline = self.simulation_overlay.timeline
        op_index = -1
        timeline_index = -1

        if self.op_map and gcode_line_idx in self.op_map.machine_code_to_op:
            op_index = self.op_map.machine_code_to_op[gcode_line_idx]
            if op_index in timeline.op_to_timeline_index:
                timeline_index = timeline.op_to_timeline_index[op_index]

        if timeline_index == -1:
            # If no op corresponds (e.g., comment), find the last valid one
            for i in range(gcode_line_idx, -1, -1):
                if self.op_map and i in self.op_map.machine_code_to_op:
                    op_idx = self.op_map.machine_code_to_op[i]
                    if op_idx in timeline.op_to_timeline_index:
                        timeline_index = timeline.op_to_timeline_index[op_idx]
                        break

        self.simulation_overlay.set_step(timeline_index)
        self._update_status_label()
        self.step_changed.send(self, line_number=gcode_line_idx)

        if self.simulation_overlay.canvas:
            self.simulation_overlay.canvas.queue_draw()

    def _on_play_pause_clicked(self, button):
        """Handles play/pause button clicks."""
        if self.playing:
            self._pause_playback()
        else:
            self._start_playback()

    def _on_step_back_clicked(self, button):
        """Handles step back button clicks."""
        self._pause_playback()
        current = int(self.slider.get_value())
        new_value = max(0, current - 1)
        self.slider.set_value(new_value)

    def _on_step_forward_clicked(self, button):
        """Handles step forward button clicks."""
        self._pause_playback()
        current = int(self.slider.get_value())
        max_step = self.num_gcode_lines - 1
        new_value = min(max_step, current + 1)
        self.slider.set_value(new_value)

    def _on_go_to_start_clicked(self, button):
        """Handles go to start button clicks."""
        self._pause_playback()
        self.slider.set_value(0)

    def _on_go_to_end_clicked(self, button):
        """Handles go to end button clicks."""
        self._pause_playback()
        max_step = self.num_gcode_lines - 1
        self.slider.set_value(max_step)

    def _start_playback(self):
        """Starts automatic playback."""
        self.playing = True
        self.play_button.set_icon_name("media-playback-pause-symbolic")

        fps = 24
        if self.num_gcode_lines > 0:
            target_frames = self.target_duration_sec * fps
            self.step_increment = self.num_gcode_lines / target_frames
        else:
            self.step_increment = 1.0

        ms_per_frame = int(1000 / fps)
        self.playback_timeout_id = GLib.timeout_add(
            ms_per_frame, self._advance_step
        )

    def _pause_playback(self):
        """Pauses playback."""
        self.playing = False
        self.play_button.set_icon_name("media-playback-start-symbolic")

        if self.playback_timeout_id:
            GLib.source_remove(self.playback_timeout_id)
            self.playback_timeout_id = None

    def _advance_step(self):
        """Advances to the next step during playback."""
        current = self.slider.get_value()
        max_step = self.num_gcode_lines - 1
        next_value = current + self.step_increment

        if next_value >= max_step:
            if self.loop_enabled:
                self.slider.set_value(0)
                return True
            else:
                self.slider.set_value(max_step)
                self._pause_playback()
                return False

        self.slider.set_value(next_value)
        return True

    def reset(self):
        """Resets the controls to initial state."""
        self._pause_playback()
        self._update_slider_range()

    def _on_close_clicked(self, button):
        """Handles close button clicks."""
        self.close_requested.send(self)
