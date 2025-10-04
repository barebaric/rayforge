"""Preview playback controls overlay."""

from gi.repository import Gtk, GLib, GObject


class PreviewControls(Gtk.Box):
    """
    Control panel for preview playback with play/pause, slider, and
    progress display. Designed to overlay on top of the canvas.
    """

    __gsignals__ = {
        "step-changed": (GObject.SignalFlags.RUN_FIRST, None, (int,))
    }

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
        self.append(self.slider)

        # Button box
        button_box = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=6,
        )
        button_box.set_halign(Gtk.Align.CENTER)
        self.append(button_box)

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
        self.append(self.status_label)

        # Initialize slider range and labels
        self._update_slider_range()
        self._update_status_label()

    def _update_slider_range(self):
        """Updates the slider range based on the number of steps."""
        step_count = self.simulation_overlay.get_step_count()
        if step_count > 0:
            self.slider.set_range(0, step_count - 1)
            self.slider.set_value(0)
        else:
            self.slider.set_range(0, 0)
            self.slider.set_value(0)
        self._update_status_label()

    def _update_status_label(self):
        """Updates the status label with step count, speed and power."""
        current = int(self.slider.get_value())
        total = self.simulation_overlay.get_step_count()
        state = self.simulation_overlay.get_current_state()

        # Check if current step is a travel command
        is_travel = False
        if self.simulation_overlay.timeline.steps and 0 <= current < len(
            self.simulation_overlay.timeline.steps
        ):
            cmd, _, _ = self.simulation_overlay.timeline.steps[current]
            is_travel = cmd.is_travel_command()

        if state:
            speed = state.cut_speed if state.cut_speed is not None else 0.0
            power = state.power if state.power is not None else 0.0
            # Convert normalized power (0.0-1.0) to percentage (0-100%)
            power_percent = power * 100.0

            # Show power as 0% for travel moves (laser is off)
            if is_travel:
                power_display = "0%"
            else:
                power_display = f"{power_percent:.0f}%"

            self.status_label.set_markup(
                (
                    f"<small>Step: {current + 1}/{total}  |  "
                    f"Speed: {speed:.0f} mm/min  |  "
                    f"Power: {power_display}</small>"
                )
            )
        else:
            self.status_label.set_markup(
                f"<small>Step: {current + 1}/{total}  |  "
                f"Speed: -  |  Power: -</small>"
            )

    def _on_slider_changed(self, slider):
        """Handles slider value changes."""
        step = int(slider.get_value())
        self.simulation_overlay.set_step(step)
        self._update_status_label()
        self.emit("step-changed", step)

        # Trigger redraw of the canvas
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
        max_step = self.simulation_overlay.get_step_count() - 1
        new_value = min(max_step, current + 1)
        self.slider.set_value(new_value)

    def _on_go_to_start_clicked(self, button):
        """Handles go to start button clicks."""
        self._pause_playback()
        self.slider.set_value(0)

    def _on_go_to_end_clicked(self, button):
        """Handles go to end button clicks."""
        self._pause_playback()
        max_step = self.simulation_overlay.get_step_count() - 1
        self.slider.set_value(max_step)

    def _start_playback(self):
        """Starts automatic playback."""
        self.playing = True
        self.play_button.set_icon_name("media-playback-pause-symbolic")

        # Calculate step increment to complete in target duration
        fps = 24
        step_count = self.simulation_overlay.get_step_count()
        if step_count > 0:
            target_frames = self.target_duration_sec * fps
            self.step_increment = step_count / target_frames
        else:
            self.step_increment = 1.0

        # Start playback timer
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
        current = self.slider.get_value()  # Use float value
        max_step = self.simulation_overlay.get_step_count() - 1

        # Advance by step increment (can be fractional)
        next_value = current + self.step_increment

        if next_value >= max_step:
            if self.loop_enabled:
                # Loop back to the beginning
                self.slider.set_value(0)
                return True
            else:
                # Reached the end, set to final step and pause
                self.slider.set_value(max_step)
                self._pause_playback()
                return False

        # Advance slider by increment
        self.slider.set_value(next_value)
        return True  # Continue playback

    def reset(self):
        """Resets the controls to initial state."""
        self._pause_playback()
        self._update_slider_range()
