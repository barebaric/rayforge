from gi.repository import Gtk, Adw


class GCodePreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title="GCode",
            icon_name='applications-engineering-symbolic',
            **kwargs
        )
        self.machine = machine

        # Preamble
        preamble_group = Adw.PreferencesGroup(title="Preamble")
        self.add(preamble_group)
        self.preamble_entry = Gtk.TextView()
        self.preamble_entry.set_size_request(300, 50)
        self.preamble_entry.get_buffer().set_text(
            "\n".join(self.machine.preamble)
        )
        self.preamble_entry.get_buffer().connect(
            "changed", self.on_preamble_changed
        )
        preamble_group.add(self.preamble_entry)

        # Postscript
        postscript_group = Adw.PreferencesGroup(title="Postscript")
        self.add(postscript_group)
        self.postscript_entry = Gtk.TextView()
        self.postscript_entry.set_size_request(300, 50)
        self.postscript_entry.get_buffer().set_text(
            "\n".join(self.machine.postscript)
        )
        self.postscript_entry.get_buffer().connect(
            "changed", self.on_postscript_changed
        )
        postscript_group.add(self.postscript_entry)

        # Air Assist Settings
        air_assist_group = Adw.PreferencesGroup(title="Air Assist")
        self.add(air_assist_group)

        # Air Assist Enable
        self.air_assist_on_row = Adw.EntryRow()
        gcode = self.machine.air_assist_on or ""
        self.air_assist_on_row.set_title(
            "Air Assist Enable GCode (blank if unsupported)"
        )
        self.air_assist_on_row.set_text(gcode)
        self.air_assist_on_row.connect(
            "changed", self.on_air_assist_on_changed
        )
        air_assist_group.add(self.air_assist_on_row)

        # Air Assist Disable
        self.air_assist_off_row = Adw.EntryRow()
        gcode = self.machine.air_assist_off or ""
        self.air_assist_off_row.set_title(
            "Air Assist Disable GCode (blank if unsupported)"
        )
        self.air_assist_off_row.set_text(gcode)
        self.air_assist_off_row.connect(
            "changed", self.on_air_assist_off_changed
        )
        air_assist_group.add(self.air_assist_off_row)

    def on_preamble_changed(self, buffer):
        """Update the preamble when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(),
            buffer.get_end_iter(),
            True
        )
        self.machine.set_preamble(text.splitlines())

    def on_postscript_changed(self, buffer):
        """Update the postscript when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(),
            buffer.get_end_iter(),
            True
        )
        self.machine.set_postscript(text.splitlines())

    def on_air_assist_on_changed(self, entry):
        """Update the air assist enable GCode when the value changes."""
        text = entry.get_text().strip()
        self.machine.set_air_assist_on(text if text else None)

    def on_air_assist_off_changed(self, entry):
        """Update the air assist disable GCode when the value changes."""
        text = entry.get_text().strip()
        self.machine.set_air_assist_off(text if text else None)
