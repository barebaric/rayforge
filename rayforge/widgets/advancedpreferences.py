import logging
from gi.repository import Gtk, Adw  # type: ignore
from ..models.dialect import DIALECTS, get_dialect


logger = logging.getLogger(__name__)


class AdvancedPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Advanced"),
            icon_name="applications-engineering-symbolic",
            **kwargs,
        )
        self.machine = machine

        # Dialect selection
        dialect_group = Adw.PreferencesGroup(title=_("Dialect"))
        dialect_group.set_description(
            _("Select the G-code flavor your machine understands.")
        )
        self.add(dialect_group)

        self.dialect_names = list(DIALECTS.keys())
        dialect_store = Gtk.StringList.new(self.dialect_names)
        self.dialect_combo_row = Adw.ComboRow(
            title=_("G-Code Dialect"), model=dialect_store
        )
        try:
            selected_index = self.dialect_names.index(
                self.machine.dialect_name
            )
            self.dialect_combo_row.set_selected(selected_index)
        except ValueError:
            self.dialect_combo_row.set_selected(0)
            self.machine.set_dialect_name(self.dialect_names[0])

        self.dialect_combo_row.connect(
            "notify::selected", self.on_dialect_changed
        )
        dialect_group.add(self.dialect_combo_row)

        # Preamble and postscript sections
        self._create_preamble_group()
        self._create_postscript_group()

    def _create_preamble_group(self):
        """Create and add preamble-related widgets."""
        group = Adw.PreferencesGroup(title=_("Preamble"))
        self.add(group)

        # The switch to toggle the override
        self.preamble_override_switch = Adw.SwitchRow(
            title=_("Override Default Preamble"),
            subtitle=_("Use custom G-code instead of the dialect's default."),
            active=self.machine.use_custom_preamble,
        )
        group.add(self.preamble_override_switch)

        # A box to hold the editor, its visibility is controlled by the switch
        self.preamble_editor_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_bottom=6,
            visible=self.machine.use_custom_preamble,
        )
        group.add(self.preamble_editor_box)

        # The text editor itself
        self.preamble_entry = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            pixels_above_lines=2,
            pixels_below_lines=2,
            left_margin=6,
            right_margin=6,
        )
        self.preamble_entry.get_buffer().set_text(
            "\n".join(self.machine.preamble), -1
        )
        self.preamble_entry.get_buffer().connect(
            "changed", self.on_preamble_changed
        )

        scrolled_window = Gtk.ScrolledWindow(
            height_request=100,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            child=self.preamble_entry,
        )
        frame = Gtk.Frame(child=scrolled_window)
        frame.add_css_class("view")
        self.preamble_editor_box.append(frame)

        # The "Append" button
        append_button = Gtk.Button(
            label=_("Append Commands from Dialect Default"),
            halign=Gtk.Align.END,
        )
        append_button.connect("clicked", self.on_append_preamble_clicked)
        self.preamble_editor_box.append(append_button)

        # Connect the switch to the box's visibility
        self.preamble_override_switch.connect(
            "notify::active", self.on_preamble_override_toggled
        )

    def _create_postscript_group(self):
        """Create and add postscript-related widgets."""
        group = Adw.PreferencesGroup(title=_("Postscript"))
        self.add(group)

        self.postscript_override_switch = Adw.SwitchRow(
            title=_("Override Default Postscript"),
            subtitle=_("Use custom G-code instead of the dialect's default."),
            active=self.machine.use_custom_postscript,
        )
        group.add(self.postscript_override_switch)

        self.postscript_editor_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=6,
            margin_bottom=6,
            visible=self.machine.use_custom_postscript,
        )
        group.add(self.postscript_editor_box)

        self.postscript_entry = Gtk.TextView(
            wrap_mode=Gtk.WrapMode.WORD_CHAR,
            pixels_above_lines=2,
            pixels_below_lines=2,
            left_margin=6,
            right_margin=6,
        )
        self.postscript_entry.get_buffer().set_text(
            "\n".join(self.machine.postscript), -1
        )
        self.postscript_entry.get_buffer().connect(
            "changed", self.on_postscript_changed
        )

        scrolled_window = Gtk.ScrolledWindow(
            height_request=100,
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            child=self.postscript_entry,
        )
        frame = Gtk.Frame(child=scrolled_window)
        frame.add_css_class("view")
        self.postscript_editor_box.append(frame)

        append_button = Gtk.Button(
            label=_("Append Commands from Dialect Default"),
            halign=Gtk.Align.END,
        )
        append_button.connect("clicked", self.on_append_postscript_clicked)
        self.postscript_editor_box.append(append_button)

        self.postscript_override_switch.connect(
            "notify::active", self.on_postscript_override_toggled
        )

    def on_dialect_changed(self, combo_row, _):
        """Update the machine's dialect when the selection changes."""
        selected_index = combo_row.get_selected()
        new_dialect_name = self.dialect_names[selected_index]
        self.machine.set_dialect_name(new_dialect_name)

    def on_preamble_override_toggled(self, switch, _):
        """Show or hide the preamble editor box based on the switch state."""
        is_active = switch.get_active()
        self.machine.set_use_custom_preamble(is_active)
        self.preamble_editor_box.set_visible(is_active)

    def on_postscript_override_toggled(self, switch, _):
        """Show or hide the postscript editor box based on the switch state."""
        is_active = switch.get_active()
        self.machine.set_use_custom_postscript(is_active)
        self.postscript_editor_box.set_visible(is_active)

    def on_append_preamble_clicked(self, button):
        """Appends the dialect default to the current text."""
        try:
            dialect = get_dialect(self.machine.dialect_name)
            default_lines = dialect.default_preamble
            if not default_lines:
                return

            buffer = self.preamble_entry.get_buffer()
            current_text = buffer.get_text(
                buffer.get_start_iter(), buffer.get_end_iter(), True
            ).strip()

            new_text = "\n".join(default_lines)
            final_text = (
                (current_text + "\n" + new_text) if current_text else new_text
            )
            buffer.set_text(final_text, -1)
        except ValueError as e:
            logger.error(f"Error getting dialect: {e}")

    def on_append_postscript_clicked(self, button):
        """Appends the dialect default to the current text."""
        try:
            dialect = get_dialect(self.machine.dialect_name)
            default_lines = dialect.default_postscript
            if not default_lines:
                return

            buffer = self.postscript_entry.get_buffer()
            current_text = buffer.get_text(
                buffer.get_start_iter(), buffer.get_end_iter(), True
            ).strip()

            new_text = "\n".join(default_lines)
            final_text = (
                (current_text + "\n" + new_text) if current_text else new_text
            )
            buffer.set_text(final_text, -1)
        except ValueError as e:
            logger.error(f"Error getting dialect: {e}")

    def on_preamble_changed(self, buffer):
        """Update the machine's custom preamble when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(), buffer.get_end_iter(), True
        )
        self.machine.set_preamble(text.splitlines())

    def on_postscript_changed(self, buffer):
        """Update the machine's custom postscript when the text changes."""
        text = buffer.get_text(
            buffer.get_start_iter(), buffer.get_end_iter(), True
        )
        self.machine.set_postscript(text.splitlines())
