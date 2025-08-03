import logging
from gi.repository import Gtk, Adw, GObject  # type: ignore

logger = logging.getLogger(__name__)


class FirmwareSettingsWidget(Gtk.Box):
    """
    A widget that ONLY displays a list of device settings. The parent
    is responsible for layout, warnings, and actions.
    """

    __gsignals__ = {
        "setting-apply": (
            GObject.SignalFlags.RUN_FIRST,
            None,
            (Gtk.Entry, str),
        ),
    }

    def __init__(self, machine, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.machine = machine

        self.list_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)

        self.scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            child=self.list_container,
        )
        # --- THE FINAL LAYOUT FIX ---
        # This frame must expand to fill the space its parent gives it.
        main_content_frame = Gtk.Frame(
            child=self.scrolled_window, vexpand=True
        )
        main_content_frame.add_css_class("view")
        self.append(main_content_frame)
        self.populate_list()

    def _create_setting_row(self, key, value, description):
        """Helper to create a single Adw.ActionRow for a setting."""
        row = Adw.ActionRow(title=f"${key}", subtitle=description)
        entry = Gtk.Entry(width_chars=10, text=str(value))
        apply_button = Gtk.Button(
            icon_name="object-select-symbolic", tooltip_text=_("Apply Change")
        )
        apply_button.add_css_class("flat")

        def on_apply(*args):
            # Emit the signal for the parent to handle
            self.emit("setting-apply", entry, key)

        entry.connect("activate", on_apply)
        apply_button.connect("clicked", on_apply)

        suffix_box = Gtk.Box(spacing=6, margin_top=6, margin_bottom=6)
        suffix_box.append(entry)
        suffix_box.append(apply_button)
        row.add_suffix(suffix_box)
        return row

    def populate_list(self):
        """
        Populates the list box with the current settings from the machine
        model.
        """
        while child := self.list_container.get_first_child():
            self.list_container.remove(child)

        new_settings = self.machine.firmware_settings
        definitions = self.machine.get_setting_definitions()

        if new_settings:
            try:
                sorted_keys = sorted(new_settings.keys(), key=int)
            except (ValueError, TypeError):
                sorted_keys = sorted(new_settings.keys())

            for key in sorted_keys:
                value = new_settings[key]
                desc = definitions.get(str(key), "Unknown setting")
                row = self._create_setting_row(key, value, desc)
                self.list_container.append(row)
