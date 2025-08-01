from gi.repository import Adw, Gtk  # type: ignore
from ..config import config


class GeneralPreferencesPage(Adw.PreferencesPage):
    """
    Preferences page for general application settings.
    This is distinct from the machine-specific general settings.
    """
    # Map for converting between UI index and config string
    THEME_MAP = ["system", "light", "dark"]
    THEME_LABELS = ["System", "Light", "Dark"]

    def __init__(self, **kwargs):
        super().__init__(
            title=_("General"),
            icon_name="preferences-system-symbolic",
            **kwargs,
        )

        app_settings_group = Adw.PreferencesGroup(
            title=_("Appearance"),
            description=_(
                "Settings related to the application's look and feel."
            ),
        )
        self.add(app_settings_group)

        self.theme_row = Adw.ComboRow(
            title=_("Theme"),
            model=Gtk.StringList.new(self.THEME_LABELS),
        )

        try:
            selected_index = self.THEME_MAP.index(config.theme)
        except ValueError:
            selected_index = 0
        self.theme_row.set_selected(selected_index)

        self.theme_row.connect("notify::selected", self.on_theme_changed)
        app_settings_group.add(self.theme_row)

    def on_theme_changed(self, combo_row, _):
        """Called when the user selects a new theme."""
        selected_index = combo_row.get_selected()
        theme_string = self.THEME_MAP[selected_index]
        config.set_theme(theme_string)
