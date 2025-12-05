from gi.repository import Adw, Gtk
from ...context import get_context
from ...shared.units.definitions import (
    get_units_for_quantity,
    get_base_unit_for_quantity,
)


class GeneralPreferencesPage(Adw.PreferencesPage):
    """
    Preferences page for general application settings.
    This is distinct from the machine-specific general settings.
    """

    # Map for converting between UI index and config string
    THEME_MAP = ["system", "light", "dark"]
    THEME_LABELS = [_("System"), _("Light"), _("Dark")]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(_("General"))
        self.set_icon_name("ui-settings-symbolic")

        app_settings_group = Adw.PreferencesGroup()
        app_settings_group.set_title(_("Appearance"))
        app_settings_group.set_description(
            _("Settings related to the application's look and feel.")
        )
        self.add(app_settings_group)

        self.theme_row = Adw.ComboRow(
            model=Gtk.StringList.new(self.THEME_LABELS)
        )
        self.theme_row.set_title(_("Theme"))

        config = get_context().config
        try:
            selected_index = self.THEME_MAP.index(config.theme)
        except ValueError:
            selected_index = 0
        self.theme_row.set_selected(selected_index)

        self.theme_row.connect("notify::selected", self.on_theme_changed)
        app_settings_group.add(self.theme_row)

        # Units Preferences
        units_group = Adw.PreferencesGroup()
        units_group.set_title(_("Units"))
        units_group.set_description(
            _(
                "Set the display units for various values throughout "
                "the application."
            )
        )
        self.add(units_group)

        # Length Unit Selector
        self.length_units = get_units_for_quantity("length")
        length_unit_labels = [u.label for u in self.length_units]
        self.length_unit_row = Adw.ComboRow(
            title=_("Length"),
            model=Gtk.StringList.new(length_unit_labels),
        )
        # Find and set the initial selection
        try:
            base_length_unit = get_base_unit_for_quantity("length")
            current_unit_name = config.unit_preferences.get(
                "length", base_length_unit.name if base_length_unit else None
            )

            if not current_unit_name:
                raise ValueError("No length unit could be determined")

            unit_names = [u.name for u in self.length_units]
            selected_index = unit_names.index(current_unit_name)
        except (ValueError, AttributeError):
            selected_index = 0  # Default to the first unit
        self.length_unit_row.set_selected(selected_index)

        self.length_unit_row.connect(
            "notify::selected", self.on_length_unit_changed
        )
        units_group.add(self.length_unit_row)

        # Speed Unit Selector
        self.speed_units = get_units_for_quantity("speed")
        speed_unit_labels = [u.label for u in self.speed_units]
        self.speed_unit_row = Adw.ComboRow(
            title=_("Speed"),
            model=Gtk.StringList.new(speed_unit_labels),
        )
        # Find and set the initial selection
        try:
            base_speed_unit = get_base_unit_for_quantity("speed")
            current_unit_name = config.unit_preferences.get(
                "speed", base_speed_unit.name if base_speed_unit else None
            )

            if not current_unit_name:
                raise ValueError("No speed unit could be determined")

            unit_names = [u.name for u in self.speed_units]
            selected_index = unit_names.index(current_unit_name)
        except (ValueError, AttributeError):
            selected_index = 0  # Default to the first unit
        self.speed_unit_row.set_selected(selected_index)

        self.speed_unit_row.connect(
            "notify::selected", self.on_speed_unit_changed
        )
        units_group.add(self.speed_unit_row)

        # Acceleration Unit Selector
        self.acceleration_units = get_units_for_quantity("acceleration")
        acceleration_unit_labels = [u.label for u in self.acceleration_units]
        self.acceleration_unit_row = Adw.ComboRow(
            title=_("Acceleration"),
            model=Gtk.StringList.new(acceleration_unit_labels),
        )
        # Find and set the initial selection
        try:
            base_acceleration_unit = get_base_unit_for_quantity("acceleration")
            current_unit_name = config.unit_preferences.get(
                "acceleration",
                base_acceleration_unit.name
                if base_acceleration_unit
                else None,
            )

            if not current_unit_name:
                raise ValueError("No acceleration unit could be determined")

            unit_names = [u.name for u in self.acceleration_units]
            selected_index = unit_names.index(current_unit_name)
        except (ValueError, AttributeError):
            selected_index = 0  # Default to the first unit
        self.acceleration_unit_row.set_selected(selected_index)

        self.acceleration_unit_row.connect(
            "notify::selected", self.on_acceleration_unit_changed
        )
        units_group.add(self.acceleration_unit_row)

    def on_theme_changed(self, combo_row, _):
        """Called when the user selects a new theme."""
        selected_index = combo_row.get_selected()
        theme_string = self.THEME_MAP[selected_index]
        get_context().config.set_theme(theme_string)

    def on_length_unit_changed(self, combo_row, _):
        """Called when the user selects a new length unit."""
        selected_index = combo_row.get_selected()
        if selected_index >= 0:
            selected_unit = self.length_units[selected_index]
            get_context().config.set_unit_preference(
                "length", selected_unit.name
            )

    def on_speed_unit_changed(self, combo_row, _):
        """Called when the user selects a new speed unit."""
        selected_index = combo_row.get_selected()
        if selected_index >= 0:
            selected_unit = self.speed_units[selected_index]
            get_context().config.set_unit_preference(
                "speed", selected_unit.name
            )

    def on_acceleration_unit_changed(self, combo_row, _):
        """Called when the user selects a new acceleration unit."""
        selected_index = combo_row.get_selected()
        if selected_index >= 0:
            selected_unit = self.acceleration_units[selected_index]
            get_context().config.set_unit_preference(
                "acceleration", selected_unit.name
            )
