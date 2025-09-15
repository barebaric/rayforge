import logging
from gi.repository import Gtk, Adw
from ..models.dialect import get_available_dialects
from ...shared.util.adwfix import get_spinrow_int
from .hook_list import HookList
from .macro_list import MacroListEditor


logger = logging.getLogger(__name__)


class AdvancedPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Advanced"),
            icon_name="applications-engineering-symbolic",
            **kwargs,
        )
        self.machine = machine

        # Output settings (was Dialect)
        output_group = Adw.PreferencesGroup(title=_("Output"))
        output_group.set_description(
            _("Configure the G-code flavor and format for your machine.")
        )
        self.add(output_group)

        # Get all available dialects from the registry
        self.available_dialects = get_available_dialects()
        dialect_display_names = [d.label for d in self.available_dialects]
        dialect_store = Gtk.StringList.new(dialect_display_names)

        self.dialect_combo_row = Adw.ComboRow(
            title=_("G-Code Dialect"), model=dialect_store
        )
        self.dialect_combo_row.set_use_subtitle(True)
        output_group.add(self.dialect_combo_row)

        # Set up a custom factory to display both title and subtitle in the
        # dropdown
        factory = Gtk.SignalListItemFactory()
        factory.connect("setup", self._on_dialect_factory_setup)
        factory.connect("bind", self._on_dialect_factory_bind)
        self.dialect_combo_row.set_factory(factory)

        # G-code precision setting
        precision_adjustment = Gtk.Adjustment(
            lower=1, upper=8, step_increment=1, page_increment=1
        )
        self.precision_row = Adw.SpinRow(
            title=_("G-code Precision"),
            subtitle=_(
                "Number of decimal places for coordinates "
                "(e.g., 3 for mm, 6 for µm)."
            ),
            adjustment=precision_adjustment,
        )
        precision_adjustment.set_value(self.machine.gcode_precision)
        self.precision_row.connect("changed", self.on_precision_changed)
        output_group.add(self.precision_row)

        # Connect the signal BEFORE setting the initial selection.
        # This ensures the handler is called to set the initial title/subtitle.
        self.dialect_combo_row.connect(
            "notify::selected", self.on_dialect_changed
        )

        # Now, set the initial selection, which will trigger on_dialect
        # changed.
        try:
            dialect_names = [d.name for d in self.available_dialects]
            selected_index = dialect_names.index(self.machine.dialect_name)
            self.dialect_combo_row.set_selected(selected_index)
        except (ValueError, AttributeError):
            # Default to the first dialect if not set or invalid
            if self.available_dialects:
                self.dialect_combo_row.set_selected(0)
            else:
                # Manually trigger handler for empty state
                self.on_dialect_changed(self.dialect_combo_row, None)

        # Add the new HookList widget, replacing the old ScriptListEditor
        hook_list_group = HookList(machine=self.machine)
        self.add(hook_list_group)

        # Add the new Macro Editor widget
        macro_editor_group = MacroListEditor(
            machine=self.machine,
            title=_("Macros"),
            description=_("Create and manage reusable G-code snippets."),
        )
        self.add(macro_editor_group)

    def _on_dialect_factory_setup(self, factory, list_item):
        """Setup handler for the dialect dropdown factory."""
        row = Adw.ActionRow()
        list_item.set_child(row)

    def _on_dialect_factory_bind(self, factory, list_item):
        """Bind handler for the dialect dropdown factory."""
        index = list_item.get_position()
        dialect = self.available_dialects[index]
        row = list_item.get_child()
        row.set_title(dialect.label)
        row.set_subtitle(dialect.description)

    def on_dialect_changed(self, combo_row, _param):
        """Update the ComboRow display and the machine's dialect."""
        selected_index = combo_row.get_selected()

        if selected_index < 0:
            self.dialect_combo_row.set_title(_("G-Code Dialect"))
            self.dialect_combo_row.set_subtitle(_("No dialects available."))
            return

        new_dialect = self.available_dialects[selected_index]

        # Update the row's own title and subtitle to reflect the selection
        self.dialect_combo_row.set_title(new_dialect.label)
        self.dialect_combo_row.set_subtitle(new_dialect.description)

        # Update the machine model if the dialect has actually changed
        if self.machine.dialect_name != new_dialect.name:
            self.machine.set_dialect_name(new_dialect.name)

    def on_precision_changed(self, spinrow):
        """Update the machine's G-code precision when the value changes."""
        value = get_spinrow_int(spinrow)
        self.machine.set_gcode_precision(value)
