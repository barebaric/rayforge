import logging
from gi.repository import Gtk, Adw
from ..models.dialect import get_available_dialects, GcodeDialect
from ...context import get_context
from ...shared.ui.adwfix import get_spinrow_int
from .hook_list import HookList
from .macro_list import MacroListEditor
from .dialect_list import DialectListEditor


logger = logging.getLogger(__name__)


class AdvancedPreferencesPage(Adw.PreferencesPage):
    def __init__(self, machine, **kwargs):
        super().__init__(
            title=_("Advanced"),
            icon_name="applications-engineering-symbolic",
            **kwargs,
        )
        self.machine = machine
        self.dialect_mgr = get_context().dialect_mgr
        self.available_dialects: list[GcodeDialect] = []

        # Output settings (was Dialect)
        output_group = Adw.PreferencesGroup(title=_("Output"))
        output_group.set_description(
            _("Configure the G-code flavor and format for your machine.")
        )
        self.add(output_group)

        self.dialect_combo_row = Adw.ComboRow(title=_("G-Code Dialect"))
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
        self.dialect_mgr.dialects_changed.connect(
            self._on_available_dialects_changed
        )
        self.connect("destroy", self._on_destroy)

        # Initial population
        self._on_available_dialects_changed()

        # Add the HookList widget
        hook_list_group = HookList(machine=self.machine)
        self.add(hook_list_group)

        # Add the Macro Editor widget
        macro_editor_group = MacroListEditor(
            machine=self.machine,
            title=_("Macros"),
            description=_("Create and manage reusable G-code snippets."),
        )
        self.add(macro_editor_group)

        # Add the Dialect Editor widget
        dialect_editor_group = DialectListEditor(
            title=_("Dialect Management"),
            description=_(
                "Create and manage custom G-code dialect definitions."
            ),
        )
        self.add(dialect_editor_group)

    def _on_destroy(self, *args):
        self.dialect_mgr.dialects_changed.disconnect(
            self._on_available_dialects_changed
        )

    def _on_available_dialects_changed(self, sender=None, **kwargs):
        """Repopulate the dropdown when the list of dialects changes."""
        current_dialect_uid = self.machine.dialect_uid
        self.available_dialects = get_available_dialects()

        dialect_display_names = [d.label for d in self.available_dialects]
        dialect_store = Gtk.StringList.new(dialect_display_names)
        self.dialect_combo_row.set_model(dialect_store)

        try:
            dialect_uids = [d.uid for d in self.available_dialects]
            selected_index = dialect_uids.index(current_dialect_uid)
            self.dialect_combo_row.set_selected(selected_index)
        except (ValueError, AttributeError):
            if self.available_dialects:
                self.dialect_combo_row.set_selected(0)
            else:
                self.on_dialect_changed(self.dialect_combo_row, None)

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

        if selected_index < 0 or not self.available_dialects:
            self.dialect_combo_row.set_title(_("G-Code Dialect"))
            self.dialect_combo_row.set_subtitle(_("No dialects available."))
            return

        new_dialect = self.available_dialects[selected_index]
        self.dialect_combo_row.set_title(new_dialect.label)
        self.dialect_combo_row.set_subtitle(new_dialect.description)

        if self.machine.dialect_uid != new_dialect.uid:
            self.machine.set_dialect_uid(new_dialect.uid)

    def on_precision_changed(self, spinrow):
        """Update the machine's G-code precision when the value changes."""
        value = get_spinrow_int(spinrow)
        self.machine.set_gcode_precision(value)
