import logging
from typing import cast
from gi.repository import Gtk, Adw
from ..settings.preferences_group import PreferencesGroupWithButton
from ...machine.models.machine import Machine
from ...machine.models.machine_hours import ResettableCounter
from ..icons import get_icon


logger = logging.getLogger(__name__)


class CounterRow(Gtk.Box):
    """A widget representing a single counter in a ListBox."""

    def __init__(self, machine: Machine, counter: ResettableCounter):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.machine = machine
        self.counter = counter
        self._setup_ui()

    def _setup_ui(self):
        """Builds the user interface for the row."""
        # Match margins exactly to MacroRow
        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        # Icon
        icon = get_icon("hourglass-symbolic")
        self.append(icon)

        # Title and Time Container
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        info_box.set_hexpand(True)  # This pushes the buttons to the right
        info_box.set_valign(Gtk.Align.CENTER)
        self.append(info_box)

        # Title
        title_label = Gtk.Label(
            label=self.counter.name,
            halign=Gtk.Align.START,
            xalign=0,
        )
        info_box.append(title_label)

        # Time Subtitle
        hours = int(self.counter.value)
        minutes = int((self.counter.value - hours) * 60)
        subtitle_text = f"{hours}h {minutes}m"

        # If there is a notification threshold, show it as a limit
        # (e.g., "10h / 100h")
        if self.counter.notify_at is not None:
            subtitle_text += f" / {self.counter.notify_at:g}h"

        value_label = Gtk.Label(
            label=subtitle_text,
            halign=Gtk.Align.START,
            xalign=0,
        )
        value_label.add_css_class("dim-label")
        info_box.append(value_label)

        # Suffix area for buttons
        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        # Reset button
        reset_button = Gtk.Button(child=get_icon("refresh-symbolic"))
        reset_button.set_tooltip_text(_("Reset Counter"))
        reset_button.add_css_class("flat")
        reset_button.connect("clicked", self._on_reset_clicked)
        suffix_box.append(reset_button)

        # Edit button
        edit_button = Gtk.Button(child=get_icon("edit-symbolic"))
        edit_button.set_tooltip_text(_("Edit Counter"))
        edit_button.add_css_class("flat")
        edit_button.connect("clicked", self._on_edit_clicked)
        suffix_box.append(edit_button)

        # Remove button
        remove_button = Gtk.Button(child=get_icon("delete-symbolic"))
        remove_button.set_tooltip_text(_("Remove Counter"))
        remove_button.add_css_class("flat")
        remove_button.connect("clicked", self._on_remove_clicked)
        suffix_box.append(remove_button)

    def _on_reset_clicked(self, button: Gtk.Button):
        """Ask for confirmation, then reset."""
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=_("Reset Counter?"),
            body=_("This will reset the accumulated hours to zero."),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("reset", _("Reset"))
        dialog.set_response_appearance(
            "reset", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.connect("response", self._on_reset_response)
        dialog.present()

    def _on_reset_response(self, dialog: Adw.MessageDialog, response: str):
        """Handle reset confirmation response."""
        if response == "reset":
            self.machine.machine_hours.reset_counter(self.counter.uid)
            logger.info(f"Reset counter: {self.counter.uid}")

    def _on_edit_clicked(self, button: Gtk.Button):
        """Handle edit button click."""
        parent_window = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        dialog = CounterEditDialog(parent_window, self.machine, self.counter)
        dialog.connect("close-request", self._on_edit_dialog_closed)
        dialog.present()

    def _on_edit_dialog_closed(self, dialog):
        """Handle edit dialog closure."""
        if dialog.saved:
            self.machine.machine_hours.update_counter(self.counter)
            logger.info(f"Edited counter: {self.counter.uid}")

    def _on_remove_clicked(self, button: Gtk.Button):
        """Ask for confirmation, then remove."""
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=_("Remove Counter?"),
            body=_(
                "Are you sure you want to remove this counter? This action "
                "cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("remove", _("Remove"))
        dialog.set_response_appearance(
            "remove", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.connect("response", self._on_remove_response)
        dialog.present()

    def _on_remove_response(self, dialog: Adw.MessageDialog, response: str):
        """Handle remove confirmation response."""
        if response == "remove":
            self.machine.machine_hours.remove_counter(self.counter.uid)
            logger.info(f"Removed counter: {self.counter.uid}")


class CounterListEditor(PreferencesGroupWithButton):
    """
    An Adwaita widget for displaying and managing a list of
    maintenance counters.
    """

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(button_label=_("Add Counter"), **kwargs)
        self.machine = machine
        self._setup_ui()

        # Only listen to machine_hours.changed.
        self.machine.machine_hours.changed.connect(
            self._on_machine_hours_changed
        )

        # Initial population
        self._on_machine_hours_changed(self.machine.machine_hours)

    def _setup_ui(self):
        """Configures the widget and its placeholder."""
        placeholder = Gtk.Label(
            label=_("No counters configured"),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)

    def _on_machine_hours_changed(self, sender, **kwargs):
        """Callback to rebuild list when machine hours change."""
        self._update_ui()

    def _update_ui(self):
        """Update the list of counters."""
        sorted_counters = sorted(
            self.machine.machine_hours.counters.values(),
            key=lambda c: c.name,
        )
        self.set_items(sorted_counters)

    def create_row_widget(self, item: ResettableCounter) -> Gtk.Widget:
        """Creates a CounterRow for the given counter item."""
        return CounterRow(self.machine, item)

    def _on_add_clicked(self, button: Gtk.Button):
        """Handle 'Add Counter' button click."""
        parent = cast(Gtk.Window, self.get_ancestor(Gtk.Window))
        new_counter = ResettableCounter(name=_("New Counter"))
        dialog = CounterEditDialog(parent, self.machine, new_counter)
        dialog.connect(
            "close-request", self._on_new_counter_dialog_closed, new_counter
        )
        dialog.present()

    def _on_new_counter_dialog_closed(
        self, dialog, new_counter: ResettableCounter
    ):
        """Handle new counter dialog closure."""
        if dialog.saved:
            self.machine.machine_hours.add_counter(new_counter)
            logger.info(f"Added new counter: {new_counter.name}")


class CounterEditDialog(Adw.Window):
    """Dialog for editing counter settings."""

    def __init__(
        self,
        transient_for: Gtk.Window,
        machine: Machine,
        counter: ResettableCounter,
    ):
        super().__init__(
            title=_("Edit Counter"),
            transient_for=transient_for,
            modal=True,
            default_width=450,
            default_height=400,
        )
        self.machine = machine
        self.counter = counter
        self.saved = False
        self._setup_ui()

    def _setup_ui(self):
        """Builds the dialog UI using Adw.ToolbarView."""
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        # Header Bar
        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        close_button = Gtk.Button(label=_("Cancel"))
        close_button.connect("clicked", self._on_close_clicked)
        header_bar.pack_start(close_button)

        save_button = Gtk.Button(label=_("Save"))
        save_button.add_css_class("suggested-action")
        save_button.connect("clicked", self._on_save_clicked)
        header_bar.pack_end(save_button)

        # Content - Use PreferencesPage for proper background styling
        page = Adw.PreferencesPage()
        toolbar_view.set_content(page)

        # Main Group
        group = Adw.PreferencesGroup()
        page.add(group)

        # Name entry
        name_row = Adw.EntryRow(title=_("Name"))
        name_row.set_text(self.counter.name)
        group.add(name_row)
        self._name_row = name_row

        # Notification interval
        notify_adjustment = Gtk.Adjustment(
            value=0,
            lower=0,
            upper=100000,
            step_increment=0.1,
            page_increment=1,
        )
        notify_row = Adw.SpinRow(
            title=_("Notification Interval"),
            subtitle=_(
                "Show notification when counter reaches this value (hours). "
                "Set to 0 to disable."
            ),
            adjustment=notify_adjustment,
            digits=1,
        )
        if self.counter.notify_at is not None:
            notify_adjustment.set_value(self.counter.notify_at)
        group.add(notify_row)
        self._notify_row = notify_row

    def _on_save_clicked(self, button: Gtk.Button):
        """Handle save button click by applying UI values to the model."""
        self.saved = True

        # Apply changes to the counter object
        self.counter.name = self._name_row.get_text()

        notify_val = self._notify_row.get_value()
        self.counter.notify_at = notify_val if notify_val > 0 else None

        self.close()

    def _on_close_clicked(self, button: Gtk.Button):
        """Handle close button click."""
        self.close()


class MachineHoursPage(Adw.PreferencesPage):
    """
    A preferences page for viewing and managing machine hours.
    """

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Machine Hours"),
            **kwargs,
        )
        self.machine = machine

        # Group for Total Hours
        total_group = Adw.PreferencesGroup(title=_("Total Hours"))
        self.add(total_group)

        self.total_hours_row = Adw.ActionRow(
            title=_("Total Operating Hours"),
            subtitle=_("Cumulative machine operating time"),
            activatable=False,
        )
        self.total_hours_row.add_prefix(get_icon("hourglass-symbolic"))

        # Add Reset Button for Total Hours
        reset_total_btn = Gtk.Button(child=get_icon("refresh-symbolic"))
        reset_total_btn.set_tooltip_text(_("Reset Total Hours"))
        reset_total_btn.add_css_class("flat")
        reset_total_btn.set_valign(Gtk.Align.CENTER)
        reset_total_btn.connect("clicked", self._on_reset_total_clicked)

        self.total_hours_row.add_suffix(reset_total_btn)
        total_group.add(self.total_hours_row)

        # Group for Resettable Counters
        self.counters_group = CounterListEditor(
            machine=machine, title=_("Maintenance Counters")
        )
        self.counters_group.set_description(
            _(
                "Track maintenance intervals with resettable "
                "counters. Use for laser tubes, lubrication, etc."
            )
        )
        self.add(self.counters_group)

        # Connect signals
        self.machine.changed.connect(self._on_machine_changed)
        self.machine.machine_hours.changed.connect(
            self._on_machine_hours_changed
        )
        self.connect("destroy", self._on_destroy)

        # Initial UI update
        self._update_ui()

    def _on_destroy(self, *args):
        """Disconnect signals to prevent memory leaks."""
        self.machine.changed.disconnect(self._on_machine_changed)
        self.machine.machine_hours.changed.disconnect(
            self._on_machine_hours_changed
        )

    def _on_machine_changed(self, sender, **kwargs):
        """Handle machine configuration changes."""
        self._update_ui()

    def _on_machine_hours_changed(self, sender, **kwargs):
        """Handle machine hours changes."""
        self._update_ui()

    def _update_ui(self):
        """Update UI with current machine hours data."""
        total_hours = self.machine.machine_hours.total_hours
        hours = int(total_hours)
        minutes = int((total_hours - hours) * 60)
        self.total_hours_row.set_subtitle(
            _("{hours}h {minutes}m total").format(hours=hours, minutes=minutes)
        )

    def _on_reset_total_clicked(self, button: Gtk.Button):
        """Ask for confirmation, then reset total hours."""
        dialog = Adw.MessageDialog(
            transient_for=cast(Gtk.Window, self.get_root()),
            heading=_("Reset Total Hours?"),
            body=_(
                "This will reset the total cumulative operating hours to "
                "zero. Maintenance counters will not be affected."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("reset", _("Reset"))
        dialog.set_response_appearance(
            "reset", Adw.ResponseAppearance.DESTRUCTIVE
        )
        dialog.connect("response", self._on_reset_total_response)
        dialog.present()

    def _on_reset_total_response(
        self, dialog: Adw.MessageDialog, response: str
    ):
        """Handle reset confirmation response for total hours."""
        if response == "reset":
            self.machine.machine_hours.reset_total_hours()
            logger.info("Reset total machine hours")
