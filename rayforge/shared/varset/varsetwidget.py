import re
import logging
from gi.repository import Gtk, Adw
from blinker import Signal
from ..util.adwfix import get_spinrow_int
from ...machine.transport.serial import SerialTransport
from ...machine.transport.validators import is_valid_hostname_or_ip
from .baudratevar import BaudrateVar
from .hostnamevar import HostnameVar
from .serialportvar import SerialPortVar
from .var import Var
from .varset import VarSet

logger = logging.getLogger(__name__)
NULL_CHOICE_LABEL = _("None Selected")


def natural_sort_key(s):
    return [
        int(text) if text.isdigit() else text.lower()
        for text in re.split("([0-9]+)", s)
    ]


class VarSetWidget(Adw.PreferencesGroup):
    """
    A self-contained Adwaita Preferences Group that populates itself with
    rows based on a VarSet. It supports two modes: immediate updates, or
    rows with explicit "Apply" buttons.
    """

    # Emits sender, key=...
    data_changed = Signal()

    def __init__(self, explicit_apply=False, **kwargs):
        super().__init__(**kwargs)
        self.explicit_apply = explicit_apply
        self.widget_map: dict[str, tuple[Adw.PreferencesRow, Var]] = {}
        self._created_rows = []
        self.data_changed = Signal()

    def clear_dynamic_rows(self):
        """Removes only the rows dynamically created by populate()."""
        for row in self._created_rows:
            self.remove(row)
        self._created_rows.clear()
        self.widget_map.clear()

    def populate(self, var_set: VarSet):
        """
        Clears previous dynamic rows and builds new ones from a VarSet.
        Any static rows added manually are preserved.
        """
        self.clear_dynamic_rows()
        for var in var_set:
            row = self._create_row_for_var(var)
            if row:
                self.add(row)
                self._created_rows.append(row)
                self.widget_map[var.key] = (row, var)

    def get_values(self) -> dict[str, object]:
        """Reads all current values from the UI widgets."""
        values = {}
        for key, (row, var) in self.widget_map.items():
            value = None
            if isinstance(row, Adw.EntryRow):
                value = row.get_text()
            elif isinstance(row, Adw.SwitchRow):
                value = row.get_active()
            elif isinstance(row, Adw.SpinRow):
                value = (
                    get_spinrow_int(row)
                    if var.var_type is int
                    else row.get_value()
                )
            elif isinstance(row, Adw.ComboRow):
                selected = row.get_selected_item()
                value_str = (
                    selected.get_string() if selected else ""  # type: ignore
                )
                value = "" if value_str == NULL_CHOICE_LABEL else value_str
            values[key] = value
        return values

    def _on_data_changed(self, key: str):
        self.data_changed.send(self, key=key)

    def _create_row_for_var(self, var: Var):
        if isinstance(var, SerialPortVar):
            return self._create_port_selection_row(var)
        if isinstance(var, BaudrateVar):
            return self._create_baud_rate_row(var)
        if isinstance(var, HostnameVar):
            return self._create_hostname_row(var)

        var_type = var.var_type
        if var_type is str:
            return self._create_string_row(var)
        elif var_type is bool:
            return self._create_boolean_row(var)
        elif var_type is int:
            return self._create_integer_row(var)
        elif var_type is float:
            return self._create_float_row(var)
        return None

    def _add_apply_button_if_needed(self, row, key):
        if not self.explicit_apply:
            return

        apply_button = Gtk.Button(
            icon_name="object-select-symbolic", tooltip_text=_("Apply Change")
        )
        apply_button.add_css_class("flat")
        apply_button.set_valign(Gtk.Align.CENTER)
        apply_button.connect("clicked", lambda b: self._on_data_changed(key))
        row.add_suffix(apply_button)

    def _create_hostname_row(self, var: HostnameVar):
        row = Adw.EntryRow(title=var.label)
        if var.description:
            row.set_tooltip_text(var.description)
        if var.value is not None:
            row.set_text(str(var.value))

        row.set_show_apply_button(True)

        def on_validate(entry_row):
            text = entry_row.get_text()
            if is_valid_hostname_or_ip(text):
                entry_row.remove_css_class("error")
            else:
                entry_row.add_css_class("error")

        row.connect("changed", on_validate)
        row.connect("apply", lambda r: self._on_data_changed(var.key))
        on_validate(row)
        return row

    def _create_string_row(self, var: Var[str]):
        row = Adw.EntryRow(title=var.label)
        if var.description:
            row.set_tooltip_text(var.description)
        if var.value is not None:
            row.set_text(str(var.value))
        row.connect("apply", lambda r: self._on_data_changed(var.key))
        if self.explicit_apply:
            self._add_apply_button_if_needed(row, var.key)
        return row

    def _create_boolean_row(self, var: Var[bool]):
        if self.explicit_apply:
            row = Adw.SwitchRow(
                title=var.label, subtitle=var.description or ""
            )
            if var.value is not None:
                row.set_active(bool(var.value))
            self._add_apply_button_if_needed(row, var.key)
        else:
            row = Adw.ActionRow(
                title=var.label, subtitle=var.description or ""
            )
            switch = Gtk.Switch(valign=Gtk.Align.CENTER)
            switch.set_active(var.value if var.value is not None else False)
            row.add_suffix(switch)
            row.set_activatable_widget(switch)
            switch.connect(
                "state-set", lambda s, a: self._on_data_changed(var.key)
            )
        return row

    def _create_integer_row(self, var: Var[int]):
        adj = Gtk.Adjustment(
            value=var.value if var.value is not None else 0,
            lower=-2147483647,
            upper=2147483647,
            step_increment=1,
        )
        row = Adw.SpinRow(
            title=var.label, subtitle=var.description or "", adjustment=adj
        )
        if not self.explicit_apply:
            row.connect("changed", lambda r: self._on_data_changed(var.key))
        else:
            self._add_apply_button_if_needed(row, var.key)
        return row

    def _create_float_row(self, var: Var[float]):
        adj = Gtk.Adjustment(
            value=var.value if var.value is not None else 0.0,
            lower=-1.0e12,
            upper=1.0e12,
            step_increment=0.1,
        )
        row = Adw.SpinRow(
            title=var.label,
            subtitle=var.description or "",
            adjustment=adj,
            digits=3,
        )
        if not self.explicit_apply:
            row.connect("changed", lambda r: self._on_data_changed(var.key))
        else:
            self._add_apply_button_if_needed(row, var.key)
        return row

    def _create_baud_rate_row(self, var: BaudrateVar):
        # Get the list of choices as strings
        choices_int = SerialTransport.list_baud_rates()
        choices_str = [str(rate) for rate in choices_int]

        # Create the ComboRow with a StringList model
        store = Gtk.StringList.new(choices_str)
        row = Adw.ComboRow(
            title=var.label, subtitle=var.description or "", model=store
        )

        # Set the initial selection based on the Var's value
        if var.value is not None:
            try:
                # Find the index of the current value in the choices
                current_value_str = str(var.value)
                if current_value_str in choices_str:
                    index = choices_str.index(current_value_str)
                    row.set_selected(index)
            except ValueError:
                logger.warning(
                    f"Baud rate '{var.value}' is not in the standard list."
                )

        if self.explicit_apply:
            self._add_apply_button_if_needed(row, var.key)
        else:
            row.connect("notify::selected-item",
                        lambda r, p: self._on_data_changed(var.key))
        return row

    def _create_port_selection_row(self, var: SerialPortVar):
        # Use a set to combine available ports with the currently saved value,
        # ensuring the saved value is always in the list even if disconnected.
        available_ports = SerialTransport.list_ports()
        port_set = set(available_ports)
        if var.value:
            port_set.add(var.value)

        # Create the final sorted list for the UI
        sorted_ports = sorted(list(port_set), key=natural_sort_key)
        choices = [NULL_CHOICE_LABEL] + sorted_ports

        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(
            title=var.label, subtitle=var.description or "", model=store
        )

        # Set initial selection from the var's value
        if var.value and var.value in choices:
            row.set_selected(choices.index(var.value))

        # This handler is called just before the dropdown opens.
        def on_open(gesture, n_press, x, y):
            # Preserve the currently selected value from the UI
            selected_obj = row.get_selected_item()
            current_selection = (
                selected_obj.get_string()  # type: ignore
                if selected_obj
                else None
            )

            # Fetch the new list of ports and ensure the current selection
            # is preserved in the list.
            new_ports = SerialTransport.list_ports()
            port_set = set(new_ports)
            if current_selection and current_selection != NULL_CHOICE_LABEL:
                port_set.add(current_selection)

            new_sorted_ports = sorted(list(port_set), key=natural_sort_key)
            new_choices = [NULL_CHOICE_LABEL] + new_sorted_ports

            # Get the existing model and update it in-place to avoid
            # closing the popover
            store = row.get_model()
            if not isinstance(store, Gtk.StringList):
                return

            # Check if an update is even needed to prevent unnecessary
            # updates
            current_choices = [
                store.get_string(i) for i in range(store.get_n_items())
            ]
            if current_choices == new_choices:
                return

            # Update the model using splice, which is less disruptive
            store.splice(0, store.get_n_items(), new_choices)

            # Restore the previous selection if it still exists
            if current_selection and current_selection in new_choices:
                row.set_selected(new_choices.index(current_selection))

        # Add a click controller to trigger the refresh when the user
        # clicks the row
        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", on_open)
        row.add_controller(click_controller)

        if self.explicit_apply:
            self._add_apply_button_if_needed(row, var.key)
        else:
            # This signal fires when the user makes a new selection
            row.connect(
                "notify::selected-item",
                lambda r, p: self._on_data_changed(var.key),
            )
        return row
