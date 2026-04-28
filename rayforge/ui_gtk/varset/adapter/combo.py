from typing import Any, List, Optional, Tuple

from gi.repository import Adw, Gtk

from ....core.varset import (
    ChoiceVar,
    Var,
)
from ....machine.transport.serial import SerialTransport
from .base import (
    RowAdapter,
    escape_title,
    natural_sort_key,
    NULL_CHOICE_LABEL,
)


class ComboAdapter(RowAdapter):
    def __init__(self, row: Adw.ComboRow, var: Var) -> None:
        self._row = row
        self._var = var

    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "ComboAdapter"]:
        assert isinstance(var, ChoiceVar)
        choices: List[str] = (
            [NULL_CHOICE_LABEL] + var.choices
            if var.allow_none
            else list(var.choices)
        )
        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        if initial_val and initial_val in choices:
            row.set_selected(choices.index(initial_val))
        else:
            row.set_selected(0)
        return row, cls(row, var)

    def get_value(self) -> Optional[Any]:
        selected = self._row.get_selected_item()
        display_str = ""
        if selected:
            display_str = selected.get_string()  # type: ignore

        if display_str == NULL_CHOICE_LABEL:
            return None
        if isinstance(self._var, ChoiceVar):
            return self._var.get_value_for_display(display_str)
        return display_str

    def set_value(self, value: Any) -> None:
        model = self._row.get_model()
        if not isinstance(model, Gtk.StringList):
            return
        display_str = NULL_CHOICE_LABEL
        if value is not None:
            if isinstance(self._var, ChoiceVar):
                display_str = self._var.get_display_for_value(
                    str(value)
                ) or str(value)
            else:
                display_str = str(value)
        for i in range(model.get_n_items()):
            if model.get_string(i) == display_str:
                self._row.set_selected(i)
                break


class BaudRateAdapter(ComboAdapter):
    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "BaudRateAdapter"]:
        choices_str = [str(rate) for rate in SerialTransport.list_baud_rates()]
        store = Gtk.StringList.new(choices_str)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        initial_val = getattr(var, target_property)
        if initial_val is not None and str(initial_val) in choices_str:
            row.set_selected(choices_str.index(str(initial_val)))
        return row, cls(row, var)


class SerialPortAdapter(ComboAdapter):
    @classmethod
    def create(
        cls, var: Var, target_property: str
    ) -> Tuple[Adw.PreferencesRow, "SerialPortAdapter"]:
        initial_val = getattr(var, target_property)
        port_set = set(SerialTransport.list_ports())
        if initial_val:
            port_set.add(initial_val)
        sorted_ports = sorted(list(port_set), key=natural_sort_key)
        choices = [NULL_CHOICE_LABEL] + sorted_ports
        store = Gtk.StringList.new(choices)
        row = Adw.ComboRow(model=store, title=escape_title(var.label))
        if var.description:
            row.set_subtitle(var.description)
        if initial_val and initial_val in choices:
            row.set_selected(choices.index(initial_val))

        def on_open(gesture, n_press, x, y):
            selected_obj = row.get_selected_item()
            current_sel = None
            if selected_obj:
                current_sel = selected_obj.get_string()  # type: ignore

            new_ports = SerialTransport.list_ports()
            port_set = set(new_ports)
            if current_sel and current_sel != NULL_CHOICE_LABEL:
                port_set.add(current_sel)
            new_sorted = sorted(list(port_set), key=natural_sort_key)
            new_choices = [NULL_CHOICE_LABEL] + new_sorted

            model = row.get_model()
            if isinstance(model, Gtk.StringList):
                model.splice(0, model.get_n_items(), new_choices)
                if current_sel in new_choices:
                    row.set_selected(new_choices.index(current_sel))

        click_controller = Gtk.GestureClick.new()
        click_controller.connect("pressed", on_open)
        row.add_controller(click_controller)
        return row, cls(row, var)
