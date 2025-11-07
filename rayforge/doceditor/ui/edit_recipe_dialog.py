import logging
from typing import Optional, Dict, Any, List
from gi.repository import Gtk, Adw
from blinker import Signal
from ...context import get_context
from ...core.recipe import Recipe
from ...core.capability import ALL_CAPABILITIES
from ...shared.units.definitions import get_unit
from ...shared.varset.varsetwidget import VarSetWidget
from .material_selector import MaterialSelectorDialog

logger = logging.getLogger(__name__)


class OptionalSpinRowController:
    """Manages an ActionRow with a SpinButton and a Switch."""

    def __init__(
        self,
        group: Adw.PreferencesGroup,
        title: str,
        subtitle: str,
        quantity: str,
    ):
        self.changed = Signal()
        self.quantity = quantity

        # Get the current unit from user preferences
        config = get_context().config
        unit_name = config.unit_preferences.get(self.quantity)
        self.unit = get_unit(unit_name) if unit_name else None
        if not self.unit:
            raise ValueError(
                f"Could not determine unit for quantity '{quantity}'"
            )

        self.row = Adw.ActionRow(title=title, subtitle=subtitle)
        group.add(self.row)

        adj = Gtk.Adjustment(lower=0, upper=9999, step_increment=0.1)
        self.spin_button = Gtk.SpinButton(
            adjustment=adj, digits=self.unit.precision
        )
        self.spin_button.set_valign(Gtk.Align.CENTER)

        self.switch = Gtk.Switch(valign=Gtk.Align.CENTER)

        # Add in reverse order of desired appearance (right to left)
        self.row.add_suffix(self.switch)
        self.row.add_suffix(self.spin_button)

        self.switch.connect("notify::active", self._on_toggled)
        self._value_changed_handler_id = self.spin_button.connect(
            "value-changed", lambda btn: self.changed.send(self)
        )

        # Set initial state
        self._on_toggled(self.switch, None)

    def _on_toggled(self, switch, _pspec):
        is_active = switch.get_active()
        self.spin_button.set_sensitive(is_active)
        self.changed.send(self)

    def get_value(self) -> Optional[float]:
        """Gets the value in base units, or None if disabled."""
        if not self.switch.get_active():
            return None
        return self.get_spin_value_in_base()

    def set_value(self, value_in_base: Optional[float]):
        """Sets the value from base units, or disables if None."""
        if value_in_base is None:
            self.switch.set_active(False)
            self.set_spin_value_in_base(0)
        else:
            self.switch.set_active(True)
            self.set_spin_value_in_base(value_in_base)

    def get_spin_value_in_base(self) -> float:
        """Gets the spinbutton's value in base units, ignoring the switch."""
        if not self.unit:
            return 0.0
        display_value = self.spin_button.get_value()
        return self.unit.to_base(display_value)

    def set_spin_value_in_base(self, value_in_base: float):
        """
        Sets the spinbutton's value from base units, without touching the
        switch.
        """
        if not self.unit:
            return
        self.spin_button.handler_block(self._value_changed_handler_id)
        display_value = self.unit.from_base(value_in_base)
        self.spin_button.set_value(display_value)
        self.spin_button.handler_unblock(self._value_changed_handler_id)


class AddEditRecipeDialog(Adw.MessageDialog):
    """A dialog for creating or editing a Recipe, with dynamic settings."""

    def __init__(
        self, parent: Optional[Gtk.Window], recipe: Optional[Recipe] = None
    ):
        super().__init__(transient_for=parent)
        self.recipe = recipe
        self._selected_material_uid: Optional[str] = (
            recipe.material_uid if recipe else None
        )
        self._machine_ids: List[Optional[str]] = []
        # Filter out the Utility capability for UI purposes
        self._ui_capabilities = [cap for cap in ALL_CAPABILITIES]

        is_editing = recipe is not None
        self.set_heading(
            _("Edit Recipe") if is_editing else _("Add New Recipe")
        )
        # Store response_id for later use (button sensitivity)
        self._response_id = "save" if is_editing else "add"
        response_label = _("Save") if is_editing else _("Add")
        self.add_response("cancel", _("Cancel"))
        self.add_response(self._response_id, response_label)
        self.set_response_appearance(
            self._response_id, Adw.ResponseAppearance.SUGGESTED
        )
        self.set_default_response(self._response_id)

        content_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL,
            spacing=24,
            margin_start=6,
            margin_end=6,
        )
        self.set_extra_child(content_box)
        self.set_size_request(600, -1)

        # New group for Recipe metadata
        recipe_group = Adw.PreferencesGroup(
            title=_("Recipe"),
            description=_(
                "A named preset of settings that can be "
                "automatically applied later."
            ),
        )
        content_box.append(recipe_group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        if recipe:
            self.name_row.set_text(recipe.name)
        self.name_row.connect("notify::text", self._on_name_changed)
        recipe_group.add(self.name_row)
        self.name_row.connect(
            "activate", lambda w: self.response(self._response_id)
        )

        self.desc_row = Adw.EntryRow(title=_("Description"))
        if recipe:
            self.desc_row.set_text(recipe.description)
        recipe_group.add(self.desc_row)

        applicability_group = Adw.PreferencesGroup(
            title=_("Applicability"),
            description=_(
                "Define when this recipe should be suggested. "
                "Leave fields blank to match any value."
            ),
        )
        content_box.append(applicability_group)

        # Use filtered capabilities
        cap_labels = [cap.label for cap in self._ui_capabilities]
        self.capability_row = Adw.ComboRow(
            title=_("Task Type"), model=Gtk.StringList.new(cap_labels)
        )
        self.capability_row.connect(
            "notify::selected", self._on_capability_changed
        )
        applicability_group.add(self.capability_row)

        # Machine Row
        machine_mgr = get_context().machine_mgr
        machines = machine_mgr.get_machines()
        machine_labels = [_("Any Machine")]
        self._machine_ids = [None]
        for machine in machines:
            machine_labels.append(machine.name)
            self._machine_ids.append(machine.id)
        self.machine_row = Adw.ComboRow(
            title=_("Machine"), model=Gtk.StringList.new(machine_labels)
        )
        applicability_group.add(self.machine_row)

        if recipe and recipe.target_machine_id:
            try:
                machine_index = self._machine_ids.index(
                    recipe.target_machine_id
                )
                self.machine_row.set_selected(machine_index)
            except ValueError:
                logger.warning(
                    "Recipe machine ID '%s' not found.",
                    recipe.target_machine_id,
                )
                self.machine_row.set_selected(0)
        else:
            self.machine_row.set_selected(0)

        # Material Row
        self.material_row = Adw.ActionRow(title=_("Material"))
        material_button = Gtk.Button(label=_("Select..."))
        material_button.set_valign(Gtk.Align.CENTER)
        material_button.connect("clicked", self._on_select_material)
        self.material_row.add_suffix(material_button)
        clear_button = Gtk.Button.new_from_icon_name("edit-clear-symbolic")
        clear_button.set_valign(Gtk.Align.CENTER)
        clear_button.set_tooltip_text(_("Clear Material Selection"))
        clear_button.connect("clicked", self._on_clear_material)
        self.material_row.add_suffix(clear_button)
        applicability_group.add(self.material_row)
        self._update_material_display()

        # Thickness Rows
        self.min_thickness_controller = OptionalSpinRowController(
            applicability_group,
            _("Min Thickness"),
            _("Minimum stock thickness for this recipe to apply"),
            "length",
        )
        self.max_thickness_controller = OptionalSpinRowController(
            applicability_group,
            _("Max Thickness"),
            _("Maximum stock thickness for this recipe to apply"),
            "length",
        )

        if recipe:
            self.min_thickness_controller.set_value(recipe.min_thickness_mm)
            self.max_thickness_controller.set_value(recipe.max_thickness_mm)

        self.min_thickness_controller.changed.connect(
            self._on_min_thickness_changed
        )
        self.max_thickness_controller.changed.connect(
            self._on_max_thickness_changed
        )

        self.varset_widget = VarSetWidget(
            title=_("Settings"),
            description=_(
                "The process settings that will be applied by this recipe."
            ),
        )
        content_box.append(self.varset_widget)

        if recipe:
            # Use filtered list for indexing
            selected_cap_index = self._ui_capabilities.index(recipe.capability)
            self.capability_row.set_selected(selected_cap_index)
        else:
            self.capability_row.set_selected(0)

        # Set initial button sensitivity and populate settings
        self._on_name_changed(self.name_row, None)
        self._on_capability_changed(self.capability_row, None)

    def _on_name_changed(self, entry_row, _pspec):
        """Updates the sensitivity of the add/save button."""
        name = self.name_row.get_text().strip()
        is_sensitive = bool(name)
        self.set_response_enabled(self._response_id, is_sensitive)

    def _on_capability_changed(self, combo_row, _pspec):
        selected_index = combo_row.get_selected()
        selected_cap = self._ui_capabilities[selected_index]

        self.varset_widget.populate(selected_cap.varset)

        if self.recipe:
            self.varset_widget.set_values(self.recipe.settings)

        self.varset_widget.set_visible(len(selected_cap.varset) > 0)

    def get_recipe_data(self) -> Dict[str, Any]:
        min_thick = self.min_thickness_controller.get_value()
        max_thick = self.max_thickness_controller.get_value()
        settings = self.varset_widget.get_values()

        # Filter out None values from settings before saving
        final_settings = {k: v for k, v in settings.items() if v is not None}

        selected_cap = self._ui_capabilities[
            self.capability_row.get_selected()
        ]

        selected_machine_index = self.machine_row.get_selected()
        selected_machine_id = self._machine_ids[selected_machine_index]

        return {
            "name": self.name_row.get_text().strip(),
            "description": self.desc_row.get_text().strip(),
            "target_machine_id": selected_machine_id,
            "material_uid": self._selected_material_uid,
            "min_thickness_mm": min_thick,
            "max_thickness_mm": max_thick,
            "target_capability_name": selected_cap.name,
            "settings": final_settings,
        }

    def _on_min_thickness_changed(self, controller: OptionalSpinRowController):
        min_val_base = controller.get_spin_value_in_base()
        max_val_base = self.max_thickness_controller.get_spin_value_in_base()

        if max_val_base < min_val_base:
            self.max_thickness_controller.set_spin_value_in_base(min_val_base)

    def _on_max_thickness_changed(self, controller: OptionalSpinRowController):
        max_val_base = controller.get_spin_value_in_base()
        min_val_base = self.min_thickness_controller.get_spin_value_in_base()

        if min_val_base > max_val_base:
            self.min_thickness_controller.set_spin_value_in_base(max_val_base)

    def _on_select_material(self, button):
        dialog = MaterialSelectorDialog(
            parent=self, on_select_callback=self._on_material_selected
        )
        dialog.present()

    def _on_material_selected(self, material_uid: str):
        self._selected_material_uid = material_uid
        self._update_material_display()

    def _on_clear_material(self, button):
        self._selected_material_uid = None
        self._update_material_display()

    def _update_material_display(self):
        if self._selected_material_uid:
            material = get_context().material_mgr.get_material(
                self._selected_material_uid
            )
            self.material_row.set_subtitle(
                material.name if material else _("Not Found")
            )
        else:
            self.material_row.set_subtitle(_("Any"))
