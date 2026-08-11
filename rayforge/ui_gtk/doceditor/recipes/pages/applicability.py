"""The recipe editor's applicability page: when a recipe matches."""

import logging
from gettext import gettext as _
from typing import Any, cast

from blinker import Signal
from gi.repository import Adw, Gtk

from .....context import get_context
from .....core.step_registry import step_registry
from ....icons import get_icon
from ....shared.optional_spin_row import OptionalSpinRowController
from ...material_selector import MaterialSelectorDialog
from ...step_type_selection_dialog import StepTypeSelectionDialog

logger = logging.getLogger(__name__)


class RecipeApplicabilityPage(Adw.PreferencesPage):
    """The applicability criteria: when a recipe should be suggested.

    Emits :attr:`selection_changed` whenever the step type selection
    changes, so the dialog can rebuild the settings pages.
    """

    def __init__(self, recipe: Any | None = None, **kwargs):
        super().__init__(**kwargs)
        self.selection_changed = Signal()

        self._recipe = recipe
        self._machine_ids: list[str | None] = [None]
        self._selected_step_types: list[str] = list(
            recipe.target_step_types if recipe else []
        )
        self._selected_material_uid: str | None = (
            recipe.material_uid if recipe else None
        )

        group = Adw.PreferencesGroup(
            title=_("Applicability"),
            description=_(
                "Define when this recipe should be suggested. "
                "Leave fields blank to match any value."
            ),
        )
        self.add(group)

        self._build_machine_row(group)
        self._build_step_types_row(group)
        self._build_material_row(group)
        self._build_thickness_rows(group)

    # --- Builders -------------------------------------------------------

    def _build_machine_row(self, group):
        machine_mgr = get_context().machine_mgr
        machine_labels = [_("Any")]
        for machine in machine_mgr.get_machines():
            machine_labels.append(machine.name)
            self._machine_ids.append(machine.id)
        self.machine_row = Adw.ComboRow(
            title=_("Machine"), model=Gtk.StringList.new(machine_labels)
        )
        group.add(self.machine_row)

        target = self._recipe.target_machine_id if self._recipe else None
        if target and target in self._machine_ids:
            self.machine_row.set_selected(self._machine_ids.index(target))
        else:
            if target:
                logger.warning("Recipe machine ID '%s' not found.", target)
            self.machine_row.set_selected(0)

    def _build_step_types_row(self, group):
        self.step_types_row = Adw.ActionRow(
            title=_("Step Types"),
            subtitle=_(
                "The step types this recipe applies to. Leave empty to "
                "match any step type."
            ),
            activatable=True,
        )
        self.step_types_row.connect("activated", self._on_step_types_clicked)
        select_btn = Gtk.Button(label=_("Select..."))
        select_btn.set_valign(Gtk.Align.CENTER)
        select_btn.connect("clicked", self._on_step_types_clicked)
        self.step_types_row.add_suffix(select_btn)
        clear_btn = Gtk.Button(child=get_icon("clear-symbolic"))
        clear_btn.set_valign(Gtk.Align.CENTER)
        clear_btn.set_tooltip_text(_("Clear Step Types Selection"))
        clear_btn.connect("clicked", self._on_clear_step_types)
        self.step_types_row.add_suffix(clear_btn)
        group.add(self.step_types_row)
        self._update_step_types_display()

    def _build_material_row(self, group):
        self.material_row = Adw.ActionRow(title=_("Material"))
        select_btn = Gtk.Button(label=_("Select..."))
        select_btn.set_valign(Gtk.Align.CENTER)
        select_btn.connect("clicked", self._on_select_material)
        self.material_row.add_suffix(select_btn)
        clear_btn = Gtk.Button(child=get_icon("clear-symbolic"))
        clear_btn.set_valign(Gtk.Align.CENTER)
        clear_btn.set_tooltip_text(_("Clear Material Selection"))
        clear_btn.connect("clicked", self._on_clear_material)
        self.material_row.add_suffix(clear_btn)
        group.add(self.material_row)
        self._update_material_display()

    def _build_thickness_rows(self, group):
        self.min_thickness_controller = OptionalSpinRowController(
            group,
            _("Min Thickness"),
            _("Minimum stock thickness for this recipe to apply"),
            "length",
        )
        self.max_thickness_controller = OptionalSpinRowController(
            group,
            _("Max Thickness"),
            _("Maximum stock thickness for this recipe to apply"),
            "length",
        )
        if self._recipe:
            self.min_thickness_controller.set_value(
                self._recipe.min_thickness_mm
            )
            self.max_thickness_controller.set_value(
                self._recipe.max_thickness_mm
            )
        self.min_thickness_controller.changed.connect(
            self._on_min_thickness_changed
        )
        self.max_thickness_controller.changed.connect(
            self._on_max_thickness_changed
        )

    # --- Selection handling --------------------------------------------

    def _on_step_types_clicked(self, _widget):
        root = self.get_root()
        parent: Gtk.Window | None = (
            root if isinstance(root, Gtk.Window) else None
        )
        dialog = StepTypeSelectionDialog(
            parent=cast(Gtk.Window, parent),
            selected=set(self._selected_step_types),
            on_select_callback=self._on_step_types_selected,
        )
        dialog.present()

    def _on_step_types_selected(self, step_types: list[str]):
        if step_types == self._selected_step_types:
            return
        self._selected_step_types = step_types
        self._update_step_types_display()
        self.selection_changed.send(self)

    def _on_clear_step_types(self, _button):
        if not self._selected_step_types:
            return
        self._selected_step_types = []
        self._update_step_types_display()
        self.selection_changed.send(self)

    def restore_selection(self, target_step_types: list[str]):
        """Restore the step type selection from a saved recipe."""
        self._selected_step_types = list(target_step_types)
        self._update_step_types_display()

    # --- Getters --------------------------------------------------------

    def get_step_types(self) -> list[str]:
        return list(self._selected_step_types)

    def get_machine_id(self) -> str | None:
        return self._machine_ids[self.machine_row.get_selected()]

    def get_material_uid(self) -> str | None:
        return self._selected_material_uid

    def get_min_thickness(self) -> float | None:
        return self.min_thickness_controller.get_value()

    def get_max_thickness(self) -> float | None:
        return self.max_thickness_controller.get_value()

    def _update_step_types_display(self):
        if not self._selected_step_types:
            self.step_types_row.set_subtitle(_("Any"))
            return
        labels = []
        for name in self._selected_step_types:
            step_class = step_registry.get(name)
            if step_class is not None:
                labels.append(step_class.TYPELABEL)
            else:
                labels.append(name)
        text = ", ".join(labels)
        if len(text) > 120:
            text = text[:120].rstrip() + _("…")
        self.step_types_row.set_subtitle(text)

    # --- Material / thickness handlers ---------------------------------

    def _on_min_thickness_changed(self, controller: OptionalSpinRowController):
        min_val = controller.get_spin_value_in_base()
        if self.max_thickness_controller.get_spin_value_in_base() < min_val:
            self.max_thickness_controller.set_spin_value_in_base(min_val)

    def _on_max_thickness_changed(self, controller: OptionalSpinRowController):
        max_val = controller.get_spin_value_in_base()
        if self.min_thickness_controller.get_spin_value_in_base() > max_val:
            self.min_thickness_controller.set_spin_value_in_base(max_val)

    def _on_select_material(self, _button):
        root = self.get_root()
        parent: Gtk.Window | None = (
            root if isinstance(root, Gtk.Window) else None
        )
        dialog = MaterialSelectorDialog(
            parent=cast(Gtk.Window, parent),
            on_select_callback=self._on_material_selected,
        )
        dialog.present()

    def _on_material_selected(self, material_uid: str):
        self._selected_material_uid = material_uid
        self._update_material_display()

    def _on_clear_material(self, _button):
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
