import logging
from gettext import gettext as _
from typing import Any, Dict, List, Optional, Tuple

from blinker import Signal
from gi.repository import Adw, Gtk

from ...core.recipe import Recipe
from ...core.step import Step
from ...core.step_registry import step_registry
from ...core.varset import VarSet
from ..icons import get_icon
from ..shared.patched_dialog_window import PatchedDialogWindow
from .recipe_pages import (
    RecipeApplicabilityPage,
    RecipeGeneralPage,
    RecipeSettingsPage,
)

logger = logging.getLogger(__name__)


class AddEditRecipeDialog(PatchedDialogWindow):
    """A multi-page window for creating or editing a Recipe.

    The dialog is a thin orchestrator over three dedicated page
    widgets: :class:`RecipeGeneralPage`,
    :class:`RecipeApplicabilityPage`, and one or more
    :class:`RecipeSettingsPage` instances (rebuilt whenever the
    task/step type selection changes).
    """

    def __init__(
        self, parent: Optional[Gtk.Window], recipe: Optional[Recipe] = None
    ):
        super().__init__(transient_for=parent, modal=True)
        self.response = Signal()
        self.recipe = recipe

        is_editing = recipe is not None
        title = _("Edit Recipe") if is_editing else _("Add New Recipe")
        self.set_title(title)
        self.set_default_size(750, 700)

        # Store the intended response ID for the positive action
        self._positive_response_id = "save" if is_editing else "add"

        # --- Layout ---
        toolbar_view = Adw.ToolbarView()
        self.set_content(toolbar_view)

        header_bar = Adw.HeaderBar()
        toolbar_view.add_top_bar(header_bar)

        # Cancel Button
        cancel_btn = Gtk.Button(label=_("Cancel"))
        cancel_btn.connect("clicked", lambda w: self._send_response("cancel"))
        header_bar.pack_start(cancel_btn)

        # Save/Add Button
        save_label = _("Save") if is_editing else _("Add")
        self.save_btn = Gtk.Button(label=save_label)
        self.save_btn.add_css_class("suggested-action")
        self.save_btn.connect(
            "clicked",
            lambda w: self._send_response(self._positive_response_id),
        )
        header_bar.pack_end(self.save_btn)

        # View Stack
        self.view_stack = Adw.ViewStack()
        toolbar_view.set_content(self.view_stack)

        # --- Custom Switcher (Icon + Text horizontal) ---
        self.switcher_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        self.switcher_box.add_css_class("linked")
        header_bar.set_title_widget(self.switcher_box)

        # Page name -> toggle button (for radio grouping + teardown).
        self._tab_buttons: Dict[str, Gtk.ToggleButton] = {}
        # View-stack name -> settings page (rebuilt dynamically).
        self._settings_pages: Dict[str, RecipeSettingsPage] = {}

        # --- Pages ---
        self.general_page = RecipeGeneralPage(recipe)
        self._add_page(
            self.general_page, "general", _("General"), "settings-symbolic"
        )
        self.general_page.name_changed.connect(self._update_save_sensitivity)
        self.general_page.submit_requested.connect(
            lambda *_: self._send_response(self._positive_response_id)
        )

        self.applicability_page = RecipeApplicabilityPage(recipe)
        self._add_page(
            self.applicability_page,
            "applicability",
            _("Applicability"),
            "query-symbolic",
        )
        self.applicability_page.selection_changed.connect(
            self._rebuild_settings
        )

        # --- Initial selection + settings ---
        self.applicability_page.restore_selection(
            recipe.target_capability_name if recipe else "",
            recipe.target_step_type if recipe else None,
        )
        self._rebuild_settings()
        self._update_save_sensitivity()

        # Default to the General tab.
        self._tab_buttons["general"].set_active(True)

    # --- Tab wiring -----------------------------------------------------

    def _create_tab_child(self, text: str, icon_name: str) -> Gtk.Widget:
        """Creates a box with an icon and a label for the toggle button."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        box.append(get_icon(icon_name))
        box.append(Gtk.Label(label=text))
        return box

    def _add_page(
        self,
        page: Gtk.Widget,
        name: str,
        title: str,
        icon_name: str,
    ):
        """Register a page in the view stack with a toggle button.

        The first page registered becomes the radio-group root; every
        subsequent button joins its group so the tabs are mutually
        exclusive.
        """
        group = self._tab_buttons["general"] if self._tab_buttons else None
        button = Gtk.ToggleButton(group=group) if group else Gtk.ToggleButton()
        button.set_child(self._create_tab_child(title, icon_name))
        button.connect("toggled", self._on_tab_toggled, name)
        self.switcher_box.append(button)
        self.view_stack.add_named(page, name)
        self._tab_buttons[name] = button

    def _on_tab_toggled(self, button, page_name):
        if button.get_active():
            self.view_stack.set_visible_child_name(page_name)

    def _send_response(self, response_id: str):
        self.response.send(self, response_id=response_id)

    def _update_save_sensitivity(self, *_args):
        self.save_btn.set_sensitive(bool(self.general_page.get_name()))

    # --- Settings pages -------------------------------------------------

    def _current_settings_groups(self) -> List[Tuple[str, VarSet]]:
        """Resolve the (title, varset) groups for the current selection."""
        step_type = self.applicability_page.get_step_type()
        if step_type:
            step_class = step_registry.get(step_type)
            if step_class is not None:
                return step_class.recipe_varset_groups()

        cap = self.applicability_page.get_capability()
        if cap and len(cap.varset) > 0:
            return [(_("Settings"), cap.varset)]

        return Step.recipe_varset_groups()

    def _rebuild_settings(self, *_args):
        """Rebuild the dynamic settings tabs from the current selection.

        Laser step types split into a "Laser" page (inherited process
        settings) and a "Step Settings" page (step-specific attributes).
        A capability-only selection yields a single "Settings" page;
        "Any"/"Any" yields the base Step settings.
        """
        groups = self._current_settings_groups()

        # Keep the user on a settings page if one was visible.
        settings_was_visible = not (
            self._tab_buttons["general"].get_active()
            or self._tab_buttons["applicability"].get_active()
        )

        # Tear down existing settings pages.
        for name, page in self._settings_pages.items():
            self.switcher_box.remove(self._tab_buttons[name])
            self.view_stack.remove(page)
        self._settings_pages.clear()
        # Drop their button entries too.
        for name in [
            n for n in list(self._tab_buttons) if n.startswith("settings-")
        ]:
            del self._tab_buttons[name]

        for index, (group_title, varset) in enumerate(groups):
            name = f"settings-{index}"
            icon_name = (
                "laser-on-symbolic"
                if group_title == _("Laser")
                else "step-settings-symbolic"
            )
            page = RecipeSettingsPage(group_title)
            page.populate(varset)
            if self.recipe:
                page.set_values(self.recipe.settings)
            self._add_page(page, name, group_title, icon_name)
            self._settings_pages[name] = page

        if settings_was_visible and self._settings_pages:
            first_name = next(iter(self._settings_pages))
            self._tab_buttons[first_name].set_active(True)

    # --- Result ---------------------------------------------------------

    def get_recipe_data(self) -> Dict[str, Any]:
        # Merge values from all settings pages.
        settings: Dict[str, Any] = {}
        for page in self._settings_pages.values():
            settings.update(page.get_values())
        final_settings = {k: v for k, v in settings.items() if v is not None}

        cap = self.applicability_page.get_capability()
        return {
            "name": self.general_page.get_name(),
            "description": self.general_page.get_description(),
            "target_machine_id": self.applicability_page.get_machine_id(),
            "target_step_type": self.applicability_page.get_step_type(),
            "material_uid": self.applicability_page.get_material_uid(),
            "min_thickness_mm": self.applicability_page.get_min_thickness(),
            "max_thickness_mm": self.applicability_page.get_max_thickness(),
            "target_capability_name": cap.name if cap else "",
            "settings": final_settings,
        }
