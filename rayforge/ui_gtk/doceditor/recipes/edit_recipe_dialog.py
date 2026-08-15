import logging
from gettext import gettext as _
from typing import Any

from blinker import Signal
from gi.repository import Adw, Gtk

from ....core.recipe import Recipe
from ....core.step import Step
from ....core.step_registry import step_registry
from ....core.varset import VarSet
from ...icons import get_icon
from ...shared.patched_dialog_window import PatchedDialogWindow
from .pages.applicability import RecipeApplicabilityPage
from .pages.general import RecipeGeneralPage
from .pages.post_processing import RecipePostProcessingPage
from .pages.settings import RecipeSettingsPage

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
        self, parent: Gtk.Window | None, recipe: Recipe | None = None
    ):
        super().__init__(transient_for=parent, modal=True)
        self.response = Signal()
        self.recipe = recipe

        is_editing = recipe is not None
        title = _("Edit Recipe") if is_editing else _("Add New Recipe")
        self.set_title(title)
        self.set_default_size(850, 700)

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
        self._tab_buttons: dict[str, Gtk.ToggleButton] = {}
        # View-stack name -> settings page (rebuilt dynamically).
        self._settings_pages: dict[str, RecipeSettingsPage] = {}
        # The post-processing page (rebuilt dynamically).
        self._post_processing_page: RecipePostProcessingPage | None = None
        # Stable name used for the post-processing view-stack page.
        self._pp_tab_name = "post-processing"

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
            list(recipe.target_step_types) if recipe else []
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

    def _current_step_classes(self) -> list[type[Step]]:
        """Resolve the step classes for the current step-type selection.

        Returns an empty list for the generic (Any) selection.
        """
        step_types = self.applicability_page.get_step_types()
        classes = [step_registry.get(name) for name in step_types]
        return [c for c in classes if c is not None]

    def _current_settings_groups(self) -> list[tuple[str, VarSet]]:
        """Resolve the (title, varset) groups for the current selection.

        With exactly one step type targeted, that step's full groups are
        shown. With several, only the settings common to all of them are
        offered (:meth:`Step.common_recipe_varset_groups`). With none,
        the base ``Step`` groups (universal motion settings) are used.
        """
        classes = self._current_step_classes()

        if len(classes) == 1:
            return classes[0].recipe_varset_groups()
        if classes:
            return Step.common_recipe_varset_groups(classes)
        return Step.recipe_varset_groups()

    def _current_transformer_dicts(self) -> list[dict[str, Any]]:
        """Resolve the transformer dicts for the current selection.

        The common transformers across the selected step types are
        overlaid with the recipe's stored values (matched by name). When
        the recipe carries a dict for a transformer that is no longer
        common, the stored dict is dropped. Dicts for transformers that
        appear in the common set but not in the recipe are taken from the
        step type defaults.
        """
        classes = self._current_step_classes()
        common_dicts = Step.common_transformer_dicts(classes)
        if not common_dicts:
            return []

        recipe_dicts = self.recipe.transformer_dicts if self.recipe else []
        recipe_by_name = {
            d.get("name"): d for d in recipe_dicts if d.get("name")
        }

        result: list[dict[str, Any]] = []
        for common_dict in common_dicts:
            name = common_dict.get("name")
            stored = recipe_by_name.get(name) if name else None
            # Keep stored params but use the structural common
            # dict as the base to guarantee key consistency.
            merged = dict(common_dict)
            if stored is not None:
                merged.update(stored)
            # Every dict carries an explicit apply state; defaults to
            # "Leave unchanged".
            merged.setdefault("recipe_apply", False)
            result.append(merged)
        return result

    def _rebuild_settings(self, *_args):
        """Rebuild the dynamic settings tabs from the current selection.

        Laser step types split into a "Laser" page (inherited process
        settings) and a "Step Settings" page (step-specific attributes).
        A capability-only selection yields a single "Settings" page;
        "Any"/"Any" yields the base Step settings.

        The post-processing tab is added (or rebuilt) when the current
        selection shares common transformers, and torn down otherwise.
        """
        groups = self._current_settings_groups()

        # Keep the user on a settings or post-processing page if one was
        # visible.
        post_processing_was_visible = (
            self._pp_tab_name in self._tab_buttons
            and self._tab_buttons[self._pp_tab_name].get_active()
        )
        settings_was_visible = not (
            self._tab_buttons["general"].get_active()
            or self._tab_buttons["applicability"].get_active()
            or post_processing_was_visible
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
                page.set_setting_dicts(self.recipe.setting_dicts)
            self._add_page(page, name, group_title, icon_name)
            self._settings_pages[name] = page

        # Rebuild the post-processing tab.
        self._rebuild_post_processing()

        if settings_was_visible and self._settings_pages:
            first_name = next(iter(self._settings_pages))
            self._tab_buttons[first_name].set_active(True)
        elif (
            post_processing_was_visible
            and self._post_processing_page is not None
        ):
            self._tab_buttons[self._pp_tab_name].set_active(True)

    def _rebuild_post_processing(self) -> None:
        """Build or tear down the post-processing tab from the selection.

        When the selected step types share common transformers, a tab is
        added (or rebuilt) with a :class:`RecipePostProcessingPage`.
        When there are no common transformers, the existing tab (if any)
        is removed.
        """
        transformer_dicts = self._current_transformer_dicts()

        # Tear down any existing post-processing page first.
        if self._post_processing_page is not None:
            self.switcher_box.remove(self._tab_buttons[self._pp_tab_name])
            self.view_stack.remove(self._post_processing_page)
            del self._tab_buttons[self._pp_tab_name]
            self._post_processing_page = None

        if not transformer_dicts:
            return

        page = RecipePostProcessingPage(transformer_dicts)
        self._add_page(
            page,
            self._pp_tab_name,
            _("Post Processing"),
            "step-settings-symbolic",
        )
        self._post_processing_page = page

    # --- Result ---------------------------------------------------------

    def get_recipe_data(self) -> dict[str, Any]:
        # Collect setting_dicts (with apply states) from all pages.
        setting_dicts: list[dict[str, Any]] = []
        for page in self._settings_pages.values():
            setting_dicts.extend(page.get_setting_dicts())

        transformer_dicts = (
            self._post_processing_page.get_transformer_dicts()
            if self._post_processing_page is not None
            else []
        )

        return {
            "name": self.general_page.get_name(),
            "description": self.general_page.get_description(),
            "target_machine_id": self.applicability_page.get_machine_id(),
            "target_step_types": self.applicability_page.get_step_types(),
            "material_uid": self.applicability_page.get_material_uid(),
            "min_thickness_mm": self.applicability_page.get_min_thickness(),
            "max_thickness_mm": self.applicability_page.get_max_thickness(),
            "setting_dicts": setting_dicts,
            "transformer_dicts": transformer_dicts,
        }
