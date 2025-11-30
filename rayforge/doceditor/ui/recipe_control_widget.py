import logging
from typing import Dict, Any, cast, Optional, TYPE_CHECKING
from gi.repository import Gtk, Adw
from blinker import Signal
from ...context import get_context
from ...core.step import Step
from ...core.recipe import Recipe
from ...core.capability import Capability
from ...undo.models.property_cmd import ChangePropertyCommand
from .recipe_selector_dialog import RecipeSelectorDialog
from .edit_recipe_dialog import AddEditRecipeDialog

if TYPE_CHECKING:
    from ..editor import DocEditor

logger = logging.getLogger(__name__)


class RecipeControlWidget(Adw.ActionRow):
    """
    A widget for managing recipe application within the StepSettingsDialog.
    """

    recipe_applied = Signal()

    def __init__(self, editor: "DocEditor", step: Step, **kwargs):
        super().__init__(**kwargs)
        self.editor = editor
        self.step = step
        self.set_title(_("Recipe"))

        # "Choose..." Button
        choose_button = Gtk.Button(label=_("Choose..."))
        choose_button.set_valign(Gtk.Align.CENTER)
        choose_button.connect("clicked", self._on_choose_clicked)
        self.add_suffix(choose_button)

        # "Save As..." Button
        save_as_button = Gtk.Button(label=_("Save As..."))
        save_as_button.set_valign(Gtk.Align.CENTER)
        save_as_button.connect("clicked", self._on_save_as_clicked)
        self.add_suffix(save_as_button)

        # "Update" Button
        self.update_button = Gtk.Button(label=_("Update"))
        self.update_button.set_valign(Gtk.Align.CENTER)
        self.update_button.add_css_class("suggested-action")
        self.update_button.connect("clicked", self._on_update_clicked)
        self.add_suffix(self.update_button)

        self.step.updated.connect(self._update_ui)
        self._update_ui(self.step)

    def _get_step_settings(self) -> Dict[str, Any]:
        """Extracts recipe-relevant settings from the step."""
        settings = {}
        capability = self._get_primary_capability()
        if not capability:
            logger.warning(
                "Could not determine primary capability for step. "
                "Cannot save recipe settings."
            )
            return {}

        # The capability defines which Step properties are part of a recipe.
        recipe_keys = capability.get_setting_keys()

        for key in recipe_keys:
            if hasattr(self.step, key):
                settings[key] = getattr(self.step, key)
        return settings

    def _get_primary_capability(self) -> Optional[Capability]:
        """
        Determines the most likely capability for the current step config.
        """
        recipe_mgr = get_context().recipe_mgr
        if self.step.applied_recipe_uid:
            recipe = recipe_mgr.get_recipe_by_id(self.step.applied_recipe_uid)
            if recipe:
                return recipe.capability
        # Fallback to the first capability in the step's supported set
        if self.step.capabilities:
            return next(iter(self.step.capabilities), None)
        return None

    def _update_ui(self, sender, **kwargs):
        """Updates the subtitle and button visibility."""
        recipe_mgr = get_context().recipe_mgr
        current_recipe = None
        is_modified = False

        if self.step.applied_recipe_uid:
            current_recipe = recipe_mgr.get_recipe_by_id(
                self.step.applied_recipe_uid
            )

        if current_recipe:
            self.set_subtitle(current_recipe.name)

            # Check if settings have diverged from the recipe by asking
            # the recipe to compare itself against the step.
            if not current_recipe.matches_step_settings(self.step):
                is_modified = True
        else:
            self.set_subtitle(_("Manual Settings"))

        self.update_button.set_visible(is_modified)

    def _on_choose_clicked(self, button: Gtk.Button):
        """Opens the recipe selector dialog."""
        if not self.step.capabilities:
            logger.warning("Step has no capabilities, cannot choose recipe.")
            return

        parent_window = cast(Gtk.Window, self.get_root())
        dialog = RecipeSelectorDialog(
            parent=parent_window,
            editor=self.editor,
            capabilities=self.step.capabilities,
            on_select_callback=self._apply_recipe,
        )
        dialog.present()

    def _apply_recipe(self, recipe: Recipe):
        """Applies a selected recipe to the step via an undoable command."""
        with self.editor.doc.history_manager.transaction(
            _("Apply Recipe '{name}'").format(name=recipe.name)
        ) as t:
            # Set recipe UID
            t.execute(
                ChangePropertyCommand(
                    target=self.step,
                    property_name="applied_recipe_uid",
                    new_value=recipe.uid,
                )
            )
            # Set individual settings from the recipe
            for key, value in recipe.settings.items():
                if hasattr(self.step, key):
                    t.execute(
                        ChangePropertyCommand(
                            target=self.step,
                            property_name=key,
                            new_value=value,
                        )
                    )
        # Signal to the parent dialog that its widgets need to be synced
        self.recipe_applied.send(self)
        self._update_ui(self.step)

    def _on_save_as_clicked(self, button: Gtk.Button):
        """Saves the current step settings as a new recipe."""
        # 1. Gather context
        active_layer = self.editor.doc.active_layer
        stock_item = active_layer.stock_item if active_layer else None
        capability = self._get_primary_capability()

        # 2. Create a template Recipe object to pre-fill the dialog
        template_recipe = Recipe(
            name=_("New {cap} Recipe").format(
                cap=capability.label if capability else "Step"
            ),
            settings=self._get_step_settings(),
            target_capability_name=capability.name if capability else "",
            target_machine_id=self.editor.context.machine.id
            if self.editor.context.machine
            else None,
            material_uid=stock_item.material_uid if stock_item else None,
            min_thickness_mm=stock_item.thickness if stock_item else None,
            max_thickness_mm=stock_item.thickness if stock_item else None,
        )

        # 3. Open the full recipe editor dialog
        parent_window = cast(Gtk.Window, self.get_root())
        dialog = AddEditRecipeDialog(
            parent=parent_window, recipe=template_recipe
        )
        dialog.connect("response", self._on_save_as_dialog_response)
        dialog.present()

    def _on_save_as_dialog_response(
        self, dialog: AddEditRecipeDialog, response_id: str
    ):
        if response_id == "add":
            data = dialog.get_recipe_data()
            if data["name"]:
                new_recipe = Recipe(**data)
                recipe_mgr = get_context().recipe_mgr
                recipe_mgr.add_recipe(new_recipe)

                # Now that the recipe is saved, apply it to the current step
                command = ChangePropertyCommand(
                    target=self.step,
                    property_name="applied_recipe_uid",
                    new_value=new_recipe.uid,
                    name=_("Set Applied Recipe"),
                )
                self.editor.doc.history_manager.execute(command)
        dialog.close()

    def _on_update_clicked(self, button: Gtk.Button):
        """Updates the applied recipe with the current step settings."""
        if not self.step.applied_recipe_uid:
            return

        recipe_mgr = get_context().recipe_mgr
        recipe = recipe_mgr.get_recipe_by_id(self.step.applied_recipe_uid)
        if not recipe:
            return

        # Show confirmation dialog
        parent_window = cast(Gtk.Window, self.get_root())
        dialog = Adw.MessageDialog(
            transient_for=parent_window,
            heading=_("Update Recipe '{name}'?").format(name=recipe.name),
            body=_(
                "This will permanently overwrite the saved recipe with the "
                "current step settings. This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("update", _("Update"))
        dialog.set_response_appearance(
            "update", Adw.ResponseAppearance.SUGGESTED
        )
        dialog.connect("response", self._on_update_dialog_response, recipe)
        dialog.present()

    def _on_update_dialog_response(
        self, dialog: Adw.MessageDialog, response_id: str, recipe: Recipe
    ):
        if response_id == "update":
            recipe.settings = self._get_step_settings()
            get_context().recipe_mgr.save_recipe(recipe)
            # Manually trigger a UI update, as the step model itself didn't
            # change
            self._update_ui(self.step)
        dialog.destroy()
