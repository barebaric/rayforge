import copy
import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, cast

from blinker import Signal
from gi.repository import Adw, Gtk

from ...context import get_context
from ...core.recipe import Recipe
from ...core.step import Step
from ...core.undo.property_cmd import ChangePropertyCommand
from .edit_recipe_dialog import AddEditRecipeDialog
from .recipe_selector_dialog import RecipeSelectorDialog

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

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

    def _get_step_settings(self) -> dict[str, Any]:
        """Extracts recipe-relevant settings from the step.

        Uses the step class's :meth:`~rayforge.core.step.Step.recipe_keys`,
        which derives the canonical list of recipe-eligible attributes
        from the step's recipe varset (replacing the older
        capability-key lookup).
        """
        settings = {}
        for key in type(self.step).recipe_keys():
            if hasattr(self.step, key):
                settings[key] = getattr(self.step, key)
        return settings

    def _get_step_transformers(self) -> list[dict[str, Any]]:
        """Deep-copy the step's transformer dicts, deduped by name.

        Returns a list of fresh dict copies combining the step's
        ``per_workpiece_transformers_dicts`` and
        ``per_step_transformers_dicts`` (a single dict can appear in
        both lists and is shared by reference, so dedup is by name
        keeping the first occurrence). Each copy is given a
        ``recipe_apply=True`` default so the recipe will stamp the
        transformer when applied.
        """
        by_name: dict[str, dict[str, Any]] = {}
        for d in list(self.step.per_workpiece_transformers_dicts) + list(
            self.step.per_step_transformers_dicts
        ):
            name = d.get("name")
            if not name or name in by_name:
                continue
            copy_d = copy.deepcopy(d)
            copy_d.setdefault("recipe_apply", True)
            by_name[name] = copy_d
        return list(by_name.values())

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
            if not current_recipe.matches_step_transformers(self.step):
                is_modified = True
        else:
            self.set_subtitle(_("Manual Settings"))

        self.update_button.set_visible(is_modified)

    def _on_choose_clicked(self, button: Gtk.Button):
        """Opens the recipe selector dialog."""
        parent_window = cast(Gtk.Window, self.get_root())
        dialog = RecipeSelectorDialog(
            parent=parent_window,
            editor=self.editor,
            on_select_callback=self._apply_recipe,
            step_type=type(self.step).__name__,
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
                    on_change_callback=(
                        lambda: (self.step.updated.send(self.step), None)[1]
                    ),
                )
            )
            # Set each setting the recipe carries; skip keys this step
            # does not own.
            for key, value in recipe.settings.items():
                if not hasattr(self.step, key):
                    continue
                t.execute(
                    ChangePropertyCommand(
                        target=self.step,
                        property_name=key,
                        new_value=value,
                        on_change_callback=(
                            lambda: (self.step.updated.send(self.step), None)[
                                1
                            ]
                        ),
                    )
                )
            # Apply transformer settings: for each recipe transformer with
            # recipe_apply=True, find the step's matching dict by name and
            # overwrite its params with undoable commands.
            self._apply_recipe_transformers(t, recipe.transformer_dicts)
        # Signal to the parent dialog that its widgets need to be synced
        self.recipe_applied.send(self)
        self._update_ui(self.step)

    def _apply_recipe_transformers(
        self, transaction: Any, transformer_dicts: list[dict[str, Any]]
    ) -> None:
        """Apply recipe transformer settings to the step's transformers.

        For each recipe dict with ``recipe_apply=True``, find the
        matching step dict by ``name`` (searching
        ``per_step_transformers_dicts`` first, then
        ``per_workpiece_transformers_dicts``). For each param key
        (except ``name`` and ``recipe_apply``), emit an undoable
        ``set_step_param`` command. The appropriate step callback
        matches the step-mode post-processing page's logic.
        """
        step_dicts_by_name: dict[str, dict[str, Any]] = {}
        for d in list(self.step.per_step_transformers_dicts) + list(
            self.step.per_workpiece_transformers_dicts
        ):
            name = d.get("name")
            if name and name not in step_dicts_by_name:
                step_dicts_by_name[name] = d

        for recipe_dict in transformer_dicts or []:
            if not recipe_dict.get("recipe_apply", True):
                continue
            name = recipe_dict.get("name")
            if not name:
                continue
            step_dict = step_dicts_by_name.get(name)
            if step_dict is None:
                continue
            is_per_step = step_dict in (self.step.per_step_transformers_dicts)
            callback = (
                self.step.per_step_transformer_changed.send
                if is_per_step
                else self._send_step_updated
            )
            for key, value in recipe_dict.items():
                if key in ("name", "recipe_apply"):
                    continue
                self.editor.step.set_step_param(
                    target_dict=step_dict,
                    key=key,
                    new_value=value,
                    name=_("Apply Recipe Transformer"),
                    on_change_callback=callback,
                )

    def _send_step_updated(self) -> None:
        self.step.updated.send(self.step)

    def _on_save_as_clicked(self, button: Gtk.Button):
        """Saves the current step settings as a new recipe."""
        # 1. Gather context - get first stock item from document
        stock_items = self.editor.doc.stock_items
        stock_item = stock_items[0] if stock_items else None
        step_class = type(self.step)

        # 2. Create a template Recipe object to pre-fill the dialog
        template_recipe = Recipe(
            name=_("New {label} Recipe").format(
                label=step_class.TYPELABEL or step_class.__name__
            ),
            settings=self._get_step_settings(),
            transformer_dicts=self._get_step_transformers(),
            target_step_types=[step_class.__name__],
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
        dialog.response.connect(self._on_save_as_dialog_response, weak=False)
        dialog.present()

    def _on_save_as_dialog_response(
        self, dialog: AddEditRecipeDialog, *, response_id: str
    ):
        if response_id in ("add", "save"):
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
            recipe.transformer_dicts = self._get_step_transformers()
            get_context().recipe_mgr.save_recipe(recipe)
            # Manually trigger a UI update, as the step model itself didn't
            # change
            self._update_ui(self.step)
        dialog.destroy()
