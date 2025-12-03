import logging
from typing import cast
from gi.repository import Gtk, Adw
from blinker import Signal
from ...context import get_context
from ...core.recipe import Recipe
from ...shared.ui.formatter import format_value
from ...shared.ui.preferences_group import PreferencesGroupWithButton
from ...icons import get_icon
from .edit_recipe_dialog import AddEditRecipeDialog

logger = logging.getLogger(__name__)


class RecipeRow(Gtk.Box):
    """A widget representing a single Recipe in a ListBox."""

    def __init__(self, recipe: Recipe, on_delete, on_edit):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        self.recipe = recipe

        self.set_margin_top(6)
        self.set_margin_bottom(6)
        self.set_margin_start(12)
        self.set_margin_end(6)

        icon_name = f"{recipe.capability.name.lower()}-symbolic"
        icon = get_icon(icon_name)
        icon.set_valign(Gtk.Align.CENTER)
        self.append(icon)

        labels_box = Gtk.Box(
            orientation=Gtk.Orientation.VERTICAL, hexpand=True
        )
        self.append(labels_box)

        title = Gtk.Label(label=recipe.name, halign=Gtk.Align.START, xalign=0)
        labels_box.append(title)

        subtitle = Gtk.Label(
            label=self._get_subtitle(),
            halign=Gtk.Align.START,
            xalign=0,
        )
        subtitle.add_css_class("dim-label")
        labels_box.append(subtitle)

        suffix_box = Gtk.Box(spacing=6, valign=Gtk.Align.CENTER)
        self.append(suffix_box)

        edit_button = Gtk.Button(child=get_icon("document-edit-symbolic"))
        edit_button.add_css_class("flat")
        edit_button.connect("clicked", lambda w: on_edit(recipe))
        suffix_box.append(edit_button)

        delete_button = Gtk.Button(child=get_icon("delete-symbolic"))
        delete_button.add_css_class("flat")
        delete_button.connect("clicked", lambda w: on_delete(recipe))
        suffix_box.append(delete_button)

    def _get_subtitle(self) -> str:
        parts = []
        context = get_context()

        # 1. Machine
        if self.recipe.target_machine_id:
            machine = context.machine_mgr.get_machine_by_id(
                self.recipe.target_machine_id
            )
            parts.append(machine.name if machine else _("Unknown Machine"))

        # 2. Capability
        parts.append(self.recipe.capability.label)

        # 3. Material
        if self.recipe.material_uid:
            material = context.material_mgr.get_material(
                self.recipe.material_uid
            )
            parts.append(material.name if material else _("Unknown Material"))

        # 4. Thickness
        if self.recipe.min_thickness_mm is not None:
            min_formatted = format_value(
                self.recipe.min_thickness_mm, "length"
            )
            if self.recipe.max_thickness_mm == self.recipe.min_thickness_mm:
                parts.append(min_formatted)
            elif self.recipe.max_thickness_mm is not None:
                max_formatted = format_value(
                    self.recipe.max_thickness_mm, "length"
                )
                parts.append(f"{min_formatted} - {max_formatted}")

        return " | ".join(parts)


class RecipeListWidget(PreferencesGroupWithButton):
    """Displays a list of recipes and allows adding/editing/deleting them."""

    def __init__(self, **kwargs):
        super().__init__(button_label=_("Add New Recipe"), **kwargs)
        self.recipes_changed = Signal()

        placeholder = Gtk.Label(
            label=_("No recipes found."),
            halign=Gtk.Align.CENTER,
            margin_top=12,
            margin_bottom=12,
        )
        placeholder.add_css_class("dim-label")
        self.list_box.set_placeholder(placeholder)
        self.list_box.set_show_separators(True)

        self.populate_recipes()

    def populate_recipes(self):
        recipe_mgr = get_context().recipe_mgr
        recipes = sorted(
            recipe_mgr.get_all_recipes(), key=lambda r: r.name.lower()
        )
        self.set_items(recipes)

    def create_row_widget(self, item: Recipe) -> Gtk.Widget:
        return RecipeRow(item, self._on_delete_recipe, self._on_edit_recipe)

    def _on_add_clicked(self, button):
        root = self.get_root()
        parent_window = (
            cast(Gtk.Window, root) if isinstance(root, Gtk.Window) else None
        )
        dialog = AddEditRecipeDialog(parent=parent_window)

        def on_response(d, response_id: str):
            if response_id == "add":
                data = d.get_recipe_data()
                if data["name"]:
                    new_recipe = Recipe(
                        name=data["name"],
                        description=data["description"],
                        target_capability_name=data["target_capability_name"],
                        target_machine_id=data["target_machine_id"],
                        material_uid=data["material_uid"],
                        min_thickness_mm=data["min_thickness_mm"],
                        max_thickness_mm=data["max_thickness_mm"],
                        settings=data["settings"],
                    )
                    get_context().recipe_mgr.add_recipe(new_recipe)
                    self.populate_recipes()
                    self.recipes_changed.send(self)
            d.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_edit_recipe(self, recipe: Recipe):
        root = self.get_root()
        parent_window = (
            cast(Gtk.Window, root) if isinstance(root, Gtk.Window) else None
        )
        dialog = AddEditRecipeDialog(parent=parent_window, recipe=recipe)

        def on_response(d, response_id: str):
            if response_id == "save":
                data = d.get_recipe_data()
                if data["name"]:
                    recipe.name = data["name"]
                    recipe.description = data["description"]
                    recipe.target_capability_name = data[
                        "target_capability_name"
                    ]
                    recipe.target_machine_id = data["target_machine_id"]
                    recipe.material_uid = data["material_uid"]
                    recipe.min_thickness_mm = data["min_thickness_mm"]
                    recipe.max_thickness_mm = data["max_thickness_mm"]
                    recipe.settings = data["settings"]
                    get_context().recipe_mgr.save_recipe(recipe)
                    self.populate_recipes()
                    self.recipes_changed.send(self)
            d.close()

        dialog.connect("response", on_response)
        dialog.present()

    def _on_delete_recipe(self, recipe: Recipe):
        root = self.get_root()
        dialog = Adw.MessageDialog(
            transient_for=(
                cast(Gtk.Window, root)
                if isinstance(root, Gtk.Window)
                else None
            ),
            heading=_("Delete '{name}'?").format(name=recipe.name),
            body=_(
                "The recipe will be permanently removed. "
                "This action cannot be undone."
            ),
        )
        dialog.add_response("cancel", _("Cancel"))
        dialog.add_response("delete", _("Delete"))
        dialog.set_response_appearance(
            "delete", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def on_response(d, response_id):
            if response_id == "delete":
                get_context().recipe_mgr.delete_recipe(recipe.uid)
                self.populate_recipes()
                self.recipes_changed.send(self)
            d.destroy()

        dialog.connect("response", on_response)
        dialog.present()
