import logging
from typing import List, Callable, TYPE_CHECKING, Set
from gi.repository import Gtk, Adw
from ...context import get_context
from ...core.recipe import Recipe
from ...core.capability import Capability
from ..icons import get_icon
from ..shared.gtk import apply_css

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


css = """
.recipe-selector-list {
    background: none;
}
"""


class RecipeSelectorDialog(Adw.MessageDialog):
    """
    A dialog for selecting a recipe from a filterable list.

    The dialog is confirmed by activating a row (double-click or Enter).
    """

    class _RecipeRow(Adw.ActionRow):
        """A custom row to hold a reference to its recipe."""

        def __init__(self, recipe: Recipe, **kwargs):
            super().__init__(**kwargs)
            self.recipe: Recipe = recipe

            # Add icon as a prefix
            icon_name = f"{recipe.capability.name.lower()}-symbolic"
            icon = get_icon(icon_name)
            self.add_prefix(icon)

    def __init__(
        self,
        parent: Gtk.Window,
        editor: "DocEditor",
        capabilities: Set[Capability],
        on_select_callback: Callable[[Recipe], None],
    ):
        super().__init__(transient_for=parent)
        self.editor = editor
        self.capabilities = capabilities
        self.on_select_callback = on_select_callback
        self._all_recipes: List[Recipe] = []

        self.set_heading(_("Select Recipe"))
        self.set_body(_("Choose a recipe to apply to the current step."))

        apply_css(css)

        # Main content area
        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        content_box.set_margin_top(12)
        self.set_extra_child(content_box)

        self.filter_switch = Adw.SwitchRow(
            title=_("Show only compatible recipes")
        )
        self.filter_switch.set_active(True)
        self.filter_switch.connect(
            "notify::active", lambda *_: self._filter_and_populate_list()
        )
        content_box.append(self.filter_switch)

        self.search_entry = Gtk.SearchEntry()
        self.search_entry.connect(
            "search-changed", lambda *_: self._filter_and_populate_list()
        )
        content_box.append(self.search_entry)

        # Scrolled window for the list
        scrolled_window = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            min_content_height=300,
            vexpand=True,
        )
        scrolled_window.add_css_class("card")
        content_box.append(scrolled_window)

        # Recipe list
        self.recipe_list = Gtk.ListBox()
        self.recipe_list.set_selection_mode(Gtk.SelectionMode.SINGLE)
        self.recipe_list.add_css_class("recipe-selector-list")
        self.recipe_list.connect("row-activated", self._on_recipe_activated)
        scrolled_window.set_child(self.recipe_list)

        # Add response button
        self.add_response("cancel", _("Cancel"))
        self.set_default_response("cancel")

        self._populate_recipes()

    def _populate_recipes(self):
        """Fetches all recipes and populates the list for the first time."""
        recipe_mgr = get_context().recipe_mgr
        self._all_recipes = sorted(
            recipe_mgr.get_all_recipes(), key=lambda r: r.name.lower()
        )
        self._filter_and_populate_list()

    def _filter_and_populate_list(self):
        """Filters recipes based on search and compatibility switch."""
        search_text = self.search_entry.get_text().lower()
        show_compatible_only = self.filter_switch.get_active()

        # Get context for compatibility check
        active_layer = self.editor.doc.active_layer
        stock_item = active_layer.stock_item if active_layer else None
        machine = self.editor.context.machine

        # Clear existing rows
        while child := self.recipe_list.get_row_at_index(0):
            self.recipe_list.remove(child)

        for recipe in self._all_recipes:
            # Filter by search text
            if search_text and search_text not in recipe.name.lower():
                continue

            # Filter by compatibility
            if show_compatible_only and not recipe.matches(
                stock_item, self.capabilities, machine
            ):
                continue

            row = self._RecipeRow(
                recipe=recipe,
                title=recipe.name,
                subtitle=recipe.capability.label,
                activatable=True,
            )
            self.recipe_list.append(row)

    def _on_recipe_activated(self, listbox: Gtk.ListBox, row: _RecipeRow):
        """Handles when a recipe is selected by activation."""
        self.on_select_callback(row.recipe)
        self.close()
