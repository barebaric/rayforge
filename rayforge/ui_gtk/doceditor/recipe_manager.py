import logging
from gi.repository import Adw
from .recipe_list import RecipeListWidget

logger = logging.getLogger(__name__)


class RecipeManager(Adw.PreferencesPage):
    """
    Widget for managing recipes.
    """

    def __init__(self):
        super().__init__(
            title=_("Recipes"),
            icon_name="recipe-symbolic",
        )

        # For now, we only have one group for all user recipes.
        # This structure allows for a library pane to be added later if needed.
        self.recipe_list_editor = RecipeListWidget(
            title=_("Recipes"),
            description=_(
                "Manage your saved recipes for different materials "
                "and processes."
            ),
        )
        self.add(self.recipe_list_editor)
