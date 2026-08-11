"""The recipe editor's settings page: one group of process settings."""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw

from .....core.varset import VarSet
from ....varset.varsetwidget import VarSetWidget


class RecipeSettingsPage(Adw.PreferencesPage):
    """One group of recipe process settings.

    Wraps a :class:`VarSetWidget` titled ``title`` (e.g. "Laser",
    "Step Settings"). The dialog creates one instance per
    :meth:`~rayforge.core.step.Step.recipe_varset_groups` entry.
    """

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.group_title = title
        self._widget = VarSetWidget(
            title=title,
            description=_(
                "The settings that will be applied by this recipe. "
                "When multiple step types are selected, only settings "
                "common to all of them are shown."
            ),
        )
        self.add(self._widget)

    def populate(self, varset: VarSet):
        self._widget.populate(varset)

    def set_values(self, values: dict[str, Any]):
        self._widget.set_values(values)

    def get_values(self) -> dict[str, Any]:
        return self._widget.get_values()

    @property
    def keys(self):
        """The setting keys rendered on this page."""
        return list(self._widget.widget_map.keys())
