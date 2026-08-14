"""The recipe editor's settings page: one group of process settings."""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw

from .....core.varset import VarSet
from ..varset_widget import RecipeVarSetWidget


class RecipeSettingsPage(Adw.PreferencesPage):
    """One group of recipe process settings.

    Wraps a :class:`RecipeVarSetWidget` titled ``title`` (e.g. "Laser",
    "Step Settings"). The dialog creates one instance per
    :meth:`~rayforge.core.step.Step.recipe_varset_groups` entry.
    """

    def __init__(self, title: str, **kwargs):
        super().__init__(**kwargs)
        self.group_title = title
        self._widget = RecipeVarSetWidget(
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

    def set_setting_dicts(self, setting_dicts: list[dict[str, Any]]):
        self._widget.set_setting_dicts(setting_dicts)

    def get_setting_dicts(self) -> list[dict[str, Any]]:
        return self._widget.get_setting_dicts()

    @property
    def keys(self):
        """The setting keys rendered on this page."""
        return list(self._widget.widget_map.keys())
