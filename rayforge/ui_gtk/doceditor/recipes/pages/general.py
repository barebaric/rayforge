"""The recipe editor's general page: name and description."""

from gettext import gettext as _
from typing import Any

from blinker import Signal
from gi.repository import Adw


class RecipeGeneralPage(Adw.PreferencesPage):
    """The recipe's name and description."""

    def __init__(self, recipe: Any | None = None, **kwargs):
        super().__init__(**kwargs)
        self.name_changed = Signal()
        self.submit_requested = Signal()

        group = Adw.PreferencesGroup(
            title=_("Recipe"),
            description=_(
                "A named preset of settings that can be "
                "automatically applied later."
            ),
        )
        self.add(group)

        self.name_row = Adw.EntryRow(title=_("Name"))
        if recipe:
            self.name_row.set_text(recipe.name)
        self.name_row.connect("notify::text", self._on_name_changed)
        self.name_row.connect("activate", self._on_name_activated)
        group.add(self.name_row)

        self.desc_row = Adw.EntryRow(title=_("Description"))
        if recipe:
            self.desc_row.set_text(recipe.description)
        group.add(self.desc_row)

    def _on_name_changed(self, entry_row, _pspec):
        self.name_changed.send(self)

    def _on_name_activated(self, _entry_row):
        self.submit_requested.send(self)

    def get_name(self) -> str:
        return self.name_row.get_text().strip()

    def get_description(self) -> str:
        return self.desc_row.get_text().strip()
