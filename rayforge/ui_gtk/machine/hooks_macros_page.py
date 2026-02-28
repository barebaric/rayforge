from gettext import gettext as _
from ...machine.models.machine import Machine
from ..shared.preferences_page import TrackedPreferencesPage
from .hook_list import HookList
from .macro_list import MacroListEditor


class HooksMacrosPage(TrackedPreferencesPage):
    key = "hooks-macros"
    path_prefix = "/machine-settings/"

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(
            title=_("Hooks & Macros"),
            icon_name="utilities-terminal-symbolic",
            **kwargs,
        )
        self.machine = machine

        hook_list = HookList(machine=self.machine)
        self.add(hook_list)

        macro_editor = MacroListEditor(
            machine=self.machine,
            title=_("Macros"),
            description=_("Create and manage reusable G-code snippets."),
        )
        self.add(macro_editor)
