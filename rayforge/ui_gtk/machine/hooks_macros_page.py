from gi.repository import Adw
from ...machine.models.machine import Machine
from .hook_list import HookList
from .macro_list import MacroListEditor


class HooksMacrosPage(Adw.PreferencesPage):
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
