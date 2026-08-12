"""Dialog for confirming the enabling of an experimental addon."""

from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw


class ExperimentalAddonDialog(Adw.MessageDialog):
    """
    Confirmation dialog shown before enabling an experimental addon.

    The addon is only enabled when the user explicitly confirms that
    they want to use an addon that may have unresolved issues.
    """

    def __init__(
        self,
        addon_name: str,
        on_enable: Callable[[], None] | None = None,
        on_cancel: Callable[[], None] | None = None,
    ):
        super().__init__()

        self._on_enable = on_enable
        self._on_cancel = on_cancel

        self.set_heading(_("Enable Experimental Addon?"))
        self.set_body(
            _(
                'The addon "{name}" is experimental and may have '
                "unresolved issues. Use it with caution."
            ).format(name=addon_name)
        )

        self.add_response("cancel", _("Cancel"))
        self.add_response("enable", _("Enable Anyway"))
        self.set_response_appearance(
            "enable", Adw.ResponseAppearance.DESTRUCTIVE
        )
        self.set_default_response("cancel")
        self.set_close_response("cancel")

        self.connect("response", self._on_response)

    def _on_response(self, dialog, response_id: str):
        if response_id == "enable":
            if self._on_enable:
                self._on_enable()
        elif self._on_cancel:
            self._on_cancel()
        self.close()
