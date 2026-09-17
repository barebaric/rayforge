from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw

from ...machine.models.machine import Machine


class PointerAlignmentDialog(Adw.MessageDialog):
    """Confirmation shown before burning while pointer alignment is on.

    Jobs are never shifted by pointer alignment: they always burn with
    the beam at the WCS positions. Because that is easy to forget while
    aiming with the pointer dot, the user confirms on every send.
    """

    def __init__(
        self,
        parent,
        machine: Machine,
        on_proceed: Callable[[], None] | None = None,
        **kwargs,
    ):
        super().__init__(transient_for=parent, **kwargs)
        self._machine = machine
        self._on_proceed = on_proceed

        self.set_heading(_("Pointer alignment is on"))
        self.set_body(
            _(
                "The job cuts with the beam at the WCS positions and is "
                "never shifted by the pointer offset."
            )
        )

        self.add_response("cancel", _("_Cancel"))
        self.add_response("burn", _("Burn _Anyway"))
        self.add_response("turn-off-burn", _("_Turn Off and Burn"))
        self.set_default_response("turn-off-burn")
        self.set_close_response("cancel")
        self.set_response_appearance(
            "turn-off-burn", Adw.ResponseAppearance.SUGGESTED
        )

        self.connect("response", self._on_response)

    def _on_response(self, dialog, response_id: str):
        self.destroy()
        if response_id == "turn-off-burn":
            self._machine.set_pointer_alignment(False)
        if response_id in ("burn", "turn-off-burn") and self._on_proceed:
            self._on_proceed()
