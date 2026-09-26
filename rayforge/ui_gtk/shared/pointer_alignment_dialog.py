from collections.abc import Callable
from gettext import gettext as _

from gi.repository import Adw

from ...machine.models.machine import Machine


class PointerAlignmentDialog(Adw.MessageDialog):
    """Confirmation shown before sending while pointer alignment is on.

    Jobs are never shifted by pointer alignment: by default they burn
    with the beam at the WCS positions. Because that is easy to forget
    while aiming with the pointer dot, the user confirms on every send
    and can instead run a pointer dry-run: the job is generated with
    the pointer offset folded in, so the pointer dot traces the
    toolpath while the beam runs displaced by the offset. All laser
    power is capped at the framing power, so the trace does not burn.
    """

    def __init__(
        self,
        parent,
        machine: Machine,
        on_proceed: Callable[[bool], None] | None = None,
        **kwargs,
    ):
        super().__init__(transient_for=parent, **kwargs)
        self._machine = machine
        self._on_proceed = on_proceed

        self.set_heading(_("Pointer alignment is on"))
        self.set_body(
            _(
                "Dry-Run with Pointer traces the toolpath with the "
                "pointer dot: the job runs with the pointer offset "
                "applied and the beam is displaced by it, with all "
                "laser power capped at the framing power so the trace "
                "does not burn. Turning alignment off burns the job "
                "normally with the beam at the WCS positions."
            )
        )

        self.add_response("cancel", _("_Cancel"))
        self.add_response("dry-run", _("Dry-Run with _Pointer"))
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
        if response_id in ("dry-run", "turn-off-burn") and self._on_proceed:
            self._on_proceed(response_id == "dry-run")
