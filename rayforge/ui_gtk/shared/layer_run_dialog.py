"""A dialog that asks which layers a job should run."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Adw, Gtk

from ...core.doc import Doc


class LayerRunDialog(Adw.MessageDialog):
    """Lists the document's layers with a checkbox each.

    Multi-layer documents contain several physical configurations, so
    Start Job asks which layers to include. The checked layers' uids
    are reported via :attr:`run_requested`; cancelling reports
    :attr:`cancelled`.
    """

    def __init__(self, parent: Gtk.Window, doc: Doc):
        super().__init__(transient_for=parent)
        self.run_requested = Signal()
        self.cancelled = Signal()

        self.set_heading(_("Start Job"))
        self.set_body(
            _(
                "This document has multiple layers. Choose which ones "
                "to run; you change the physical setup between jobs."
            )
        )

        content = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        content.set_margin_top(6)
        self._checks: list[tuple[str, Gtk.CheckButton]] = []
        for layer in doc.layers:
            check = Gtk.CheckButton(label=layer.name, active=True)
            content.append(check)
            self._checks.append((layer.uid, check))
        self.set_extra_child(content)

        self.add_response("cancel", _("_Cancel"))
        self.add_response("run", _("_Run"))
        self.set_default_response("run")
        self.set_close_response("cancel")
        self.set_response_appearance("run", Adw.ResponseAppearance.SUGGESTED)
        self.connect("response", self._on_response)

    def _on_response(self, dialog, response):
        if response == "run":
            uids = [uid for uid, check in self._checks if check.get_active()]
            self.run_requested.send(self, layer_uids=uids)
        else:
            self.cancelled.send(self)
