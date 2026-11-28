"""A dialog that asks which layers a job should run."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...core.doc import Doc


class LayerRunDialog(Gtk.Dialog):
    """Lists the document's layers with a checkbox each.

    Multi-layer documents run one physical configuration at a time, so
    Start Job asks which layers to include. The checked layers' uids
    are reported via :attr:`run_requested`; cancelling reports
    :attr:`cancelled`.
    """

    def __init__(self, parent: Gtk.Window, doc: Doc):
        super().__init__(transient_for=parent)
        self.run_requested = Signal()
        self.cancelled = Signal()

        self.set_title(_("Start Job"))
        self.set_modal(True)
        self.add_button(_("_Cancel"), Gtk.ResponseType.CANCEL)
        run_button = self.add_button(_("_Run"), Gtk.ResponseType.OK)
        run_button.add_css_class("suggested")

        content = self.get_content_area()
        content.set_spacing(6)
        intro = Gtk.Label(
            label=_(
                "This document has multiple layers. Choose which ones "
                "to run; you change the physical layer between jobs."
            ),
            wrap=True,
            xalign=0,
            max_width_chars=45,
        )
        content.append(intro)

        self._checks: list[tuple[str, Gtk.CheckButton]] = []
        for layer in doc.layers:
            check = Gtk.CheckButton(label=layer.name, active=True)
            content.append(check)
            self._checks.append((layer.uid, check))

        self.connect("response", self._on_response)

    def _on_response(self, dialog, response):
        if response == Gtk.ResponseType.OK:
            uids = [uid for uid, check in self._checks if check.get_active()]
            self.run_requested.send(self, layer_uids=uids)
        else:
            self.cancelled.send(self)
        self.destroy()
