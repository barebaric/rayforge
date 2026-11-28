"""The "+ Workflow" pill for adding workflows."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...core.color import COLOR_PALETTE
from .node import swatch_button


class WorkflowBar(Gtk.Box):
    """The quick-add workflow bar shown below the Operations tree.

    Each swatch adds a workflow with that color; the colors are not
    tied to existing workflows.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.workflow_requested = Signal()

        self.add_css_class("rf-pill")
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)

        title = Gtk.Label(label=_("Workflow"), xalign=0)
        title.add_css_class("rf-pill-title")
        self.append(title)

        for color in COLOR_PALETTE:
            swatch = swatch_button(color, diameter=16)
            swatch.add_css_class("rf-pill-swatch")
            swatch.set_valign(Gtk.Align.CENTER)
            swatch.set_tooltip_text(_("Add workflow"))
            swatch.connect(
                "clicked",
                lambda _button, color=color: self.workflow_requested.send(
                    self, color=color
                ),
            )
            self.append(swatch)
