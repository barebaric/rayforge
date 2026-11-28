"""The "+ Workflow" pill for adding and picking workflows."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import Gtk

from ...core.color import COLOR_PALETTE
from ...core.workflow import Workflow
from ...ui_gtk.icons import get_icon
from .node import swatch_button


class WorkflowBar(Gtk.Box):
    """The "Workflow [swatches] +" pill shown below the Operations tree.

    The swatches show the workflows that exist, in run order; picking
    one reports :attr:`workflow_activated`. The plus button opens the
    color picker and reports :attr:`workflow_requested` with the picked
    color. Creating the workflow is the caller's job.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.workflow_requested = Signal()
        self.workflow_activated = Signal()

        self.add_css_class("rf-pill")
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)

        title = Gtk.Label(label=_("Workflow"), xalign=0)
        title.add_css_class("rf-pill-title")
        self.append(title)

        self._swatches = Gtk.Box(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6
        )
        self._swatches.set_valign(Gtk.Align.CENTER)
        self.append(self._swatches)

        plus = Gtk.MenuButton(css_classes=["flat", "rf-pill-swatch"])
        icon = get_icon("add-symbolic")
        icon.set_pixel_size(14)
        plus.set_child(icon)
        plus.set_tooltip_text(_("Add workflow"))
        plus.set_valign(Gtk.Align.CENTER)
        self._popover = self._build_color_popover()
        plus.set_popover(self._popover)
        self.append(plus)

    def _build_color_popover(self) -> Gtk.Popover:
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        row.set_margin_start(10)
        row.set_margin_end(10)
        row.set_margin_top(8)
        row.set_margin_bottom(8)
        for color in COLOR_PALETTE:
            swatch = swatch_button(color)
            swatch.connect("clicked", self._on_color_clicked, color)
            row.append(swatch)
        popover = Gtk.Popover()
        popover.set_child(row)
        return popover

    def set_workflows(self, workflows: list[Workflow]):
        """Rebuilds the swatches from the workflows that exist."""
        child = self._swatches.get_first_child()
        while child:
            next_child = child.get_next_sibling()
            self._swatches.remove(child)
            child = next_child
        for workflow in workflows:
            swatch = swatch_button(workflow.color, diameter=16)
            swatch.add_css_class("rf-pill-swatch")
            swatch.set_valign(Gtk.Align.CENTER)
            swatch.set_tooltip_text(workflow.name)
            swatch.connect(
                "clicked",
                lambda b, workflow=workflow: self.workflow_activated.send(
                    self, workflow=workflow
                ),
            )
            self._swatches.append(swatch)

    def _on_color_clicked(self, button, color):
        self._popover.popdown()
        self.workflow_requested.send(self, color=color)
