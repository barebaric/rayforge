"""The "+ Workflow" pill for adding workflows."""

from gettext import gettext as _

from blinker import Signal
from gi.repository import GLib, Gtk

from ...core.color import pick_contrasting_colors
from ...ui_gtk.icons import get_icon
from .node import swatch_button

_QUICK_COLORS = 3


class WorkflowBar(Gtk.Box):
    """The quick-add workflow bar shown below the Operations tree.

    The swatches are quick-add buttons for the most contrasting colors;
    the plus button opens a custom color dialog.
    """

    def __init__(self):
        super().__init__(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.workflow_requested = Signal()

        self.add_css_class("rf-pill")
        self.set_halign(Gtk.Align.CENTER)
        self.set_valign(Gtk.Align.CENTER)
        self.set_margin_top(10)
        self.set_margin_bottom(4)

        title = Gtk.Label(label=_("Workflow"), xalign=0)
        title.add_css_class("rf-pill-title")
        self.append(title)

        for color in pick_contrasting_colors(_QUICK_COLORS):
            swatch = swatch_button(color, diameter=16)
            swatch.add_css_class("rf-pill-swatch")
            swatch.set_valign(Gtk.Align.CENTER)
            swatch.set_tooltip_text(
                _("Add workflow with color {color}").format(color=color)
            )
            swatch.connect(
                "clicked",
                lambda _button, color=color: self.workflow_requested.send(
                    self, color=color
                ),
            )
            self.append(swatch)

        custom = Gtk.Button(css_classes=["flat", "rf-pill-swatch"])
        icon = get_icon("add-symbolic")
        icon.set_pixel_size(14)
        custom.set_child(icon)
        custom.set_tooltip_text(_("Add workflow with a custom color"))
        custom.set_valign(Gtk.Align.CENTER)
        custom.connect("clicked", self._on_custom_color_clicked)
        self.append(custom)

    def _on_custom_color_clicked(self, button):
        root = self.get_root()
        parent = root if isinstance(root, Gtk.Window) else None
        dialog = Gtk.ColorDialog()
        dialog.set_with_alpha(False)
        dialog.choose_rgba(parent, None, None, self._on_color_chosen)

    def _on_color_chosen(self, dialog, result):
        try:
            rgba = dialog.choose_rgba_finish(result)
        except GLib.Error:
            return
        if rgba is None:
            return
        red = round(rgba.red * 255)
        green = round(rgba.green * 255)
        blue = round(rgba.blue * 255)
        color = f"#{red:02x}{green:02x}{blue:02x}"
        self.workflow_requested.send(self, color=color)
