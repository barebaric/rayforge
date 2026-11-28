"""The "+ Workflow" pill for adding workflows."""

from gettext import gettext as _
from itertools import combinations

from blinker import Signal
from gi.repository import GLib, Gtk

from ...core.color import COLOR_PALETTE
from ...ui_gtk.icons import get_icon
from .node import swatch_button

_QUICK_COLORS = 3


def _rgb(color: str) -> tuple[int, int, int]:
    value = color.lstrip("#")
    return (
        int(value[0:2], 16),
        int(value[2:4], 16),
        int(value[4:6], 16),
    )


def _distance(first: str, second: str) -> int:
    a = _rgb(first)
    b = _rgb(second)
    return sum((x - y) ** 2 for x, y in zip(a, b))


def _contrasting_colors(count: int) -> tuple[str, ...]:
    """Picks the palette colors that are most distinct from each other."""
    candidates = combinations(COLOR_PALETTE, count)
    return max(
        candidates,
        key=lambda combo: min(
            _distance(first, second)
            for first, second in combinations(combo, 2)
        ),
    )


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

        for color in _contrasting_colors(_QUICK_COLORS):
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
