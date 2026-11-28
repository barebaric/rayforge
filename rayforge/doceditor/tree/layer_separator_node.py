"""A dimmed separator row that marks the start of a layer."""

from gi.repository import Gtk

from .node import NodeRow
from .specs import NodeSpec


class LayerSeparatorNode(NodeRow):
    """A non-interactive row announcing the layer of the rows below."""

    def __init__(self, layer, callbacks=None):
        spec = NodeSpec(
            kind="layer-separator",
            uid=f"layer-separator:{layer.uid}",
            label=layer.name,
            selectable=False,
        )
        super().__init__(spec, callbacks)
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        box.set_margin_start(8)
        box.set_margin_end(8)
        box.set_margin_top(10)
        box.set_margin_bottom(2)
        label = Gtk.Label(label=layer.name)
        label.add_css_class("rf-layer-separator-label")
        left = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        left.set_hexpand(True)
        left.set_valign(Gtk.Align.CENTER)
        right = Gtk.Separator(orientation=Gtk.Orientation.HORIZONTAL)
        right.set_hexpand(True)
        right.set_valign(Gtk.Align.CENTER)
        box.append(left)
        box.append(label)
        box.append(right)
        self.set_child(box)
        self.set_activatable(False)
        self.set_selectable(False)
