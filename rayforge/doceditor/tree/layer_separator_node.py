"""A dimmed separator row that marks the start of a layer."""

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
