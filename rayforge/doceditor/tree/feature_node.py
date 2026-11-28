"""A read-only feature row shown beneath a part."""

from .node import NodeRow
from .specs import NodeSpec


class FeatureNode(NodeRow):
    """A non-editable summary of a feature detected in a workpiece."""

    def __init__(
        self,
        workpiece_uid: str,
        key: str,
        label: str,
        detail: str = "",
        depth: int = 2,
        callbacks=None,
    ):
        spec = NodeSpec(
            kind="feature",
            uid=f"feature:{workpiece_uid}:{key}",
            depth=depth,
            label=label,
            detail=detail,
            selectable=False,
        )
        super().__init__(spec, callbacks)
