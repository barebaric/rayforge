"""The row for one stock item."""

from collections.abc import Set as AbstractSet

from ...core.stock import StockItem
from .node import NodeRow
from .specs import NodeSpec

_STOCK_ICON = "stock-symbolic"


class StockNode(NodeRow):
    """The row for one stock item placed in the document."""

    def __init__(
        self,
        stock: StockItem,
        depth: int = 1,
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(self.build_spec(stock, depth, selected), callbacks)

    @staticmethod
    def build_spec(
        stock: StockItem,
        depth: int = 1,
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        return NodeSpec(
            kind="stock",
            uid=stock.uid,
            depth=depth,
            label=stock.name,
            icon=_STOCK_ICON,
            detail=_thickness_detail(stock),
            eye=True,
            on=stock.visible,
            selected=stock.uid in selected,
        )


def _thickness_detail(stock: StockItem) -> str:
    if stock.thickness is None:
        return ""
    return f"{stock.thickness:.1f} mm"
