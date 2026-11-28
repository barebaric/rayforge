"""The row for one group of parts."""

from collections.abc import Set as AbstractSet
from gettext import ngettext

from gi.repository import Gtk

from ...core.group import Group
from .node import NodeRow
from .specs import NodeSpec

_GROUP_ICON = "layers-symbolic"


class GroupNode(NodeRow):
    """The row for one group, optionally followed by the parts inside."""

    def __init__(
        self,
        group: Group,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(group, depth, expanded, selected), callbacks
        )

    @staticmethod
    def build_spec(
        group: Group,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        members = group.all_workpieces
        return NodeSpec(
            kind="group",
            uid=group.uid,
            depth=depth,
            label=group.name,
            icon=_GROUP_ICON,
            detail=ngettext("{n} part", "{n} parts", len(members)).format(
                n=len(members)
            ),
            eye=True,
            on=group.visible,
            selected=group.uid in selected,
            has_children=bool(members),
            expanded=group.uid in expanded,
        )

    @classmethod
    def build_rows(
        cls,
        group: Group,
        depth: int,
        expanded: AbstractSet[str],
        selected: AbstractSet[str],
        callbacks,
    ) -> list[Gtk.Widget]:
        from .workpiece_node import WorkPieceNode

        rows: list[Gtk.Widget] = [
            cls(group, depth, expanded, selected, callbacks)
        ]
        if group.uid not in expanded:
            return rows
        for member in group.all_workpieces:
            rows.extend(
                WorkPieceNode.build_rows(
                    member, depth + 1, expanded, selected, callbacks
                )
            )
        return rows
