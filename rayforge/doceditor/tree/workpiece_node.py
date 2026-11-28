"""The row for one workpiece, with the features detected in it."""

from collections.abc import Set as AbstractSet
from gettext import gettext as _
from gettext import ngettext

from gi.repository import Gtk
from raygeo.geo import Move

from ...core.workpiece import WorkPiece
from .feature_node import FeatureNode
from .node import NodeRow
from .specs import NodeSpec

_WORKPIECE_ICON = "model-symbolic"


class WorkPieceNode(NodeRow):
    """The row for one part, optionally followed by its features."""

    def __init__(
        self,
        workpiece: WorkPiece,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
        callbacks=None,
    ):
        super().__init__(
            self.build_spec(workpiece, depth, expanded, selected), callbacks
        )

    @staticmethod
    def build_spec(
        workpiece: WorkPiece,
        depth: int = 1,
        expanded: AbstractSet[str] = frozenset(),
        selected: AbstractSet[str] = frozenset(),
    ) -> NodeSpec:
        return NodeSpec(
            kind="workpiece",
            uid=workpiece.uid,
            depth=depth,
            label=workpiece.name,
            icon=_WORKPIECE_ICON,
            detail=_size_detail(workpiece),
            eye=True,
            on=workpiece.visible,
            selected=workpiece.uid in selected,
            has_children=_has_features(workpiece),
            expanded=workpiece.uid in expanded,
        )

    @classmethod
    def build_rows(
        cls,
        workpiece: WorkPiece,
        depth: int,
        expanded: AbstractSet[str],
        selected: AbstractSet[str],
        callbacks,
    ) -> list[Gtk.Widget]:
        rows: list[Gtk.Widget] = [
            cls(workpiece, depth, expanded, selected, callbacks)
        ]
        if workpiece.uid not in expanded:
            return rows
        rows.extend(_feature_rows(workpiece, depth, callbacks))
        return rows


def _has_features(workpiece: WorkPiece) -> bool:
    boundaries = workpiece.boundaries
    if boundaries is not None and not boundaries.is_empty():
        return True
    return bool(workpiece.fills)


def _feature_rows(
    workpiece: WorkPiece, depth: int, callbacks
) -> list[Gtk.Widget]:
    rows: list[Gtk.Widget] = []
    boundaries = workpiece.boundaries
    if boundaries is not None and not boundaries.is_empty():
        inner, outer = boundaries.split_inner_and_outer_contours()
        closed = len(inner) + len(outer)
        open_paths = max(0, _contour_count(boundaries) - closed)
        if outer:
            rows.append(
                FeatureNode(
                    workpiece.uid,
                    "outer",
                    _("Outer paths"),
                    _("{n} paths").format(n=len(outer)),
                    depth,
                    callbacks,
                )
            )
        if inner:
            rows.append(
                FeatureNode(
                    workpiece.uid,
                    "inner",
                    _("Inner paths"),
                    _("{n} paths").format(n=len(inner)),
                    depth,
                    callbacks,
                )
            )
        if open_paths:
            rows.append(
                FeatureNode(
                    workpiece.uid,
                    "open",
                    _("Open paths"),
                    _("{n} paths").format(n=open_paths),
                    depth,
                    callbacks,
                )
            )
    fills = workpiece.fills
    if fills:
        count = len(fills)
        rows.append(
            FeatureNode(
                workpiece.uid,
                "fills",
                _("Fills"),
                ngettext("{n} fill", "{n} fills", count).format(n=count),
                depth,
                callbacks,
            )
        )
    return rows


def _contour_count(geometry) -> int:
    return sum(1 for command in geometry.data if isinstance(command, Move))


def _size_detail(workpiece: WorkPiece) -> str:
    width = workpiece.natural_width_mm
    height = workpiece.natural_height_mm
    if width <= 0 and height <= 0:
        return ""
    return f"{width:.0f} × {height:.0f} mm"
