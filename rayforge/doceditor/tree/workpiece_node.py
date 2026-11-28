"""The row for one workpiece, with the features detected in it."""

from collections.abc import Set as AbstractSet
from gettext import gettext as _
from gettext import ngettext

from gi.repository import Gtk

from ...core.workpiece import WorkPiece, WorkpieceFeatureKind
from .feature_node import FeatureNode
from .node import NodeRow
from .specs import NodeSpec

_WORKPIECE_ICON = "model-symbolic"
_FEATURE_LABELS = {
    WorkpieceFeatureKind.OUTER_PATHS: _("Outer paths"),
    WorkpieceFeatureKind.INNER_PATHS: _("Inner paths"),
    WorkpieceFeatureKind.OPEN_PATHS: _("Open paths"),
    WorkpieceFeatureKind.FILLS: _("Fills"),
}


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
    for feature in workpiece.detect_features():
        if feature.kind is WorkpieceFeatureKind.FILLS:
            detail = ngettext("{n} fill", "{n} fills", feature.count).format(
                n=feature.count
            )
        else:
            detail = _("{n} paths").format(n=feature.count)
        rows.append(
            FeatureNode(
                workpiece.uid,
                feature.kind.value,
                _FEATURE_LABELS[feature.kind],
                detail,
                depth,
                callbacks,
            )
        )
    return rows


def _size_detail(workpiece: WorkPiece) -> str:
    width = workpiece.natural_width_mm
    height = workpiece.natural_height_mm
    if width <= 0 and height <= 0:
        return ""
    return f"{width:.0f} × {height:.0f} mm"
