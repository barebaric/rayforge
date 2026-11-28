"""Row specifications shared by the Parts and Operations trees.

Node classes build a :class:`NodeSpec` from their model item; the
:class:`~rayforge.doceditor.tree.tree.TreeWidget` renders the rows.
Specs are plain frozen data so the builders stay testable without GTK
and survive model changes untouched.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class NodeSpec:
    """One row of a tree.

    ``kind`` selects the row type ("stock", "group", "workpiece",
    "step", "workflow", "part"). ``depth`` controls the indent,
    ``color`` renders a colored dot, ``badge`` a numbered circle and
    ``icon`` a symbolic icon; at most one of the three is shown.
    ``detail`` is a dimmed summary and ``duration`` a right-aligned
    text column. ``eye`` shows a visibility toggle whose state is
    ``on``.
    """

    kind: str
    uid: str
    depth: int = 0
    label: str = ""
    icon: str | None = None
    color: str | None = None
    badge: str | None = None
    detail: str = ""
    duration: str = ""
    eye: bool = False
    on: bool = True
    selectable: bool = True
    selected: bool = False
    has_children: bool = False
    expanded: bool = False
