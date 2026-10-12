from rayforge.core.item import DocItem
from rayforge.core.workpiece import WorkPiece

from .sketch import Sketch


def is_sketch_workpiece(item: DocItem | None) -> bool:
    """
    Whether *item* is a workpiece drawn by a sketch. Other geometry
    providers (barcodes, for example) also set a provider uid, so the
    uid alone does not tell.
    """
    return isinstance(item, WorkPiece) and isinstance(
        item.get_geometry_provider(), Sketch
    )
