from typing import Any

from blinker import Signal
from raygeo.geo import Geometry
from sketcher.core import Sketch
from sketcher.core.workpieces import is_sketch_workpiece

from rayforge.core.doc import Doc
from rayforge.core.workpiece import WorkPiece


class _OtherProvider:
    def __init__(self):
        self.uid = "other-provider"
        self.name = "Other"
        self.updated = Signal()
        self.provider_type_name = "other"
        self.renderer = None

    def get_geometry(self, params=None, *, resolved_text_cache=None):
        geometry = Geometry()
        geometry.move_to(0.0, 0.0)
        geometry.line_to(5.0, 0.0)
        geometry.line_to(5.0, 5.0)
        geometry.close_path()
        return geometry, []

    def to_dict(self) -> dict[str, Any]:
        return {"uid": self.uid, "type": "other"}


def test_sketch_workpiece():
    doc = Doc()
    sketch = Sketch(name="s")
    sketch.add_point(0, 0)
    doc.add_asset(sketch)
    wp = WorkPiece.from_geometry_provider(sketch)
    doc.active_layer.add_child(wp)
    assert is_sketch_workpiece(wp)


def test_other_provider_is_not_a_sketch():
    wp = WorkPiece.from_geometry_provider(_OtherProvider())
    assert wp.geometry_provider_uid
    assert not is_sketch_workpiece(wp)


def test_plain_items_are_not_sketches():
    assert not is_sketch_workpiece(WorkPiece("plain"))
    assert not is_sketch_workpiece(None)
