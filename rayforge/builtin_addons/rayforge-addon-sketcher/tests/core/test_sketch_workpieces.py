from typing import Any

from blinker import Signal
from raygeo.geo import Geometry
from sketcher.core.sketch import Sketch
from sketcher.core.workpieces import is_sketch_workpiece

from rayforge.core.workpiece import WorkPiece


class _OtherProvider:
    def __init__(self):
        self.uid = "other-provider"
        self.name = "Other"
        self.updated = Signal()

    @property
    def provider_type_name(self) -> str:
        return "other"

    @property
    def renderer(self):
        return None

    def get_geometry(
        self, params=None, *, text_values=None, resolved_text_cache=None
    ):
        geometry = Geometry()
        geometry.move_to(0.0, 0.0)
        geometry.line_to(5.0, 0.0)
        geometry.line_to(5.0, 5.0)
        geometry.close_path()
        return geometry, []

    def get_reference_point(self):
        return None

    def variables_used(self) -> set[str]:
        return set()

    def to_dict(self) -> dict[str, Any]:
        return {"uid": self.uid, "type": "other"}


def test_sketch_workpiece():
    sketch = Sketch(name="s")
    wp = WorkPiece.from_geometry_provider(sketch)
    wp._transient_geometry_provider = sketch
    assert is_sketch_workpiece(wp)


def test_other_provider_is_not_a_sketch():
    provider = _OtherProvider()
    wp = WorkPiece.from_geometry_provider(provider)
    wp._transient_geometry_provider = provider
    assert wp.geometry_provider_uid
    assert not is_sketch_workpiece(wp)


def test_plain_items_are_not_sketches():
    assert not is_sketch_workpiece(WorkPiece("plain"))
    assert not is_sketch_workpiece(None)
