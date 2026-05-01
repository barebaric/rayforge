import pytest
from rayforge.core.ops import Ops
from rayforge.machine.models.zone import Zone, ZoneShape


@pytest.fixture
def make_line_ops():
    def _make(moves):
        ops = Ops()
        for x, y, cutting in moves:
            if cutting:
                ops.line_to(x, y)
            else:
                ops.move_to(x, y)
        return ops

    return _make


@pytest.fixture
def make_rect_zone():
    def _make(x, y, w, h, name="Test Zone"):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.name = name
        zone.params = {"x": x, "y": y, "w": w, "h": h}
        return zone

    return _make
