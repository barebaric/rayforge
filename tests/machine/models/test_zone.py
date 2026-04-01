from rayforge.core.ops import Ops
from rayforge.machine.models.zone import (
    Zone,
    ZoneShape,
    check_ops_collides_with_zones,
)


class TestZoneCollidesWithLine:
    def test_rect_zone_hit(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 5, "y": 5, "w": 10, "h": 10}
        assert zone.collides_with_line((0, 10), (20, 10))

    def test_rect_zone_miss(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 5, "y": 5, "w": 10, "h": 10}
        assert not zone.collides_with_line((0, 0), (4, 0))

    def test_box_zone_hit(self):
        zone = Zone()
        zone.shape = ZoneShape.BOX
        zone.params = {"x": 0, "y": 0, "z": 0, "w": 10, "h": 10, "d": 5}
        assert zone.collides_with_line((0, 5), (15, 5))

    def test_cylinder_zone_hit(self):
        zone = Zone()
        zone.shape = ZoneShape.CYLINDER
        zone.params = {"x": 10, "y": 10, "z": 0, "radius": 5, "height": 10}
        assert zone.collides_with_line((0, 10), (20, 10))

    def test_cylinder_zone_miss(self):
        zone = Zone()
        zone.shape = ZoneShape.CYLINDER
        zone.params = {"x": 10, "y": 10, "z": 0, "radius": 2, "height": 10}
        assert not zone.collides_with_line((0, 0), (5, 0))


class TestZoneCollidesWithArc:
    def test_rect_zone_arc_hit(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 5, "y": 5, "w": 10, "h": 10}
        start = (10, 0)
        end = (0, 10)
        center = (0, 0)
        assert zone.collides_with_arc(start, end, center, False)

    def test_rect_zone_arc_miss(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 50, "y": 50, "w": 10, "h": 10}
        start = (10, 0)
        end = (0, 10)
        center = (0, 0)
        assert not zone.collides_with_arc(start, end, center, False)

    def test_cylinder_zone_arc_hit(self):
        zone = Zone()
        zone.shape = ZoneShape.CYLINDER
        zone.params = {
            "x": 7,
            "y": 7,
            "z": 0,
            "radius": 5,
            "height": 10,
        }
        start = (10, 0)
        end = (0, 10)
        center = (0, 0)
        assert zone.collides_with_arc(start, end, center, False)


class TestCheckOpsCollidesWithZones:
    def _make_line_ops(self, moves):
        ops = Ops()
        for x, y, cutting in moves:
            if cutting:
                ops.line_to(x, y)
            else:
                ops.move_to(x, y)
        return ops

    def test_no_zones(self):
        ops = self._make_line_ops([(10, 10, False), (20, 20, True)])
        assert not check_ops_collides_with_zones(ops, {})

    def test_line_hits_rect_zone(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 5, "y": 5, "w": 10, "h": 10}
        ops = self._make_line_ops([(0, 10, False), (20, 10, True)])
        assert check_ops_collides_with_zones(ops, {"z1": zone})

    def test_line_misses_rect_zone(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 50, "y": 50, "w": 10, "h": 10}
        ops = self._make_line_ops([(0, 0, False), (10, 10, True)])
        assert not check_ops_collides_with_zones(ops, {"z1": zone})

    def test_travel_triggers(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 0, "y": 0, "w": 20, "h": 20}
        ops = Ops()
        ops.move_to(0, 0)
        ops.move_to(50, 50)
        assert check_ops_collides_with_zones(ops, {"z1": zone})

    def test_arc_hits_zone(self):
        zone = Zone()
        zone.shape = ZoneShape.CYLINDER
        zone.params = {
            "x": 7,
            "y": 7,
            "z": 0,
            "radius": 5,
            "height": 10,
        }
        ops = Ops()
        ops.move_to(10, 0)
        ops.arc_to(0, 10, -10, 0, False)
        assert check_ops_collides_with_zones(ops, {"z1": zone})

    def test_multiple_zones(self):
        zone1 = Zone()
        zone1.shape = ZoneShape.RECT
        zone1.params = {"x": 50, "y": 50, "w": 5, "h": 5}

        zone2 = Zone()
        zone2.shape = ZoneShape.RECT
        zone2.params = {"x": 5, "y": 5, "w": 5, "h": 5}

        ops = self._make_line_ops([(0, 0, False), (10, 10, True)])
        assert check_ops_collides_with_zones(ops, {"z1": zone1, "z2": zone2})

    def test_empty_ops(self):
        zone = Zone()
        zone.shape = ZoneShape.RECT
        zone.params = {"x": 0, "y": 0, "w": 10, "h": 10}
        ops = Ops()
        assert not check_ops_collides_with_zones(ops, {"z1": zone})
