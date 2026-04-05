from rayforge.core.ops import Ops
from rayforge.core.ops.commands import ScanLinePowerCommand
from rayforge.simulator.vertex_map import (
    build_vertex_map,
    build_scanline_overlay,
)


def test_empty_ops():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 0


def test_single_line():
    ops = Ops()
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 2
    assert len(m.command_vertex_offset) == len(list(ops)) + 1


def test_multiple_lines():
    ops = Ops()
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.line_to(0.0, 10.0, 0.0)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 6
    assert m.command_vertex_offset[0] == 0
    assert m.command_vertex_offset[-1] == 6


def test_zero_power_produces_no_powered_vertices():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 0


def test_state_commands_zero_vertices():
    ops = Ops()
    ops.set_power(0.5)
    ops.set_cut_speed(800)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 2
    assert m.command_vertex_offset[0] == 0
    assert m.command_vertex_offset[1] == 0
    assert m.command_vertex_offset[2] == 0


def test_offset_at_each_command():
    ops = Ops()
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(10.0, 10.0, 0.0)
    m = build_vertex_map(ops)
    assert m.command_vertex_offset == [0, 0, 0, 2, 2, 4]
    assert m.total_powered_vertices == 4


def test_scanline_zero_powered_vertices():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100, 200])))
    m = build_vertex_map(ops)
    assert m.total_powered_vertices == 0


def test_arc_produces_multiple_vertices():
    ops = Ops()
    ops.set_power(0.5)
    ops.move_to(0.0, 0.0, 0.0)
    ops.arc_to(100.0, 0.0, 50.0, 0.0, clockwise=True)
    m = build_vertex_map(ops)
    assert m.total_powered_vertices >= 4


def test_match_vertex_encoder():
    from rayforge.pipeline.encoder.vertexencoder import VertexEncoder

    ops = Ops()
    ops.set_power(0.8)
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    ops.set_power(0.0)
    ops.line_to(10.0, 10.0, 0.0)
    ops.set_power(1.0)
    ops.line_to(20.0, 10.0, 0.0)

    encoder = VertexEncoder()
    vd = encoder.encode(ops)
    m = build_vertex_map(ops)

    assert m.total_powered_vertices == vd.powered_vertices.shape[0]

    assert m.total_powered_vertices == vd.powered_vertices.shape[0]


def test_scanline_overlay_empty():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.line_to(10.0, 0.0, 0.0)
    overlay = build_scanline_overlay(ops)
    assert overlay.total_overlay_vertices == 0
    assert len(overlay.positions) == 0


def test_scanline_overlay_single_command():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100, 200])))
    overlay = build_scanline_overlay(ops)
    assert overlay.total_overlay_vertices == 2
    assert overlay.positions.shape == (6,)
    assert overlay.colors.shape == (8,)
    assert overlay.cmd_vertex_offset[-1] == 2


def test_scanline_overlay_zero_power_gaps():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(
        ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100, 0, 0, 200]))
    )
    overlay = build_scanline_overlay(ops)
    assert overlay.total_overlay_vertices == 4


def test_scanline_overlay_all_zero():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([0, 0, 0])))
    overlay = build_scanline_overlay(ops)
    assert overlay.total_overlay_vertices == 0


def test_scanline_overlay_multiple_scanlines():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([100])))
    ops.add(ScanLinePowerCommand((10.0, 5.0, 0.0), bytearray([200])))
    overlay = build_scanline_overlay(ops)
    assert overlay.total_overlay_vertices == 4
    assert len(overlay.cmd_vertex_offset) == len(list(ops)) + 1


def test_scanline_overlay_positions_match_ops():
    ops = Ops()
    ops.move_to(0.0, 0.0, 0.0)
    ops.add(ScanLinePowerCommand((10.0, 0.0, 0.0), bytearray([255, 255, 255])))
    overlay = build_scanline_overlay(ops)
    pos = overlay.positions.reshape(-1, 3)
    assert abs(pos[0][0] - 0.0) < 1e-6
    assert abs(pos[-1][0] - 10.0) < 1e-6
