import pytest
import numpy as np

from rayforge.core.ops import Ops
from rayforge.core.ops.commands import ScanLinePowerCommand
from compile_scene_helper import (
    make_test_config,
    make_assembled_ops,
    make_flat_layer_config,
    make_rotary_layer_config,
)
from rayforge.simulator.scene3d.scene_compiler import compile_scene


def _flat_config():
    return make_test_config(layer_configs={"layer1": make_flat_layer_config()})


def _rotary_config(diameter=50.0):
    return make_test_config(
        layer_configs={"layer1": make_rotary_layer_config(diameter=diameter)}
    )


def _single_layer_ops(step_ops, layer_uid="layer1"):
    return make_assembled_ops([(layer_uid, step_ops)])


class TestCompileLineTo:
    def test_single_powered_line(self):
        ops = Ops()
        ops.move_to(1.0, 2.0, 0.0)
        ops.set_power(0.5)
        ops.line_to(4.0, 6.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 1
        vl = artifact.vertex_layers[0]
        assert not vl.is_rotary
        pv = vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2
        np.testing.assert_allclose(pv[0], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(pv[1], [4.0, 6.0, 0.0])

        pvv = vl.power_values
        assert pvv.shape[0] == 2

    def test_travel_move(self):
        ops = Ops()
        ops.move_to(1.0, 0.0, 0.0)
        ops.move_to(5.0, 0.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 1
        vl = artifact.vertex_layers[0]
        assert not vl.is_rotary
        tv = vl.travel_verts.reshape(-1, 3)
        assert tv.shape[0] == 2
        np.testing.assert_allclose(tv[0], [1.0, 0.0, 0.01])
        np.testing.assert_allclose(tv[1], [5.0, 0.0, 0.01])


class TestCompileScanline:
    def test_scanline_zero_power_segments(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        powers = bytearray([0, 0, 255, 255, 0, 0])
        ops.add(ScanLinePowerCommand((6.0, 0.0, 0.0), powers))

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        vl = artifact.vertex_layers[0]
        zpv = vl.zero_power_verts.reshape(-1, 3)
        assert zpv.shape[0] == 4

        assert len(artifact.overlay_layers) == 1
        ol = artifact.overlay_layers[0]
        assert not ol.is_rotary
        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 2

    def test_scanline_overlay_power_values(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        powers = bytearray([128, 128])
        ops.add(ScanLinePowerCommand((2.0, 0.0, 0.0), powers))

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.overlay_layers) == 1
        ol = artifact.overlay_layers[0]
        ov_pow = ol.power_values
        assert ov_pow.shape[0] == 2
        assert all(p > 0 for p in ov_pow)


class TestCompileRotary:
    def test_rotary_cylinder_wrapping(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(10.0, 0.0, 0.0)

        diameter = 50.0
        assembled = _single_layer_ops(ops)
        config = _rotary_config(diameter=diameter)

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 1
        vl = artifact.vertex_layers[0]
        assert vl.is_rotary
        pv = vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2

        assert abs(pv[0, 1]) < 1e-5
        assert abs(pv[0, 2] - diameter / 2) < 1e-3


class TestCompileCancel:
    def test_cancel_raises(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.line_to(1.0, 1.0, 0.0)
        ops.line_to(2.0, 2.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        with pytest.raises(RuntimeError, match="Cancelled"):
            compile_scene(
                assembled,
                config,
                cancel_check=lambda: True,
            )


class TestCompileEmpty:
    def test_empty_ops(self):
        ops = Ops()
        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 0
        assert len(artifact.overlay_layers) == 0

    def test_empty_job_markers_only(self):
        assembled = Ops()
        assembled.job_start()
        assembled.job_end()

        config = _flat_config()
        artifact = compile_scene(assembled, config)
        assert len(artifact.vertex_layers) == 0
        assert len(artifact.overlay_layers) == 0


class TestCompileMultiLayer:
    def test_flat_and_rotary_separated(self):
        ops_flat = Ops()
        ops_flat.move_to(0.0, 0.0, 0.0)
        ops_flat.set_power(1.0)
        ops_flat.line_to(5.0, 0.0, 0.0)

        ops_rot = Ops()
        ops_rot.move_to(0.0, 0.0, 0.0)
        ops_rot.set_power(1.0)
        ops_rot.line_to(5.0, 0.0, 0.0)

        assembled = make_assembled_ops(
            [
                ("flat_layer", ops_flat),
                ("rot_layer", ops_rot),
            ]
        )
        config = make_test_config(
            layer_configs={
                "flat_layer": make_flat_layer_config(),
                "rot_layer": make_rotary_layer_config(diameter=50.0),
            }
        )

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 2

        flat_vl = [vl for vl in artifact.vertex_layers if not vl.is_rotary][0]
        pv_flat = flat_vl.powered_verts.reshape(-1, 3)
        assert pv_flat[0, 2] == 0.0

        rot_vl = [vl for vl in artifact.vertex_layers if vl.is_rotary][0]
        pv_rot = rot_vl.powered_verts.reshape(-1, 3)
        assert abs(pv_rot[0, 2] - 25.0) < 1e-3


class TestPoweredOffsets:
    def test_single_powered_line_offsets(self):
        ops = Ops()
        ops.move_to(1.0, 2.0, 0.0)
        ops.set_power(0.5)
        ops.line_to(4.0, 6.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        vl = artifact.vertex_layers[0]
        off = vl.powered_cmd_offsets
        assert len(off) == len(assembled.commands) + 1
        assert off[0] == 0
        assert off[-1] == 2

    def test_multi_layer_cumulative_powered_offsets(self):
        ops1 = Ops()
        ops1.move_to(0.0, 0.0, 0.0)
        ops1.set_power(1.0)
        ops1.line_to(5.0, 0.0, 0.0)

        ops2 = Ops()
        ops2.move_to(0.0, 0.0, 0.0)
        ops2.set_power(1.0)
        ops2.line_to(10.0, 0.0, 0.0)
        ops2.line_to(15.0, 0.0, 0.0)

        assembled = make_assembled_ops(
            [
                ("layer1", ops1),
                ("layer2", ops2),
            ]
        )
        config = make_test_config(
            layer_configs={
                "layer1": make_flat_layer_config(),
                "layer2": make_flat_layer_config(),
            }
        )

        artifact = compile_scene(assembled, config)

        vl = artifact.vertex_layers[0]
        off = vl.powered_cmd_offsets
        assert off[0] == 0
        assert off[-1] == 6

    def test_travel_offsets(self):
        ops = Ops()
        ops.move_to(1.0, 0.0, 0.0)
        ops.move_to(5.0, 0.0, 0.0)
        ops.move_to(10.0, 0.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        vl = artifact.vertex_layers[0]
        off = vl.travel_cmd_offsets
        assert len(off) == len(assembled.commands) + 1
        assert off[0] == 0
        assert off[-1] == 4

    def test_offsets_match_vertex_counts(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(5.0, 0.0, 0.0)
        ops.line_to(10.0, 0.0, 0.0)
        ops.move_to(20.0, 0.0, 0.0)
        ops.line_to(25.0, 0.0, 0.0)

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        vl = artifact.vertex_layers[0]
        n_powered = vl.powered_verts.size // 3
        n_travel = vl.travel_verts.size // 3
        assert vl.powered_cmd_offsets[-1] == n_powered
        assert vl.travel_cmd_offsets[-1] == n_travel

    def test_empty_ops_offsets(self):
        ops = Ops()
        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.vertex_layers) == 0


class TestOverlayOffsets:
    def test_per_command_offsets(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(0.5)
        ops.line_to(1.0, 0.0, 0.0)
        powers = bytearray([128, 128])
        ops.add(ScanLinePowerCommand((3.0, 0.0, 0.0), powers))
        ops.line_to(4.0, 0.0, 0.0)
        powers2 = bytearray([64])
        ops.add(ScanLinePowerCommand((5.0, 0.0, 0.0), powers2))

        assembled = _single_layer_ops(ops)
        config = _flat_config()

        artifact = compile_scene(assembled, config)

        assert len(artifact.overlay_layers) == 1
        ol = artifact.overlay_layers[0]

        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 4

        off = ol.cmd_offsets
        assert off[0] == 0
        assert off[-1] == 4
        assert len(off) == len(assembled.commands) + 1

    def test_multi_layer_cumulative_offsets(self):
        ops1 = Ops()
        ops1.move_to(0.0, 0.0, 0.0)
        powers1 = bytearray([128, 128])
        ops1.add(ScanLinePowerCommand((2.0, 0.0, 0.0), powers1))

        ops2 = Ops()
        ops2.move_to(0.0, 0.0, 0.0)
        powers2 = bytearray([64])
        ops2.add(ScanLinePowerCommand((1.0, 0.0, 0.0), powers2))

        assembled = make_assembled_ops(
            [
                ("layer1", ops1),
                ("layer2", ops2),
            ]
        )
        config = make_test_config(
            layer_configs={
                "layer1": make_flat_layer_config(),
                "layer2": make_flat_layer_config(),
            }
        )

        artifact = compile_scene(assembled, config)

        assert len(artifact.overlay_layers) == 1
        ol = artifact.overlay_layers[0]
        off = ol.cmd_offsets
        assert off[0] == 0
        assert off[-1] == 4

        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 4
