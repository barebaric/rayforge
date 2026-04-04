import pytest
import numpy as np

from rayforge.core.ops import Ops
from rayforge.core.ops.commands import ScanLinePowerCommand
from compile_scene_helper import make_test_config
from rayforge.ui_gtk.canvas3d.render_config import StepRenderConfig
from rayforge.ui_gtk.canvas3d.scene_compiler import compile_scene


def _identity_config(**step_overrides):
    cfg = make_test_config()
    if step_overrides:
        sc = StepRenderConfig(
            rotary_enabled=step_overrides.get("rotary_enabled", False),
            rotary_diameter=step_overrides.get("rotary_diameter", 25.0),
            laser_uid=step_overrides.get("laser_uid", ""),
        )
        cfg.step_configs["step1"] = sc
    return cfg


class TestCompileLineTo:
    def test_single_powered_line(self):
        ops = Ops()
        ops.move_to(1.0, 2.0, 0.0)
        ops.set_power(0.5)
        ops.line_to(4.0, 6.0, 0.0)

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.vertex_layers) == 2
        vl = artifact.vertex_layers[0]
        pv = vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2
        np.testing.assert_allclose(pv[0], [1.0, 2.0, 0.0])
        np.testing.assert_allclose(pv[1], [4.0, 6.0, 0.0])

        pc = vl.powered_colors.reshape(-1, 4)
        assert pc.shape[0] == 2
        assert pc.shape[1] == 4

        assert artifact.vertex_layers[1].powered_verts.size == 0

    def test_travel_move(self):
        ops = Ops()
        ops.move_to(1.0, 0.0, 0.0)
        ops.move_to(5.0, 0.0, 0.0)

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.vertex_layers) == 2
        vl = artifact.vertex_layers[0]
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

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        vl = artifact.vertex_layers[0]
        zpv = vl.zero_power_verts.reshape(-1, 3)
        assert zpv.shape[0] == 4

        assert len(artifact.overlay_layers) == 2
        ol = artifact.overlay_layers[0]
        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 2
        assert artifact.overlay_layers[1].positions.size == 0

    def test_scanline_overlay_colors_with_lut(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        powers = bytearray([128, 128])
        ops.add(ScanLinePowerCommand((2.0, 0.0, 0.0), powers))

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.overlay_layers) == 2
        ol = artifact.overlay_layers[0]
        ov_col = ol.colors.reshape(-1, 4)
        assert ov_col.shape[0] == 2
        assert ov_col[0, 3] == 1.0


class TestCompileRotary:
    def test_rotary_cylinder_wrapping(self):
        ops = Ops()
        ops.move_to(0.0, 0.0, 0.0)
        ops.set_power(1.0)
        ops.line_to(10.0, 0.0, 0.0)

        diameter = 50.0
        config = _identity_config(
            rotary_enabled=True, rotary_diameter=diameter
        )
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.vertex_layers) == 2
        assert artifact.vertex_layers[0].powered_verts.size == 0
        vl = artifact.vertex_layers[1]
        pv = vl.powered_verts.reshape(-1, 3)
        assert pv.shape[0] == 2

        assert abs(pv[0, 1]) < 1e-5
        assert abs(pv[0, 2] - diameter / 2) < 1e-3


class TestCompileCancel:
    def test_cancel_raises(self):
        ops1 = Ops()
        ops1.move_to(0.0, 0.0, 0.0)
        ops1.line_to(1.0, 1.0, 0.0)

        ops2 = Ops()
        ops2.move_to(0.0, 0.0, 0.0)
        ops2.line_to(2.0, 2.0, 0.0)

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        with pytest.raises(RuntimeError, match="Cancelled"):
            compile_scene(
                [(ops1, step_cfg), (ops2, step_cfg)],
                config,
                cancel_check=lambda: True,
            )


class TestCompileEmpty:
    def test_empty_ops(self):
        ops = Ops()
        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.vertex_layers) == 2
        assert artifact.vertex_layers[0].powered_verts.size == 0
        assert artifact.vertex_layers[1].powered_verts.size == 0
        assert len(artifact.overlay_layers) == 2
        assert artifact.overlay_layers[0].positions.size == 0
        assert artifact.overlay_layers[1].positions.size == 0

    def test_empty_ops_list(self):
        config = _identity_config()
        artifact = compile_scene([], config)
        assert len(artifact.vertex_layers) == 2
        assert len(artifact.overlay_layers) == 2


class TestCompileMultiStep:
    def test_flat_and_rotary_separated(self):
        ops_flat = Ops()
        ops_flat.move_to(0.0, 0.0, 0.0)
        ops_flat.set_power(1.0)
        ops_flat.line_to(5.0, 0.0, 0.0)

        ops_rot = Ops()
        ops_rot.move_to(0.0, 0.0, 0.0)
        ops_rot.set_power(1.0)
        ops_rot.line_to(5.0, 0.0, 0.0)

        config = _identity_config()

        flat_cfg = StepRenderConfig(
            rotary_enabled=False, rotary_diameter=0.0, laser_uid=""
        )
        rot_cfg = StepRenderConfig(
            rotary_enabled=True, rotary_diameter=50.0, laser_uid=""
        )

        artifact = compile_scene(
            [(ops_flat, flat_cfg), (ops_rot, rot_cfg)], config
        )

        assert len(artifact.vertex_layers) == 2

        flat_vl = artifact.vertex_layers[0]
        pv_flat = flat_vl.powered_verts.reshape(-1, 3)
        assert pv_flat[0, 2] == 0.0

        rot_vl = artifact.vertex_layers[1]
        pv_rot = rot_vl.powered_verts.reshape(-1, 3)
        assert abs(pv_rot[0, 2] - 25.0) < 1e-3


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

        config = _identity_config()
        step_cfg = config.step_configs["step1"]

        artifact = compile_scene([(ops, step_cfg)], config)

        assert len(artifact.overlay_layers) == 2
        ol = artifact.overlay_layers[0]

        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 4

        off = ol.cmd_offsets
        assert off[0] == 0
        assert off[-1] == 4
        assert len(off) == len(ops.commands) + 1

    def test_multi_step_cumulative_offsets(self):
        ops1 = Ops()
        ops1.move_to(0.0, 0.0, 0.0)
        powers1 = bytearray([128, 128])
        ops1.add(ScanLinePowerCommand((2.0, 0.0, 0.0), powers1))

        ops2 = Ops()
        ops2.move_to(0.0, 0.0, 0.0)
        powers2 = bytearray([64])
        ops2.add(ScanLinePowerCommand((1.0, 0.0, 0.0), powers2))

        config = _identity_config()
        flat_cfg = StepRenderConfig(
            rotary_enabled=False, rotary_diameter=0.0, laser_uid=""
        )

        artifact = compile_scene([(ops1, flat_cfg), (ops2, flat_cfg)], config)

        assert len(artifact.overlay_layers) == 2
        ol = artifact.overlay_layers[0]
        off = ol.cmd_offsets
        assert off[0] == 0
        assert off[-1] == 4

        ov_pos = ol.positions.reshape(-1, 3)
        assert ov_pos.shape[0] == 4
