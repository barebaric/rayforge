"""UI tests for RenderContext.update() and the per-section contexts."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from raygeo.ops.axis import Axis

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.camera import Camera
from rayforge.ui_gtk.sim3d.render_context import FrameInputs, RenderContext
from rayforge.ui_gtk.sim3d.viewport import ViewportConfig


def _make_camera(width=800, height=600):
    return Camera(
        position=np.array([0.0, 0.0, 100.0]),
        target=np.array([0.0, 0.0, 0.0]),
        up=np.array([0.0, 1.0, 0.0]),
        width=width,
        height=height,
    )


def _make_scene(had_rotary_layers=False):
    scene = MagicMock()
    scene.had_rotary_layers = had_rotary_layers
    scene.cylinder_transform = np.eye(4, dtype=np.float64)
    return scene


def _update(
    *,
    camera=None,
    viewport=None,
    scene=None,
    op_player=None,
    machine=None,
    show_travel_moves=False,
    show_grid=True,
    show_nogo_zones=True,
    show_models=True,
):
    frame = FrameInputs(
        camera=camera or _make_camera(),
        viewport=viewport or ViewportConfig.default(),
        color_set=ColorSet(),
        op_player=op_player,
        compiled_artifact=MagicMock(),
        doc=MagicMock(),
        machine=machine,
        cylinder_transform=(
            scene.cylinder_transform
            if scene
            else _make_scene().cylinder_transform
        ),
        had_rotary_layers=scene.had_rotary_layers if scene else False,
        show_travel_moves=show_travel_moves,
        show_grid=show_grid,
        show_nogo_zones=show_nogo_zones,
        show_models=show_models,
    )
    ctx = RenderContext()
    ctx.update(frame)
    return ctx


@pytest.mark.ui
def test_update_populates_render_context():
    ctx = _update()
    assert isinstance(ctx, RenderContext)
    assert ctx.camera.mvp_ui is not None
    assert ctx.kinematics.mvp_for(False) is not None
    assert ctx.kinematics.mvp_for(True) is not None
    for m in (
        ctx.camera.mvp_ui,
        ctx.kinematics.mvp_for(False),
        ctx.kinematics.mvp_for(True),
    ):
        assert m.shape == (4, 4)


@pytest.mark.ui
def test_update_flat_branch_matches_ui_mvp():
    ctx = _update(viewport=ViewportConfig.default())
    assert ctx.kinematics.rotary_axis is None
    assert ctx.kinematics.is_rotary is False
    mvp_flat = ctx.camera.mvp_ui
    mvp_rot = ctx.kinematics.mvp_for(True)
    assert mvp_flat is not None
    assert mvp_rot is not None
    np.testing.assert_allclose(mvp_rot, mvp_flat)
    assert ctx.kinematics.cylinder_mesh_mvp() is None


@pytest.mark.ui
def test_update_rotary_branch_applies_rotation():
    machine = MagicMock()
    machine.get_default_laser_head.return_value = None
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = True
    machine.assembly.model_world_transforms.return_value = {}
    machine.assembly.head_positions.return_value = {}
    machine.assembly.head_rotary_positions.return_value = {}
    op_player = MagicMock()
    op_player.rotary_axis = Axis.A
    op_player.state.axes = {Axis.A: 90.0}
    op_player.get_current_layer.return_value = None

    scene = _make_scene(had_rotary_layers=True)
    ctx = _update(scene=scene, op_player=op_player, machine=machine)

    assert ctx.kinematics.rotary_axis is Axis.A
    assert ctx.kinematics.is_rotary is True
    mvp_rot = ctx.kinematics.mvp_for(True)
    mvp_flat = ctx.camera.mvp_ui
    assert mvp_rot is not None
    assert mvp_flat is not None
    assert not np.allclose(mvp_rot, mvp_flat)


@pytest.mark.ui
def test_update_populates_machine_kinematics():
    machine = MagicMock()
    machine.get_default_laser_head.return_value = None
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = True
    machine.assembly.model_world_transforms.return_value = {
        "gantry": np.eye(4, dtype=np.float64)
    }
    machine.assembly.head_positions.return_value = {
        "head_0": (10.0, 20.0, 30.0)
    }
    machine.assembly.head_rotary_positions.return_value = {
        "head_0": np.array([0.0, 0.0, 25.0])
    }
    op_player = MagicMock()
    op_player.rotary_axis = Axis.A
    op_player.state.axes = {Axis.A: 90.0}
    current_layer = MagicMock()
    current_layer.rotary_diameter = 30.0
    op_player.get_current_layer.return_value = current_layer

    ctx = _update(
        op_player=op_player,
        machine=machine,
        scene=_make_scene(had_rotary_layers=True),
    )

    assert ctx.kinematics.has_rotary is True
    assert list(ctx.kinematics.model_world_transforms) == ["gantry"]
    assert ctx.kinematics.head_positions == {"head_0": (10.0, 20.0, 30.0)}
    assert set(ctx.kinematics.rotary_head_positions) == {"head_0"}
    assert set(ctx.kinematics.focused_rotary_head_positions) == {"head_0"}
    machine.assembly.head_positions.assert_called_once()
    machine.assembly.model_world_transforms.assert_called_once()


@pytest.mark.ui
def test_update_flat_kinematics_without_op_player():
    machine = MagicMock()
    machine.get_default_laser_head.return_value = None
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = False
    machine.assembly.model_world_transforms.return_value = {}
    machine.assembly.head_positions.return_value = {}

    ctx = _update(machine=machine, op_player=None)

    assert ctx.kinematics.has_rotary is False
    assert ctx.kinematics.model_world_transforms == {}
    assert ctx.kinematics.head_positions == {}
    assert ctx.kinematics.rotary_head_positions == {}
    machine.assembly.head_positions.assert_called_once()


@pytest.mark.ui
def test_update_skips_kinematics_without_machine():
    ctx = _update(machine=None, op_player=None)

    assert ctx.kinematics.is_rotary is False
    assert ctx.kinematics.model_world_transforms == {}
    assert ctx.kinematics.head_positions == {}
    assert ctx.kinematics.rotary_head_positions == {}
    assert ctx.kinematics.cylinder_mesh_mvp() is None


@pytest.mark.ui
def test_update_propagates_show_toggles():
    ctx = _update(
        show_travel_moves=True,
        show_grid=False,
        show_nogo_zones=False,
        show_models=False,
    )
    assert ctx.camera.show_travel_moves is True
    assert ctx.camera.show_grid is False
    assert ctx.camera.show_nogo_zones is False
    assert ctx.camera.show_models is False


@pytest.mark.ui
def test_update_line_width_at_least_two():
    ctx = _update()
    assert ctx.camera.line_width >= 2.0


@pytest.mark.ui
def test_update_line_width_scales_with_laser_spot():
    big_machine = MagicMock()
    big_head = MagicMock()
    big_head.spot_size_mm = (50.0, 50.0)
    big_machine.get_default_laser_head.return_value = big_head

    small_machine = MagicMock()
    small_head = MagicMock()
    small_head.spot_size_mm = (0.01, 0.01)
    small_machine.get_default_laser_head.return_value = small_head

    big_ctx = _update(machine=big_machine)
    small_ctx = _update(machine=small_machine)
    assert big_ctx.camera.line_width >= small_ctx.camera.line_width
