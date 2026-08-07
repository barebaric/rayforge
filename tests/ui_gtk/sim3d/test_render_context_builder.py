"""UI tests for the RenderContextBuilder."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from raygeo.ops.axis import Axis

from rayforge.core.color import ColorSet
from rayforge.ui_gtk.sim3d.camera import Camera
from rayforge.ui_gtk.sim3d.gl_utils import RenderContext
from rayforge.ui_gtk.sim3d.render_context_builder import RenderContextBuilder
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


def _build(
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
    builder = RenderContextBuilder()
    ctx = builder.build(
        camera=camera or _make_camera(),
        viewport=viewport or ViewportConfig.default(),
        color_set=ColorSet(),
        scene=scene or _make_scene(),
        op_player=op_player,
        compiled_artifact=MagicMock(),
        doc=MagicMock(),
        machine=machine,
        show_travel_moves=show_travel_moves,
        show_grid=show_grid,
        show_nogo_zones=show_nogo_zones,
        show_models=show_models,
    )
    return builder, ctx


@pytest.mark.ui
def test_build_returns_render_context():
    _, ctx = _build()
    assert isinstance(ctx, RenderContext)
    assert ctx.mvp_flat_gl is not None
    assert ctx.mvp_rot_gl is not None
    assert ctx.cyl_mesh_mvp_gl is not None
    assert ctx.rot_4x4 is not None
    for m in (ctx.mvp_flat_gl, ctx.mvp_rot_gl, ctx.cyl_mesh_mvp_gl):
        assert m.shape == (4, 4)


@pytest.mark.ui
def test_build_flat_branch_matches_ui_mvp():
    _, ctx = _build(viewport=ViewportConfig.default())
    assert ctx.rotary_axis is None
    assert ctx.had_rotary_layers is False
    mvp_flat = ctx.mvp_flat_gl
    mvp_rot = ctx.mvp_rot_gl
    cyl_mesh = ctx.cyl_mesh_mvp_gl
    rot = ctx.rot_4x4
    assert mvp_flat is not None
    assert mvp_rot is not None
    assert cyl_mesh is not None
    assert rot is not None
    assert np.allclose(mvp_rot, mvp_flat)
    assert np.allclose(cyl_mesh, mvp_flat)
    assert np.allclose(rot, np.eye(4))


@pytest.mark.ui
def test_build_rotary_branch_applies_rotation():
    machine = MagicMock()
    machine.get_default_laser_head.return_value = None
    machine.assembly = MagicMock()
    machine.assembly.has_rotary = True
    op_player = MagicMock()
    op_player.rotary_axis = Axis.A
    op_player.state.axes = {Axis.A: 90.0}

    scene = _make_scene(had_rotary_layers=True)
    _, ctx = _build(scene=scene, op_player=op_player, machine=machine)

    assert ctx.rotary_axis is Axis.A
    assert ctx.had_rotary_layers is True
    rot = ctx.rot_4x4
    mvp_rot = ctx.mvp_rot_gl
    mvp_flat = ctx.mvp_flat_gl
    assert rot is not None
    assert mvp_rot is not None
    assert mvp_flat is not None
    assert not np.allclose(rot, np.eye(4))
    assert not np.allclose(mvp_rot, mvp_flat)


@pytest.mark.ui
def test_build_propagates_show_toggles():
    _, ctx = _build(
        show_travel_moves=True,
        show_grid=False,
        show_nogo_zones=False,
        show_models=False,
    )
    assert ctx.show_travel_moves is True
    assert ctx.show_grid is False
    assert ctx.show_nogo_zones is False
    assert ctx.show_models is False


@pytest.mark.ui
def test_build_line_width_at_least_two():
    _, ctx = _build()
    assert ctx.line_width >= 2.0


@pytest.mark.ui
def test_build_line_width_scales_with_laser_spot():
    big_machine = MagicMock()
    big_head = MagicMock()
    big_head.spot_size_mm = (50.0, 50.0)
    big_machine.get_default_laser_head.return_value = big_head

    small_machine = MagicMock()
    small_head = MagicMock()
    small_head.spot_size_mm = (0.01, 0.01)
    small_machine.get_default_laser_head.return_value = small_head

    _, big_ctx = _build(machine=big_machine)
    _, small_ctx = _build(machine=small_machine)
    assert big_ctx.line_width >= small_ctx.line_width
