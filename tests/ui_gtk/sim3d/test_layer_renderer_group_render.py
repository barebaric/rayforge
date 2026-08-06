"""Tests for LayerRendererGroup.render."""

from unittest.mock import MagicMock

import numpy as np

from rayforge.ui_gtk.sim3d.layer_renderer_group import LayerRendererGroup


class _FakePlayer:
    def __init__(self, current_index):
        self.current_index = current_index


def _make_group(
    powered_offsets,
    travel_offsets,
    ring_offsets,
    is_rotary=False,
    ring_vertex_count=0,
):
    group = LayerRendererGroup(is_rotary=is_rotary)
    ops = MagicMock()
    ring = MagicMock()
    ring.vertex_count = ring_vertex_count
    group.ops_renderer = ops
    group.ring_renderer = ring
    group.powered_offsets = powered_offsets
    group.travel_offsets = travel_offsets
    group.ring_offsets = ring_offsets
    return group, ops, ring


def _make_ctx(mvp_flat, mvp_rot):
    ctx = MagicMock()
    ctx.mvp_flat_gl = mvp_flat
    ctx.mvp_rot_gl = mvp_rot
    return ctx


def test_group_render_uses_rotary_mvp_and_deferred_ring():
    group, ops, ring = _make_group(
        [0, 10, 20], [0, 5], [0, 1, 2], is_rotary=True, ring_vertex_count=8
    )
    mvp_flat = np.zeros((4, 4), dtype=np.float32)
    mvp_rot = np.ones((4, 4), dtype=np.float32)
    ctx = _make_ctx(mvp_flat, mvp_rot)
    shaders = MagicMock()

    result = group.render(ctx, shaders, _FakePlayer(0))

    ops.render.assert_called_once()
    assert ops.render.call_args.args[0] is ctx
    assert ops.render.call_args.args[1] is shaders
    assert ctx.executed_vertex_count == 10
    assert ctx.executed_travel_vertex_count == 5

    assert result is not None
    assert result[0] is ring
    assert result[1] == 1


def test_group_render_flat_mvp_no_ring():
    group, ops, _ = _make_group([0, 10], [0], [0], is_rotary=False)
    mvp_flat = np.zeros((4, 4), dtype=np.float32)
    mvp_rot = np.ones((4, 4), dtype=np.float32)
    ctx = _make_ctx(mvp_flat, mvp_rot)

    result = group.render(ctx, MagicMock(), _FakePlayer(0))

    ops.render.assert_called_once()
    assert ctx.executed_vertex_count == 10
    assert result is None


def test_group_render_mid_and_last_index():
    offsets = [0, 100, 200, 300]
    group, ops, _ = _make_group(offsets, [0], [0])

    ctx = _make_ctx(None, None)
    group.render(ctx, MagicMock(), _FakePlayer(1))
    assert ctx.executed_vertex_count == 200

    ctx = _make_ctx(None, None)
    group.render(ctx, MagicMock(), _FakePlayer(2))
    assert ctx.executed_vertex_count == 300
