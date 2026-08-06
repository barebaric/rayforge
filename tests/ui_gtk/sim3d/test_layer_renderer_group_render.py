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


def test_group_render_uses_rotary_mvp_and_deferred_ring():
    group, ops, ring = _make_group(
        [0, 10, 20], [0, 5], [0, 1, 2], is_rotary=True, ring_vertex_count=8
    )
    mvp_flat = np.zeros((4, 4), dtype=np.float32)
    mvp_rot = np.ones((4, 4), dtype=np.float32)

    result = group.render(
        MagicMock(), MagicMock(), _FakePlayer(0), mvp_flat, mvp_rot
    )

    ops.render.assert_called_once()
    args, kwargs = ops.render.call_args
    assert args[2] is mvp_rot
    assert kwargs["executed_vertex_count"] == 10
    assert kwargs["executed_travel_vertex_count"] == 5

    assert result is not None
    assert result[0] is ring
    assert result[2] == 1


def test_group_render_flat_mvp_no_ring():
    group, ops, _ = _make_group([0, 10], [0], [0], is_rotary=False)
    mvp_flat = np.zeros((4, 4), dtype=np.float32)
    mvp_rot = np.ones((4, 4), dtype=np.float32)

    result = group.render(
        MagicMock(), MagicMock(), _FakePlayer(0), mvp_flat, mvp_rot
    )

    args, _ = ops.render.call_args
    assert args[2] is mvp_flat
    assert result is None


def test_group_render_mid_and_last_index():
    offsets = [0, 100, 200, 300]
    group, ops, _ = _make_group(offsets, [0], [0])

    group.render(MagicMock(), MagicMock(), _FakePlayer(1), None, None)
    _, kwargs = ops.render.call_args
    assert kwargs["executed_vertex_count"] == 200

    group.render(MagicMock(), MagicMock(), _FakePlayer(2), None, None)
    _, kwargs = ops.render.call_args
    assert kwargs["executed_vertex_count"] == 300
