"""Tests for LayerRendererGroup.update_from_artifact."""

from unittest.mock import MagicMock

import numpy as np

from rayforge.simulator.scene3d import ScanlineOverlayLayer, VertexLayer
from rayforge.ui_gtk.sim3d.layer_renderer_group import LayerRendererGroup


def _make_vl():
    return VertexLayer(
        powered_verts=np.zeros(6, dtype=np.float32),
        powered_attrib=np.zeros((2, 4), dtype=np.float32),
        travel_verts=np.zeros(6, dtype=np.float32),
        zero_power_verts=np.zeros(6, dtype=np.float32),
        powered_cmd_offsets=[1, 2],
        travel_cmd_offsets=[3, 4],
        is_rotary=False,
    )


def _make_ol():
    return ScanlineOverlayLayer(
        positions=np.zeros(6, dtype=np.float32),
        overlay_attrib=np.zeros(8, dtype=np.float32),
        cmd_offsets=[5, 6],
        is_rotary=False,
    )


def test_update_from_artifact_uploads_ops_and_overlay():
    group = LayerRendererGroup(is_rotary=False)
    group.ops_renderer = MagicMock()
    group.ring_renderer = MagicMock()

    vl = _make_vl()
    ol = _make_ol()
    group.update_from_artifact(vl, ol, show_travel_moves=True)

    group.ops_renderer.update_from_vertex_layer.assert_called_once_with(
        vl, True
    )
    group.ring_renderer.update_from_overlay_layer.assert_called_once_with(ol)
    assert group.powered_offsets == [1, 2]
    assert group.travel_offsets == [3, 4]
    assert group.ring_offsets == [5, 6]


def test_update_from_artifact_without_overlay_clears_ring():
    group = LayerRendererGroup(is_rotary=False)
    group.ops_renderer = MagicMock()
    group.ring_renderer = MagicMock()

    group.update_from_artifact(_make_vl(), None, show_travel_moves=False)

    group.ring_renderer.clear.assert_called_once_with()
    group.ring_renderer.update_from_overlay_layer.assert_not_called()
    assert group.ring_offsets == []
