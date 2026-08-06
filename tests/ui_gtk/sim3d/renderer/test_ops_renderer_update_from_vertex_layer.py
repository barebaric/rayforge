"""Tests for OpsRenderer.update_from_vertex_layer."""

from unittest.mock import patch

import numpy as np
import pytest

from rayforge.simulator.scene3d import VertexLayer
from rayforge.ui_gtk.sim3d.renderer.ops_renderer import OpsRenderer


def _make_vertex_layer(powered_count=2, zero_power_count=3, travel_count=4):
    return VertexLayer(
        powered_verts=np.zeros(powered_count * 3, dtype=np.float32),
        powered_attrib=np.zeros((powered_count, 4), dtype=np.float32),
        travel_verts=np.zeros(travel_count * 3, dtype=np.float32),
        zero_power_verts=np.zeros(zero_power_count * 3, dtype=np.float32),
        powered_cmd_offsets=[0, 1],
        travel_cmd_offsets=[2, 3],
        is_rotary=False,
    )


@pytest.fixture
def renderer():
    r = OpsRenderer()
    with (
        patch.object(r, "_create_vbo", return_value=1),
        patch.object(r, "_create_vao", return_value=1),
        patch.object(r, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
    ):
        r.init_gl()
    return r


@pytest.mark.ui
def test_update_from_vertex_layer_hides_travel(renderer):
    vl = _make_vertex_layer()

    with patch.object(renderer, "_load_buffer_data") as mock_load:
        renderer.update_from_vertex_layer(vl, show_travel_moves=False)

    assert renderer.powered_vertex_count == 2
    assert renderer.travel_vertex_count == 0

    attrib = mock_load.call_args_list[1].args[1]
    assert attrib.size == 2 * 4


@pytest.mark.ui
def test_update_from_vertex_layer_shows_travel(renderer):
    vl = _make_vertex_layer()

    with patch.object(renderer, "_load_buffer_data") as mock_load:
        renderer.update_from_vertex_layer(vl, show_travel_moves=True)

    assert renderer.powered_vertex_count == 2 + 3
    assert renderer.travel_vertex_count == 4

    attrib = mock_load.call_args_list[1].args[1]
    assert attrib.size == (2 + 3) * 4


@pytest.mark.ui
def test_update_from_vertex_layer_fills_zero_power_alpha(renderer):
    vl = _make_vertex_layer()

    with patch.object(renderer, "_load_buffer_data") as mock_load:
        renderer.update_from_vertex_layer(vl, show_travel_moves=True)

    attrib = mock_load.call_args_list[1].args[1]
    zero_attrib = attrib[2 * 4 :]
    assert zero_attrib.size == 3 * 4
    assert np.all(zero_attrib[3::4] == 1.0)
