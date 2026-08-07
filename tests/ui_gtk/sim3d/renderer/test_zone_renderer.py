"""
Tests for the ZoneRenderer resource lifecycle.
"""

from unittest.mock import patch

import pytest

from rayforge.machine.models.zone import Zone
from rayforge.ui_gtk.sim3d.renderer.zone_renderer import ZoneRenderer


@pytest.mark.ui
def test_update_zones_twice_then_cleanup_no_double_delete():
    """Repeated updates must untrack deleted GL objects so that cleanup
    does not delete live or recycled IDs twice."""
    renderer = ZoneRenderer()
    zones = [Zone()]

    vao_ids = iter(range(1, 1000))
    vbo_ids = iter(range(1, 1000))

    def gen_vaos(count):
        return next(vao_ids)

    def gen_vbos(count):
        return next(vbo_ids)

    deleted_vaos = []
    deleted_vbos = []

    def record_delete_vaos(count, ids):
        deleted_vaos.extend(ids)

    def record_delete_buffers(count, ids):
        deleted_vbos.extend(ids)

    with (
        patch("OpenGL.GL.glGenVertexArrays", side_effect=gen_vaos),
        patch("OpenGL.GL.glGenBuffers", side_effect=gen_vbos),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
        patch(
            "OpenGL.GL.glDeleteVertexArrays",
            side_effect=record_delete_vaos,
        ),
        patch(
            "OpenGL.GL.glDeleteBuffers",
            side_effect=record_delete_buffers,
        ),
    ):
        renderer.update_zones(zones)
        first_fill_vao, first_fill_vbo = (
            renderer._fill_vao,
            renderer._fill_vbo,
        )
        first_edge_vao, first_edge_vbo = (
            renderer._edge_vao,
            renderer._edge_vbo,
        )
        assert renderer._owned_vaos == [
            first_fill_vao,
            first_edge_vao,
        ]
        assert renderer._owned_vbos == [
            first_fill_vbo,
            first_edge_vbo,
        ]

        renderer.update_zones(zones)
        assert renderer._owned_vaos == [
            renderer._fill_vao,
            renderer._edge_vao,
        ]
        assert renderer._owned_vbos == [
            renderer._fill_vbo,
            renderer._edge_vbo,
        ]
        assert first_fill_vao not in renderer._owned_vaos
        assert first_fill_vbo not in renderer._owned_vbos
        assert first_edge_vao not in renderer._owned_vaos
        assert first_edge_vbo not in renderer._owned_vbos

        renderer.cleanup()

    assert sorted(deleted_vaos) == [1, 2, 3, 4]
    assert sorted(deleted_vbos) == [1, 2, 3, 4]
    assert renderer._owned_vaos == []
    assert renderer._owned_vbos == []
