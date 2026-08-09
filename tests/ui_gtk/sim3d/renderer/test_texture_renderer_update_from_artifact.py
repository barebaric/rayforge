"""Tests for TextureArtifactRenderer texture-layer ingestion."""

from unittest.mock import patch

import numpy as np
import pytest
from raygeo.compressed_array import CompressedArray

from rayforge.simulator.scene3d import CompiledSceneArtifact, TextureLayer
from rayforge.ui_gtk.sim3d.renderer.texture_renderer import (
    TextureArtifactRenderer,
)


@pytest.fixture
def renderer():
    r = TextureArtifactRenderer()
    with (
        patch.object(r, "_create_vbo", return_value=1),
        patch.object(r, "_create_vao", return_value=1),
        patch.object(r, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
        patch("OpenGL.GL.glTexParameteri"),
        patch("OpenGL.GL.glPixelStorei"),
        patch("OpenGL.GL.glTexImage2D"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDeleteTextures"),
        patch("OpenGL.GL.glGenTextures", return_value=1),
        patch(
            "OpenGL.GL.glGetIntegerv",
            side_effect=lambda name, out: setattr(out, "value", 8192),
        ),
    ):
        r.init_gl()
        yield r


def _make_texture_layer(laser_uid=""):
    return TextureLayer(
        power_texture=CompressedArray.from_uint8_2d(
            np.zeros((2, 2), dtype=np.uint8)
        ),
        width_px=2,
        height_px=2,
        model_matrix=np.eye(4, dtype=np.float32),
        cylinder_vertices=None,
        rotary_diameter=0.0,
        rotary_enabled=False,
        activation_cmd_idx=-1,
        laser_uid=laser_uid,
    )


def _make_artifact(texture_layers, laser_uid_order=None):
    return CompiledSceneArtifact(
        generation_id=1,
        vertex_layers=[],
        texture_layers=texture_layers,
        overlay_layers=[],
        laser_uid_order=laser_uid_order,
    )


@pytest.mark.ui
def test_add_instance_from_texture_layer_laser_index(renderer):
    tl = _make_texture_layer(laser_uid="L1")
    renderer.add_instance_from_texture_layer(tl, ["L0", "L1", "L2"])

    assert len(renderer.instances) == 1
    assert renderer.instances[0]["laser_index"] == 1


@pytest.mark.ui
def test_add_instance_from_texture_layer_missing_laser_uid(renderer):
    renderer.add_instance_from_texture_layer(
        _make_texture_layer(laser_uid=""), ["L0", "L1"]
    )
    assert renderer.instances[0]["laser_index"] == 0

    renderer.add_instance_from_texture_layer(
        _make_texture_layer(laser_uid="L9"), ["L0", "L1"]
    )
    assert renderer.instances[1]["laser_index"] == 0


@pytest.mark.ui
def test_add_instance_from_texture_layer_rotary_flag(renderer):
    tl = _make_texture_layer()
    tl.rotary_enabled = True
    tl.rotary_diameter = 30.0
    tl.cylinder_vertices = np.zeros((5, 5), dtype=np.float32)
    renderer.add_instance_from_texture_layer(tl)

    inst = renderer.instances[0]
    assert inst["rotary_enabled"] is True
    assert inst["rotary_diameter"] == 30.0
    assert inst["cylinder_vertices"] is tl.cylinder_vertices


@pytest.mark.ui
def test_update_from_artifact_clears_and_uploads(renderer):
    tl_a = _make_texture_layer(laser_uid="A")
    tl_b = _make_texture_layer(laser_uid="B")
    renderer.add_instance_from_texture_layer(tl_a, ["A"])
    assert len(renderer.instances) == 1

    renderer.update_from_artifact(_make_artifact([tl_a, tl_b], ["A", "B"]))

    assert len(renderer.instances) == 2
    assert renderer.instances[0]["laser_index"] == 0
    assert renderer.instances[1]["laser_index"] == 1
