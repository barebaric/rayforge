"""
Tests for the TextureArtifactRenderer class.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from OpenGL import GL

from rayforge.core.color import ColorSet
from rayforge.pipeline.artifact.base import TextureData
from rayforge.ui_gtk.sim3d.gl_utils import ShaderSet
from rayforge.ui_gtk.sim3d.render_context import (
    CameraContext,
    KinematicsContext,
    RenderContext,
)
from rayforge.ui_gtk.sim3d.renderer.texture_renderer import (
    TextureArtifactRenderer,
)


def _make_ctx():
    return RenderContext(
        camera=CameraContext(color_set=ColorSet()),
        kinematics=KinematicsContext(mvp_ui=np.eye(4, dtype=np.float32)),
    )


def _init_renderer(renderer):
    with (
        patch.object(renderer, "_create_vbo", return_value=1),
        patch.object(renderer, "_create_vao", return_value=1),
        patch.object(renderer, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
        patch("OpenGL.GL.glTexParameteri"),
        patch("OpenGL.GL.glPixelStorei"),
        patch("OpenGL.GL.glTexImage2D"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glGenTextures", side_effect=[1, 2]),
        patch(
            "OpenGL.GL.glGetIntegerv",
            side_effect=lambda name, out: setattr(out, "value", 8192),
        ),
    ):
        renderer.init_gl()
    return renderer


@pytest.mark.ui
def test_build_mipmaps_max_reduces():
    """The power-map mip pyramid must down-sample with the 2x2 maximum
    so scanline rows survive minification instead of aliasing away."""
    data = np.arange(25, dtype=np.uint8).reshape(5, 5) % 7
    mips = TextureArtifactRenderer._build_mipmaps(data)
    # Level sizes follow the GL floor-halving rule (max(1, size // 2)),
    # not ceil: an inconsistent chain would leave the texture incomplete
    # and every lookup black.
    assert [m.shape for m in mips] == [(5, 5), (2, 2), (1, 1)]
    for level in range(1, len(mips)):
        prev = mips[level - 1]
        cur = mips[level]
        for i in range(cur.shape[0]):
            for j in range(cur.shape[1]):
                block = prev[2 * i : 2 * i + 2, 2 * j : 2 * j + 2]
                assert cur[i, j] == block.max()


@pytest.mark.ui
def test_render_writes_depth_for_texture_quad():
    """The texture must write depth across its whole quad (including
    zero-power gaps) so occluders behind it cannot band the preview."""
    renderer = _init_renderer(TextureArtifactRenderer())
    renderer.prepare(_make_ctx())

    power_texture = np.full((4, 4), 128, dtype=np.uint8)
    with (
        patch("OpenGL.GL.glGenTextures", return_value=1),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glTexParameteri"),
        patch("OpenGL.GL.glPixelStorei"),
        patch("OpenGL.GL.glTexImage2D"),
    ):
        renderer.add_instance(
            TextureData(
                power_texture_data=power_texture,
                dimensions_mm=(10.0, 10.0),
                position_mm=(0.0, 0.0),
            ),
            np.eye(4, dtype=np.float32),
        )

    shader = MagicMock()
    with (
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glEnable"),
        patch("OpenGL.GL.glBlendFunc"),
        patch("OpenGL.GL.glDepthMask") as mock_depth_mask,
        patch("OpenGL.GL.glDepthFunc") as mock_depth_func,
        patch("OpenGL.GL.glActiveTexture"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glDrawArrays"),
    ):
        renderer.render(_make_ctx(), ShaderSet(texture=shader))

    assert GL.GL_TRUE in [c.args[0] for c in mock_depth_mask.call_args_list]
    assert GL.GL_LEQUAL in [c.args[0] for c in mock_depth_func.call_args_list]


@pytest.mark.ui
def test_texture_coordinates_orientation():
    """
    Test that texture coordinates are set up correctly.

    The texture coordinates use OpenGL's standard convention with T=1.0 at
    the bottom and T=0.0 at the top. The actual fix for Y displacement is
    in the depth.py file where the texture data is filled with flipped Y
    coordinates to match the flipped Y coordinates used in the operations.
    """
    # Define the expected quad vertices (original orientation)
    # fmt: off
    expected_vertices = np.array(
        [
            # Position (x, y, z)  Texture Coords (s, t)
            0.0, 0.0, 0.0, 0.0, 1.0,  # Bottom-left
            1.0, 0.0, 0.0, 1.0, 1.0,  # Bottom-right
            1.0, 1.0, 0.0, 1.0, 0.0,  # Top-right
            0.0, 1.0, 0.0, 0.0, 0.0,  # Top-left
        ],
        dtype=np.float32,
    )
    # fmt: on

    # Check texture coordinates (last 2 attributes of each vertex)
    # Bottom-left vertex (index 0): should have T=1.0
    assert expected_vertices[4] == 1.0, (
        f"Bottom-left T coordinate should be 1.0, got {expected_vertices[4]}"
    )

    # Top-left vertex (index 3): should have T=0.0
    assert expected_vertices[19] == 0.0, (
        f"Top-left T coordinate should be 0.0, got {expected_vertices[19]}"
    )

    # Bottom-right vertex (index 1): should have T=1.0
    assert expected_vertices[9] == 1.0, (
        f"Bottom-right T coordinate should be 1.0, got {expected_vertices[9]}"
    )

    # Top-right vertex (index 2): should have T=0.0
    assert expected_vertices[14] == 0.0, (
        f"Top-right T coordinate should be 0.0, got {expected_vertices[14]}"
    )


@pytest.mark.ui
def test_add_instance_with_different_sizes():
    """
    Test that texture instances are added correctly with different transforms.

    This test verifies that the renderer correctly stores the pre-computed
    world transformation matrix passed to it. The renderer itself does not
    perform scaling based on artifact dimensions; it relies on the caller
    to provide the final model matrix for the unit quad.
    """
    # Create a dummy 1x1 power texture, its content doesn't matter for
    # this test
    power_texture = np.zeros((1, 1), dtype=np.uint8)

    # Create a renderer instance
    renderer = TextureArtifactRenderer()

    # Mock all OpenGL calls
    with (
        patch.object(renderer, "_create_vbo", return_value=1),
        patch.object(renderer, "_create_vao", return_value=1),
        patch.object(renderer, "_create_texture", return_value=1),
        patch("OpenGL.GL.glBindBuffer"),
        patch("OpenGL.GL.glBufferData"),
        patch("OpenGL.GL.glBindVertexArray"),
        patch("OpenGL.GL.glVertexAttribPointer"),
        patch("OpenGL.GL.glEnableVertexAttribArray"),
        patch("OpenGL.GL.glTexParameteri"),
        patch("OpenGL.GL.glPixelStorei"),
        patch("OpenGL.GL.glTexImage2D"),
        patch("OpenGL.GL.glBindTexture"),
        patch("OpenGL.GL.glGenTextures", side_effect=[1, 2]),
        patch(
            "OpenGL.GL.glGetIntegerv",
            side_effect=lambda name, out: setattr(out, "value", 8192),
        ),
    ):
        # Initialize the renderer
        renderer.init_gl()

        # --- Test Case 1: A 100x100mm artifact at (10, 20) ---
        texture_data_1 = TextureData(
            power_texture_data=power_texture,
            dimensions_mm=(100.0, 100.0),
            position_mm=(10.0, 20.0),
        )
        # The caller is responsible for creating this matrix (T * S).
        scale_mat1 = np.diag([100.0, 100.0, 1.0, 1.0])
        translate_mat1 = np.identity(4)
        translate_mat1[:3, 3] = [10.0, 20.0, 0.0]
        world_transform_1 = (translate_mat1 @ scale_mat1).astype(np.float32)

        renderer.add_instance(texture_data_1, world_transform_1)

        # Check that an instance was added
        assert len(renderer.instances) == 1
        # Check that the stored model matrix is the one we passed in
        stored_matrix_1 = renderer.instances[0]["model_matrix"]
        assert np.allclose(stored_matrix_1, world_transform_1), (
            "Stored matrix does not match provided matrix for instance 1."
        )

        # --- Test Case 2: A 200x50mm artifact at (0, 0) ---
        texture_data_2 = TextureData(
            power_texture_data=power_texture,
            dimensions_mm=(200.0, 50.0),
            position_mm=(0.0, 0.0),
        )
        # Create the corresponding world transform matrix (just scaling).
        world_transform_2 = np.diag([200.0, 50.0, 1.0, 1.0]).astype(np.float32)

        renderer.add_instance(texture_data_2, world_transform_2)

        # Check that a second instance was added
        assert len(renderer.instances) == 2
        # Check the model matrix for the large artifact
        stored_matrix_2 = renderer.instances[1]["model_matrix"]
        assert np.allclose(stored_matrix_2, world_transform_2), (
            "Stored matrix does not match provided matrix for instance 2."
        )
