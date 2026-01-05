"""
Tests for the TextureArtifactRenderer class.
"""

import numpy as np
from unittest.mock import patch
from rayforge.ui_gtk.canvas3d.texture_renderer import (
    TextureArtifactRenderer,
)
from rayforge.pipeline.artifact.base import TextureData


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
