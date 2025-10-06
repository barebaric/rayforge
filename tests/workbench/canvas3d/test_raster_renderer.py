"""
Tests for the RasterArtifactRenderer class.
"""

import numpy as np
from unittest.mock import patch
from rayforge.workbench.canvas3d.raster_renderer import RasterArtifactRenderer
from rayforge.pipeline import CoordinateSystem
from rayforge.pipeline.producer.base import HybridRasterArtifact
from rayforge.core.ops.container import Ops


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
    Test that raster instances are added correctly with different workpiece
    sizes.

    This test verifies that the model matrix correctly scales the quad to
    match the artifact's dimensions, which should prevent Y displacement at
    different sizes.
    """
    # Create a simple 10x10 power texture with a diagonal line
    power_texture = np.zeros((10, 10), dtype=np.uint8)
    for i in range(10):
        power_texture[i, i] = 255  # Diagonal line at full power

    # Create a sample artifact with default size (100mm x 100mm)
    sample_artifact = HybridRasterArtifact(
        ops=Ops(),
        is_scalable=True,
        source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
        power_texture_data=power_texture,
        dimensions_mm=(100.0, 100.0),  # 100mm x 100mm
        position_mm=(0.0, 0.0),
    )

    # Create a renderer instance
    renderer = RasterArtifactRenderer()

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
    ):
        # Initialize the renderer
        renderer.init_gl()

        # Test with default size (100mm x 100mm)
        transform_matrix = np.identity(4, dtype=np.float32)

        with patch("OpenGL.GL.glGenTextures", return_value=1):
            renderer.add_instance(sample_artifact, transform_matrix)

            # Check that an instance was added
            assert len(renderer.instances) == 1

            # Check the model matrix
            instance = renderer.instances[0]
            model_matrix = instance["model_matrix"]

            # The model matrix should include the scaling from dimensions_mm
            # The local model matrix scales the unit quad to the artifact's
            # dimensions and positions it at the artifact's position

            # Check that the scaling is correct (100mm in both X and Y)
            assert np.isclose(model_matrix[0, 0], 100.0), (
                f"X scale should be 100.0, got {model_matrix[0, 0]}"
            )
            assert np.isclose(model_matrix[1, 1], 100.0), (
                f"Y scale should be 100.0, got {model_matrix[1, 1]}"
            )

            # Check that the position is correct (0, 0)
            assert np.isclose(model_matrix[0, 3], 0.0), (
                f"X position should be 0.0, got {model_matrix[0, 3]}"
            )
            assert np.isclose(model_matrix[1, 3], 0.0), (
                f"Y position should be 0.0, got {model_matrix[1, 3]}"
            )

        # Test with larger size (200mm x 200mm)
        large_artifact = HybridRasterArtifact(
            ops=Ops(),
            is_scalable=True,
            source_coordinate_system=CoordinateSystem.PIXEL_SPACE,
            power_texture_data=power_texture,
            dimensions_mm=(200.0, 200.0),  # 200mm x 200mm
            position_mm=(0.0, 0.0),
        )

        with patch("OpenGL.GL.glGenTextures", return_value=2):
            renderer.add_instance(large_artifact, transform_matrix)

            # Check that a second instance was added
            assert len(renderer.instances) == 2

            # Check the model matrix for the large artifact
            instance = renderer.instances[1]
            model_matrix = instance["model_matrix"]

            # Check that the scaling is correct (200mm in both X and Y)
            assert np.isclose(model_matrix[0, 0], 200.0), (
                f"X scale should be 200.0, got {model_matrix[0, 0]}"
            )
            assert np.isclose(model_matrix[1, 1], 200.0), (
                f"Y scale should be 200.0, got {model_matrix[1, 1]}"
            )
