"""
A renderer for visualizing texture-based artifacts using GPU texture rendering.
"""

import logging
import math
import time
from typing import List, Dict, Any, Optional
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from ...pipeline.artifact.base import TextureData


logger = logging.getLogger(__name__)


class TextureArtifactRenderer(BaseRenderer):
    """
    Renders texture-based artifacts as textured quads for high-performance
    visualization.

    This renderer uses a single quad with a texture containing power values,
    allowing for instant rendering of complex raster operations that would
    otherwise require millions of individual lines.
    """

    def __init__(self):
        """Initializes the TextureArtifactRenderer."""
        super().__init__()
        self.vao: int = 0
        self.vbo: int = 0
        self.texture: int = 0
        self.color_lut_texture: int = 0
        self.is_initialized: bool = False
        self.max_texture_size: int = 0
        self.instances: List[Dict[str, Any]] = []
        self.cylinder_vao: int = 0
        self.cylinder_vbo: int = 0
        self._cylinder_cache: Optional[Dict[str, Any]] = None

    def init_gl(self):
        """
        Initializes OpenGL resources for rendering textured quads.

        Creates the VAO/VBO for a quad and OpenGL Textures for the texture
        data and color lookup table (LUT).
        """
        if self.is_initialized:
            return

        max_size = GL.GLint()
        GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE, max_size)
        self.max_texture_size = max_size.value
        logger.debug(f"OpenGL max texture size: {self.max_texture_size}")

        self.vbo = self._create_vbo()
        self.vao = self._create_vao()
        self.texture = self._create_texture()
        self.color_lut_texture = self._create_texture()

        # Define quad vertices (position, texture coordinates)
        # fmt: off
        quad_vertices = np.array(
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

        # Upload vertex data
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            quad_vertices.nbytes,
            quad_vertices,
            GL.GL_STATIC_DRAW,
        )

        # Set up vertex attributes
        GL.glBindVertexArray(self.vao)

        # Position attribute (location 0)
        GL.glVertexAttribPointer(
            0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(0)
        )
        GL.glEnableVertexAttribArray(0)

        # Texture coordinate attribute (location 1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(3 * 4)
        )
        GL.glEnableVertexAttribArray(1)

        GL.glBindVertexArray(0)

        # Set up 2D texture for power data
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        # Set up 2D texture (with height=1) for color LUT for compatibility
        # with the sampler2D in the shader.
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        self.cylinder_vbo = self._create_vbo()
        self.cylinder_vao = self._create_vao()

        self.is_initialized = True
        logger.debug("TextureArtifactRenderer initialized")

    def _cleanup_self(self):
        """Cleans up OpenGL resources specific to this renderer."""
        if not self.is_initialized:
            return

        try:
            self.clear()
            self.is_initialized = False
        except Exception as e:
            logger.warning(f"TextureArtifactRenderer cleanup warning: {e}")

    def _downsample_texture(
        self, data: np.ndarray, new_height: int, new_width: int
    ) -> np.ndarray:
        """Downsamples texture data using nearest-neighbor sampling."""
        h, w = data.shape
        y_step = h / new_height
        x_step = w / new_width
        y_coords = (np.arange(new_height) * y_step).astype(int)
        x_coords = (np.arange(new_width) * x_step).astype(int)
        return data[y_coords][:, x_coords].astype(np.uint8)

    def _generate_cylinder_vertices_from_matrix(
        self,
        grid_matrix: np.ndarray,
        diameter: float,
        grid_s: int = 8,
        grid_t: int = 64,
    ) -> np.ndarray:
        """
        Generate vertices for a cylinder segment directly from
        a transform matrix.

        The grid_matrix transforms [0,1] texture coords to
        Local Cylinder Space. The resulting Y coordinate is
        then natively wrapped into the cylinder Z/Y planes.
        """
        radius = diameter / 2.0
        circumference = diameter * math.pi

        i_vals = np.arange(grid_s, dtype=np.float32)
        j_vals = np.arange(grid_t, dtype=np.float32)
        lx0 = i_vals / grid_s
        lx1 = (i_vals + 1) / grid_s
        ly0 = j_vals / grid_t
        ly1 = (j_vals + 1) / grid_t

        lx_grid, ly_grid = np.meshgrid(lx0, ly1, indexing="ij")
        lx1_grid, _ = np.meshgrid(lx1, ly1, indexing="ij")
        _, ly0_grid = np.meshgrid(lx0, ly0, indexing="ij")

        def _transform_points(lx, ly):
            shape = lx.shape
            ones = np.ones(shape, dtype=np.float32)
            pts = np.stack(
                [
                    lx.ravel(),
                    ly.ravel(),
                    np.zeros(lx.size, dtype=np.float32),
                    ones.ravel(),
                ],
                axis=-1,
            )
            p_cyl = pts @ grid_matrix.T
            theta = (p_cyl[:, 1] / circumference) * 2.0 * np.pi
            x = p_cyl[:, 0].reshape(shape)
            y = (radius * np.sin(theta)).reshape(shape)
            z = (radius * np.cos(theta)).reshape(shape)
            return x, y, z

        x00, y00, z00 = _transform_points(lx_grid, ly0_grid)
        x10, y10, z10 = _transform_points(lx1_grid, ly0_grid)
        x01, y01, z01 = _transform_points(lx_grid, ly_grid)
        x11, y11, z11 = _transform_points(lx1_grid, ly_grid)

        s0 = lx_grid
        s1 = lx1_grid
        t0 = 1.0 - ly0_grid
        t1 = 1.0 - ly_grid

        s0_f = s0.flatten()
        s1_f = s1.flatten()
        t0_f = t0.flatten()
        t1_f = t1.flatten()
        x00_f, y00_f, z00_f = x00.flatten(), y00.flatten(), z00.flatten()
        x10_f, y10_f, z10_f = x10.flatten(), y10.flatten(), z10.flatten()
        x01_f, y01_f, z01_f = x01.flatten(), y01.flatten(), z01.flatten()
        x11_f, y11_f, z11_f = x11.flatten(), y11.flatten(), z11.flatten()

        n = s0_f.size
        vertices = np.empty(n * 30, dtype=np.float32)

        # Triangle 1: (x00,y00,z00,s0,t0), (x10,y10,z10,s1,t0),
        #              (x01,y01,z01,s0,t1)
        vertices[0::30] = x00_f
        vertices[1::30] = y00_f
        vertices[2::30] = z00_f
        vertices[3::30] = s0_f
        vertices[4::30] = t0_f
        vertices[5::30] = x10_f
        vertices[6::30] = y10_f
        vertices[7::30] = z10_f
        vertices[8::30] = s1_f
        vertices[9::30] = t0_f
        vertices[10::30] = x01_f
        vertices[11::30] = y01_f
        vertices[12::30] = z01_f
        vertices[13::30] = s0_f
        vertices[14::30] = t1_f

        # Triangle 2: (x10,y10,z10,s1,t0), (x11,y11,z11,s1,t1),
        #              (x01,y01,z01,s0,t1)
        vertices[15::30] = x10_f
        vertices[16::30] = y10_f
        vertices[17::30] = z10_f
        vertices[18::30] = s1_f
        vertices[19::30] = t0_f
        vertices[20::30] = x11_f
        vertices[21::30] = y11_f
        vertices[22::30] = z11_f
        vertices[23::30] = s1_f
        vertices[24::30] = t1_f
        vertices[25::30] = x01_f
        vertices[26::30] = y01_f
        vertices[27::30] = z01_f
        vertices[28::30] = s0_f
        vertices[29::30] = t1_f

        return vertices

    def clear(self):
        """Clears all instances and their associated textures."""
        if not self.is_initialized:
            return
        # This needs to be called on the GL thread.
        textures_to_delete = [
            instance["texture_id"] for instance in self.instances
        ]
        if textures_to_delete:
            GL.glDeleteTextures(textures_to_delete)
        self.instances.clear()

    def add_instance(
        self,
        texture_data: TextureData,
        final_model_matrix: np.ndarray,
        color_lut: Optional[np.ndarray] = None,
        rotary_enabled: bool = False,
        rotary_diameter: float = 25.0,
    ):
        """Adds a texture artifact to be rendered in the next frame."""
        if not self.is_initialized:
            return

        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )

        height, width = texture_data.power_texture_data.shape

        if width > self.max_texture_size or height > self.max_texture_size:
            scale = min(
                self.max_texture_size / width,
                self.max_texture_size / height,
            )
            new_width = int(width * scale)
            new_height = int(height * scale)
            logger.warning(
                f"Texture size {width}x{height} exceeds max "
                f"{self.max_texture_size}, downsampling to "
                f"{new_width}x{new_height}"
            )
            power_data = self._downsample_texture(
                texture_data.power_texture_data, new_height, new_width
            )
            height, width = new_height, new_width
        else:
            power_data = texture_data.power_texture_data

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R8,
            width,
            height,
            0,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            power_data,
        )
        # Restore the default alignment.
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        instance_data = {
            "texture_id": texture_id,
            "model_matrix": final_model_matrix,
            "color_lut": color_lut,
            "rotary_enabled": rotary_enabled,
            "rotary_diameter": rotary_diameter,
        }

        if rotary_enabled and rotary_diameter > 0:
            instance_data["cylinder_vertices"] = (
                self._generate_cylinder_vertices_from_matrix(
                    grid_matrix=final_model_matrix,
                    diameter=rotary_diameter,
                    grid_s=8,
                    grid_t=64,
                )
            )

        self.instances.append(instance_data)

    def update_color_lut(self, lut_data: np.ndarray):
        """
        Updates the color lookup table texture, now using GL_TEXTURE_2D.
        """
        if not self.is_initialized:
            return

        lut_data = np.ascontiguousarray(lut_data, dtype=np.float32)

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            lut_data.shape[0],  # width = 256
            1,  # height = 1
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            lut_data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render(self, view_proj_scene_matrix: np.ndarray, shader: Shader):
        """
        Renders all flat (non-rotary) texture instances.

        Args:
            view_proj_scene_matrix: The combined Projection * View * SceneModel
              matrix (P*V*M_scene), not transposed.
            shader: The shader to use for rendering.
        """
        if not self.is_initialized or not self.instances:
            return

        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)
        GL.glBindVertexArray(self.vao)

        for instance in self.instances:
            if instance["rotary_enabled"]:
                continue

            final_mvp = view_proj_scene_matrix @ instance["model_matrix"]
            shader.set_mat4("uMVP", final_mvp.T)

            if instance["color_lut"] is not None:
                lut_data = np.ascontiguousarray(
                    instance["color_lut"], dtype=np.float32
                )
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA32F,
                    lut_data.shape[0],
                    1,
                    0,
                    GL.GL_RGBA,
                    GL.GL_FLOAT,
                    lut_data,
                )
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

        GL.glBindVertexArray(0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def render_cylinder(
        self,
        view_proj_scene_matrix: np.ndarray,
        shader: Shader,
    ):
        """
        Renders all texture instances mapped onto a cylinder.

        Args:
            view_proj_scene_matrix: The combined Projection * View * SceneModel
              matrix (P*V*M_scene), not transposed.
            shader: The shader to use for rendering.
        """
        if not self.is_initialized or not self.instances:
            return

        t_cyl_start = time.perf_counter()

        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)

        num_rotary = 0

        for instance in self.instances:
            if not instance["rotary_enabled"]:
                continue
            num_rotary += 1

            vertices = instance.get("cylinder_vertices")
            if vertices is None:
                continue

            vertex_count = len(vertices) // 5

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.cylinder_vbo)
            GL.glBufferData(
                GL.GL_ARRAY_BUFFER,
                vertices.nbytes,
                vertices,
                GL.GL_DYNAMIC_DRAW,
            )

            GL.glBindVertexArray(self.cylinder_vao)
            GL.glVertexAttribPointer(
                0, 3, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(0)
            )
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(
                1, 2, GL.GL_FLOAT, GL.GL_FALSE, 5 * 4, GL.GLvoidp(3 * 4)
            )
            GL.glEnableVertexAttribArray(1)
            GL.glBindVertexArray(0)

            if instance["color_lut"] is not None:
                lut_data = np.ascontiguousarray(
                    instance["color_lut"], dtype=np.float32
                )
                GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
                GL.glTexImage2D(
                    GL.GL_TEXTURE_2D,
                    0,
                    GL.GL_RGBA32F,
                    lut_data.shape[0],
                    1,
                    0,
                    GL.GL_RGBA,
                    GL.GL_FLOAT,
                    lut_data,
                )
                GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

            # Draw using the full Scene Matrix, so it correctly
            # inherits WCS and _model_matrix.
            shader.set_mat4("uMVP", view_proj_scene_matrix.T)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glBindVertexArray(self.cylinder_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, vertex_count)
            GL.glBindVertexArray(0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        t_cyl_elapsed = (time.perf_counter() - t_cyl_start) * 1000
        if t_cyl_elapsed > 5:
            logger.info(
                f"[TEX3D] render_cylinder took {t_cyl_elapsed:.1f}ms "
                f"(rotary={num_rotary})"
            )
