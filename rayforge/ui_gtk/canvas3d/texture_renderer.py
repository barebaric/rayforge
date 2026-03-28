"""
A renderer for visualizing texture-based artifacts using GPU texture rendering.
"""

import logging
import math
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

    def _generate_cylinder_vertices(
        self,
        length: float,
        angle_start: float,
        angle_end: float,
        diameter: float,
        segments: int = 32,
    ) -> np.ndarray:
        """
        Generate vertices for a cylinder segment.

        Args:
            length: Length along the X-axis (cylinder axis).
            angle_start: Start angle in radians.
            angle_end: End angle in radians.
            diameter: Cylinder diameter.
            segments: Number of segments around the arc.

        Returns:
            Array of vertices with position (x, y, z) and tex coords (s, t).
        """
        radius = diameter / 2.0
        vertices = []

        for i in range(segments):
            t0 = i / segments
            t1 = (i + 1) / segments

            theta0 = angle_start + t0 * (angle_end - angle_start)
            theta1 = angle_start + t1 * (angle_end - angle_start)

            y0 = radius * math.sin(theta0)
            z0 = radius * math.cos(theta0)
            y1 = radius * math.sin(theta1)
            z1 = radius * math.cos(theta1)

            s0 = 1.0 - t0
            s1 = 1.0 - t1

            vertices.extend(
                [
                    0.0,
                    y0,
                    z0,
                    0.0,
                    s0,
                    length,
                    y0,
                    z0,
                    1.0,
                    s0,
                    length,
                    y1,
                    z1,
                    1.0,
                    s1,
                    0.0,
                    y0,
                    z0,
                    0.0,
                    s0,
                    length,
                    y1,
                    z1,
                    1.0,
                    s1,
                    0.0,
                    y1,
                    z1,
                    0.0,
                    s1,
                ]
            )

        return np.array(vertices, dtype=np.float32)

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

        self.instances.append(
            {
                "texture_id": texture_id,
                "model_matrix": final_model_matrix,
                "color_lut": color_lut,
            }
        )

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
        Renders all texture instances.

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
        rotary_diameter: float,
        wcs_x_offset: float = 0.0,
        wcs_y_offset: float = 0.0,
    ):
        """
        Renders all texture instances mapped onto a cylinder.

        Args:
            view_proj_scene_matrix: The combined Projection * View * SceneModel
              matrix (P*V*M_scene), not transposed.
            shader: The shader to use for rendering.
            rotary_diameter: The diameter of the cylinder in mm.
            wcs_x_offset: The X offset of the WCS origin in workarea coords.
            wcs_y_offset: The Y offset of the WCS origin in workarea coords.
        """
        if not self.is_initialized or not self.instances:
            return

        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)

        for instance in self.instances:
            model_matrix = instance["model_matrix"]

            origin = model_matrix @ np.array([0.0, 0.0, 0.0, 1.0])
            corner_x = model_matrix @ np.array([1.0, 0.0, 0.0, 1.0])
            corner_y = model_matrix @ np.array([0.0, 1.0, 0.0, 1.0])

            tex_width_mm = corner_x[0] - origin[0]
            tex_height_mm = corner_y[1] - origin[1]
            tex_x_offset = origin[0] - wcs_x_offset
            tex_y_offset = origin[1] - wcs_y_offset

            logger.debug(
                f"Texture cylinder: width={tex_width_mm}, "
                f"height={tex_height_mm}, x_off={tex_x_offset}, "
                f"y_off={tex_y_offset}"
            )

            circumference = rotary_diameter * math.pi
            angle_start = (tex_y_offset / circumference) * 2.0 * math.pi
            angle_end = (
                ((tex_y_offset + tex_height_mm) / circumference)
                * 2.0
                * math.pi
            )

            cache_key = (tex_width_mm, angle_start, angle_end, rotary_diameter)
            if (
                self._cylinder_cache is None
                or self._cylinder_cache["key"] != cache_key
            ):
                vertices = self._generate_cylinder_vertices(
                    length=tex_width_mm,
                    angle_start=angle_start,
                    angle_end=angle_end,
                    diameter=rotary_diameter,
                    segments=48,
                )
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

                self._cylinder_cache = {
                    "key": cache_key,
                    "vertex_count": len(vertices) // 5,
                }

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

            translation = np.identity(4, dtype=np.float32)
            translation[0, 3] = tex_x_offset

            final_mvp = view_proj_scene_matrix @ translation
            shader.set_mat4("uMVP", final_mvp.T)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glBindVertexArray(self.cylinder_vao)
            GL.glDrawArrays(
                GL.GL_TRIANGLES, 0, self._cylinder_cache["vertex_count"]
            )
            GL.glBindVertexArray(0)

        GL.glActiveTexture(GL.GL_TEXTURE1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
