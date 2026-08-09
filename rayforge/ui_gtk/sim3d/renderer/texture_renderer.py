"""
A renderer for visualizing texture-based artifacts using GPU texture rendering.
"""

import logging
import time
from typing import Any

import numpy as np
from OpenGL import GL
from OpenGL.error import GLError

from ....pipeline.artifact.base import TextureData
from ....simulator.scene3d import CompiledSceneArtifact, TextureLayer
from ...shared.color_lut_provider import ColorLutProvider
from ..gl_utils import ShaderSet
from ..render_context import RenderContext
from .base import BaseRenderer

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
        self.instances: list[dict[str, Any]] = []
        self.cylinder_vao: int = 0
        self.cylinder_vbo: int = 0
        self._num_laser_luts: int = 1
        self._flat_mvp: np.ndarray | None = None
        self._cyl_mvp: np.ndarray | None = None

    def prepare(self, ctx: RenderContext) -> None:
        """Caches the per-frame MVP matrices for the texture quads."""
        self._flat_mvp = ctx.camera.mvp_ui
        self._cyl_mvp = ctx.kinematics.cylinder_mesh_mvp()

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
        except GLError as e:
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

    @staticmethod
    def _max_reduce(data: np.ndarray) -> np.ndarray:
        """Halves a power map taking the 2x2 block maximum."""
        h, w = data.shape
        nh, nw = (h + 1) // 2, (w + 1) // 2
        if h % 2 == 1:
            data = np.pad(data, ((0, 1), (0, 0)), mode="edge")
        if w % 2 == 1:
            data = np.pad(data, ((0, 0), (0, 1)), mode="edge")
        return data.reshape(nh, 2, nw, 2).max(axis=(1, 3))

    @classmethod
    def _build_mipmaps(cls, data: np.ndarray) -> list[np.ndarray]:
        """Builds a max-reduction mip pyramid of a power map.

        Down-sampling with the maximum (rather than the average) keeps
        the scanline structure intact at every zoom level, so the shader
        can pick a mip level that matches its texel footprint instead of
        aliasing between rows when the texture is minified.
        """
        mips = [data]
        level = data
        while min(level.shape) > 1:
            level = cls._max_reduce(level)
            mips.append(level)
        return mips

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
        rotary_enabled: bool = False,
        rotary_diameter: float = 25.0,
        cylinder_vertices: np.ndarray | None = None,
        laser_index: int = 0,
    ):
        """Adds a texture artifact to be rendered in the next frame."""
        if not self.is_initialized:
            return

        texture_id = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture_id)
        # NEAREST_MIPMAP_NEAREST keeps the texture mipmap complete so
        # texelFetch() can read the explicit LOD the shader computes
        # from the texel footprint (fixes minification moire banding).
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D,
            GL.GL_TEXTURE_MIN_FILTER,
            GL.GL_NEAREST_MIPMAP_NEAREST,
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

        mips = self._build_mipmaps(power_data)

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        for level, mip in enumerate(mips):
            mh, mw = mip.shape
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D,
                level,
                GL.GL_R8,
                mw,
                mh,
                0,
                GL.GL_RED,
                GL.GL_UNSIGNED_BYTE,
                mip,
            )
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 4)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

        instance_data = {
            "texture_id": texture_id,
            "model_matrix": final_model_matrix,
            "rotary_enabled": rotary_enabled,
            "rotary_diameter": rotary_diameter,
            "laser_index": laser_index,
            "max_mip": len(mips) - 1,
        }

        if rotary_enabled and cylinder_vertices is not None:
            instance_data["cylinder_vertices"] = cylinder_vertices

        self.instances.append(instance_data)

    def add_instance_from_texture_layer(
        self,
        tl: TextureLayer,
        laser_uid_order: list[str] | None = None,
    ):
        """Adds a texture instance from a compiled texture layer."""
        laser_index = 0
        if (
            tl.laser_uid
            and laser_uid_order
            and tl.laser_uid in laser_uid_order
        ):
            laser_index = laser_uid_order.index(tl.laser_uid)
        tex_data = TextureData(
            power_texture_data=tl.power_texture.to_numpy(),
            dimensions_mm=(0.0, 0.0),
            position_mm=(0.0, 0.0),
        )
        self.add_instance(
            tex_data,
            tl.model_matrix,
            rotary_enabled=tl.rotary_enabled,
            rotary_diameter=tl.rotary_diameter,
            cylinder_vertices=tl.cylinder_vertices,
            laser_index=laser_index,
        )

    def update_from_artifact(self, artifact: CompiledSceneArtifact):
        """Clears existing instances and uploads the artifact's layers."""
        self.clear()
        for tl in artifact.texture_layers:
            self.add_instance_from_texture_layer(tl, artifact.laser_uid_order)

    def update_color_lut(self, lut_data: np.ndarray, num_lasers: int = 1):
        """
        Updates the color lookup table texture, now using GL_TEXTURE_2D.
        """
        if not self.is_initialized:
            return

        self._num_laser_luts = num_lasers
        lut_data = np.ascontiguousarray(lut_data, dtype=np.float32)

        if lut_data.ndim == 3:
            width, height = lut_data.shape[1], lut_data.shape[0]
        else:
            width, height = lut_data.shape[0], 1

        GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_RGBA32F,
            width,
            height,
            0,
            GL.GL_RGBA,
            GL.GL_FLOAT,
            lut_data,
        )
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)

    def update_color_lut_from(self, provider: ColorLutProvider):
        """Updates the colour LUT from a shared ColorLutProvider."""
        self.update_color_lut(provider.engrave_lut_2d(), provider.num_lasers)

    def render(self, ctx: RenderContext, shaders: ShaderSet, **kwargs) -> None:
        """
        Renders all texture instances: flat quads first, then the
        cylinder-mapped (rotary) ones.

        Args:
            ctx: The current render context; carries the reached count.
            shaders: The shader set; the ``texture`` program is used.
        """
        if not self.is_initialized or not self.instances:
            return

        shader = shaders.texture
        if shader is None:
            return

        pending_alpha = 0.3
        self._draw_flat(shader, pending_alpha)
        self._draw_cylinder(shader, pending_alpha)

    def _draw_flat(
        self,
        shader,
        pending_alpha: float = 0.3,
    ):
        """Draws all flat (non-rotary) texture instances."""
        if self._flat_mvp is None:
            return

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        # Fill depth across the whole raster quad (including the
        # zero-power gaps) so occluders behind it cannot show through.
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)
        shader.set_int("uNumLaserLUTs", self._num_laser_luts)
        GL.glBindVertexArray(self.vao)

        for i, instance in enumerate(self.instances):
            if instance["rotary_enabled"]:
                continue

            shader.set_float("uAlpha", pending_alpha)
            shader.set_float("uMaxMip", float(instance.get("max_mip", 0)))

            shader.set_int("uLaserIndex", instance.get("laser_index", 0))

            final_mvp = self._flat_mvp @ instance["model_matrix"]
            shader.set_mat4("uMVP", final_mvp)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glDrawArrays(GL.GL_TRIANGLE_FAN, 0, 4)

    def _draw_cylinder(
        self,
        shader,
        pending_alpha: float = 0.3,
    ):
        """Draws all texture instances mapped onto a cylinder."""
        if self._cyl_mvp is None:
            return

        t_cyl_start = time.perf_counter()

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDepthFunc(GL.GL_LEQUAL)
        shader.use()

        GL.glActiveTexture(GL.GL_TEXTURE0)
        shader.set_int("uTexture", 0)
        GL.glActiveTexture(GL.GL_TEXTURE1)
        shader.set_int("uColorLUT", 1)
        shader.set_int("uNumLaserLUTs", self._num_laser_luts)

        num_rotary = 0

        for i, instance in enumerate(self.instances):
            if not instance["rotary_enabled"]:
                continue
            num_rotary += 1

            shader.set_float("uAlpha", pending_alpha)
            shader.set_float("uMaxMip", float(instance.get("max_mip", 0)))

            shader.set_int("uLaserIndex", instance.get("laser_index", 0))

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

            # Draw using the full Scene Matrix, so it correctly
            # inherits WCS and _model_matrix.
            shader.set_mat4("uMVP", self._cyl_mvp)

            GL.glActiveTexture(GL.GL_TEXTURE1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.color_lut_texture)
            GL.glActiveTexture(GL.GL_TEXTURE0)
            GL.glBindTexture(GL.GL_TEXTURE_2D, instance["texture_id"])

            GL.glBindVertexArray(self.cylinder_vao)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, vertex_count)

        t_cyl_elapsed = (time.perf_counter() - t_cyl_start) * 1000
        if t_cyl_elapsed > 5:
            logger.info(
                f"[TEX3D] render_cylinder took {t_cyl_elapsed:.1f}ms "
                f"(rotary={num_rotary})"
            )
