"""
Renders text in a 3D OpenGL scene.

This module provides a class `TextRenderer3D` for rendering text that faces
the camera (billboarding) in a 3D environment. It creates a texture atlas
from a specified font for the characters '0'-'9'.
"""

import logging
from typing import Dict, Optional, Tuple, Union
import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw, ImageFont
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class TextRenderer3D(BaseRenderer):
    """Renders billboarded text in a 3D scene."""

    def __init__(self, font_path: Optional[str] = None, font_size: int = 32):
        """
        Initializes the text renderer on the CPU.

        Args:
            font_path: Path to TTF font. If None, uses a system default.
            font_size: The size of the font for the texture atlas.
        """
        super().__init__()
        self.char_data: Dict[str, Dict[str, Union[float, int]]] = {}
        self.texture_id: int = 0
        self.atlas_width: int = 0
        self.atlas_height: int = 0
        self.vao: int = 0
        self.vbo: int = 0
        self._font_size: int = font_size
        self._atlas_image: Optional[Image.Image] = None

        try:
            # A common default font path for Debian/Ubuntu systems.
            if font_path is None:
                font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
            self.font = ImageFont.truetype(font_path, font_size)
            logger.info(f"Loaded font: {font_path}")
        except (IOError, TypeError):
            logger.warning(
                f"Font not found at '{font_path}'. Falling back to default."
            )
            self.font = ImageFont.load_default()

        self._prepare_texture_atlas()

    def _prepare_texture_atlas(self) -> None:
        """Creates a texture atlas for numeric characters on the CPU."""
        chars_to_render = "0123456789"
        total_width, max_height = 0, 0

        # First pass: calculate atlas dimensions.
        for char in chars_to_render:
            try:
                # Pillow >= 9.2.0 uses getbbox (left, top, right, bottom)
                bbox = self.font.getbbox(char)
                total_width += bbox[2] - bbox[0]
                max_height = max(max_height, bbox[3] - bbox[1])
            except AttributeError:
                # Fallback for Pillow < 9.2.0.
                width, height = self.font.getsize(char)  # type: ignore
                total_width += width
                max_height = max(max_height, height)

        self.atlas_width = int(total_width)
        self.atlas_height = int(max_height or self._font_size)

        atlas_img = Image.new("L", (self.atlas_width, self.atlas_height), 0)
        draw = ImageDraw.Draw(atlas_img)
        x_offset = 0

        # Second pass: draw characters and store rendering metadata.
        for char in chars_to_render:
            try:
                bbox = self.font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                y_offset = -bbox[1]  # Vertical offset to align baseline
            except AttributeError:
                char_width, char_height = self.font.getsize(  # type: ignore
                    char
                )
                y_offset = 0

            draw.text((x_offset, y_offset), char, font=self.font, fill=255)

            self.char_data[char] = {
                "u0": x_offset / self.atlas_width,
                "v0": 0,
                "u1": (x_offset + char_width) / self.atlas_width,
                "v1": char_height / self.atlas_height,
                "width": char_width,
                "height": char_height,
            }
            x_offset += char_width

        self._atlas_image = atlas_img

    def init_gl(self) -> None:
        """Initializes all OpenGL resources."""
        if self._atlas_image:
            self._upload_atlas_to_gpu()
            self._atlas_image = None  # Release CPU memory

        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 16 * 4, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glBindVertexArray(0)

    def _upload_atlas_to_gpu(self) -> None:
        """Helper to create and configure the OpenGL texture."""
        if not self._atlas_image:
            return
        self.texture_id = self._create_texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R8,
            self.atlas_width,
            self.atlas_height,
            0,
            GL.GL_RED,
            GL.GL_UNSIGNED_BYTE,
            self._atlas_image.tobytes(),
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR
        )
        GL.glTexParameteri(
            GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR
        )

    def render_text(
        self,
        shader: Shader,
        text: str,
        position: np.ndarray,
        scale: float,
        color: Tuple[float, float, float, float],
        mvp_matrix: np.ndarray,
        view_matrix: np.ndarray,
    ) -> None:
        """
        Renders a string of text at a given 3D position.

        Args:
            shader: The shader program to use for rendering text.
            text: The string to render (must contain '0'-'9').
            position: A numpy array (vec3) for the center of the text.
            scale: A float to scale the size of the text.
            color: A tuple (r, g, b, a) for the text color.
            mvp_matrix: The combined Model-View-Projection matrix.
            view_matrix: The camera's view matrix for billboarding.
        """
        if not self.vao:
            return

        shader.use()
        shader.set_vec4("uTextColor", color)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glBindVertexArray(self.vao)

        billboard_matrix = np.transpose(view_matrix[:3, :3])
        total_text_width = (
            sum(
                self.char_data[c]["width"] for c in text if c in self.char_data
            )
            * scale
        )
        current_x = -total_text_width / 2.0

        for char in text:
            if char not in self.char_data:
                continue

            char_info = self.char_data[char]
            char_width = char_info["width"] * scale
            char_height = char_info["height"] * scale
            char_center_offset = billboard_matrix @ np.array(
                [current_x + char_width / 2, char_height / 2, 0]
            )
            world_pos = position + char_center_offset

            shader.set_mat4("uMVP", mvp_matrix)
            shader.set_mat3("uBillboard", billboard_matrix)
            shader.set_vec3("uTextWorldPos", world_pos)
            shader.set_vec2("uQuadSize", (char_width, char_height))

            u0, v0 = char_info["u0"], char_info["v0"]
            u1, v1 = char_info["u1"], char_info["v1"]
            vertices = np.array(
                [
                    -0.5, 0.5, u0, v0,
                    -0.5, -0.5, u0, v1,
                    0.5, 0.5, u1, v0,
                    0.5, -0.5, u1, v1,
                ],
                dtype=np.float32,
            )

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBufferSubData(
                GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices
            )
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)
            current_x += char_width

        GL.glBindVertexArray(0)
