"""
Renders text in a 3D OpenGL scene.

This module provides a class `TextRenderer3D` for rendering text that faces
the camera (billboarding) in a 3D environment. It creates a texture atlas
from a specified font for the characters '0'-'9'.
"""

import logging
import math
from typing import Dict, Optional, Tuple, Union
import numpy as np
from OpenGL import GL
import cairo
from gi.repository import Pango, PangoCairo
from .gl_utils import BaseRenderer, Shader

logger = logging.getLogger(__name__)


class TextRenderer3D(BaseRenderer):
    """Renders billboarded text in a 3D scene."""

    def __init__(
        self, font_family: Optional[str] = None, font_size: int = 128
    ):
        """
        Initializes the text renderer on the CPU.

        Args:
            font_family: The name of the font to use (e.g. "Arial").
            font_size: The size of the font for the texture atlas.
        """
        super().__init__()
        self.char_data: Dict[str, Dict[str, Union[float, int]]] = {}
        self.texture_id: int = 0
        self.atlas_width: int = 0
        self.atlas_height: int = 0
        self.vao: int = 0
        self.vbo: int = 0
        self._font_size_px = font_size
        self._atlas_buffer: Optional[bytes] = None

        # This part no longer loads a font object, it just stores the
        # description
        self.font_desc = Pango.FontDescription()
        font_name = font_family if font_family else "sans-serif"
        self.font_desc.set_family(font_name)
        self.font_desc.set_size(self._font_size_px * Pango.SCALE)
        logger.info(
            f"Using Pango font description: {self.font_desc.to_string()}"
        )

        self._prepare_texture_atlas_pango()

    def _prepare_texture_atlas_pango(self) -> None:
        """
        Creates a texture atlas for numeric characters using Pango and Cairo.
        This version uses font-wide metrics (ascent/descent) to ensure all
        characters are vertically aligned to a common baseline.
        """
        chars_to_render = "0123456789-"
        padding_px = 2

        # Create a dummy Cairo surface to create a context for Pango
        dummy_surface = cairo.ImageSurface(cairo.FORMAT_A8, 1, 1)
        cr = cairo.Context(dummy_surface)
        layout = PangoCairo.create_layout(cr)
        layout.set_font_description(self.font_desc)

        # Get font-wide metrics to establish a common baseline. This is key
        # to correctly aligning characters with different vertical extents,
        # like '8' and '-'.
        pango_context = layout.get_context()
        metrics = pango_context.get_metrics(self.font_desc, None)
        ascent = metrics.get_ascent() / Pango.SCALE
        descent = metrics.get_descent() / Pango.SCALE

        # The atlas height is the full logical line height of the font.
        self.atlas_height = math.ceil(ascent + descent)

        char_metrics = {}
        total_advance_px = 0

        for char in chars_to_render:
            layout.set_text(char, -1)
            ink_rect, logical_rect = layout.get_pixel_extents()
            # We use the logical width (advance) for spacing calculation
            advance_px = logical_rect.width
            char_metrics[char] = {
                "ink_rect": ink_rect,
                "advance_px": advance_px,
            }
            total_advance_px += advance_px + padding_px
            logger.debug(
                f"Char '{char}': advance={advance_px}px, ink_rect={ink_rect}"
            )

        self.atlas_width = int(total_advance_px)

        if self.atlas_width <= 0 or self.atlas_height <= 0:
            logger.error(
                "Failed to calculate valid atlas size: %dx%d",
                self.atlas_width,
                self.atlas_height,
            )
            return

        # Create the real surface for the atlas
        atlas_surface = cairo.ImageSurface(
            cairo.FORMAT_A8, self.atlas_width, self.atlas_height
        )
        cr = cairo.Context(atlas_surface)
        layout = PangoCairo.create_layout(cr)
        layout.set_font_description(self.font_desc)
        cr.set_source_rgba(1.0, 1.0, 1.0, 1.0)  # Draw in white

        x_cursor = 0
        for char in chars_to_render:
            metrics = char_metrics[char]
            ink_rect = metrics["ink_rect"]
            advance_px = metrics["advance_px"]

            # We position the layout at y=0. Pango draws text relative to its
            # logical box. Since our atlas height is exactly ascent+descent,
            # this aligns the font baseline to `y = ascent`, keeping all
            # characters vertically aligned correctly.
            # We shift X to remove the left-side bearing for tighter packing,
            # but we allocate the full logical width for the texture slot.
            cr.move_to(x_cursor - ink_rect.x, 0)
            layout.set_text(char, -1)
            PangoCairo.show_layout(cr, layout)

            self.char_data[char] = {
                "u0": x_cursor / self.atlas_width,
                "v0": 0.0,
                "u1": min((x_cursor + advance_px) / self.atlas_width, 1.0),
                "v1": 1.0,
                "width_px": advance_px,
                "height_px": float(self.atlas_height),
            }
            x_cursor += advance_px + padding_px

        # Get the raw byte data and stride from the Cairo surface
        atlas_surface.flush()
        buffer = atlas_surface.get_data()
        stride = atlas_surface.get_stride()

        # Repack buffer if stride != width (remove padding bytes)
        if stride != self.atlas_width:
            logger.debug(
                f"Atlas stride ({stride}) != width ({self.atlas_width}). "
                "Repacking buffer."
            )
            unpacked_buffer = bytearray(self.atlas_width * self.atlas_height)
            for i in range(self.atlas_height):
                row_start_in = i * stride
                row_end_in = row_start_in + self.atlas_width
                row_start_out = i * self.atlas_width
                unpacked_buffer[
                    row_start_out : row_start_out + self.atlas_width
                ] = buffer[row_start_in:row_end_in]
            self._atlas_buffer = bytes(unpacked_buffer)
        else:
            self._atlas_buffer = bytes(buffer)

    def init_gl(self) -> None:
        """Initializes all OpenGL resources."""
        if self._atlas_buffer:
            self._upload_atlas_to_gpu()
            self._atlas_buffer = None  # Free CPU memory

        self.vao = self._create_vao()
        self.vbo = self._create_vbo()

        GL.glBindVertexArray(self.vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 16 * 4, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 4, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glBindVertexArray(0)

    def _upload_atlas_to_gpu(self) -> None:
        """Helper to create and configure the OpenGL texture."""
        if not self._atlas_buffer:
            return
        self.texture_id = self._create_texture()
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)

        # Tell OpenGL how to unpack the pixel data. We have 1-byte alignment
        # since we manually created a tightly-packed buffer.
        old_alignment = GL.glGetIntegerv(GL.GL_UNPACK_ALIGNMENT)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)

        GL.glTexImage2D(
            GL.GL_TEXTURE_2D,
            0,
            GL.GL_R8,  # Internal format: 8-bit red channel
            self.atlas_width,
            self.atlas_height,
            0,
            GL.GL_RED,  # Source fmt: also red (from our single-channel data)
            GL.GL_UNSIGNED_BYTE,
            self._atlas_buffer,
        )

        # Restore the original alignment
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, old_alignment)

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
        height_in_world_units: float,
        color: Tuple[float, float, float, float],
        mvp_matrix: np.ndarray,
        view_matrix: np.ndarray,
        align: str = "center",
    ) -> None:
        """
        Renders a string of text at a given 3D position, facing the camera.
        The entire string billboards as a single unit.

        Args:
            shader: The shader program to use for rendering text.
            text: The string to render (must contain characters of the atlas).
            position: A numpy array (vec3) for the text's anchor point.
            height_in_world_units: Desired height of the text in world units.
            color: A tuple (r, g, b, a) for the text color.
            mvp_matrix: The column-major Model-View-Projection matrix.
            view_matrix: The camera's view matrix (row-major).
            align: Horizontal alignment ('left', 'center', 'right').
        """
        if not self.vao or not text or self.atlas_height < 1:
            return

        shader.use()
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.texture_id)
        GL.glBindVertexArray(self.vao)

        # --- UNIFORMS CONSTANT FOR THE ENTIRE STRING ---
        shader.set_vec4("uTextColor", color)
        shader.set_mat4("uMVP", mvp_matrix)

        # Pass the string's anchor position. This is the pivot for the
        # billboard.
        shader.set_vec3("uTextWorldPos", position)

        # Get the camera's rotation matrix to billboard the text.
        try:
            inv_view = np.linalg.inv(view_matrix)
            camera_rotation_matrix_row_major = inv_view[:3, :3]
            u = camera_rotation_matrix_row_major[:, 0]
            u /= np.linalg.norm(u)
            v = camera_rotation_matrix_row_major[:, 1]
            v -= np.dot(v, u) * u
            v /= np.linalg.norm(v)
            w = np.cross(u, v)
            camera_rotation_matrix_row_major = np.column_stack((u, v, w))
        except np.linalg.LinAlgError:
            logger.warning(
                "View matrix inversion failed, using identity for billboard."
            )
            camera_rotation_matrix_row_major = np.identity(3)

        billboard_matrix_col_major = camera_rotation_matrix_row_major.T
        shader.set_mat3("uBillboard", billboard_matrix_col_major)

        # --- CALCULATE LAYOUT AND RENDER CHARACTERS ---
        pixel_to_world_scale = height_in_world_units / self.atlas_height
        total_text_width_px = sum(
            self.char_data[c]["width_px"] for c in text if c in self.char_data
        )
        total_text_width_world = total_text_width_px * pixel_to_world_scale

        if align == "right":
            current_x_local = -total_text_width_world
        elif align == "left":
            current_x_local = 0.0
        else:  # 'center'
            current_x_local = -total_text_width_world / 2.0

        for char in text:
            if char not in self.char_data:
                continue

            char_info = self.char_data[char]
            char_width_world = char_info["width_px"] * pixel_to_world_scale
            char_height_world = char_info["height_px"] * pixel_to_world_scale

            # Send the character's size and its offset from the string's
            # anchor.
            shader.set_vec2("uQuadSize", (char_width_world, char_height_world))
            GL.glUniform1f(
                shader.get_uniform_location("uCharOffsetX"), current_x_local
            )

            # The vertex data is the same for every character (a unit quad).
            # The shader now does all the work of positioning it.
            u0, v0 = char_info["u0"], char_info["v0"]
            u1, v1 = char_info["u1"], char_info["v1"]
            vertices = np.array(
                [
                    -0.5,
                    0.5,
                    u0,
                    v0,  # Top-left
                    -0.5,
                    -0.5,
                    u0,
                    v1,  # Bottom-left
                    0.5,
                    0.5,
                    u1,
                    v0,  # Top-right
                    0.5,
                    -0.5,
                    u1,
                    v1,  # Bottom-right
                ],
                dtype=np.float32,
            )

            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.vbo)
            GL.glBufferSubData(
                GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices
            )
            GL.glDrawArrays(GL.GL_TRIANGLE_STRIP, 0, 4)

            current_x_local += char_width_world

        GL.glBindVertexArray(0)
