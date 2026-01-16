"""
Renders a 3D grid and axes for a scene.

This module provides the AxisRenderer3D class, which is responsible for
creating and drawing a grid on the XY plane, along with labeled X and Y
axes. It uses a composed PlaneRenderer for the background.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple
import numpy as np
from OpenGL import GL
from .gl_utils import BaseRenderer, Shader
from .text_renderer_3d import TextRenderer3D
from .plane_renderer import PlaneRenderer

logger = logging.getLogger(__name__)


class AxisRenderer3D(BaseRenderer):
    """Renders a 3D grid with axes, background, and labels on the XY plane."""

    def __init__(
        self,
        width_mm: float,
        height_mm: float,
        grid_size_mm: float = 10.0,
        font_family: Optional[str] = None,
    ):
        """Initializes the AxisRenderer3D with scene dimensions.

        Args:
            width_mm: The total width of the grid along the X-axis in mm.
            height_mm: The total height of the grid along the Y-axis in mm.
            grid_size_mm: The spacing between grid lines in mm.
            font_family: The name of the font to use for labels
            (e.g. "Cantarell").
        """
        super().__init__()
        self.width_mm = float(width_mm)
        self.height_mm = float(height_mm)
        self.grid_size_mm = float(grid_size_mm)
        self.font_family = font_family

        # Colors
        self.background_color = 0.8, 0.8, 0.8, 0.1
        self.grid_color = 0.4, 0.4, 0.4, 1.0
        self.axis_color = 1.0, 1.0, 1.0, 1.0
        self.wcs_marker_color = 1.0, 0.0, 1.0, 1.0  # Magenta
        self.label_color = 0.9, 0.9, 0.9, 1.0

        # Composition
        self.background_renderer = PlaneRenderer(
            width=self.width_mm,
            height=self.height_mm,
            color=self.background_color,
            z_offset=-0.002,
        )
        self._add_child_renderer(self.background_renderer)

        self.text_renderer: Optional[TextRenderer3D] = None

        # Grid and Axes resources
        self.grid_vao, self.grid_vbo, self.grid_vertex_count = 0, 0, 0
        self.axes_vao, self.axes_vbo, self.axes_vertex_count = 0, 0, 0
        (
            self.wcs_marker_vao,
            self.wcs_marker_vbo,
            self.wcs_marker_vertex_count,
        ) = (0, 0, 0)

    def set_background_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the background plane."""
        self.background_color = color
        self.background_renderer.color = color

    def set_grid_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the grid lines."""
        self.grid_color = color

    def set_axis_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the main X and Y axis lines."""
        self.axis_color = color

    def set_label_color(self, color: Tuple[float, float, float, float]):
        """Sets the color for the axis labels."""
        self.label_color = color

    def init_gl(self) -> None:
        """Initializes OpenGL resources for all components."""
        # Delegate initialization to child renderers
        self.background_renderer.init_gl()

        self.text_renderer = TextRenderer3D(font_family=self.font_family)
        self.text_renderer.init_gl()
        self._add_child_renderer(self.text_renderer)

        # Initialize self-managed components using base class helpers
        self._init_grid_and_axes()

    def _init_grid_and_axes(self):
        """Creates VAOs/VBOs for the grid and axis lines."""
        grid_z_pos = -0.001
        w, h = self.width_mm, self.height_mm

        # Grid vertices
        grid_verts = []
        for x in np.arange(self.grid_size_mm, w, self.grid_size_mm):
            grid_verts.extend([x, 0.0, grid_z_pos, x, h, grid_z_pos])
        for y in np.arange(self.grid_size_mm, h, self.grid_size_mm):
            grid_verts.extend([0.0, y, grid_z_pos, w, y, grid_z_pos])

        # Axis vertices
        axis_verts = [0.0, 0.0, 0.0, w, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, h, 0.0]

        # WCS Marker vertices (a cross)
        marker_size = self.grid_size_mm * 0.5
        marker_z_pos = 0.001  # Slightly above the axes
        wcs_marker_verts = [
            -marker_size,
            0.0,
            marker_z_pos,
            marker_size,
            0.0,
            marker_z_pos,
            0.0,
            -marker_size,
            marker_z_pos,
            0.0,
            marker_size,
            marker_z_pos,
        ]

        # Create Grid resources
        self.grid_vao = self._create_vao()
        self.grid_vbo = self._create_vbo()
        self.grid_vertex_count = len(grid_verts) // 3
        GL.glBindVertexArray(self.grid_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.grid_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(grid_verts, dtype=np.float32).nbytes,
            np.array(grid_verts, dtype=np.float32),
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Create Axis resources
        self.axes_vao = self._create_vao()
        self.axes_vbo = self._create_vbo()
        self.axes_vertex_count = len(axis_verts) // 3
        GL.glBindVertexArray(self.axes_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.axes_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(axis_verts, dtype=np.float32).nbytes,
            np.array(axis_verts, dtype=np.float32),
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        # Create WCS Marker resources
        self.wcs_marker_vao = self._create_vao()
        self.wcs_marker_vbo = self._create_vbo()
        self.wcs_marker_vertex_count = len(wcs_marker_verts) // 3
        GL.glBindVertexArray(self.wcs_marker_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self.wcs_marker_vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            np.array(wcs_marker_verts, dtype=np.float32).nbytes,
            np.array(wcs_marker_verts, dtype=np.float32),
            GL.GL_STATIC_DRAW,
        )
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(0)

        GL.glBindVertexArray(0)

    def render(
        self,
        line_shader: Shader,
        text_shader: Shader,
        scene_mvp: np.ndarray,
        text_mvp: np.ndarray,
        view_matrix: np.ndarray,
        model_matrix: np.ndarray,
        origin_offset_mm: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        x_right: bool = False,
        y_down: bool = False,
        x_negative: bool = False,
        y_negative: bool = False,
    ) -> None:
        """
        Orchestrates the rendering of all components in the correct order.

        Args:
            line_shader: The shader program for drawing lines/solids.
            text_shader: The shader program for drawing text.
            scene_mvp: The final MVP matrix for the grid and background.
            text_mvp: The MVP matrix for the text labels (no model transform).
            view_matrix: The view matrix, used for billboarding text.
            model_matrix: The model matrix for coordinate system transforms.
            origin_offset_mm: The (x, y, z) offset for the work coordinate
              system.
            x_right: True if the machine origin is on the right side.
            y_down: True if the machine origin is at the top.
            x_negative: True if the X-axis counts down from the origin.
            y_negative: True if the Y-axis counts down from the origin.
        """
        if not all(
            (
                self.grid_vao,
                self.axes_vao,
                self.wcs_marker_vao,
                self.text_renderer,
            )
        ):
            return

        # 1. Calculate the world-space position of the WCS origin.
        # The offset is in machine coordinates. `model_matrix` transforms
        # the machine bed to world space. We apply the same transform to the
        # offset vector to find its world-space position.
        off_x, off_y, off_z = origin_offset_mm

        # Invert coordinates if the axis is configured as negative, so they
        # represent positive magnitudes relative to the origin for the
        # model matrix transform.
        if x_negative:
            off_x = -off_x
        if y_negative:
            off_y = -off_y

        # Use w=1.0 so that translations in model_matrix are applied.
        offset_vec = np.array([off_x, off_y, off_z, 1.0], dtype=np.float32)
        world_offset_vec = model_matrix @ offset_vec

        # 2. Construct the MVP for the static grid/axes.
        # The grid's model matrix is just the machine's base model_matrix
        # (handling origin flips), without any WCS translation.
        grid_mvp = model_matrix.T @ text_mvp

        # Enable blending for transparent objects
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        line_shader.use()

        # Draw background plane
        GL.glDepthMask(GL.GL_FALSE)
        self.background_renderer.render(line_shader, grid_mvp)
        GL.glDepthMask(GL.GL_TRUE)

        # Draw grid and axes
        line_shader.set_mat4("uMVP", grid_mvp)
        line_shader.set_vec4("uColor", self.grid_color)
        GL.glLineWidth(1.0)
        GL.glBindVertexArray(self.grid_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.grid_vertex_count)

        line_shader.set_vec4("uColor", self.axis_color)
        GL.glLineWidth(2.0)
        GL.glBindVertexArray(self.axes_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.axes_vertex_count)

        # 3. Draw the WCS origin marker
        wcs_translation_matrix = np.identity(4, dtype=np.float32)
        wcs_translation_matrix[:3, 3] = world_offset_vec[:3]
        wcs_marker_mvp = wcs_translation_matrix.T @ text_mvp

        line_shader.set_mat4("uMVP", wcs_marker_mvp)
        line_shader.set_vec4("uColor", self.wcs_marker_color)
        GL.glBindVertexArray(self.wcs_marker_vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self.wcs_marker_vertex_count)

        GL.glBindVertexArray(0)

        # 5. Pass the correct world-space offset vector to the label renderer.
        self._render_axis_labels(
            text_shader,
            text_mvp,
            view_matrix,
            model_matrix,
            origin_offset_mm=origin_offset_mm,
            x_right=x_right,
            y_down=y_down,
            x_negative=x_negative,
            y_negative=y_negative,
        )
        GL.glDisable(GL.GL_BLEND)

    def _render_axis_labels(
        self,
        text_shader: Shader,
        text_mvp_matrix: np.ndarray,
        view_matrix: np.ndarray,
        model_matrix: np.ndarray,
        origin_offset_mm: Tuple[float, float, float],
        x_right: bool = False,
        y_down: bool = False,
        x_negative: bool = False,
        y_negative: bool = False,
    ) -> None:
        """Helper method to render text labels along the axes."""
        if not self.text_renderer:
            return
        label_height_mm = 2.5
        x_axis_label_y_offset = label_height_mm * 1.2
        y_axis_label_x_offset = label_height_mm * 0.6

        # 1. Get the WCS offset in machine coordinates
        work_origin_x, work_origin_y, _ = origin_offset_mm

        # 2. Calculate the visual position of the WCS origin in the local
        #    (0..width, 0..height, Y-up) grid coordinate space.
        #    The "eff_wcs" is the positive magnitude from the origin corner.
        eff_wcs_x = -work_origin_x if x_negative else work_origin_x
        eff_wcs_y = -work_origin_y if y_negative else work_origin_y

        # Since model_matrix handles the reflection (scale -1), the local
        # coordinate system always starts at the origin corner (0,0 local).
        # So the WCS local position is simply the effective distance.
        wcs_local_x = eff_wcs_x
        wcs_local_y = eff_wcs_y

        # X-axis labels
        # Find the range of grid lines that are on the machine bed.
        # Grid covers 0..Width.
        # We label relative to WCS.
        # delta = x_phys_local - wcs_local_x
        # min_delta = 0 - wcs_local_x
        # max_delta = Width - wcs_local_x
        min_delta_x = 0.0 - wcs_local_x
        max_delta_x = self.width_mm - wcs_local_x
        k_start_x = math.ceil(min_delta_x / self.grid_size_mm)
        k_end_x = math.floor(max_delta_x / self.grid_size_mm)

        for k in range(k_start_x, k_end_x + 1):
            delta = k * self.grid_size_mm

            # Physical position of the grid line in local space
            x_phys_local = wcs_local_x + delta

            # Position of the label text below the grid line
            pos_local = np.array(
                [x_phys_local, -x_axis_label_y_offset, 0.0, 1.0]
            )
            pos_final = (model_matrix @ pos_local)[:3]

            # Label value logic:
            # If negative axis, movement into the bed (+delta) corresponds
            # to more negative values.
            # If positive axis, movement into the bed (+delta) corresponds
            # to more positive values.
            # This holds true regardless of origin corner (x_right) because
            # delta is defined in the flipped local space.
            label_val = -delta if x_negative else delta
            label_text = str(int(round(label_val)))

            self.text_renderer.render_text(
                text_shader,
                label_text,
                pos_final,
                label_height_mm,
                self.label_color,
                text_mvp_matrix,
                view_matrix,
            )

        # Y-axis labels
        y_label_align = "right"
        if x_right:
            y_label_align = "left"

        min_delta_y = 0.0 - wcs_local_y
        max_delta_y = self.height_mm - wcs_local_y
        k_start_y = math.ceil(min_delta_y / self.grid_size_mm)
        k_end_y = math.floor(max_delta_y / self.grid_size_mm)

        for k in range(k_start_y, k_end_y + 1):
            delta = k * self.grid_size_mm

            # Physical position of the grid line in local space
            y_phys_local = wcs_local_y + delta

            # Position of the label text next to the grid line
            pos_local = np.array(
                [-y_axis_label_x_offset, y_phys_local, 0.0, 1.0]
            )
            pos_final = (model_matrix @ pos_local)[:3]

            # Label value logic: same as X
            label_val = -delta if y_negative else delta
            label_text = str(int(round(label_val)))

            self.text_renderer.render_text(
                text_shader,
                label_text,
                pos_final,
                label_height_mm,
                self.label_color,
                text_mvp_matrix,
                view_matrix,
                align=y_label_align,
            )
