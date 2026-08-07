"""
Per-frame RenderContext assembly for the 3D canvas.

Builds the full per-frame :class:`RenderContext` (view/projection matrix
math plus the scene and playback state) that the renderers consume.
Pure matrix math with no GL calls, so it is directly unit-testable.
"""

import math
from typing import TYPE_CHECKING, Optional

import numpy as np

from .camera import Camera
from .gl_utils import RenderContext, rotation_4x4

if TYPE_CHECKING:
    from ...core.color import ColorSet
    from ...core.doc import Doc
    from ...machine.models.machine import Machine
    from ...simulator.op_player import OpPlayer
    from ...simulator.scene3d import CompiledSceneArtifact
    from .renderer.scene_renderer import SceneRenderer
    from .viewport import ViewportConfig


class RenderContextBuilder:
    """Assembles the per-frame RenderContext for the scene renderers."""

    @staticmethod
    def _world_size_to_pixels(
        mvp_gl: np.ndarray,
        world_mm: float,
        viewport_w: int,
        viewport_h: int,
    ) -> float:
        mvp = mvp_gl.T
        p0 = mvp @ np.array([0, 0, 0, 1], dtype=np.float32)
        p1 = mvp @ np.array([world_mm, 0, 0, 1], dtype=np.float32)
        if abs(p0[3]) < 1e-9 or abs(p1[3]) < 1e-9:
            return 1.0
        ndc_dx = (p1[0] / p1[3]) - (p0[0] / p0[3])
        return abs(ndc_dx) * viewport_w * 0.5

    def _compute_spot_line_width(
        self,
        machine: Optional["Machine"],
        camera: Camera,
        mvp_gl: np.ndarray,
    ) -> float:
        spot_mm = 0.1
        laser_head = machine.get_default_laser_head() if machine else None
        if laser_head is not None:
            spot_mm = laser_head.spot_size_mm[0]
        if not camera:
            return 2.0
        px = self._world_size_to_pixels(
            mvp_gl,
            spot_mm,
            camera.width,
            camera.height,
        )
        return max(2.0, px)

    def build(
        self,
        *,
        camera: Camera,
        viewport: "ViewportConfig",
        color_set: "ColorSet",
        scene: "SceneRenderer",
        op_player: Optional["OpPlayer"],
        compiled_artifact: Optional["CompiledSceneArtifact"],
        doc: "Doc",
        machine: Optional["Machine"],
        show_travel_moves: bool,
        show_grid: bool,
        show_nogo_zones: bool,
        show_models: bool,
    ) -> RenderContext:
        """Build the full per-frame RenderContext for the scene renderers.

        ``camera`` must be available before calling; the canvas guards for
        this before invoking ``build``.
        """
        proj_matrix = camera.get_projection_matrix()
        view_matrix = camera.get_view_matrix()

        # Base MVP for UI elements that should not be model-transformed
        mvp_matrix_ui = proj_matrix @ view_matrix

        # Create WCS translation matrix
        offset_x, offset_y, offset_z = viewport.wcs_offset_mm
        wcs_translation_matrix = np.array(
            [
                [1, 0, 0, offset_x],
                [0, 1, 0, offset_y],
                [0, 0, 1, offset_z],
                [0, 0, 0, 1],
            ],
            dtype=np.float32,
        )

        # Final model matrix for the grid combines the origin flip and WCS
        # translation. Grid/Axes vertices are in local (0..W, 0..H).
        # 1. Apply wcs_translation (shift by offset).
        # 2. Apply model_matrix (orient to machine coords).
        # Note: matrix order A @ B applies B then A.
        # This order applies WCS translation locally, THEN applies the
        # machine flip/origin shift.
        grid_model_matrix = viewport.model_matrix @ wcs_translation_matrix

        # Final MVP for scene geometry (grid, axes, rotary)
        mvp_matrix_scene = mvp_matrix_ui @ grid_model_matrix

        # Convert to column-major for OpenGL
        mvp_matrix_ui_gl = mvp_matrix_ui.T

        # Build the shared render context for this frame
        spot_line_width = self._compute_spot_line_width(
            machine, camera, mvp_matrix_ui_gl
        )
        rotary_axis = op_player.rotary_axis if op_player else None
        ctx = RenderContext(
            proj_matrix=proj_matrix,
            view_matrix=view_matrix,
            mvp_ui=mvp_matrix_ui,
            mvp_scene=mvp_matrix_scene,
            margin_shift=viewport.margin_shift,
            model_matrix=viewport.model_matrix,
            viewport_height=camera.height,
            camera_position=camera.position,
            color_set=color_set,
            show_travel_moves=show_travel_moves,
            line_width=spot_line_width,
            machine=machine,
            doc=doc,
            op_player=op_player,
            compiled_artifact=compiled_artifact,
            viewport=viewport,
            rotary_axis=rotary_axis,
            had_rotary_layers=scene.had_rotary_layers,
            show_grid=show_grid,
            show_nogo_zones=show_nogo_zones,
            show_models=show_models,
            wcs_offset_mm=viewport.wcs_offset_mm,
            x_right=viewport.x_right,
            y_down=viewport.y_down,
            x_negative=viewport.x_negative,
            y_negative=viewport.y_negative,
        )

        # Compute the rotary helper fields once per frame so the
        # renderers can consume them.
        ctx.mvp_flat_gl = mvp_matrix_ui_gl
        cyl_angle = 0.0
        if (
            op_player
            and scene.had_rotary_layers
            and machine
            and rotary_axis is not None
        ):
            asm = machine.assembly
            if asm.has_rotary:
                degrees = op_player.state.axes.get(rotary_axis, 0.0)
                cyl_angle = math.radians(degrees)

        if scene.had_rotary_layers and machine:
            cyl_base_mvp = (
                mvp_matrix_ui.astype(np.float64)
                @ viewport.margin_shift.astype(np.float64)
                @ scene.cylinder_transform
            )
        else:
            cyl_base_mvp = mvp_matrix_ui.astype(
                np.float64
            ) @ viewport.margin_shift.astype(np.float64)

        vis_rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        rot_4x4 = rotation_4x4(vis_rot_axis, cyl_angle)
        ctx.rot_4x4 = rot_4x4
        ctx.mvp_rot_gl = (cyl_base_mvp @ rot_4x4).T.astype(np.float32)
        cyl_mesh_mvp = (
            mvp_matrix_ui
            @ viewport.margin_shift
            @ scene.cylinder_transform
            @ rot_4x4
        ).astype(np.float64)
        ctx.cyl_mesh_mvp_gl = cyl_mesh_mvp.T.astype(np.float32)

        return ctx
