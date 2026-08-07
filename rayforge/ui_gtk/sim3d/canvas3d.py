import logging
import math
import time
from typing import TYPE_CHECKING, Optional

import numpy as np
from gi.repository import Gdk, Gtk, Pango
from OpenGL import GL

from ...context import RayforgeContext
from ...pipeline.pipeline import Pipeline
from ...shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from .camera import ViewDirection
from .camera_controller import CameraController
from .chunked_upload import ChunkedUploadController
from .doc_signals import DocSignalHub
from .gl_utils import RenderContext, rotation_4x4
from .renderer.scene_renderer import SceneRenderer
from .scene_presenter import ScenePresenter
from .theme_resolver import ThemeResolver
from .viewport import ViewportConfig

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Canvas3D(Gtk.GLArea):
    """A GTK Widget for rendering a 3D scene with OpenGL."""

    def __init__(
        self,
        context: RayforgeContext,
        doc_editor: "DocEditor",
        viewport: ViewportConfig,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._doc_editor = doc_editor
        self._viewport = viewport

        self._scene = SceneRenderer()
        self._show_travel_moves = False
        self._show_nogo_zones = True
        self._show_models = True
        self._show_grid = True
        self._gl_initialized = False
        self._scene_gl_dirty = False

        self._theme_resolver = ThemeResolver(
            self,
            scene=self._scene,
            get_machine=lambda: self._context.machine,
            get_gl_initialized=lambda: self._gl_initialized,
            request_render=self.queue_render,
        )

        self._presenter = ScenePresenter(
            self._context,
            self._doc_editor,
            self._scene,
            theme_resolver=self._theme_resolver,
            get_viewport=lambda: self._viewport,
            get_gl_initialized=lambda: self._gl_initialized,
            get_show_travel_moves=lambda: self._show_travel_moves,
            get_has_stale_job=lambda: self._doc_hub.has_stale_job(),
            get_camera_available=lambda: self._cam_ctrl.camera is not None,
            make_current=self.make_current,
            mark_scene_dirty=lambda: setattr(self, "_scene_gl_dirty", True),
            mark_artifact_dirty=lambda: (
                self._upload_ctrl.mark_artifact_dirty()
            ),
            reset_view=self.reset_view,
            request_render=self.queue_render,
        )

        self._upload_ctrl = ChunkedUploadController(
            self._scene,
            get_artifact=lambda: self._presenter.compiled_artifact,
            get_show_travel_moves=lambda: self._show_travel_moves,
            get_gl_initialized=lambda: self._gl_initialized,
            make_current=self.make_current,
            request_render=self.queue_render,
            on_luts_required=self._theme_resolver.update_renderer_color_luts,
            on_op_player_required=self._presenter._on_op_player_required,
        )

        self._cam_ctrl = CameraController(
            self,
            get_viewport=lambda: self._viewport,
            request_render=self.queue_render,
            on_key_pressed=self._on_key_pressed,
        )

        self._doc_hub = DocSignalHub(
            self._context,
            self._doc_editor,
            set_viewport=lambda vp: setattr(self, "_viewport", vp),
            mark_scene_dirty=lambda: setattr(self, "_scene_gl_dirty", True),
            request_render=self.queue_render,
            refresh_scene=self._presenter.update_scene_from_doc,
            get_gl_initialized=lambda: self._gl_initialized,
            get_job_handle=lambda: self._presenter.job_handle,
            on_pipeline_state_changed=(
                self._presenter._on_pipeline_state_changed
            ),
            on_job_generation_finished=(
                self._presenter._on_job_generation_finished
            ),
        )

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self._cam_ctrl.on_resize)
        self.connect("notify::style", self._theme_resolver.on_style_changed)

        self._doc_hub.connect()

        self._context.config.changed.connect(self._on_config_changed)

    def set_machine(self, viewport: Optional[ViewportConfig] = None):
        self._doc_hub.set_machine(viewport)

    def has_stale_job(self) -> bool:
        """True if the cached job handle is from an older generation."""
        return self._doc_hub.has_stale_job()

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_hub.doc

    @property
    def pipeline(self) -> "Pipeline":
        """Returns the current pipeline from the editor."""
        return self._doc_hub.pipeline

    def reset_view(self, direction: ViewDirection):
        """Resets the camera to the specified preset view."""
        self._cam_ctrl.reset_view(direction)

    def set_perspective(self, enabled: bool) -> bool:
        """Toggles the 3D camera between perspective and orthographic.

        Returns True if the camera was available and updated.
        """
        camera = self._cam_ctrl.camera
        if camera is None:
            return False
        camera.is_perspective = enabled
        self.queue_render()
        return True

    def set_playback_overlay(self, overlay):
        """Attach the playback overlay widget and bind it to this canvas."""
        self._presenter.set_playback_overlay(overlay)
        overlay.set_canvas(self)

    def _on_style_changed(self, widget, gparam):
        """Marks theme resources as dirty when the GTK theme changes."""
        self._theme_resolver.mark_dirty()
        self.queue_render()

    def _on_config_changed(self, sender, **kwargs):
        """Updates renderer color LUTs when config settings change."""
        if not self._gl_initialized:
            return
        axis_renderer = self._scene.axis_renderer
        if axis_renderer:
            new_step = get_default_grid_step_mm()
            if not math.isclose(axis_renderer.grid_size_mm, new_step):
                self.make_current()
                axis_renderer.set_grid_size(new_step)
            axis_renderer.set_grid_unit_factor(
                get_preferred_unit_factor("length")
            )
        self._theme_resolver.update_renderer_color_luts()
        self.queue_render()

    def on_realize(self, area) -> None:
        """Called when the GLArea is ready to have its context made current."""
        logger.info("GLArea realized.")

        self._cam_ctrl.create_camera(self.get_width(), self.get_height())

        self._init_gl_resources()
        self._theme_resolver.mark_dirty()

        self.reset_view(ViewDirection.ISO)
        self._theme_resolver.update_theme_and_colors()
        self._doc_hub.connect_pipeline()

        if self._presenter.job_handle is None and self.pipeline:
            self._presenter.job_handle = self.pipeline.last_completed_handle

        self._presenter.update_scene_from_doc()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        overlay = self._presenter.playback_overlay
        if keyval == Gdk.KEY_space and overlay:
            overlay.handle_space()
            return True
        return False

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self._doc_hub.disconnect()
        self._context.config.changed.disconnect(self._on_config_changed)
        self._upload_ctrl.cancel()
        try:
            self.make_current()
            self._presenter.cancel_scene_preparation()
            self._scene.cleanup()
        except Exception as e:
            logger.debug("Error during GL cleanup on unrealize: %s", e)
        finally:
            self._gl_initialized = False
        logger.debug("on_unrealize: finished.")

    def _init_gl_resources(self) -> None:
        """Initializes OpenGL state, shaders, and renderer objects."""
        try:
            self.make_current()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            # Get the theme's default font family from GTK
            font_family = "sans-serif"  # A safe fallback
            settings = Gtk.Settings.get_default()
            if settings:
                font_name_str = settings.get_property("gtk-font-name")
                logger.debug(f"Gtk uses font {font_name_str}")
                if font_name_str:
                    # Use Pango to reliably parse the string
                    # (e.g., "Ubuntu Sans")
                    font_desc = Pango.FontDescription.from_string(
                        font_name_str
                    )
                    font_family = font_desc.get_family() or "sans-serif"
                    logger.debug(f"Pango normalized font to {font_family}")

            self._scene.init_gl(self._viewport, font_family)

            self._gl_initialized = True
        except Exception as e:
            logger.error(f"OpenGL Initialization Error: {e}", exc_info=True)
            self._gl_initialized = False

    def _process_pending_gl_updates(self):
        if self._scene_gl_dirty:
            self._scene_gl_dirty = False
            if self._scene.update_axis_from_viewport(self._viewport):
                self._theme_resolver.mark_dirty()
            self.make_current()
            self._scene.update_cylinders_from_doc(
                self.doc, self._viewport, self._context.machine
            )
            self._scene.update_models_from_context(
                self._context, self._context.machine
            )
            machine = self._context.machine
            if self._scene.zone_renderer and machine:
                self._scene.update_zones_from_machine(machine)
        self._upload_ctrl.process_pending()

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

    def _compute_spot_line_width(self, mvp_gl: np.ndarray) -> float:
        machine = self._context.machine
        spot_mm = 0.1
        laser_head = machine.get_default_laser_head() if machine else None
        if laser_head is not None:
            spot_mm = laser_head.spot_size_mm[0]
        if not self._cam_ctrl.camera:
            return 2.0
        px = self._world_size_to_pixels(
            mvp_gl,
            spot_mm,
            self._cam_ctrl.camera.width,
            self._cam_ctrl.camera.height,
        )
        return max(2.0, px)

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self._cam_ctrl.camera or not self._gl_initialized:
            return False

        self._process_pending_gl_updates()

        if self._theme_resolver.theme_is_dirty:
            self._theme_resolver.update_theme_and_colors()

        if not self._theme_resolver.color_set:
            return False

        t_render_start = time.perf_counter()
        try:
            GL.glViewport(
                0, 0, self._cam_ctrl.camera.width, self._cam_ctrl.camera.height
            )
            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
            )

            proj_matrix = self._cam_ctrl.camera.get_projection_matrix()
            view_matrix = self._cam_ctrl.camera.get_view_matrix()

            # Base MVP for UI elements that should not be model-transformed
            mvp_matrix_ui = proj_matrix @ view_matrix

            # Create WCS translation matrix
            offset_x, offset_y, offset_z = self._viewport.wcs_offset_mm
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
            grid_model_matrix = (
                self._viewport.model_matrix @ wcs_translation_matrix
            )

            # Final MVP for scene geometry (grid, axes, rotary)
            mvp_matrix_scene = mvp_matrix_ui @ grid_model_matrix

            # Convert to column-major for OpenGL
            mvp_matrix_ui_gl = mvp_matrix_ui.T

            # Build the shared render context for this frame
            spot_line_width = self._compute_spot_line_width(mvp_matrix_ui_gl)
            op_player = self._presenter.op_player
            machine = self._context.machine
            rotary_axis = op_player.rotary_axis if op_player else None
            ctx = RenderContext(
                proj_matrix=proj_matrix,
                view_matrix=view_matrix,
                mvp_ui=mvp_matrix_ui,
                mvp_scene=mvp_matrix_scene,
                margin_shift=self._viewport.margin_shift,
                model_matrix=self._viewport.model_matrix,
                viewport_height=self._cam_ctrl.camera.height,
                camera_position=self._cam_ctrl.camera.position,
                color_set=self._theme_resolver.color_set,
                show_travel_moves=self._show_travel_moves,
                line_width=spot_line_width,
                machine=machine,
                doc=self.doc,
                op_player=op_player,
                compiled_artifact=self._presenter.compiled_artifact,
                viewport=self._viewport,
                rotary_axis=rotary_axis,
                had_rotary_layers=self._scene.had_rotary_layers,
                show_grid=self._show_grid,
                show_nogo_zones=self._show_nogo_zones,
                show_models=self._show_models,
                wcs_offset_mm=self._viewport.wcs_offset_mm,
                x_right=self._viewport.x_right,
                y_down=self._viewport.y_down,
                x_negative=self._viewport.x_negative,
                y_negative=self._viewport.y_negative,
            )

            # Compute the rotary helper fields once per frame so the
            # renderers can consume them.
            ctx.mvp_flat_gl = mvp_matrix_ui_gl
            cyl_angle = 0.0
            if (
                op_player
                and self._scene.had_rotary_layers
                and machine
                and rotary_axis is not None
            ):
                asm = machine.assembly
                if asm.has_rotary:
                    degrees = op_player.state.axes.get(rotary_axis, 0.0)
                    cyl_angle = math.radians(degrees)

            if self._scene.had_rotary_layers and machine:
                cyl_base_mvp = (
                    mvp_matrix_ui.astype(np.float64)
                    @ self._viewport.margin_shift.astype(np.float64)
                    @ self._scene.cylinder_transform
                )
            else:
                cyl_base_mvp = mvp_matrix_ui.astype(
                    np.float64
                ) @ self._viewport.margin_shift.astype(np.float64)

            vis_rot_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            rot_4x4 = rotation_4x4(vis_rot_axis, cyl_angle)
            ctx.rot_4x4 = rot_4x4
            ctx.mvp_rot_gl = (cyl_base_mvp @ rot_4x4).T.astype(np.float32)
            cyl_mesh_mvp = (
                mvp_matrix_ui
                @ self._viewport.margin_shift
                @ self._scene.cylinder_transform
                @ rot_4x4
            ).astype(np.float64)
            ctx.cyl_mesh_mvp_gl = cyl_mesh_mvp.T.astype(np.float32)

            self._scene.render(ctx)

        except Exception as e:
            logger.error("OpenGL Render Error: %s", e, exc_info=True)
            return False

        t_render_elapsed = (time.perf_counter() - t_render_start) * 1000
        if t_render_elapsed > 16:
            logger.info(f"[CANVAS3D] on_render took {t_render_elapsed:.1f}ms")
        return True

    def set_show_travel_moves(self, visible: bool):
        """Sets the visibility of travel moves in the 3D view."""
        if self._show_travel_moves == visible:
            return
        self._show_travel_moves = visible
        self._presenter.update_renderers_from_artifact()

    def set_show_nogo_zones(self, visible: bool):
        if self._show_nogo_zones == visible:
            return
        self._show_nogo_zones = visible
        self.queue_render()

    def set_show_models(self, visible: bool):
        if self._show_models == visible:
            return
        self._show_models = visible
        self.queue_render()

    def set_show_grid(self, visible: bool):
        if self._show_grid == visible:
            return
        self._show_grid = visible
        self.queue_render()

    def update_scene_from_doc(self):
        """Refreshes the 3D scene content from the document."""
        self._presenter.update_scene_from_doc()

    def scene_is_ready(self) -> bool:
        """True when the compiled scene is uploaded and rendered.

        Tooling (e.g. the screenshot harness) polls this to wait for the
        3D view to reach a stable, rendered state.
        """
        if not self._gl_initialized:
            return False
        if self._presenter.compiled_artifact is None:
            return False
        if self._upload_ctrl.is_dirty:
            return False
        task = self._presenter.scene_preparation_task
        return task is None or task.is_final()
