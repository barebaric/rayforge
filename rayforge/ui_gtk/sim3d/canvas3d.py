import logging
import math
import time
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from gi.repository import Gdk, GLib, Gtk, Pango
from OpenGL import GL
from raygeo.ops import Ops

from ...context import RayforgeContext
from ...machine.kinematic_mapping import (
    KinematicMapping,
    resolve_layer_rotary,
)
from ...pipeline.artifact.handle import BaseArtifactHandle
from ...pipeline.artifact.job import JobArtifact
from ...pipeline.pipeline import Pipeline
from ...shared.tasker import Task, task_mgr
from ...shared.units.formatter import (
    get_default_grid_step_mm,
    get_preferred_unit_factor,
)
from ...simulator.op_player import OpPlayer, SnapshotBuilder
from ...simulator.scene3d import (
    CompiledSceneArtifact,
    LayerRenderConfig,
    RenderConfig3D,
    compile_scene_in_thread,
)
from .camera import ViewDirection
from .camera_controller import CameraController
from .gl_utils import RenderContext, rotation_4x4
from .renderer.scene_renderer import SceneRenderer
from .theme_resolver import ThemeResolver
from .viewport import ViewportConfig

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor
    from ...machine.models.machine import Machine

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
        self._scene_preparation_task: Optional[Task] = None
        self._compiled_artifact: Optional[CompiledSceneArtifact] = None
        self._current_job_handle: Optional[BaseArtifactHandle] = None
        self._op_player: Optional[OpPlayer] = None
        self._playback_overlay = None
        self._show_travel_moves = False
        self._show_nogo_zones = True
        self._show_models = True
        self._show_grid = True
        self._gl_initialized = False
        self._scene_gl_dirty = False
        self._artifact_gl_dirty = False
        self._upload_state = None

        self._theme_resolver = ThemeResolver(
            self,
            scene=self._scene,
            get_machine=lambda: self._context.machine,
            get_gl_initialized=lambda: self._gl_initialized,
            request_render=self.queue_render,
        )

        self._cam_ctrl = CameraController(
            self,
            get_viewport=lambda: self._viewport,
            request_render=self.queue_render,
            on_key_pressed=self._on_key_pressed,
        )

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self._cam_ctrl.on_resize)
        self.connect("notify::style", self._theme_resolver.on_style_changed)

        # Connect to machine for WCS updates
        machine = self._context.machine
        if machine:
            machine.wcs_updated.connect(self._on_wcs_updated)
            machine.changed.connect(self._on_wcs_updated)
            self._on_wcs_updated(machine)

        # Connect to doc for per-layer WCS updates
        self.doc.active_layer_changed.connect(self._on_active_layer_changed)
        self._active_layer_wcs_conn = None
        self._connect_active_layer_wcs()

        self._context.config.changed.connect(self._on_config_changed)

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_editor.doc

    @property
    def pipeline(self) -> "Pipeline":
        """Returns the current pipeline from the editor."""
        return self._doc_editor.pipeline

    @property
    def rotary_enabled(self) -> bool:
        """Returns True if the active layer has rotary mode enabled."""
        if self.doc and self.doc.active_layer:
            return self.doc.active_layer.rotary_enabled
        return False

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
        self._playback_overlay = overlay
        overlay.set_canvas(self)

    def has_stale_job(self) -> bool:
        """True if the cached job handle is from an older generation."""
        if self._current_job_handle is None:
            return True
        return (
            self._current_job_handle.generation_id
            != self.pipeline.data_generation_id
        )

    def _on_wcs_updated(self, machine: "Machine", **kwargs):
        """Handler for when the machine's WCS state changes."""
        if machine:
            self._viewport = self._build_viewport(machine)
        self._scene_gl_dirty = True
        self.queue_render()

    def _get_active_layer_wcs_offset(self, machine: "Machine"):
        """Returns the WCS offset for the active layer."""
        layer = self.doc.active_layer if self.doc else None
        if layer and layer.wcs:
            return machine.get_wcs_offset(layer.wcs)
        return machine.get_active_wcs_offset()

    def _build_viewport(self, machine: "Machine") -> "ViewportConfig":
        """Build a ViewportConfig using the active layer's WCS."""
        return ViewportConfig.from_machine_with_wcs(
            machine, self._get_active_layer_wcs_offset(machine)
        )

    def _connect_active_layer_wcs(self):
        """Connect to the active layer's updated signal for WCS changes."""
        if self._active_layer_wcs_conn is not None:
            old_layer = self.doc.active_layer
            old_layer.updated.disconnect(self._active_layer_wcs_conn)
            self._active_layer_wcs_conn = None

        layer = self.doc.active_layer
        if layer:
            self._active_layer_wcs_conn = layer.updated.connect(
                self._on_active_layer_updated
            )

    def _on_active_layer_changed(self, sender):
        """Reconnect WCS tracking to the new active layer."""
        self._connect_active_layer_wcs()
        machine = self._context.machine
        if machine:
            self._on_wcs_updated(machine)

    def _on_active_layer_updated(self, layer):
        """Handle property changes on the active layer, including WCS."""
        machine = self._context.machine
        if machine:
            self._on_wcs_updated(machine)

    def set_machine(self, viewport: Optional[ViewportConfig] = None):
        old_machine = self._context.machine
        if old_machine:
            old_machine.wcs_updated.disconnect(self._on_wcs_updated)
            old_machine.changed.disconnect(self._on_wcs_updated)

        if viewport is None:
            viewport = ViewportConfig.default()

        self._viewport = viewport

        new_machine = self._context.machine
        if new_machine:
            new_machine.wcs_updated.connect(self._on_wcs_updated)
            new_machine.changed.connect(self._on_wcs_updated)
            self._on_wcs_updated(new_machine)

        if self._gl_initialized:
            self.update_scene_from_doc()

    def _on_pipeline_state_changed(self, sender, *, is_processing: bool):
        """
        Handler for when the pipeline's busy state changes. When it becomes
        not busy, the document has settled and the scene should be updated.
        """
        if not is_processing and self._current_job_handle is not None:
            if self.has_stale_job():
                logger.debug(
                    "Pipeline settled with stale job. Clearing 3D scene."
                )
                self._current_job_handle = None
                self._compiled_artifact = None
                self._artifact_gl_dirty = True
                self.queue_render()
            else:
                logger.debug("Pipeline has settled. Updating 3D scene.")
                self.update_scene_from_doc()

    def _on_job_generation_finished(self, sender, **kwargs):
        task_status = kwargs.get("task_status")
        handle = kwargs.get("handle")
        logger.debug(
            f"[CANVAS3D] _on_job_generation_finished: "
            f"status={task_status}, handle={'yes' if handle else 'none'}"
        )
        if task_status == "completed":
            if handle is not None:
                self._current_job_handle = handle
                self.update_scene_from_doc()
                self.queue_render()
            else:
                logger.debug(
                    "[CANVAS3D] Job completed with no output. Clearing scene."
                )
                self._current_job_handle = None
                self._compiled_artifact = None
                self._artifact_gl_dirty = True
                self.queue_render()

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

    def _connect_pipeline_signals(self):
        if self.pipeline:
            self.pipeline.processing_state_changed.connect(
                self._on_pipeline_state_changed
            )
            self.pipeline.job_generation_finished.connect(
                self._on_job_generation_finished
            )

    def _disconnect_pipeline_signals(self):
        if self.pipeline:
            self.pipeline.processing_state_changed.disconnect(
                self._on_pipeline_state_changed
            )
            self.pipeline.job_generation_finished.disconnect(
                self._on_job_generation_finished
            )

    def on_realize(self, area) -> None:
        """Called when the GLArea is ready to have its context made current."""
        logger.info("GLArea realized.")

        self._cam_ctrl.create_camera(self.get_width(), self.get_height())

        self._init_gl_resources()
        self._theme_resolver.mark_dirty()

        self.reset_view(ViewDirection.ISO)
        self._theme_resolver.update_theme_and_colors()
        self._connect_pipeline_signals()

        if self._current_job_handle is None and self.pipeline:
            self._current_job_handle = self.pipeline.last_completed_handle

        self.update_scene_from_doc()

    def _on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_space and self._playback_overlay:
            self._playback_overlay.handle_space()
            return True
        return False

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self._disconnect_pipeline_signals()
        machine = self._context.machine
        if machine:
            machine.wcs_updated.disconnect(self._on_wcs_updated)
            machine.changed.disconnect(self._on_wcs_updated)
        self._context.config.changed.disconnect(self._on_config_changed)
        try:
            self.make_current()
            if self._scene_preparation_task:
                self._scene_preparation_task.cancel()
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
        if self._artifact_gl_dirty:
            self._artifact_gl_dirty = False
            self._start_chunked_artifact_upload()

    def _start_chunked_artifact_upload(self):
        if not self._compiled_artifact:
            for group in self._scene.layer_groups:
                group.ops_renderer.clear()
            if self._scene.texture_renderer:
                self._scene.texture_renderer.clear()
            self.queue_render()
            return

        if not self._gl_initialized:
            return

        self.make_current()

        # Upload the power colour LUTs before any vertex data. The chunked
        # upload runs on idle callbacks, which can be pre-empted by a
        # redraw between items; a redraw that renders powered lines against
        # an uninitialised LUT would draw them at full brightness.
        upload_items = self._scene.prepare_chunked_upload(
            self._compiled_artifact, self._show_travel_moves
        )

        self._upload_state = {
            "items": upload_items,
            "index": 0,
        }
        GLib.idle_add(self._step_chunked_upload)

    def _step_chunked_upload(self) -> bool:
        if self._upload_state is None:
            return False

        items = self._upload_state["items"]
        idx = self._upload_state["index"]

        if idx >= len(items):
            self._upload_state = None
            self.queue_render()
            return False

        item = items[idx]
        self._upload_state["index"] = idx + 1

        try:
            kind = item[0]

            if kind == "ops":
                _, group, vl, show_travel_moves = item
                group.ops_renderer.update_from_vertex_layer(
                    vl, show_travel_moves
                )

            elif kind == "overlay":
                _, group, ol = item
                group.ring_renderer.update_from_overlay_layer(ol)

            elif kind == "textures":
                _, artifact = item
                if self._scene.texture_renderer:
                    self._scene.texture_renderer.update_from_artifact(artifact)

            elif kind == "color_luts":
                self._theme_resolver.update_renderer_color_luts()

            elif kind == "op_player":
                self._build_op_player_async()
                if self._compiled_artifact and self._op_player:
                    self._scene.extract_playback_offsets(
                        self._compiled_artifact
                    )

        except Exception:
            logger.exception("[CANVAS3D] Error during chunked upload")
            self._upload_state = None
            return False

        return True

    def _build_op_player_async(self):
        ops = self._get_ops_for_playback()
        machine = self._context.machine
        if machine is None:
            return

        if ops is None or ops.is_empty():
            self._op_player = None
            for group in self._scene.layer_groups:
                group.powered_offsets = np.array([], dtype=np.int32)
                group.travel_offsets = np.array([], dtype=np.int32)
                group.ring_offsets = np.array([], dtype=np.int32)
            if self._playback_overlay:
                self._playback_overlay.set_player(None)
            self.queue_render()
            return

        # Preserve the playhead and seek snapshots when the underlying
        # ops object has not changed (e.g. only the viewport moved).
        saved_index = None
        reused_snapshots = []
        if self._op_player is not None and self._op_player.ops is ops:
            saved_index = self._op_player.current_index
            reused_snapshots = self._op_player.snapshots

        player = OpPlayer(ops, machine, self.doc, build_snapshots=False)
        player.set_snapshots(reused_snapshots)

        # Make the player available right away so that the next render
        # can dim textures that have not been reached yet.  Seeking to
        # the first layer is cheap (the first LAYER_START is near the
        # start of the ops), and reused snapshots keep restores of a
        # previous playhead fast as well.
        if saved_index is not None:
            player.seek(saved_index)
            initial_index = saved_index
        else:
            player.seek_to_first_layer()
            initial_index = 0
        self._op_player = player
        if self._playback_overlay:
            self._playback_overlay.set_player(player, initial_index)
        self.queue_render()

        # Build seek-acceleration snapshots in the background.  They are
        # collected into a fresh list and attached from the main thread
        # to avoid racing with concurrent seeks reading _snapshots.
        def _on_snapshots_done(task):
            if task.get_status() != "completed":
                return
            if self._op_player is player:
                player.set_snapshots(task.result())

        def _build_snapshots_thread(ops, machine, doc):
            n = ops.len()
            if n <= 1000:
                return []
            temp = SnapshotBuilder(
                ops, machine, doc, player._create_home_state()
            )
            interval = 1000
            snapshots = []
            for target in range(interval, n, interval):
                temp.advance_to(target - 1)
                temp.state.reached_textures.clear()
                snapshots.append(
                    (
                        target,
                        temp.state.copy(),
                        temp._source_axis,
                        temp._rotary_axis,
                    )
                )
            return snapshots

        task_mgr.run_thread(
            _build_snapshots_thread,
            ops,
            machine,
            self.doc,
            key=(id(self), "build-snapshots"),
            when_done=_on_snapshots_done,
        )

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
            op_player = self._op_player
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
                compiled_artifact=self._compiled_artifact,
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
        self._update_renderers_from_artifact()

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

    def _on_scene_prepared(self, task: Task):
        """
        Callback for when the background scene compilation task is
        finished.  The compiled artifact is available directly as
        ``task.result_value`` since the compilation runs in-process.
        """
        if task.get_status() != "completed":
            if task.is_cancelled():
                logger.debug(
                    "[CANVAS3D] Scene preparation task cancelled (superseded)."
                )
            else:
                self._compiled_artifact = None
                self._op_player = None
                logger.error("[CANVAS3D] Scene preparation task failed.")
                self._artifact_gl_dirty = True
                self.queue_render()
            return

        self._scene_preparation_task = None

        artifact = task.result()
        if artifact is None:
            logger.warning(
                "[CANVAS3D] Scene task completed but produced no "
                "artifact (possibly empty scene)."
            )
            self._compiled_artifact = None
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        if not isinstance(artifact, CompiledSceneArtifact):
            logger.error(
                f"[CANVAS3D] Expected CompiledSceneArtifact, got "
                f"{type(artifact).__name__}"
            )
            self._compiled_artifact = None
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        logger.debug("[CANVAS3D] Scene compilation finished.")
        self._compiled_artifact = artifact
        self._artifact_gl_dirty = True
        self.queue_render()

    def _update_renderers_from_artifact(self):
        if not self._compiled_artifact:
            for group in self._scene.layer_groups:
                group.ops_renderer.clear()
                group.ring_renderer.clear()
                group.ring_offsets = []
            if self._scene.texture_renderer:
                self._scene.texture_renderer.clear()
            self.queue_render()
            return

        if not self._gl_initialized:
            return

        self.make_current()

        self._scene.update_from_artifact(
            self._compiled_artifact, self._show_travel_moves
        )

        self._theme_resolver.update_renderer_color_luts()

        logger.debug(
            "[CANVAS3D] Scanline overlay uploaded. Groups: %s"
            % ", ".join(
                "%s:%d"
                % (
                    "rot" if g.is_rotary else "flat",
                    g.ring_renderer.vertex_count,
                )
                for g in self._scene.layer_groups
            )
        )

        self.queue_render()

    def _get_ops_for_playback(self) -> Optional[Ops]:
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact):
                if artifact.mapped_ops is not None:
                    return artifact.mapped_ops
                if artifact.ops is not None:
                    return artifact.ops
        return None

    def update_scene_from_doc(self):
        """
        Updates the entire scene content from the document. This is the main
        entry point for refreshing the 3D view.
        """
        if not self._gl_initialized:
            return
        if not self._scene.texture_renderer:
            return

        t_update_start = time.perf_counter()
        logger.debug("Canvas3D: Updating scene from document.")

        # Theme/color updates only need to happen once per theme change
        if self._theme_resolver.theme_is_dirty:
            self._theme_resolver.update_theme_and_colors()
        if not self._theme_resolver.color_set:
            logger.warning("Cannot update scene, color set not resolved.")
            return

        # Update cylinder renderers and camera based on layer rotary state
        any_rotary = any(layer.rotary_enabled for layer in self.doc.layers)
        self._scene_gl_dirty = True
        if (
            self._scene.had_rotary_layers
            and not any_rotary
            and self._cam_ctrl.camera
        ):
            self.reset_view(ViewDirection.ISO)
        self._scene.had_rotary_layers = any_rotary

        world_to_visual = np.identity(4, dtype=np.float32)
        world_to_cyl_local = np.identity(4, dtype=np.float32)

        machine = self._context.machine
        if machine:
            ms = self._viewport.margin_shift
            wcs = self._viewport.wcs_offset_mm
            world_to_visual[0, 3] = ms[0, 3]
            world_to_visual[1, 3] = ms[1, 3]
            world_to_visual[2, 3] = wcs[2]

            asm = machine.assembly
            if asm.has_rotary:
                self._scene.set_cylinder_transform(
                    asm.cylinder_base_transform()
                )
            else:
                self._scene.set_cylinder_transform(np.eye(4, dtype=np.float64))

        layer_configs: Dict[str, LayerRenderConfig] = {}
        for layer in self.doc.layers:
            axis_position = 0.0
            reverse = False
            axis_position_3d = None
            cylinder_dir = None
            if layer.rotary_enabled and machine:
                cfg = resolve_layer_rotary(layer, machine)
                module = cfg.module
                if module is not None:
                    mapping = KinematicMapping.from_rotary_module(
                        module,
                        layer.rotary_diameter,
                        apply_gear_ratio=False,
                    )
                    if mapping is not None:
                        axis_position = mapping.axis_position
                        axis_position_3d = tuple(
                            mapping.axis_position_3d.tolist()
                        )
                        cylinder_dir = tuple(mapping.cylinder_dir.tolist())
                        reverse = mapping.reverse
            layer_configs[layer.uid] = LayerRenderConfig(
                rotary_enabled=layer.rotary_enabled,
                rotary_diameter=layer.rotary_diameter,
                axis_position=axis_position,
                reverse=reverse,
                axis_position_3d=axis_position_3d,
                cylinder_dir=cylinder_dir,
            )

        render_config = RenderConfig3D(
            world_to_visual=world_to_visual,
            world_to_cyl_local=world_to_cyl_local,
            layer_configs=layer_configs,
        )

        self._schedule_scene_preparation(render_config.to_dict())

        t_update_elapsed = (time.perf_counter() - t_update_start) * 1000
        if t_update_elapsed > 5:
            logger.info(
                f"[CANVAS3D] update_scene_from_doc took "
                f"{t_update_elapsed:.1f}ms"
            )

    def _schedule_scene_preparation(
        self,
        render_config_dict: Dict,
    ):
        task_key = (id(self), "prepare-3d-scene-vertices")

        if not self._gl_initialized or self._theme_resolver.color_set is None:
            return

        job_handle = self._current_job_handle
        if job_handle is None:
            logger.debug("[CANVAS3D] No job artifact, skipping compilation.")
            return

        if self._scene_preparation_task:
            self._scene_preparation_task.cancel()
            self._scene_preparation_task = None
            logger.debug(
                "[CANVAS3D] Cancelled in-progress compilation, "
                "scheduling new one."
            )

        logger.debug("[CANVAS3D] Scheduling scene compilation task.")
        assert render_config_dict is not None
        self._scene_preparation_task = task_mgr.run_thread(
            compile_scene_in_thread,
            self._context.artifact_store,
            job_handle.to_dict(),
            render_config_dict,
            key=task_key,
            when_done=self._on_scene_prepared,
        )
