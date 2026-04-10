import logging
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
import numpy as np
from gi.repository import GLib, Gdk, Gtk, Pango
from OpenGL import GL
from ....context import RayforgeContext
from ....core.geo import Point
from ....core.model import Model
from ....core.ops import Ops
from ....core.ops.commands import LayerStartCommand
from ....machine.assembly import LinkRole
from ....machine.models.colors import OpsColorSet
from ....pipeline.artifact.base import TextureData
from ....pipeline.artifact.handle import create_handle_from_dict
from ....pipeline.artifact.job import JobArtifact, JobArtifactHandle
from ....pipeline.pipeline import Pipeline
from ....shared.util.colors import ColorSet, hex_to_rgba
from ....shared.tasker import task_mgr, Task
from ....simulator.machine_state import MachineState
from ....simulator.op_player import OpPlayer
from ...shared.gtk_color import GtkColorResolver, ColorSpecDict
from .axis_renderer_3d import AxisRenderer3D
from .background_renderer import BackgroundRenderer
from .camera import Camera, ViewDirection, rotation_matrix_from_axis_angle
from ..scene3d import (
    CompiledSceneArtifact,
    LayerRenderConfig,
    RenderConfig3D,
    compile_scene_in_subprocess,
)
from .cylinder_renderer import CylinderRenderer
from .gl_utils import Shader
from .laser_beam_renderer import LaserBeamRenderer
from .model_renderer import ModelRenderer
from .ops_renderer import OpsRenderer
from .ring_buffer_renderer import RingBufferRenderer
from .shaders import (
    SIMPLE_FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER,
    TEXT_FRAGMENT_SHADER,
    TEXT_VERTEX_SHADER,
    TEXTURE_FRAGMENT_SHADER,
    TEXTURE_VERTEX_SHADER,
)
from .sphere_renderer import SphereRenderer
from .texture_renderer import TextureArtifactRenderer
from .viewport import ViewportConfig
from .zone_renderer import ZoneRenderer

if TYPE_CHECKING:
    from ....core.doc import Doc
    from ....doceditor.editor import DocEditor
    from ....machine.models.machine import Machine

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class _LayerRendererGroup:
    """Owns the GPU renderers and playback offsets for one compiled layer."""

    def __init__(self, is_rotary: bool):
        self.is_rotary = is_rotary
        self.ops_renderer = OpsRenderer()
        self.ring_renderer = RingBufferRenderer()
        self.powered_offsets: List[int] = []
        self.travel_offsets: List[int] = []
        self.ring_offsets: List[int] = []

    def init_gl(self):
        self.ops_renderer.init_gl()
        self.ring_renderer.init_gl()

    def cleanup(self):
        self.ops_renderer.cleanup()
        self.ring_renderer.cleanup()

    def clear(self):
        self.ops_renderer.clear()
        self.ring_renderer.clear()
        self.powered_offsets = []
        self.travel_offsets = []
        self.ring_offsets = []


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

        self.camera: Optional[Camera] = None
        self._main_shader: Optional[Shader] = None
        self._text_shader: Optional[Shader] = None
        self._axis_renderer: Optional[AxisRenderer3D] = None
        self._layer_groups: List[_LayerRendererGroup] = []
        self._sphere_renderer: Optional[SphereRenderer] = None
        self._laser_beam_renderer: Optional[LaserBeamRenderer] = None
        self._background_renderer: Optional[BackgroundRenderer] = None
        self._texture_renderer: Optional[TextureArtifactRenderer] = None
        self._texture_shader: Optional[Shader] = None
        self._cylinder_renderers: Dict[float, CylinderRenderer] = {}
        self._model_renderers: List[Tuple[ModelRenderer, str]] = []
        self._zone_renderer: Optional[ZoneRenderer] = None
        self._had_rotary_layers = False
        self._scene_preparation_task: Optional[Task] = None
        self._compiled_artifact: Optional[CompiledSceneArtifact] = None
        self._pending_scene_handle_dict: Optional[dict] = None
        self._current_job_handle: Optional[JobArtifactHandle] = None
        self._op_player: Optional[OpPlayer] = None
        self._playback_overlay = None
        self._world_to_cyl_local = np.identity(4, dtype=np.float32)
        self._show_travel_moves = False
        self._show_nogo_zones = True
        self._show_models = True
        self._is_orbiting = False
        self._is_z_rotating = False
        self._gl_initialized = False
        self._scene_gl_dirty = False
        self._artifact_gl_dirty = False
        self._current_layer_uid = None

        self._color_spec: ColorSpecDict = {
            "cut": ("#ff00ff22", "#ff00ff"),
            "engrave": ("#00000009", "#000000"),
            "travel": ("#FF6600", 0.7),
            "zero_power": ("@accent_color", 0.5),
        }
        self._color_set: Optional[ColorSet] = None
        self._laser_color_sets: Dict[str, ColorSet] = {}
        self._theme_is_dirty = True

        # State for interactions
        self._last_pan_offset: Optional[Point] = None
        self._rotation_pivot: Optional[np.ndarray] = None
        self._last_orbit_pos: Optional[Point] = None
        self._last_z_rotate_screen_pos: Optional[Point] = None

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self.on_resize)
        self.connect("notify::style", self._on_style_changed)
        self._setup_interactions()

        # Connect to machine for WCS updates
        machine = self._context.machine
        if machine:
            machine.wcs_updated.connect(self._on_wcs_updated)
            machine.changed.connect(self._on_wcs_updated)
            self._on_wcs_updated(machine)

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

    def set_playback_overlay(self, overlay):
        """Set the playback overlay widget for slider sync."""
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
            self._viewport = ViewportConfig.from_machine(machine)
        self._scene_gl_dirty = True
        self.queue_render()

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
            logger.debug("Pipeline has settled. Updating 3D scene.")
            self.update_scene_from_doc()

    def _on_job_generation_finished(self, sender, **kwargs):
        task_status = kwargs.get("task_status")
        handle = kwargs.get("handle")
        logger.debug(
            f"[CANVAS3D] _on_job_generation_finished: "
            f"status={task_status}, handle={'yes' if handle else 'none'}"
        )
        if task_status == "completed" and handle is not None:
            self._current_job_handle = handle
            self.update_scene_from_doc()
            self.queue_render()

    def _on_style_changed(self, widget, gparam):
        """Marks theme resources as dirty when the GTK theme changes."""
        self._theme_is_dirty = True
        self.queue_render()

    def get_world_coords_on_plane(
        self, x: float, y: float, camera: Camera
    ) -> Optional[np.ndarray]:
        """Calculates the 3D world coordinates on the XY plane from 2D."""
        ndc_x = (2.0 * x) / camera.width - 1.0
        ndc_y = 1.0 - (2.0 * y) / camera.height

        try:
            inv_proj = np.linalg.inv(camera.get_projection_matrix())
            inv_view = np.linalg.inv(camera.get_view_matrix())
        except np.linalg.LinAlgError:
            return None

        clip_coords = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        eye_coords = inv_proj @ clip_coords
        eye_coords[2] = -1.0
        eye_coords[3] = 0.0

        world_coords_vec4 = inv_view @ eye_coords
        ray_dir = world_coords_vec4[:3] / np.linalg.norm(world_coords_vec4[:3])
        ray_origin = camera.position

        plane_normal = np.array([0, 0, 1], dtype=np.float64)
        denom = np.dot(plane_normal, ray_dir)
        if abs(denom) < 1e-6:
            return None

        t = -np.dot(plane_normal, ray_origin) / denom
        if t < 0:
            return None

        return ray_origin + t * ray_dir

    def _setup_interactions(self):
        """Connects GTK4 gesture and event controllers for interaction."""
        # Middle mouse drag for Pan/Orbit
        drag_middle = Gtk.GestureDrag.new()
        drag_middle.set_button(Gdk.BUTTON_MIDDLE)
        drag_middle.connect("drag-begin", self.on_drag_begin)
        drag_middle.connect("drag-update", self.on_drag_update)
        drag_middle.connect("drag-end", self.on_drag_end)
        self.add_controller(drag_middle)

        # Left mouse drag for Z-axis rotation
        drag_left = Gtk.GestureDrag.new()
        drag_left.set_button(Gdk.BUTTON_PRIMARY)
        drag_left.connect("drag-begin", self.on_z_rotate_begin)
        drag_left.connect("drag-update", self.on_z_rotate_update)
        drag_left.connect("drag-end", self.on_z_rotate_end)
        self.add_controller(drag_left)

        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self.on_scroll)
        self.add_controller(scroll)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self._on_key_pressed)
        self.add_controller(key_controller)

    def _on_key_pressed(self, controller, keyval, keycode, state):
        if keyval == Gdk.KEY_space and self._playback_overlay:
            if self._playback_overlay._play_button.get_sensitive():
                self._playback_overlay._on_play_clicked(
                    self._playback_overlay._play_button
                )
            return True
        return False

    def _clear_drag_state(self):
        """Resets all state variables related to any drag operation."""
        self._is_orbiting = False
        self._is_z_rotating = False
        self._last_pan_offset = None
        self._rotation_pivot = None
        self._last_orbit_pos = None
        self._last_z_rotate_screen_pos = None

    def reset_view(self, direction: ViewDirection):
        """Resets the camera to the specified preset view."""
        if not self.camera:
            return
        logger.info("Resetting to %s view.", direction.value)
        self.camera.set_view(
            direction,
            self._viewport.width_mm,
            self._viewport.depth_mm,
        )
        self._clear_drag_state()
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

        self.camera = Camera(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            self.get_width(),
            self.get_height(),
        )

        self._sphere_renderer = SphereRenderer(1.0, 16, 32)
        self._laser_beam_renderer = LaserBeamRenderer()
        self._background_renderer = BackgroundRenderer()

        self._init_gl_resources()
        self._theme_is_dirty = True

        self.reset_view(ViewDirection.ISO)
        self._update_theme_and_colors()
        self._connect_pipeline_signals()

        if self._current_job_handle is None and self.pipeline:
            self._current_job_handle = self.pipeline.last_completed_handle

        self.update_scene_from_doc()

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self._disconnect_pipeline_signals()
        machine = self._context.machine
        if machine:
            machine.wcs_updated.disconnect(self._on_wcs_updated)
            machine.changed.disconnect(self._on_wcs_updated)
        self._release_pending_scene_handle()
        try:
            self.make_current()
            if self._scene_preparation_task:
                self._scene_preparation_task.cancel()
            if self._axis_renderer:
                self._axis_renderer.cleanup()
            for group in self._layer_groups:
                group.cleanup()
            if self._sphere_renderer:
                self._sphere_renderer.cleanup()
            if self._laser_beam_renderer:
                self._laser_beam_renderer.cleanup()
            if self._background_renderer:
                self._background_renderer.cleanup()
            if self._texture_renderer:
                self._texture_renderer.cleanup()
            if self._zone_renderer:
                self._zone_renderer.cleanup()
            for renderer in self._cylinder_renderers.values():
                renderer.cleanup()
            for renderer, _ in self._model_renderers:
                renderer.cleanup()
            if self._main_shader:
                self._main_shader.cleanup()
            if self._text_shader:
                self._text_shader.cleanup()
            if self._texture_shader:
                self._texture_shader.cleanup()
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

            self._main_shader = Shader(
                SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER
            )
            self._text_shader = Shader(
                TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER
            )
            self._texture_shader = Shader(
                TEXTURE_VERTEX_SHADER, TEXTURE_FRAGMENT_SHADER
            )

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
                    font_family = font_desc.get_family()
                    logger.debug(f"Pango normalized font to {font_family}")

            self._axis_renderer = AxisRenderer3D(
                self._viewport.width_mm,
                self._viewport.depth_mm,
                font_family=font_family,
            )
            self._apply_extent_frame(self._viewport)
            self._axis_renderer.init_gl()
            self._texture_renderer = TextureArtifactRenderer()
            self._texture_renderer.init_gl()
            if self._sphere_renderer:
                self._sphere_renderer.init_gl()
            if self._laser_beam_renderer:
                self._laser_beam_renderer.init_gl()
            try:
                if self._background_renderer:
                    self._background_renderer.init_gl()
            except Exception as e:
                logger.warning(
                    "Background renderer init failed, "
                    "falling back to clear color: %s",
                    e,
                )
                self._background_renderer = None
            self._zone_renderer = ZoneRenderer()
            self._zone_renderer.init_gl()

            self._gl_initialized = True
        except Exception as e:
            logger.error(f"OpenGL Initialization Error: {e}", exc_info=True)
            self._gl_initialized = False

    def _update_theme_and_colors(self):
        """
        Resolves the ColorSet and updates other theme-dependent elements.
        """
        if not self._axis_renderer or not self._texture_renderer:
            return

        resolver = GtkColorResolver(self)
        self._color_set = resolver.resolve(self._color_spec)

        if self._color_set:
            self._texture_renderer.update_color_lut(
                self._color_set.get_lut("engrave")
            )

        style_context = self.get_style_context()
        found, bg_rgba = style_context.lookup_color("theme_bg_color")
        if not found:
            found, bg_rgba = style_context.lookup_color("theme_bg_color")

        if found:
            bg_color = (
                bg_rgba.red * 0.35,
                bg_rgba.green * 0.35,
                bg_rgba.blue * 0.35,
            )
            bg_light = (
                min(1.0, bg_rgba.red * 0.9),
                min(1.0, bg_rgba.green * 0.9),
                min(1.0, bg_rgba.blue * 0.9),
            )
            clear_color = (
                bg_rgba.red,
                bg_rgba.green,
                bg_rgba.blue,
                bg_rgba.alpha,
            )
        else:
            bg_color = (0.11, 0.11, 0.14)
            bg_light = (0.2, 0.2, 0.25)
            clear_color = (0.2, 0.2, 0.25, 1.0)

        if self._background_renderer:
            self._background_renderer.set_colors(bg_color, bg_light)

        GL.glClearColor(*clear_color)

        # Get the foreground color for axes and labels
        found, fg_rgba = style_context.lookup_color("view_fg_color")
        if found:
            axis_color = (
                fg_rgba.red,
                fg_rgba.green,
                fg_rgba.blue,
                fg_rgba.alpha,
            )
            # Grid color is derived from fg color to be less prominent
            grid_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.5
            bg_plane_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.08

            self._axis_renderer.set_background_color(bg_plane_color)
            self._axis_renderer.set_axis_color(axis_color)
            self._axis_renderer.set_label_color(axis_color)
            self._axis_renderer.set_grid_color(grid_color)

        self._update_laser_colors()
        self._theme_is_dirty = False

    def _update_laser_colors(self):
        """
        Resolve ColorSet for each laser in the machine.

        This builds per-laser color sets using the machine's laser list
        and the current theme colors for travel/zero_power.
        """
        machine = self._context.machine
        if not machine or not self._color_set:
            self._laser_color_sets = {}
            return

        self._laser_color_sets = {}
        for laser in machine.heads:
            laser_color_set = OpsColorSet.from_laser(laser, self._color_set)
            self._laser_color_sets[laser.uid] = laser_color_set.to_color_set()

    def _process_pending_gl_updates(self):
        if self._scene_gl_dirty:
            self._scene_gl_dirty = False
            self._update_axis_renderer()
            self._update_cylinder_renderers()
            self._update_model_renderers()
            self._update_zone_renderer()
        if self._artifact_gl_dirty:
            self._artifact_gl_dirty = False
            self._update_renderers_from_artifact()
            self._update_op_player()

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
        if machine and machine.heads:
            spot_mm = machine.heads[0].spot_size_mm[0]
        if not self.camera:
            return 2.0
        px = self._world_size_to_pixels(
            mvp_gl, spot_mm, self.camera.width, self.camera.height
        )
        return max(2.0, px)

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self.camera or not self._gl_initialized:
            return False

        self._process_pending_gl_updates()

        machine = self._context.machine
        if (
            machine
            and self._op_player
            and self._op_player.state.current_layer_uid
            != self._current_layer_uid
        ):
            uid = self._op_player.state.current_layer_uid
            self._current_layer_uid = uid
            layer = None
            if uid:
                for lay in self.doc.layers:
                    if lay.uid == uid:
                        layer = lay
                        break
            machine.configure_for_layer(layer)
            self._scene_gl_dirty = True
            self._process_pending_gl_updates()

        if self._theme_is_dirty:
            self._update_theme_and_colors()

        if not self._color_set:
            return False

        t_render_start = time.perf_counter()
        try:
            GL.glViewport(0, 0, self.camera.width, self.camera.height)
            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
            )

            if self._background_renderer:
                self._background_renderer.render()

            proj_matrix = self.camera.get_projection_matrix()
            view_matrix = self.camera.get_view_matrix()

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

            # Final MVP for scene geometry (grid, axes)
            mvp_matrix_scene = mvp_matrix_ui @ grid_model_matrix

            # Convert to column-major for OpenGL
            mvp_matrix_ui_gl = mvp_matrix_ui.T
            mvp_matrix_scene_gl = mvp_matrix_scene.T

            if (
                self._axis_renderer is not None
                and self._main_shader is not None
                and self._text_shader is not None
            ):
                self._axis_renderer.render(
                    self._main_shader,
                    self._text_shader,
                    mvp_matrix_scene_gl,
                    mvp_matrix_ui_gl,
                    view_matrix,
                    self._viewport.model_matrix,
                    origin_offset_mm=self._viewport.wcs_offset_mm,
                    x_right=self._viewport.x_right,
                    y_down=self._viewport.y_down,
                    x_negative=self._viewport.x_negative,
                    y_negative=self._viewport.y_negative,
                )

            margin_shift = self._viewport.margin_shift
            machine = self._context.machine

            if (
                self._zone_renderer
                and self._main_shader
                and self._show_nogo_zones
            ):
                zone_mvp_gl = (mvp_matrix_ui @ margin_shift).T
                self._zone_renderer.render(self._main_shader, zone_mvp_gl)

            # Compute cylinder rotation from assembly.
            cyl_angle = 0.0
            if self._op_player and self._had_rotary_layers and machine:
                asm = machine.assembly
                if asm.has_rotary:
                    chuck_links = asm.get_links_by_role(LinkRole.CHUCK)
                    if chuck_links:
                        driver_axis = chuck_links[0].driver_axis
                        assert driver_axis is not None
                        raw_y = self._op_player.state.axes.get(
                            driver_axis, 0.0
                        )
                        cyl_pos = self._world_to_cyl_local @ np.array(
                            [0, raw_y, 0, 1.0], dtype=np.float32
                        )
                        cyl_local_y = cyl_pos[1]
                        diameter = asm.rotary_diameter
                        assert diameter is not None
                        circumference = diameter * np.pi
                        cyl_angle = (cyl_local_y / circumference) * 2 * np.pi

            rot_cyl_gl = mvp_matrix_scene_gl
            if abs(cyl_angle) > 1e-9:
                rot_3x3 = rotation_matrix_from_axis_angle(
                    np.array([1.0, 0.0, 0.0]), cyl_angle
                )
                rot_4x4 = np.eye(4, dtype=np.float64)
                rot_4x4[:3, :3] = rot_3x3
                rot_cyl = mvp_matrix_scene @ rot_4x4
                rot_cyl_gl = rot_cyl.T.astype(np.float32)

            # Render each layer group (ops + ring buffer)
            spot_line_width = self._compute_spot_line_width(mvp_matrix_ui_gl)
            for group in self._layer_groups:
                mvp = rot_cyl_gl if group.is_rotary else mvp_matrix_ui_gl

                exec_powered = -1
                exec_travel = -1
                exec_ring = -1
                if self._op_player:
                    idx = self._op_player.current_index
                    if group.powered_offsets and idx + 1 < len(
                        group.powered_offsets
                    ):
                        exec_powered = group.powered_offsets[idx + 1]
                    if group.travel_offsets and idx + 1 < len(
                        group.travel_offsets
                    ):
                        exec_travel = group.travel_offsets[idx + 1]
                    if group.ring_offsets and idx + 1 < len(
                        group.ring_offsets
                    ):
                        exec_ring = group.ring_offsets[idx + 1]

                    pv_total = group.ops_renderer.powered_vertex_count
                    off_len = len(group.powered_offsets)
                    if exec_powered >= 0 and idx % 50 == 0:
                        tag = "rot" if group.is_rotary else "flat"
                        logger.info(
                            f"[PLAYBACK-DIAG] {tag} "
                            f"idx={idx}/{off_len - 1} "
                            f"exec={exec_powered}/{pv_total} "
                            f"off[-3:]="
                            f"{group.powered_offsets[-3:]}"
                        )

                if self._main_shader:
                    group.ops_renderer.render(
                        self._main_shader,
                        mvp,
                        colors=self._color_set,
                        show_travel_moves=self._show_travel_moves,
                        executed_vertex_count=exec_powered,
                        executed_travel_vertex_count=exec_travel,
                        line_width=spot_line_width,
                    )

                if group.ring_renderer.vertex_count > 0 and self._main_shader:
                    tag = "rot" if group.is_rotary else "flat"
                    logger.debug(
                        f"[RING-PLAYBACK] {tag} "
                        f"exec={exec_ring} "
                        f"total={group.ring_renderer.vertex_count}"
                    )
                    group.ring_renderer.render(
                        self._main_shader,
                        mvp,
                        executed_vertex_count=exec_ring,
                        line_width=spot_line_width,
                    )

            laser_light_pos = None

            # Laser head markers
            if (
                self._op_player
                and self._sphere_renderer
                and self._main_shader
                and machine
            ):
                asm = machine.assembly
                wcs = self._viewport.wcs_offset_mm
                heads = asm.head_positions(
                    self._op_player.state, wcs_offset=wcs
                )
                for name, (hx, hy, hz) in heads.items():
                    head_pos = margin_shift @ np.array(
                        [hx, hy, hz, 1.0], dtype=np.float32
                    )
                    beam_height = 50.0
                    beam_color = (1.0, 0.3, 0.1, 1.0)
                    if name.startswith("head_"):
                        try:
                            idx = int(name.split("_")[1])
                            laser = machine.heads[idx]
                            if laser.focal_distance > 0:
                                beam_height = laser.focal_distance
                            beam_color = hex_to_rgba(laser.cut_color)
                        except (ValueError, IndexError):
                            pass
                    if (
                        self._laser_beam_renderer
                        and self._main_shader
                        and self._op_player.state.laser_on
                    ):
                        if self._had_rotary_layers and asm.has_rotary:
                            beam_pos = head_pos.copy()
                            beam_pos[1] = grid_model_matrix[1, 3]
                            diameter = asm.rotary_diameter
                            if diameter and diameter > 0:
                                beam_pos[2] += diameter / 2.0
                        else:
                            beam_pos = head_pos.copy()
                        self._laser_beam_renderer.render(
                            self._main_shader,
                            proj_matrix,
                            view_matrix,
                            beam_pos[:3],
                            beam_height=beam_height,
                            viewport_height=self.camera.height,
                            color=beam_color,
                        )
                        laser_light_pos = beam_pos[:3].astype(np.float32)

            for renderer in self._cylinder_renderers.values():
                if self._main_shader:
                    renderer.render(self._main_shader, rot_cyl_gl)

            if self._show_models and self._model_renderers and machine:
                asm = machine.assembly
                wcs = self._viewport.wcs_offset_mm
                if self._op_player:
                    model_transforms = asm.model_world_transforms(
                        self._op_player.state, wcs_offset=wcs
                    )
                else:
                    model_transforms = asm.model_world_transforms(
                        MachineState(), wcs_offset=wcs
                    )
                is_rotary = self._had_rotary_layers and asm.has_rotary
                for renderer, link_name in self._model_renderers:
                    if self._main_shader:
                        t = model_transforms.get(link_name)
                        if t is None:
                            continue
                        module_transform = t.astype(np.float32)
                        if is_rotary:
                            link = asm.get_link(link_name)
                            if link and link.role != LinkRole.CHUCK:
                                module_transform[1, 3] = (
                                    grid_model_matrix[1, 3]
                                    - margin_shift[1, 3]
                                )
                                diameter = asm.rotary_diameter
                                if diameter and diameter > 0:
                                    module_transform[2, 3] += diameter / 2.0
                        combined = (
                            mvp_matrix_ui @ margin_shift @ module_transform
                        )
                        renderer.render(
                            self._main_shader,
                            combined.T,
                            model_matrix=margin_shift @ module_transform,
                            camera_position=self.camera.position,
                            point_light_pos=laser_light_pos,
                        )

            tex_reached = None
            if self._op_player and self._compiled_artifact:
                tex_reached = 0
                playhead = self._op_player.current_index
                for tl in self._compiled_artifact.texture_layers:
                    if playhead >= tl.activation_cmd_idx:
                        tex_reached += 1
            if self._texture_renderer and self._texture_shader:
                self._texture_renderer.render(
                    mvp_matrix_ui,
                    self._texture_shader,
                    reached_count=tex_reached,
                )
                rot_cyl_mvp = mvp_matrix_scene
                if abs(cyl_angle) > 1e-9:
                    rot_3x3 = rotation_matrix_from_axis_angle(
                        np.array([1.0, 0.0, 0.0]), cyl_angle
                    )
                    rot_4x4 = np.eye(4, dtype=np.float64)
                    rot_4x4[:3, :3] = rot_3x3
                    rot_cyl_mvp = mvp_matrix_scene @ rot_4x4
                self._texture_renderer.render_cylinder(
                    rot_cyl_mvp,
                    self._texture_shader,
                    reached_count=tex_reached,
                )

        except Exception as e:
            logger.error("OpenGL Render Error: %s", e, exc_info=True)
            return False

        t_render_elapsed = (time.perf_counter() - t_render_start) * 1000
        if t_render_elapsed > 16:
            logger.info(f"[CANVAS3D] on_render took {t_render_elapsed:.1f}ms")
        return True

    def on_resize(self, area, width: int, height: int):
        """Handles the window resize event."""
        if self.camera:
            self.camera.width, self.camera.height = int(width), int(height)
        self.queue_render()

    def on_drag_begin(self, gesture, x: float, y: float):
        """Handles the start of a middle-mouse-button drag."""
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        state = gesture.get_current_event_state()
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        if not is_shift and self.camera:
            if self.camera.is_perspective:
                # For perspective, pick a point on the floor plane to orbit.
                self._rotation_pivot = self.get_world_coords_on_plane(
                    x, y, self.camera
                )
                if self._rotation_pivot is None:
                    self._rotation_pivot = self.camera.target.copy()
            else:  # Orthographic
                # For ortho, always orbit around the camera's current look-at
                # point. This is stable and intuitive.
                self._rotation_pivot = self.camera.target.copy()

            self._last_orbit_pos = None
            self._is_orbiting = True
        else:
            self._last_pan_offset = 0.0, 0.0
            self._is_orbiting = False

    def on_drag_update(self, gesture, offset_x: float, offset_y: float):
        """Handles updates during a drag operation (panning or orbiting)."""
        if not self.camera:
            return

        state = gesture.get_current_event_state()
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        if is_shift:
            if self._last_pan_offset is None:
                self._last_pan_offset = 0.0, 0.0
            dx = offset_x - self._last_pan_offset[0]
            dy = offset_y - self._last_pan_offset[1]
            self.camera.pan(-dx, -dy)
            self._last_pan_offset = offset_x, offset_y
        else:  # CAD-style Orbit Logic
            if not self._is_orbiting or self._rotation_pivot is None:
                return

            event = gesture.get_last_event()
            if not event:
                return
            _, x_curr, y_curr = event.get_position()

            if self._last_orbit_pos is None:
                self._last_orbit_pos = x_curr, y_curr
                return

            prev_x, prev_y = self._last_orbit_pos
            self._last_orbit_pos = x_curr, y_curr
            delta_x = x_curr - prev_x
            delta_y = y_curr - prev_y

            sensitivity = 0.004

            if self.camera.is_perspective:
                # Perspective orbit (Turntable Style)
                if abs(delta_x) > 1e-6:
                    axis_yaw = np.array([0, 1, 0], dtype=np.float64)
                    self.camera.orbit(
                        self._rotation_pivot, axis_yaw, -delta_x * sensitivity
                    )
                if abs(delta_y) > 1e-6:
                    forward = self.camera.target - self.camera.position
                    axis_pitch = np.cross(forward, self.camera.up)
                    if np.linalg.norm(axis_pitch) > 1e-6:
                        self.camera.orbit(
                            self._rotation_pivot,
                            axis_pitch,
                            -delta_y * sensitivity,
                        )
            else:
                # Orthographic orbit (Z-Up Turntable)
                yaw_angle = -delta_x * sensitivity
                pitch_angle = -delta_y * sensitivity

                # Yaw Rotation (around World Z axis)
                if abs(yaw_angle) > 1e-6:
                    axis_yaw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
                    rot_yaw = rotation_matrix_from_axis_angle(
                        axis_yaw, yaw_angle
                    )
                    # Apply to position and up vectors
                    self.camera.position = self._rotation_pivot + rot_yaw @ (
                        self.camera.position - self._rotation_pivot
                    )
                    self.camera.up = rot_yaw @ self.camera.up

                # Pitch Rotation (around Camera's local right axis)
                if abs(pitch_angle) > 1e-6:
                    # Get camera's state *after* the yaw rotation
                    forward_vec = self.camera.target - self.camera.position
                    world_z_axis = np.array([0.0, 0.0, 1.0])

                    # Gimbal Lock Prevention
                    norm_fwd = np.linalg.norm(forward_vec)
                    if norm_fwd > 1e-6:
                        dot_prod = np.dot(forward_vec / norm_fwd, world_z_axis)
                        # Stop if looking down and trying to pitch more down
                        if dot_prod < -0.999 and pitch_angle < 0:
                            pitch_angle = 0.0
                        # Stop if looking up and trying to pitch more up
                        elif dot_prod > 0.999 and pitch_angle > 0:
                            pitch_angle = 0.0

                    if abs(pitch_angle) > 1e-6:
                        axis_pitch = np.cross(forward_vec, self.camera.up)
                        if np.linalg.norm(axis_pitch) > 1e-6:
                            rot_pitch = rotation_matrix_from_axis_angle(
                                axis_pitch, pitch_angle
                            )
                            # Apply to position and up vectors
                            self.camera.position = (
                                self._rotation_pivot
                                + rot_pitch
                                @ (self.camera.position - self._rotation_pivot)
                            )
                            self.camera.up = rot_pitch @ self.camera.up

                # Ensure target is always correct
                self.camera.target = self._rotation_pivot

        self.queue_render()

    def on_drag_end(self, gesture, offset_x, offset_y):
        """Handles the end of a drag operation."""
        self._clear_drag_state()
        self.queue_render()

    def on_z_rotate_begin(self, gesture, x: float, y: float):
        """
        Handles the start of a left-mouse-button drag for Z-axis rotation.
        """
        if not self.camera:
            return
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        self._is_z_rotating = True
        self._last_z_rotate_screen_pos = None  # Will be set on first update

    def on_z_rotate_update(self, gesture, offset_x: float, offset_y: float):
        """Handles updates during a Z-axis rotation drag (linear motion)."""
        if not self.camera or not self._is_z_rotating:
            return

        # Initialize the last position with the current offset if it's None.
        # This handles the start of the drag smoothly.
        if self._last_z_rotate_screen_pos is None:
            self._last_z_rotate_screen_pos = (0.0, 0.0)

        prev_off_x, _ = self._last_z_rotate_screen_pos

        # Calculate delta from the last frame's offset
        delta_x = offset_x - prev_off_x

        # Update the stored offset for the next frame
        self._last_z_rotate_screen_pos = (offset_x, offset_y)

        # Apply rotation. Dragging left/right rotates around world Z.
        # Sensitivity: Radians per pixel.
        sensitivity = 0.01
        angle = -delta_x * sensitivity

        axis_z = np.array([0, 0, 1], dtype=np.float64)
        pivot_world = self.camera.target
        self.camera.orbit(pivot_world, axis_z, angle)
        self.queue_render()

    def on_z_rotate_end(self, gesture, offset_x, offset_y):
        """Handles the end of a Z-axis rotation drag."""
        self._clear_drag_state()
        self.queue_render()

    def on_scroll(self, controller, dx, dy):
        """Handles the mouse scroll wheel for zooming."""
        if self.camera:
            self.camera.dolly(dy)
            self.queue_render()

    def set_show_travel_moves(self, visible: bool):
        """Sets the visibility of travel moves in the 3D view."""
        if self._show_travel_moves == visible:
            return
        self._show_travel_moves = visible
        self._update_renderers_from_artifact()
        self._extract_playback_offsets_from_artifact()
        self._upload_scanline_overlay()

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

    def _release_scene_shm(self, handle_dict: dict):
        """Release a compiled scene SHM block from the artifact store."""
        try:
            handle = create_handle_from_dict(handle_dict)
            self._context.artifact_store.release(handle)
        except Exception:
            logger.debug(
                "[CANVAS3D] Failed to release scene SHM", exc_info=True
            )

    def _release_pending_scene_handle(self):
        """Release the pending compiled scene SHM, if any."""
        if self._pending_scene_handle_dict is not None:
            self._release_scene_shm(self._pending_scene_handle_dict)
            self._pending_scene_handle_dict = None

    def _on_scene_compiled_event(
        self, task: Task, event_name: str, data: dict
    ):
        """
        Handles the scene_compiled event from the subprocess.
        Adopts the shared memory handle so it survives worker cleanup.
        """
        if event_name == "scene_compiled":
            if task.is_cancelled():
                logger.debug(
                    "[CANVAS3D] Ignoring scene_compiled event from "
                    "cancelled task."
                )
                return
            if (
                self._scene_preparation_task
                and task.id != self._scene_preparation_task.id
            ):
                logger.debug(
                    "[CANVAS3D] Ignoring stale scene_compiled event "
                    "(task mismatch)."
                )
                return
            handle_dict = data.get("handle_dict")
            if handle_dict is not None:
                try:
                    self._context.artifact_store.adopt_from_dict(handle_dict)
                    self._pending_scene_handle_dict = handle_dict
                    logger.debug(
                        f"[CANVAS3D] Adopted scene artifact: "
                        f"{handle_dict.get('shm_name', '?')}"
                    )
                except Exception as e:
                    logger.error(
                        f"[CANVAS3D] Failed to adopt scene artifact: {e}"
                    )
                    self._pending_scene_handle_dict = None

    def _on_scene_prepared(self, task: Task):
        """
        Callback for when the background scene compilation task is finished.
        """
        if task.get_status() != "completed":
            self._release_pending_scene_handle()
            self._compiled_artifact = None
            self._op_player = None
            logger.error(
                "[CANVAS3D] Scene preparation task failed or was cancelled."
            )
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        handle_dict = self._pending_scene_handle_dict
        self._pending_scene_handle_dict = None

        if handle_dict is None:
            logger.warning(
                "[CANVAS3D] Scene task completed but no artifact handle "
                "was received (possibly empty scene)."
            )
            self._compiled_artifact = None
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        logger.debug(
            "[CANVAS3D] Scene compilation finished. Loading artifact."
        )
        try:
            handle = create_handle_from_dict(handle_dict)
            artifact = self._context.artifact_store.get(handle)
        except Exception as e:
            logger.error(f"[CANVAS3D] Failed to load compiled scene: {e}")
            self._release_scene_shm(handle_dict)
            self._compiled_artifact = None
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        self._release_scene_shm(handle_dict)

        if not isinstance(artifact, CompiledSceneArtifact):
            logger.error(
                f"[CANVAS3D] Expected CompiledSceneArtifact, got "
                f"{type(artifact).__name__}"
            )
            self._compiled_artifact = None
            self._artifact_gl_dirty = True
            self.queue_render()
            return

        self._compiled_artifact = artifact
        self._artifact_gl_dirty = True
        self.queue_render()

    def _update_renderers_from_artifact(self):
        if not self._compiled_artifact:
            for group in self._layer_groups:
                group.ops_renderer.clear()
            if self._texture_renderer:
                self._texture_renderer.clear()
            self.queue_render()
            return

        if not self._gl_initialized:
            return

        self.make_current()

        for group in self._layer_groups:
            group.cleanup()
        self._layer_groups.clear()

        artifact = self._compiled_artifact
        for vl in artifact.vertex_layers:
            group = _LayerRendererGroup(is_rotary=vl.is_rotary)
            group.init_gl()
            self._layer_groups.append(group)

            if self._show_travel_moves:
                pv_final = np.concatenate(
                    (vl.powered_verts, vl.zero_power_verts)
                )
                pc_final = np.concatenate(
                    (vl.powered_colors, vl.zero_power_colors)
                )
                tv_final = vl.travel_verts
            else:
                pv_final = vl.powered_verts
                pc_final = vl.powered_colors
                tv_final = np.array([], dtype=np.float32)

            powered_count = vl.powered_verts.size // 3
            zero_count = vl.zero_power_verts.size // 3
            logger.debug(
                f"[UPLOAD] is_rotary={vl.is_rotary} "
                f"powered={powered_count} "
                f"zero_power={zero_count} "
                f"total={pv_final.size // 3} "
                f"travel={tv_final.size // 3} "
                f"show_travel={self._show_travel_moves}"
            )

            group.ops_renderer.update_from_vertex_data(
                pv_final, pc_final, tv_final
            )

        if self._texture_renderer:
            self._texture_renderer.clear()
            for tl in artifact.texture_layers:
                tex_data = TextureData(
                    power_texture_data=tl.power_texture,
                    dimensions_mm=(0.0, 0.0),
                    position_mm=(0.0, 0.0),
                )
                rotary_enabled = tl.rotary_enabled
                self._texture_renderer.add_instance(
                    tex_data,
                    tl.model_matrix,
                    color_lut=tl.color_lut,
                    rotary_enabled=rotary_enabled,
                    rotary_diameter=tl.rotary_diameter,
                    cylinder_vertices=tl.cylinder_vertices,
                )

        self.queue_render()

    def _update_op_player(self):
        ops = self._get_ops_for_playback()
        if ops is None or ops.is_empty():
            logger.debug("[CANVAS3D] _update_op_player: no ops available.")
            self._op_player = None
            self._current_layer_uid = None
            for group in self._layer_groups:
                group.powered_offsets = []
                group.travel_offsets = []
                group.ring_offsets = []
            GLib.idle_add(self._notify_playback_overlay, 0)
            return
        logger.debug(
            "[CANVAS3D] _update_op_player: got ops with "
            "%d commands." % len(ops)
        )

        saved_index = None
        if self._op_player is not None and self._op_player.ops is ops:
            saved_index = self._op_player.current_index

        machine = self._context.machine
        if machine is None:
            return
        self._op_player = OpPlayer(ops, machine=machine, doc=self.doc)
        self._current_layer_uid = None
        self._extract_playback_offsets_from_artifact()

        if saved_index is not None:
            self._op_player.seek(saved_index)
            initial_index = saved_index
        else:
            # Auto-advance to the first LayerStartCommand so the canvas
            # shows the first layer's surface (e.g. correct workpiece
            # diameter) immediately instead of an empty default scene.
            initial_index = 0
            for i, cmd in enumerate(self._op_player.ops):
                if isinstance(cmd, LayerStartCommand):
                    self._op_player.seek(i)
                    initial_index = i
                    break

        GLib.idle_add(
            self._notify_playback_overlay,
            len(self._op_player.ops),
            initial_index,
        )
        self._upload_scanline_overlay()

    def _get_ops_for_playback(self) -> Optional[Ops]:
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact) and artifact.ops is not None:
                return artifact.ops
        return None

    def _notify_playback_overlay(
        self, command_count: int, initial_index: int = 0
    ):
        if self._playback_overlay is not None:
            self._playback_overlay.update_ops_range(
                command_count, initial_index
            )

    def _upload_scanline_overlay(self):
        """
        Upload pre-compiled scanline overlay data from the artifact's
        overlay layers to per-group ring buffer renderers.
        """
        if not self._layer_groups:
            return

        if not self._compiled_artifact:
            for group in self._layer_groups:
                group.ring_renderer.clear()
                group.ring_offsets = []
            return

        overlay_layers = self._compiled_artifact.overlay_layers
        if not overlay_layers:
            for group in self._layer_groups:
                group.ring_renderer.clear()
                group.ring_offsets = []
            return

        self.make_current()

        for group in self._layer_groups:
            ol = None
            for overlay in overlay_layers:
                if overlay.is_rotary == group.is_rotary:
                    ol = overlay
                    break

            if ol is not None:
                group.ring_renderer.upload(
                    ol.positions.ravel(),
                    ol.colors.ravel(),
                )
            else:
                group.ring_renderer.clear()

        logger.debug(
            "[CANVAS3D] Scanline overlay uploaded. Groups: %s"
            % ", ".join(
                "%s:%d"
                % (
                    "rot" if g.is_rotary else "flat",
                    g.ring_renderer.vertex_count,
                )
                for g in self._layer_groups
            )
        )

    def _extract_playback_offsets_from_artifact(self):
        if not self._compiled_artifact or not self._op_player:
            return

        for group in self._layer_groups:
            vl = None
            for vertex_layer in self._compiled_artifact.vertex_layers:
                if vertex_layer.is_rotary == group.is_rotary:
                    vl = vertex_layer
                    break

            if vl is not None:
                group.powered_offsets = vl.powered_cmd_offsets
                group.travel_offsets = vl.travel_cmd_offsets
            else:
                group.powered_offsets = []
                group.travel_offsets = []

            ol = None
            for overlay in self._compiled_artifact.overlay_layers:
                if overlay.is_rotary == group.is_rotary:
                    ol = overlay
                    break

            if ol is not None:
                group.ring_offsets = ol.cmd_offsets
            else:
                group.ring_offsets = []

    def _update_cylinder_renderers(self):
        """
        Reads chuck diameters from the assembly and creates one
        CylinderRenderer per unique diameter. Removes renderers for
        diameters no longer in use.
        """
        self.make_current()

        machine = self._context.machine

        desired_diameters: Dict[float, bool] = {}
        if machine and self._had_rotary_layers:
            for diameter in machine.assembly.chuck_diameters.values():
                desired_diameters[diameter] = True

        max_length = self._viewport.width_mm
        if machine:
            default_rm = machine.get_default_rotary_module()
            if default_rm:
                max_length = min(max_length, default_rm.max_workpiece_length)

        # Remove renderers for diameters no longer needed
        for diameter in list(self._cylinder_renderers.keys()):
            if diameter not in desired_diameters:
                self._cylinder_renderers[diameter].cleanup()
                del self._cylinder_renderers[diameter]
                logger.debug(
                    f"Removed cylinder renderer for diameter={diameter}mm"
                )

        # Create renderers for new diameters
        for diameter in desired_diameters:
            if diameter not in self._cylinder_renderers:
                renderer = CylinderRenderer(
                    diameter=diameter,
                    length=max_length,
                    rings=24,
                    length_segments=12,
                )
                renderer.set_color((0.4, 0.6, 0.8, 0.25))
                renderer.init_gl()
                self._cylinder_renderers[diameter] = renderer
                logger.debug(
                    f"Initialized cylinder renderer: "
                    f"diameter={diameter}mm, length={max_length}mm"
                )

    def _clear_model_renderers(self):
        """Remove all model renderers without rebuilding."""
        for r, _ in self._model_renderers:
            r.cleanup()
        self._model_renderers.clear()

    def _apply_extent_frame(self, vp):
        if not self._axis_renderer or vp.extent_frame is None:
            return
        fx, fy, fw, fh = vp.extent_frame
        ml = -fx
        mb = -fy
        mt = fh - vp.depth_mm - mb
        mr = fw - vp.width_mm - ml
        if vp.x_right:
            fx = -mr
        if vp.y_down:
            fy = -mt
        self._axis_renderer.set_extent_frame(fx, fy, fw, fh, show=True)

    def _update_axis_renderer(self):
        if not self._axis_renderer:
            return
        vp = self._viewport
        if (
            self._axis_renderer.width_mm == vp.width_mm
            and self._axis_renderer.height_mm == vp.depth_mm
        ):
            return
        self.make_current()
        font_family = self._axis_renderer.font_family
        self._axis_renderer.cleanup()
        self._axis_renderer = AxisRenderer3D(
            vp.width_mm, vp.depth_mm, font_family=font_family
        )
        self._apply_extent_frame(vp)
        self._axis_renderer.init_gl()
        self._theme_is_dirty = True

    def _update_zone_renderer(self):
        if not self._zone_renderer:
            return
        machine = self._context.machine
        if not machine:
            return
        self.make_current()
        zones = list(machine.nogo_zones.values())
        self._zone_renderer.update_zones(zones)

    def _update_model_renderers(self):
        """Rebuild renderers for all assembly links with 3D models."""
        self.make_current()
        self._clear_model_renderers()

        machine = self._context.machine
        if not machine:
            return

        assembly = machine.assembly
        if assembly is None:
            return

        model_links = assembly.get_model_links()
        logger.debug("Model renderers: %d links with models", len(model_links))

        for link in model_links:
            assert link.model_id is not None
            logger.debug(
                "Model renderers: resolving model_id=%s for link %s",
                link.model_id,
                link.name,
            )
            resolved = self._context.model_mgr.resolve(
                Model(name="", path=Path(link.model_id))
            )
            if resolved is None:
                logger.warning(
                    "Model file not found: %s, skipping.",
                    link.model_id,
                )
                continue

            renderer = ModelRenderer(resolved)
            renderer.init_gl()
            logger.debug(
                "Model renderer created: vao=%d, vertex_count=%d, bounds=%s",
                renderer._vao,
                renderer._vertex_count,
                renderer.bounds,
            )
            self._model_renderers.append((renderer, link.name))

    def update_scene_from_doc(self):
        """
        Updates the entire scene content from the document. This is the main
        entry point for refreshing the 3D view.
        """
        if not self._gl_initialized:
            return
        if not self._texture_renderer:
            return

        t_update_start = time.perf_counter()
        logger.debug("Canvas3D: Updating scene from document.")

        # Theme/color updates only need to happen once per theme change
        if self._theme_is_dirty:
            self._update_theme_and_colors()
        if not self._color_set:
            logger.warning("Cannot update scene, color set not resolved.")
            return

        # Update cylinder renderers and camera based on layer rotary state
        any_rotary = any(layer.rotary_enabled for layer in self.doc.layers)
        self._scene_gl_dirty = True
        if self._had_rotary_layers and not any_rotary and self.camera:
            self.reset_view(ViewDirection.ISO)
        self._had_rotary_layers = any_rotary

        world_to_visual = np.identity(4, dtype=np.float32)
        world_to_grid = np.identity(4, dtype=np.float32)
        world_to_cyl_local = np.identity(4, dtype=np.float32)

        machine = self._context.machine
        if machine:
            ms = self._viewport.margin_shift
            wcs = self._viewport.wcs_offset_mm
            world_to_visual[0, 3] = ms[0, 3]
            world_to_visual[1, 3] = ms[1, 3]
            world_to_visual[2, 3] = wcs[2]

            try:
                visual_to_grid = np.linalg.inv(self._viewport.model_matrix)
            except np.linalg.LinAlgError:
                visual_to_grid = np.identity(4, dtype=np.float32)

            world_to_grid = visual_to_grid @ world_to_visual

            cyl_to_grid = np.identity(4, dtype=np.float32)
            cyl_to_grid[0, 3] = wcs[0]
            cyl_to_grid[1, 3] = wcs[1]
            cyl_to_grid[2, 3] = wcs[2]

            try:
                grid_to_cyl = np.linalg.inv(cyl_to_grid)
            except np.linalg.LinAlgError:
                grid_to_cyl = np.identity(4, dtype=np.float32)

            world_to_cyl_local = grid_to_cyl @ world_to_grid

        laser_color_luts: Dict[str, dict] = {}
        for uid, cs in self._laser_color_sets.items():
            laser_color_luts[uid] = {
                "cut": cs.get_lut("cut").astype(np.float32).tobytes(),
                "engrave": cs.get_lut("engrave").astype(np.float32).tobytes(),
            }

        layer_configs: Dict[str, LayerRenderConfig] = {}
        for layer in self.doc.layers:
            layer_configs[layer.uid] = LayerRenderConfig(
                rotary_enabled=layer.rotary_enabled,
                rotary_diameter=layer.rotary_diameter,
            )

        render_config = RenderConfig3D(
            world_to_visual=world_to_visual,
            world_to_cyl_local=world_to_cyl_local,
            default_color_lut_cut=self._color_set.get_lut("cut")
            .astype(np.float32)
            .tobytes(),
            default_color_lut_engrave=self._color_set.get_lut("engrave")
            .astype(np.float32)
            .tobytes(),
            laser_color_luts=laser_color_luts,
            zero_power_rgba=self._color_set.get_rgba("zero_power"),
            layer_configs=layer_configs,
        )

        self._world_to_cyl_local = world_to_cyl_local
        self._schedule_scene_preparation(render_config.to_dict())

        t_update_elapsed = (time.perf_counter() - t_update_start) * 1000
        if t_update_elapsed > 5:
            logger.info(
                f"[CANVAS3D] update_scene_from_doc took "
                f"{t_update_elapsed:.1f}ms"
            )

    def _schedule_scene_preparation(
        self,
        render_config_dict: dict,
    ):
        task_key = (id(self), "prepare-3d-scene-vertices")

        if self._gl_initialized and self._color_set is not None:
            if self._scene_preparation_task:
                self._scene_preparation_task.cancel()

            self._release_pending_scene_handle()

            job_handle = self._current_job_handle
            if job_handle is None:
                logger.debug(
                    "[CANVAS3D] No job artifact, skipping compilation."
                )
                return

            logger.debug("[CANVAS3D] Scheduling scene compilation task.")
            assert render_config_dict is not None
            self._scene_preparation_task = task_mgr.run_process(
                compile_scene_in_subprocess,
                self._context.artifact_store,
                job_handle.to_dict(),
                render_config_dict,
                key=task_key,
                when_done=self._on_scene_prepared,
                when_event=self._on_scene_compiled_event,
            )
