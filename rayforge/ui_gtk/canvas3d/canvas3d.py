import logging
import math
import time
from pathlib import Path
from typing import Optional, Tuple, List, Dict, TYPE_CHECKING
import numpy as np
from gi.repository import Gdk, Gtk, Pango
from OpenGL import GL
from ...context import RayforgeContext
from ...core.geo import Point, Point3D, Rect
from ...core.model import Model
from ...core.ops import Ops
from ...core.ops.commands import MovingCommand
from ...machine.driver.driver import Axis
from ...machine.models.colors import OpsColorSet
from ...pipeline.artifact.base import TextureData
from ...pipeline.artifact.handle import create_handle_from_dict
from ...pipeline.artifact.job import JobArtifact, JobArtifactHandle
from ...pipeline.pipeline import Pipeline
from ...shared.util.colors import ColorSet
from ...shared.tasker import task_mgr, Task
from ...simulator.op_player import OpPlayer
from ..shared.gtk_color import GtkColorResolver, ColorSpecDict
from .axis_renderer_3d import AxisRenderer3D
from .camera import Camera, rotation_matrix_from_axis_angle
from .compiled_scene import CompiledSceneArtifact
from .cylinder_renderer import CylinderRenderer
from .gl_utils import Shader
from .model_renderer import ModelRenderer
from .ops_renderer import OpsRenderer
from .render_config import (
    LayerRenderConfig,
    RenderConfig3D,
)
from .ring_buffer_renderer import RingBufferRenderer
from .scene_compiler_runner import compile_scene_in_subprocess
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
from .zone_renderer import ZoneRenderer

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor
    from ...machine.models.machine import Machine

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
        width_mm: float,
        depth_mm: float,
        x_right: bool = False,
        y_down: bool = False,
        x_negative: bool = False,
        y_negative: bool = False,
        extent_frame: Optional[Rect] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._context = context
        self._doc_editor = doc_editor
        self._width_mm = width_mm
        self._depth_mm = depth_mm
        self._x_right = x_right
        self._y_down = y_down
        self._x_negative = x_negative
        self._y_negative = y_negative
        self._extent_frame = extent_frame

        self.camera: Optional[Camera] = None
        self._main_shader: Optional[Shader] = None
        self._text_shader: Optional[Shader] = None
        self._axis_renderer: Optional[AxisRenderer3D] = None
        self._layer_groups: List[_LayerRendererGroup] = []
        self._sphere_renderer: Optional[SphereRenderer] = None
        self._texture_renderer: Optional[TextureArtifactRenderer] = None
        self._texture_shader: Optional[Shader] = None
        self._cylinder_renderers: Dict[float, CylinderRenderer] = {}
        self._model_renderers: List[Tuple[ModelRenderer, np.ndarray]] = []
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
        self._wcs_offset_mm: Point3D = (0.0, 0.0, 0.0)

        # This matrix transforms the grid and axes from a standard Y-up,
        # X-right system to match the machine's coordinate system.
        translate_mat = np.identity(4, dtype=np.float32)
        scale_mat = np.identity(4, dtype=np.float32)

        if self._y_down:
            translate_mat[1, 3] = self._depth_mm
            scale_mat[1, 1] = -1.0

        if self._x_right:
            translate_mat[0, 3] = self._width_mm
            scale_mat[0, 0] = -1.0

        self._model_matrix = translate_mat @ scale_mat

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

    def _on_wcs_updated(self, machine: "Machine", **kwargs):
        """Handler for when the machine's WCS state changes."""
        if machine.wcs_origin_is_workarea_origin:
            self._wcs_offset_mm = (0.0, 0.0, 0.0)
        else:
            wcs_x, wcs_y, wcs_z = machine.get_active_wcs_offset()
            ml, mt, mr, mb = machine.work_margins

            if machine.x_axis_right:
                machine_x = -mr
            else:
                machine_x = -ml
            if machine.y_axis_down:
                machine_y = -mt
            else:
                machine_y = -mb

            if machine.reverse_x_axis:
                local_x = machine_x - wcs_x
            else:
                local_x = machine_x + wcs_x
            if machine.reverse_y_axis:
                local_y = machine_y - wcs_y
            else:
                local_y = machine_y + wcs_y
            self._wcs_offset_mm = (local_x, local_y, wcs_z)
        self.queue_render()

    def set_machine(
        self,
        width_mm: float,
        depth_mm: float,
        x_right: bool = False,
        y_down: bool = False,
        x_negative: bool = False,
        y_negative: bool = False,
        extent_frame: Optional[Rect] = None,
    ):
        old_machine = self._context.machine
        if old_machine:
            old_machine.wcs_updated.disconnect(self._on_wcs_updated)
            old_machine.changed.disconnect(self._on_wcs_updated)

        self._width_mm = width_mm
        self._depth_mm = depth_mm
        self._x_right = x_right
        self._y_down = y_down
        self._x_negative = x_negative
        self._y_negative = y_negative
        self._extent_frame = extent_frame

        translate_mat = np.identity(4, dtype=np.float32)
        scale_mat = np.identity(4, dtype=np.float32)

        if self._y_down:
            translate_mat[1, 3] = self._depth_mm
            scale_mat[1, 1] = -1.0

        if self._x_right:
            translate_mat[0, 3] = self._width_mm
            scale_mat[0, 0] = -1.0

        self._model_matrix = translate_mat @ scale_mat

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
        if not is_processing:
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
            self._update_op_player()
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

    def _clear_drag_state(self):
        """Resets all state variables related to any drag operation."""
        self._is_orbiting = False
        self._is_z_rotating = False
        self._last_pan_offset = None
        self._rotation_pivot = None
        self._last_orbit_pos = None
        self._last_z_rotate_screen_pos = None

    def reset_view_top(self):
        """Resets the camera to a top-down orthographic view (Z-up)."""
        if not self.camera:
            return
        logger.info("Resetting to top view.")
        # The camera class now handles all orientation logic internally.
        self.camera.set_top_view(self._width_mm, self._depth_mm)

        # A view reset can interrupt a drag operation, leaving stale state.
        self._clear_drag_state()
        self.queue_render()

    def reset_view_front(self):
        """Resets the camera to a front-facing perspective view."""
        if not self.camera:
            return
        logger.info("Resetting to front view.")
        self.camera.set_front_view(self._width_mm, self._depth_mm)
        # A view reset can interrupt a drag operation, leaving stale state.
        self._clear_drag_state()
        self.queue_render()

    def reset_view_iso(self):
        """Resets to a standard isometric perspective view (Z-up)."""
        if not self.camera:
            return
        logger.info("Resetting to isometric view.")
        self.camera.set_iso_view(self._width_mm, self._depth_mm)

        # A view reset can interrupt a drag operation, leaving stale state.
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
        self._init_gl_resources()
        self._theme_is_dirty = True

        self.camera = Camera(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            self.get_width(),
            self.get_height(),
        )

        self._sphere_renderer = SphereRenderer(1.0, 16, 32)
        self.reset_view_front()
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
        self.make_current()
        try:
            if self._scene_preparation_task:
                self._scene_preparation_task.cancel()
            if self._axis_renderer:
                self._axis_renderer.cleanup()
            for group in self._layer_groups:
                group.cleanup()
            if self._sphere_renderer:
                self._sphere_renderer.cleanup()
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
        finally:
            self._gl_initialized = False

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
                self._width_mm, self._depth_mm, font_family=font_family
            )
            if self._extent_frame is not None:
                fx, fy, fw, fh = self._extent_frame
                ml = -fx
                mb = -fy
                mt = fh - self._depth_mm - mb
                mr = fw - self._width_mm - ml
                if self._x_right:
                    fx = -mr
                if self._y_down:
                    fy = -mt
                self._axis_renderer.set_extent_frame(fx, fy, fw, fh, show=True)
            self._axis_renderer.init_gl()
            self._texture_renderer = TextureArtifactRenderer()
            self._texture_renderer.init_gl()
            if self._sphere_renderer:
                self._sphere_renderer.init_gl()
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
            GL.glClearColor(
                bg_rgba.red, bg_rgba.green, bg_rgba.blue, bg_rgba.alpha
            )
        else:
            GL.glClearColor(0.2, 0.2, 0.25, 1.0)  # Final fallback

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
            bg_plane_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.25

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

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self.camera or not self._gl_initialized:
            return False

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

            proj_matrix = self.camera.get_projection_matrix()
            view_matrix = self.camera.get_view_matrix()

            # Base MVP for UI elements that should not be model-transformed
            mvp_matrix_ui = proj_matrix @ view_matrix

            # Create WCS translation matrix
            offset_x, offset_y, offset_z = self._wcs_offset_mm
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
            # 2. Apply _model_matrix (orient to machine coords).
            # Note: matrix order A @ B applies B then A.
            # This order applies WCS translation locally, THEN applies the
            # machine flip/origin shift.
            grid_model_matrix = self._model_matrix @ wcs_translation_matrix

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
                    mvp_matrix_scene_gl,  # Pass the final grid MVP
                    mvp_matrix_ui_gl,  # For text (no model/WCS transform)
                    view_matrix,
                    self._model_matrix,
                    origin_offset_mm=self._wcs_offset_mm,
                    x_right=self._x_right,
                    y_down=self._y_down,
                    x_negative=self._x_negative,
                    y_negative=self._y_negative,
                )

            machine = self._context.machine
            margin_shift = np.identity(4, dtype=np.float32)
            if machine:
                ml, _, _, mb = machine.work_margins
                margin_shift[0, 3] = -ml
                margin_shift[1, 3] = -mb

            if (
                self._zone_renderer
                and self._main_shader
                and self._show_nogo_zones
            ):
                zone_mvp_gl = (mvp_matrix_ui @ margin_shift).T
                self._zone_renderer.render(self._main_shader, zone_mvp_gl)

            # Compute cylinder rotation from kinematics.
            cyl_angle = 0.0
            if self._op_player and self._had_rotary_layers and machine:
                kin = machine.kinematics
                if kin.has_rotary:
                    world_y = self._op_player.state.axes[Axis.Y]
                    cyl_pos = self._world_to_cyl_local @ np.array(
                        [0, world_y, 0, 1.0], dtype=np.float32
                    )
                    cyl_local_y = cyl_pos[1]
                    diameter = kin.rotary_diameter
                    assert diameter is not None
                    circumference = diameter * math.pi
                    cyl_angle = (cyl_local_y / circumference) * 2 * math.pi

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
                    )

            # Laser head marker
            if (
                self._op_player
                and self._sphere_renderer
                and self._main_shader
                and machine
            ):
                heads = machine.kinematics.head_positions(
                    self._op_player.state
                )
                head_name = next(iter(heads), None)
                if head_name is not None:
                    hx, hy, hz = heads[head_name]
                    head_pos = margin_shift @ np.array(
                        [hx, hy, hz, 1.0], dtype=np.float32
                    )
                    self._sphere_renderer.render(
                        self._main_shader,
                        proj_matrix,
                        view_matrix,
                        head_pos[:3],
                        color=(1.0, 0.2, 0.0, 1.0),
                        scale=2.0,
                    )

            for renderer in self._cylinder_renderers.values():
                if self._main_shader:
                    renderer.render(self._main_shader, rot_cyl_gl)

            if self._show_models:
                for renderer, module_transform in self._model_renderers:
                    if self._main_shader:
                        combined = (
                            mvp_matrix_ui @ margin_shift @ module_transform
                        )
                        renderer.render(
                            self._main_shader,
                            combined.T,
                            model_matrix=margin_shift @ module_transform,
                            camera_position=self.camera.position,
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
            logger.error(f"OpenGL Render Error: {e}", exc_info=True)
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

    def _on_scene_compiled_event(
        self, task: Task, event_name: str, data: dict
    ):
        """
        Handles the scene_compiled event from the subprocess.
        Adopts the shared memory handle so it survives worker cleanup.
        """
        if event_name == "scene_compiled":
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
            self._compiled_artifact = None
            self._op_player = None
            for group in self._layer_groups:
                group.clear()
            self._pending_scene_handle_dict = None
            logger.error(
                "[CANVAS3D] Scene preparation task failed or was cancelled."
            )
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
            self._update_renderers_from_artifact()
            self._update_op_player()
            return

        logger.debug(
            "[CANVAS3D] Scene compilation finished. Loading artifact."
        )
        try:
            handle = create_handle_from_dict(handle_dict)
            artifact = self._context.artifact_store.get(handle)
        except Exception as e:
            logger.error(f"[CANVAS3D] Failed to load compiled scene: {e}")
            self._compiled_artifact = None
            self._update_renderers_from_artifact()
            self._update_op_player()
            return

        if not isinstance(artifact, CompiledSceneArtifact):
            logger.error(
                f"[CANVAS3D] Expected CompiledSceneArtifact, got "
                f"{type(artifact).__name__}"
            )
            self._compiled_artifact = None
            self._update_renderers_from_artifact()
            self._update_op_player()
            return

        self._compiled_artifact = artifact
        self._update_renderers_from_artifact()
        self._update_op_player()

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
            for group in self._layer_groups:
                group.powered_offsets = []
                group.travel_offsets = []
                group.ring_offsets = []
            self._notify_playback_overlay(0)
            return
        logger.debug(
            "[CANVAS3D] _update_op_player: got ops with "
            "%d commands." % len(ops)
        )
        self._op_player = OpPlayer(ops)
        self._extract_playback_offsets_from_artifact()
        last_idx = self._op_player.seek_last_movement()
        if last_idx is not None:
            last_cmd = self._op_player.ops.commands[last_idx]
            assert isinstance(last_cmd, MovingCommand)
            assert last_cmd.end is not None
            assert (
                abs(self._op_player.state.axes[Axis.X] - last_cmd.end[0])
                < 1e-6
            )
            assert (
                abs(self._op_player.state.axes[Axis.Y] - last_cmd.end[1])
                < 1e-6
            )
            assert (
                abs(self._op_player.state.axes[Axis.Z] - last_cmd.end[2])
                < 1e-6
            )
        self._notify_playback_overlay(len(self._op_player.ops))
        self._upload_scanline_overlay()

    def _get_ops_for_playback(self) -> Optional[Ops]:
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact) and artifact.ops is not None:
                return artifact.ops
        return None

    def _notify_playback_overlay(self, command_count: int):
        if self._playback_overlay is not None:
            self._playback_overlay.update_ops_range(command_count)

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
        Scans all layers for rotary-enabled ones and creates one
        CylinderRenderer per unique diameter. Removes renderers for
        diameters no longer in use.
        """
        self.make_current()

        machine = self._context.machine

        desired_diameters: Dict[float, bool] = {}
        for layer in self.doc.layers:
            if layer.rotary_enabled:
                desired_diameters[layer.rotary_diameter] = True

        max_length = self._width_mm
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
        """Rebuild renderers for modules referenced by rotary layers."""
        self.make_current()
        self._clear_model_renderers()

        machine = self._context.machine
        if not machine:
            return

        used_uids = {
            layer.rotary_module_uid
            for layer in self.doc.layers
            if layer.rotary_enabled and layer.rotary_module_uid
        }

        logger.debug("Model renderers: used_uids=%s", used_uids or "none")

        for uid in used_uids:
            module = machine.get_rotary_module_by_uid(uid)
            if module is None:
                logger.debug("Model renderers: uid %s not found", uid)
                continue

            if module.model_id is None:
                logger.debug("Model renderers: module %s has no model_id", uid)
                continue

            logger.debug(
                "Model renderers: resolving model_id=%s", module.model_id
            )
            resolved = self._context.model_mgr.resolve(
                Model(name="", path=Path(module.model_id))
            )
            if resolved is None:
                logger.warning(
                    "Model file not found: %s, skipping.",
                    module.model_id,
                )
                continue

            logger.debug("Model renderers: resolved to %s", resolved)
            renderer = ModelRenderer(resolved)
            renderer.init_gl()
            logger.debug(
                "Model renderer created: vao=%d, vertex_count=%d, bounds=%s",
                renderer._vao,
                renderer._vertex_count,
                renderer.bounds,
            )
            self._model_renderers.append(
                (renderer, module.transform.astype(np.float32))
            )

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
        if self._gl_initialized:
            self._update_cylinder_renderers()
            if any_rotary:
                self._update_model_renderers()
            else:
                self._clear_model_renderers()
            self._update_zone_renderer()
        if self._had_rotary_layers and not any_rotary and self.camera:
            self.reset_view_front()
        self._had_rotary_layers = any_rotary

        world_to_visual = np.identity(4, dtype=np.float32)
        world_to_grid = np.identity(4, dtype=np.float32)
        world_to_cyl_local = np.identity(4, dtype=np.float32)

        machine = self._context.machine
        if machine:
            ml, mt, mr, mb = machine.work_margins
            world_to_visual[0, 3] = -ml
            world_to_visual[1, 3] = -mb

            try:
                visual_to_grid = np.linalg.inv(self._model_matrix)
            except np.linalg.LinAlgError:
                visual_to_grid = np.identity(4, dtype=np.float32)

            world_to_grid = visual_to_grid @ world_to_visual

            cyl_to_grid = np.identity(4, dtype=np.float32)
            cyl_to_grid[0, 3] = self._wcs_offset_mm[0]
            cyl_to_grid[1, 3] = self._wcs_offset_mm[1]
            cyl_to_grid[2, 3] = self._wcs_offset_mm[2]

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

            job_handle = self._current_job_handle
            if job_handle is None:
                logger.debug(
                    "[CANVAS3D] No job artifact, skipping compilation."
                )
                return

            logger.debug("[CANVAS3D] Scheduling scene compilation task.")
            assert render_config_dict is not None
            self._pending_scene_handle_dict = None
            self._scene_preparation_task = task_mgr.run_process(
                compile_scene_in_subprocess,
                self._context.artifact_store,
                job_handle.to_dict(),
                render_config_dict,
                key=task_key,
                when_done=self._on_scene_prepared,
                when_event=self._on_scene_compiled_event,
            )
