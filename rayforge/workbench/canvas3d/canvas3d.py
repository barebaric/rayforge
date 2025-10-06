import logging
import math
from typing import Optional, Tuple, List
import numpy as np
from gi.repository import Gdk, Gtk, Pango
from OpenGL import GL
from .camera import Camera, rotation_matrix_from_axis_angle
from .gl_utils import Shader
from .ops_renderer import Ops, OpsRenderer
from .sphere_renderer import SphereRenderer
from .axis_renderer_3d import AxisRenderer3D
from .raster_renderer import RasterArtifactRenderer
from .shaders import (
    SIMPLE_FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER,
    TEXT_FRAGMENT_SHADER,
    TEXT_VERTEX_SHADER,
    RASTER_FRAGMENT_SHADER,
    RASTER_VERTEX_SHADER,
)
from ...core.ops import ScanLinePowerCommand
from ...shared.util.colors import ColorSet
from ...shared.util.gtk_color import GtkColorResolver, ColorSpecDict
from ...shared.tasker import task_mgr, Task
from ...pipeline.producer.base import HybridRasterArtifact


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Canvas3D(Gtk.GLArea):
    """A GTK Widget for rendering a 3D scene with OpenGL."""

    def __init__(
        self,
        doc,
        width_mm: float,
        depth_mm: float,
        y_down: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.doc = doc
        self.width_mm = width_mm
        self.depth_mm = depth_mm
        self.y_down = y_down

        self.camera: Optional[Camera] = None
        self.main_shader: Optional[Shader] = None
        self.text_shader: Optional[Shader] = None
        self.axis_renderer: Optional[AxisRenderer3D] = None
        self.ops_renderer: Optional[OpsRenderer] = None
        self.sphere_renderer: Optional[SphereRenderer] = None
        self.raster_renderer: Optional[RasterArtifactRenderer] = None
        self.raster_shader: Optional[Shader] = None
        self._ops_preparation_task: Optional[Task] = None
        self._ops_vtx_cache: Optional[
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ] = None
        self._show_travel_moves = False
        self._is_orbiting = False
        self._is_z_rotating = False
        self._gl_initialized = False
        self._model_matrix = np.identity(4, dtype=np.float32)

        self._color_spec: ColorSpecDict = {
            "cut": ("#ff00ff22", "#ff00ff"),
            "engrave": ("#00000009", "#000000"),
            "travel": ("#FF6600", 0.7),
            "zero_power": ("@accent_color", 0.5),
        }
        self._color_set: Optional[ColorSet] = None
        self._theme_is_dirty = True

        if self.y_down:
            # This matrix transforms from a Y-up coordinate system (used by
            # the Ops data) to a Y-down visual representation. It scales Y by
            # -1 and then translates by the depth of the bed, effectively
            # moving the origin from bottom-left to top-left.
            translate_mat = np.identity(4, dtype=np.float32)
            translate_mat[1, 3] = self.depth_mm
            scale_mat = np.identity(4, dtype=np.float32)
            scale_mat[1, 1] = -1.0
            self._model_matrix = translate_mat @ scale_mat

        # State for interactions
        self._last_pan_offset: Optional[Tuple[float, float]] = None
        self._rotation_pivot: Optional[np.ndarray] = None
        self._last_orbit_pos: Optional[Tuple[float, float]] = None
        self._last_z_rotate_screen_pos: Optional[Tuple[float, float]] = None

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self.on_resize)
        self.connect("notify::style", self._on_style_changed)
        self._setup_interactions()

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
        self.camera.set_top_view(
            self.width_mm, self.depth_mm, y_down=self.y_down
        )

        # A view reset can interrupt a drag operation, leaving stale state.
        self._clear_drag_state()
        self.queue_render()

    def reset_view_front(self):
        """Resets the camera to a front-facing perspective view."""
        if not self.camera:
            return
        logger.info("Resetting to front view.")
        self.camera.set_front_view(self.width_mm, self.depth_mm)
        # A view reset can interrupt a drag operation, leaving stale state.
        self._clear_drag_state()
        self.queue_render()

    def reset_view_iso(self):
        """Resets to a standard isometric perspective view (Z-up)."""
        if not self.camera:
            return
        logger.info("Resetting to isometric view.")
        self.camera.set_iso_view(self.width_mm, self.depth_mm)

        # A view reset can interrupt a drag operation, leaving stale state.
        self._clear_drag_state()
        self.queue_render()

    def on_realize(self, area) -> None:
        """Called when the GLArea is ready to have its context made current."""
        logger.info("GLArea realized.")
        self._init_gl_resources()
        self._theme_is_dirty = True

        # Create the camera with placeholder values. The correct initial view
        # will be set by reset_view_iso() below.
        self.camera = Camera(
            np.array([0.0, 0.0, 1.0]),  # position
            np.array([0.0, 0.0, 0.0]),  # target
            np.array([0.0, 1.0, 0.0]),  # up
            self.get_width(),
            self.get_height(),
        )

        self.sphere_renderer = SphereRenderer(1.0, 16, 32)
        self.reset_view_front()

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self.make_current()
        try:
            if self._ops_preparation_task:
                self._ops_preparation_task.cancel()
            if self.axis_renderer:
                self.axis_renderer.cleanup()
            if self.ops_renderer:
                self.ops_renderer.cleanup()
            if self.sphere_renderer:
                self.sphere_renderer.cleanup()
            if self.raster_renderer:
                self.raster_renderer.cleanup()
            if self.main_shader:
                self.main_shader.cleanup()
            if self.text_shader:
                self.text_shader.cleanup()
            if self.raster_shader:
                self.raster_shader.cleanup()
        finally:
            self._gl_initialized = False

    def _init_gl_resources(self) -> None:
        """Initializes OpenGL state, shaders, and renderer objects."""
        try:
            self.make_current()
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glDepthFunc(GL.GL_LEQUAL)
            GL.glEnable(GL.GL_MULTISAMPLE)
            GL.glEnable(GL.GL_BLEND)
            GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

            self.main_shader = Shader(
                SIMPLE_VERTEX_SHADER, SIMPLE_FRAGMENT_SHADER
            )
            self.text_shader = Shader(TEXT_VERTEX_SHADER, TEXT_FRAGMENT_SHADER)
            self.raster_shader = Shader(
                RASTER_VERTEX_SHADER, RASTER_FRAGMENT_SHADER
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

            self.axis_renderer = AxisRenderer3D(
                self.width_mm, self.depth_mm, font_family=font_family
            )
            self.axis_renderer.init_gl()
            self.ops_renderer = OpsRenderer()
            self.ops_renderer.init_gl()
            self.raster_renderer = RasterArtifactRenderer()
            self.raster_renderer.init_gl()
            if self.sphere_renderer:
                self.sphere_renderer.init_gl()

            self._gl_initialized = True
        except Exception as e:
            logger.error(f"OpenGL Initialization Error: {e}", exc_info=True)
            self._gl_initialized = False

    def _update_theme_and_colors(self):
        """
        Resolves the ColorSet and updates other theme-dependent elements.
        """
        if not self.axis_renderer or not self.raster_renderer:
            return

        style_context = self.get_style_context()
        resolver = GtkColorResolver(style_context)
        self._color_set = resolver.resolve(self._color_spec)

        if self._color_set:
            self.raster_renderer.update_color_lut(
                self._color_set.get_lut("engrave")
            )

        # Get background color and set it for OpenGL. Prioritize the specific
        # 'view_bg_color', but fall back to the generic 'theme_bg_color'.
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

            self.axis_renderer.set_background_color(bg_plane_color)
            self.axis_renderer.set_axis_color(axis_color)
            self.axis_renderer.set_label_color(axis_color)
            self.axis_renderer.set_grid_color(grid_color)

        self._theme_is_dirty = False

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self.camera or not self._gl_initialized:
            return False

        if self._theme_is_dirty:
            self._update_theme_and_colors()

        if not self._color_set:
            return False  # Cannot render without resolved colors

        try:
            GL.glViewport(0, 0, self.camera.width, self.camera.height)
            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
            )

            proj_matrix = self.camera.get_projection_matrix()
            view_matrix = self.camera.get_view_matrix()

            # Base MVP for UI elements that should not be model-transformed
            mvp_matrix_ui = proj_matrix @ view_matrix

            # MVP for scene geometry that IS model-transformed (e.g., Y-down)
            mvp_matrix_scene = mvp_matrix_ui @ self._model_matrix

            # Convert to column-major for OpenGL
            mvp_matrix_ui_gl = mvp_matrix_ui.T
            mvp_matrix_scene_gl = mvp_matrix_scene.T

            if self.axis_renderer and self.main_shader and self.text_shader:
                self.axis_renderer.render(
                    self.main_shader,
                    self.text_shader,
                    mvp_matrix_scene_gl,  # For grid/axes
                    mvp_matrix_ui_gl,  # For text
                    view_matrix,
                    self._model_matrix,  # Pass the model matrix for positions
                )
            if self.ops_renderer and self.main_shader:
                # The ops vertices are already in world space, so we do not
                # apply the scene's _model_matrix. We pass the UI matrix which
                # only contains camera projection and view.
                self.ops_renderer.render(
                    self.main_shader,
                    mvp_matrix_ui_gl,
                    colors=self._color_set,
                    show_travel_moves=self._show_travel_moves,
                )
            if self.raster_renderer and self.raster_shader:
                # The raster quads are part of the scene and need the scene's
                # model matrix. Their vertices are local, so the renderer
                # combines the scene VP with the instance's world model matrix.
                self.raster_renderer.render(
                    mvp_matrix_scene, self.raster_shader
                )

        except Exception as e:
            logger.error(f"OpenGL Render Error: {e}", exc_info=True)
            return False
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
        """Handles updates during a Z-axis 'turntable' rotation."""
        if not self.camera or not self._is_z_rotating:
            return

        event = gesture.get_last_event()
        if not event:
            return
        _, x_curr, y_curr = event.get_position()

        if self._last_z_rotate_screen_pos is None:
            self._last_z_rotate_screen_pos = (x_curr, y_curr)
            return

        x_prev, y_prev = self._last_z_rotate_screen_pos
        self._last_z_rotate_screen_pos = (x_curr, y_curr)

        # Use the center of the widget as the screen-space pivot
        pivot_x = self.get_width() / 2.0
        pivot_y = self.get_height() / 2.0

        # Calculate angle of the previous and current mouse positions
        # relative to the screen pivot.
        angle_prev = math.atan2(y_prev - pivot_y, x_prev - pivot_x)
        angle_curr = math.atan2(y_curr - pivot_y, x_curr - pivot_y)
        delta_angle = angle_curr - angle_prev

        # Handle atan2 wrap-around from -pi to pi
        if delta_angle > math.pi:
            delta_angle -= 2 * math.pi
        elif delta_angle < -math.pi:
            delta_angle += 2 * math.pi

        axis_z = np.array([0, 0, 1], dtype=np.float64)
        pivot_world = self.camera.target  # Rotate around the look-at point
        self.camera.orbit(pivot_world, axis_z, delta_angle)
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
        self._update_renderer_from_cache()

    def _on_ops_prepared(self, task: Task):
        """
        Callback for when the background ops preparation task is finished.
        """
        if task.get_status() != "completed":
            self._ops_vtx_cache = None
            if self.ops_renderer:
                self.ops_renderer.clear()
            self.queue_render()
            return

        # Cache the full vertex data from the background task
        self._ops_vtx_cache = task.result()
        self._update_renderer_from_cache()

    def _update_renderer_from_cache(self):
        """
        Helper to update the renderer based on current visibility flags.
        """
        if not self.ops_renderer or not self._ops_vtx_cache:
            if self.ops_renderer:
                self.ops_renderer.clear()
            self.queue_render()
            return

        (
            powered_verts,
            powered_colors,
            travel_verts,
            zero_power_verts,
            zero_power_colors,
        ) = self._ops_vtx_cache

        if self._show_travel_moves:
            # When travel is shown, zero-power cuts are also shown. They are
            # conceptually similar (non-cutting moves). We append them to the
            # powered buffer as they use vertex colors.
            powered_verts_final = np.concatenate(
                (powered_verts, zero_power_verts)
            )
            powered_colors_final = np.concatenate(
                (powered_colors, zero_power_colors)
            )
            travel_verts_final = travel_verts
        else:
            # When travel is hidden, hide zero-power cuts as well.
            powered_verts_final = powered_verts
            powered_colors_final = powered_colors
            travel_verts_final = np.array([], dtype=np.float32)

        # This part runs on the main thread and is fast (just GPU uploads)
        self.ops_renderer.update_from_vertex_data(
            powered_verts_final,
            powered_colors_final,
            travel_verts_final,
        )
        self.queue_render()

    def set_scene_content(
        self,
        vector_ops: Ops,
        raster_instances: List[Tuple[HybridRasterArtifact, np.ndarray]],
    ):
        """
        Sets the entire scene content from pre-aggregated and transformed data.
        """
        if not self.ops_renderer or not self.raster_renderer:
            return

        # --- INPUT VALIDATION LOGGING ---
        scanline_count = sum(
            1
            for cmd in vector_ops.commands
            if isinstance(cmd, ScanLinePowerCommand)
        )
        logger.debug(
            f"Canvas3D.set_scene_content: Received {len(vector_ops.commands)} "
            f"vector commands, including {scanline_count} "
            "ScanLinePowerCommands."
        )
        # --- END LOGGING ---

        # Theme/color updates only need to happen once per theme change
        if self._theme_is_dirty:
            self._update_theme_and_colors()
        if not self._color_set:
            logger.warning("Cannot set artifacts, color set not resolved.")
            return

        # Handle raster instances by clearing and re-adding them
        self.raster_renderer.clear()
        for artifact, transform_matrix in raster_instances:
            self.raster_renderer.add_instance(artifact, transform_matrix)

        # Schedule the vector ops for preparation.
        self._schedule_ops_preparation(vector_ops)

    def _schedule_ops_preparation(self, ops: Ops):
        """
        Schedules the given Ops to be prepared for rendering in a
        background thread.
        """
        task_key = (id(self), "prepare-3d-ops-vertices")

        if self.ops_renderer and self._color_set is not None:
            self._ops_preparation_task = task_mgr.run_thread(
                self.ops_renderer.prepare_vertex_data,
                ops,
                self._color_set,
                key=task_key,
                when_done=self._on_ops_prepared,
            )
