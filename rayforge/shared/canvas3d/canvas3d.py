import logging
from typing import Optional, Tuple
import numpy as np
from gi.repository import Gdk, Gtk  # type: ignore
from OpenGL import GL
from .camera import Camera
from .gl_utils import Shader
from .ops_renderer import Ops, OpsRenderer
from .sphere_renderer import SphereRenderer
from .axis_renderer_3d import AxisRenderer3D
from .shaders import (
    SIMPLE_FRAGMENT_SHADER,
    SIMPLE_VERTEX_SHADER,
    TEXT_FRAGMENT_SHADER,
    TEXT_VERTEX_SHADER,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Canvas3D(Gtk.GLArea):
    """A GTK Widget for rendering a 3D scene with OpenGL."""

    def __init__(self, doc, machine, **kwargs):
        super().__init__(**kwargs)
        self.doc, self.machine = doc, machine
        self.width_mm, self.depth_mm = (
            machine.dimensions if machine else (100.0, 100.0)
        )
        self.camera: Optional[Camera] = None
        self.main_shader: Optional[Shader] = None
        self.text_shader: Optional[Shader] = None
        self.axis_renderer: Optional[AxisRenderer3D] = None
        self.ops_renderer: Optional[OpsRenderer] = None
        self.sphere_renderer: Optional[SphereRenderer] = None
        self._is_orbiting = False
        self._gl_initialized = False

        # State for interactions
        self._last_pan_offset: Optional[Tuple[float, float]] = None
        self._rotation_pivot: Optional[np.ndarray] = None
        self._last_orbit_pos: Optional[Tuple[float, float]] = None

        self.set_has_depth_buffer(True)
        self.set_focusable(True)
        self.connect("realize", self.on_realize)
        self.connect("unrealize", self.on_unrealize)
        self.connect("render", self.on_render)
        self.connect("resize", self.on_resize)
        self._setup_interactions()
        self.reset_view_top()

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
        drag = Gtk.GestureDrag.new()
        drag.set_button(Gdk.BUTTON_MIDDLE)
        drag.connect("drag-begin", self.on_drag_begin)
        drag.connect("drag-update", self.on_drag_update)
        drag.connect("drag-end", self.on_drag_end)
        self.add_controller(drag)

        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self.on_scroll)
        self.add_controller(scroll)

        key_controller = Gtk.EventControllerKey.new()
        key_controller.connect("key-pressed", self.on_key_pressed)
        self.add_controller(key_controller)

    def on_key_pressed(self, controller, keyval, keycode, state):
        """Handles key press events for the canvas."""
        logger.debug(f"key pressed: {keyval}")
        if keyval == Gdk.KEY_p and self.camera:
            self.camera.is_perspective = not self.camera.is_perspective
            self.queue_render()
            return True
        if keyval in (Gdk.KEY_1, Gdk.KEY_KP_1):
            self.reset_view_top()
            return True
        # Using '7' (and numpad 7) for isometric view, a common convention.
        if keyval in (Gdk.KEY_7, Gdk.KEY_KP_7):
            self.reset_view_iso()
            return True
        return False

    def reset_view_top(self):
        """Resets the camera to a top-down orthographic view (Z-up)."""
        if not self.camera:
            return
        logger.info("Resetting to top view.")
        center_x, center_y = self.width_mm / 2.0, self.depth_mm / 2.0
        max_dim = max(self.width_mm, self.depth_mm)

        # Look from above (positive Z) down to the XY plane.
        new_pos = np.array(
            [center_x, center_y, max_dim * 1.5], dtype=np.float64
        )
        new_target = np.array([center_x, center_y, 0.0], dtype=np.float64)
        new_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)

        self.camera.is_perspective = True
        self.camera.position = new_pos
        self.camera.target = new_target
        self.camera.up = new_up

        # A view reset can interrupt a drag operation, leaving stale state.
        # This state must be cleared to prevent the next drag from failing.
        self._is_orbiting = False
        self._last_pan_offset = None
        self._rotation_pivot = None
        self._last_orbit_pos = None

        self.queue_render()

    def reset_view_iso(self):
        """Resets to a standard isometric perspective view (Z-up)."""
        if not self.camera:
            return
        logger.info("Resetting to isometric view.")
        center_x, center_y = self.width_mm / 2.0, self.depth_mm / 2.0
        max_dim = max(self.width_mm, self.depth_mm)

        # Target the center of the XY "floor" plane.
        new_target = np.array([center_x, center_y, 0.0], dtype=np.float64)

        # Position the camera for a view from the top-front-left.
        # This corresponds to looking from negative x, negative y, and
        # positive z.
        direction = np.array([-1.0, -1.0, 1.0])
        direction = direction / np.linalg.norm(direction)

        distance = max_dim * 1.7  # A bit more distance for perspective
        new_pos = new_target + direction * distance

        # In a Z-up system, the Z-axis is the natural "up" vector for an
        # isometric view, making it appear upright.
        new_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        self.camera.is_perspective = True
        self.camera.position = new_pos
        self.camera.target = new_target
        self.camera.up = new_up
        self.queue_render()

    def on_realize(self, area) -> None:
        """Called when the GLArea is ready to have its context made current."""
        logger.info("GLArea realized.")
        center_x, center_y = self.width_mm / 2.0, self.depth_mm / 2.0
        max_dim = max(self.width_mm, self.depth_mm)

        cam_pos = np.array(
            [center_x, center_y, max_dim * 1.5], dtype=np.float64
        )
        cam_target = np.array([center_x, center_y, 0.0], dtype=np.float64)
        up_vector = np.array([0, 1, 0], dtype=np.float64)  # Y-axis is "up"

        self.camera = Camera(
            cam_pos, cam_target, up_vector, self.get_width(), self.get_height()
        )
        self.sphere_renderer = SphereRenderer(1.0, 16, 32)

        self.queue_render()

    def on_unrealize(self, area) -> None:
        """Called before the GLArea is unrealized."""
        logger.info("GLArea unrealized. Cleaning up GL resources.")
        self.make_current()
        try:
            if self.axis_renderer:
                self.axis_renderer.cleanup()
            if self.ops_renderer:
                self.ops_renderer.cleanup()
            if self.sphere_renderer:
                self.sphere_renderer.cleanup()
            if self.main_shader:
                self.main_shader.cleanup()
            if self.text_shader:
                self.text_shader.cleanup()
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

            self.axis_renderer = AxisRenderer3D(self.width_mm, self.depth_mm)
            self.axis_renderer.init_gl()
            self.ops_renderer = OpsRenderer()
            self.ops_renderer.init_gl()
            if self.sphere_renderer:
                self.sphere_renderer.init_gl()

            self._gl_initialized = True
            self.update_from_doc()
        except Exception as e:
            logger.error(f"OpenGL Initialization Error: {e}", exc_info=True)
            self._gl_initialized = False

    def _update_theme_colors(self):
        """
        Reads the current theme colors from the widget's style context
        and applies them to the renderer.
        """
        if not self.axis_renderer:
            return

        style_context = self.get_style_context()

        # Get background color and set it for OpenGL
        found, bg_rgba = style_context.lookup_color("view_bg_color")
        if found:
            GL.glClearColor(
                bg_rgba.red, bg_rgba.green, bg_rgba.blue, bg_rgba.alpha
            )
        else:
            GL.glClearColor(0.2, 0.2, 0.25, 1.0)  # Fallback

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
            bg_plane_color = fg_rgba.red, fg_rgba.green, fg_rgba.blue, 0.2

            self.axis_renderer.set_background_color(bg_plane_color)
            self.axis_renderer.set_axis_color(axis_color)
            self.axis_renderer.set_label_color(axis_color)
            self.axis_renderer.set_grid_color(grid_color)

    def on_render(self, area, ctx) -> bool:
        """The main rendering loop."""
        if not self._gl_initialized:
            self._init_gl_resources()
        if not self._gl_initialized or not self.camera:
            return False

        self._update_theme_colors()

        try:
            GL.glViewport(0, 0, self.camera.width, self.camera.height)
            GL.glClear(
                GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT  # type: ignore
            )

            proj_matrix = self.camera.get_projection_matrix()
            view_matrix = self.camera.get_view_matrix()
            mvp_matrix = proj_matrix @ view_matrix

            if self.axis_renderer and self.main_shader and self.text_shader:
                self.axis_renderer.render(
                    self.main_shader,
                    self.text_shader,
                    mvp_matrix,
                    view_matrix,
                )
            if self.ops_renderer and self.main_shader:
                self.ops_renderer.render(self.main_shader, mvp_matrix)

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
            self._rotation_pivot = self.get_world_coords_on_plane(
                x, y, self.camera
            )
            if self._rotation_pivot is None:
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
            delta_x = x_curr - prev_x
            delta_y = y_curr - prev_y
            self._last_orbit_pos = x_curr, y_curr

            sensitivity = 0.004

            # Yaw (Horizontal drag): Rotate around the fixed world Y-axis.
            if abs(delta_x) > 1e-6:
                axis_yaw = np.array([0, 1, 0], dtype=np.float64)
                self.camera.orbit(
                    self._rotation_pivot, axis_yaw, -delta_x * sensitivity
                )

            # Pitch (Vertical drag): Rotate around the camera's local
            # right-axis.
            if abs(delta_y) > 1e-6:
                forward = self.camera.target - self.camera.position
                axis_pitch = np.cross(forward, self.camera.up)

                if np.linalg.norm(axis_pitch) > 1e-6:
                    self.camera.orbit(
                        self._rotation_pivot,
                        axis_pitch,
                        -delta_y * sensitivity,
                    )

        self.queue_render()

    def on_drag_end(self, gesture, offset_x, offset_y):
        """Handles the end of a drag operation."""
        self._last_pan_offset = None
        if self._is_orbiting:
            self._is_orbiting = False
            self._last_orbit_pos = None
            self._rotation_pivot = None
            self.queue_render()

    def on_scroll(self, controller, dx, dy):
        """Handles the mouse scroll wheel for zooming."""
        if self.camera:
            self.camera.dolly(dy)
            self.queue_render()

    def update_from_doc(self):
        """Updates the ops renderer with new data and redraws."""
        if self.ops_renderer and self._gl_initialized:
            self.ops_renderer.update_ops(self.create_sample_ops())
            self.queue_render()

    def create_sample_ops(self):
        """Creates a sample set of operations for visualization."""
        ops = Ops()
        ops.move_to(10, 10, 5)
        ops.line_to(90, 10, 0)
        ops.line_to(90, 90, 0)
        ops.line_to(10, 90, 0)
        ops.line_to(10, 10, 0)
        return ops
