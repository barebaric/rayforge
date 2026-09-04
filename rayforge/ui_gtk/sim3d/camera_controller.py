"""
Camera + interaction controller for the 3D canvas.

Owns the :class:`Camera` instance, the drag/scroll gesture wiring, the
orbit/pan/dolly math, view resetting, and viewport resizing.  The canvas
stays a thin ``Gtk.GLArea`` that consumes the camera during rendering.
"""

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from gi.repository import Gdk, Gtk
from raygeo.geo.types import Point

from .camera import Camera, ViewDirection, rotation_matrix_from_axis_angle
from .gl_utils import rotation_4x4
from .picking import PickScene, camera_ray, pick_point

if TYPE_CHECKING:
    from .viewport import ViewportConfig

logger = logging.getLogger(__name__)


class CameraController:
    """
    Manages the 3D camera and all mouse/key interactions for the canvas.

    The controller attaches its own GTK gesture/event controllers to the
    widget it is given and requests redraws through ``request_render``.
    """

    def __init__(
        self,
        widget: Gtk.Widget,
        get_viewport: Callable[[], "ViewportConfig"],
        request_render: Callable[[], None],
        on_key_pressed: Callable | None = None,
        get_pick_scene: Callable[[], PickScene | None] | None = None,
    ):
        self.camera: Camera | None = None
        self._widget = widget
        self._get_viewport = get_viewport
        self._request_render = request_render
        self._get_pick_scene = get_pick_scene

        # State for interactions
        self._is_orbiting = False
        self._is_z_rotating = False
        self._last_pan_offset: Point | None = None
        self._pan_anchor: np.ndarray | None = None
        self._pan_start_screen: tuple[float, float] | None = None
        self._rotation_pivot: np.ndarray | None = None
        self._last_orbit_pos: Point | None = None
        self._last_z_rotate_screen_pos: Point | None = None

        # The EventControllerScroll provides no access to the pointer
        # position, so it is tracked here via a motion controller.
        self._mouse_pos: tuple[float, float] | None = None

        self._setup_interactions(on_key_pressed)

    def create_camera(self, width: int, height: int) -> Camera:
        """Create the camera at the given widget size and store it."""
        self.camera = Camera(
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            width,
            height,
        )
        return self.camera

    def on_resize(self, area, width: int, height: int):
        """Handles the window resize event."""
        if self.camera:
            self.camera.width, self.camera.height = int(width), int(height)
        self._request_render()

    def get_world_coords_on_plane(
        self, x: float, y: float
    ) -> np.ndarray | None:
        """Calculates the 3D world coordinates on the XY plane from 2D."""
        camera = self.camera
        if camera is None:
            return None

        ray = camera_ray(camera, x, y)
        if ray is None:
            return None
        ray_origin, ray_dir = ray

        plane_normal = np.array([0, 0, 1], dtype=np.float64)
        denom = np.dot(plane_normal, ray_dir)
        if abs(denom) < 1e-6:
            return None

        t = -np.dot(plane_normal, ray_origin) / denom
        if t < 0:
            return None

        # When the ray grazes the plane (a near-flat view), the
        # intersection is almost infinitely far away.  Anchoring a pan or
        # zoom on such a point amplifies mouse movement into extreme,
        # erratic camera motion, so bail out and let callers fall back to
        # stable pixel-based panning / dolly-only zoom.
        if abs(denom) < 0.1:
            return None

        return ray_origin + t * ray_dir

    def _setup_interactions(self, on_key_pressed: Callable | None = None):
        """Connects GTK4 gesture and event controllers for interaction."""
        # Middle mouse drag for Pan/Orbit
        drag_middle = Gtk.GestureDrag.new()
        drag_middle.set_button(Gdk.BUTTON_MIDDLE)
        drag_middle.connect("drag-begin", self.on_drag_begin)
        drag_middle.connect("drag-update", self.on_drag_update)
        drag_middle.connect("drag-end", self.on_drag_end)
        self._widget.add_controller(drag_middle)

        # Left mouse drag for Z-axis rotation
        drag_left = Gtk.GestureDrag.new()
        drag_left.set_button(Gdk.BUTTON_PRIMARY)
        drag_left.connect("drag-begin", self.on_z_rotate_begin)
        drag_left.connect("drag-update", self.on_z_rotate_update)
        drag_left.connect("drag-end", self.on_z_rotate_end)
        self._widget.add_controller(drag_left)

        scroll = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        scroll.connect("scroll", self.on_scroll)
        self._widget.add_controller(scroll)

        # Track the pointer position for zooming towards the cursor.
        motion = Gtk.EventControllerMotion.new()
        motion.connect("motion", self.on_motion)
        motion.connect("leave", self.on_motion_leave)
        self._widget.add_controller(motion)

        # Grab keyboard focus when the canvas is clicked so that the
        # EventControllerKey actually receives key events.  Without this,
        # the previously-focused widget keeps consuming them.
        click = Gtk.GestureClick.new()
        click.set_button(0)
        click.connect("pressed", self._on_click_focus)
        self._widget.add_controller(click)

        key_controller = Gtk.EventControllerKey.new()
        if on_key_pressed is not None:
            key_controller.connect("key-pressed", on_key_pressed)
        self._widget.add_controller(key_controller)

    def _on_click_focus(self, gesture, n_press, x, y):
        """Grab keyboard focus so the canvas receives key events."""
        logger.debug(
            "Canvas3D click: button=%d, n_press=%d, pos=(%.2f, %.2f)",
            gesture.get_current_button(),
            n_press,
            x,
            y,
        )
        self._widget.grab_focus()

    def _clear_drag_state(self):
        """Resets all state variables related to any drag operation."""
        self._is_orbiting = False
        self._is_z_rotating = False
        self._last_pan_offset = None
        self._pan_anchor = None
        self._pan_start_screen = None
        self._rotation_pivot = None
        self._last_orbit_pos = None
        self._last_z_rotate_screen_pos = None

    def reset_view(self, direction: ViewDirection):
        """Resets the camera to the specified preset view."""
        if not self.camera:
            return
        logger.info("Resetting to %s view.", direction.value)
        viewport = self._get_viewport()
        self.camera.set_view(
            direction,
            viewport.width_mm,
            viewport.depth_mm,
        )
        self._clear_drag_state()
        self._request_render()

    def on_drag_begin(self, gesture, x: float, y: float):
        """Handles the start of a middle-mouse-button drag."""
        gesture.set_state(Gtk.EventSequenceState.CLAIMED)
        state = gesture.get_current_event_state()
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)
        logger.debug("Middle drag begin at (%.2f, %.2f)", x, y)

        if not is_shift and self.camera:
            # Orbit around the point on the object under the cursor,
            # falling back to the point on the floor plane.
            self._rotation_pivot = self._pick_pivot(x, y)
            if self._rotation_pivot is None:
                # Fall back to the camera target projected onto the grid
                # plane and clamped to the grid.  The raw target can drift
                # far below the plane during navigation, and orbiting around
                # such a point would sweep the camera through an enormous
                # arc.
                fallback = self.camera.target.copy()
                fallback[2] = 0.0
                self._rotation_pivot = self._clamp_to_grid(fallback)

            self._last_orbit_pos = None
            self._is_orbiting = True
            logger.debug(
                "Middle drag orbits, pivot=%s",
                self._rotation_pivot.tolist(),
            )
        else:
            self._pan_anchor = self.get_world_coords_on_plane(x, y)
            self._pan_start_screen = x, y
            self._last_pan_offset = 0.0, 0.0
            self._is_orbiting = False
            if self._pan_anchor is not None:
                self._pan_anchor = self._clamp_to_grid(self._pan_anchor)
            anchor = (
                self._pan_anchor.tolist()
                if self._pan_anchor is not None
                else None
            )
            logger.debug("Middle drag pans, anchor=%s", anchor)

    def on_drag_update(self, gesture, offset_x: float, offset_y: float):
        """Handles updates during a drag operation (panning or orbiting)."""
        logger.debug(
            "Middle drag update: offset=(%.2f, %.2f)", offset_x, offset_y
        )
        if not self.camera:
            return
        camera = self.camera

        state = gesture.get_current_event_state()
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        if is_shift:
            self._update_pan(camera, offset_x, offset_y)
            self._request_render()
            return

        delta = self._get_orbit_delta(gesture)
        if delta is not None and self._rotation_pivot is not None:
            self._apply_orbit(camera, self._rotation_pivot, *delta)
            self._request_render()

    def _pick_pivot(self, x: float, y: float) -> np.ndarray | None:
        """Returns the point on scene geometry under the cursor.

        Uses the scene's pickable geometry when available and falls
        back to the floor plane point otherwise.  A picked object point is
        kept as-is (bounded only by distance), while a plane point with no
        object under the cursor is clamped onto the grid so that orbiting
        around far-away empty space does not rotate the view extremely
        rapidly.
        """
        if self._get_pick_scene is not None and self.camera is not None:
            scene = self._get_pick_scene()
            if scene is not None:
                point = pick_point(scene, self.camera, x, y)
                if point is not None:
                    return self._clamp_pivot(point)
        pivot = self.get_world_coords_on_plane(x, y)
        if pivot is None:
            return None
        return self._clamp_to_grid(pivot)

    def _clamp_to_grid(self, point: np.ndarray) -> np.ndarray:
        """Clamps a point on the plane to the grid bounds.

        When the cursor is over empty plane away from the grid, the picked
        point would otherwise be far outside the working area, making orbit
        sweep the camera through an enormous arc or making pan track a
        distant point and move the camera extremely fast.  Pulling the point
        back onto the grid keeps navigation comfortable.  The z component is
        flattened onto the plane as well, since these points live on it.
        """
        viewport = self._get_viewport()
        point = point.copy()
        point[0] = min(max(point[0], 0.0), viewport.width_mm)
        point[1] = min(max(point[1], 0.0), viewport.depth_mm)
        point[2] = 0.0
        return point

    def _clamp_pivot(self, pivot: np.ndarray) -> np.ndarray:
        """Limits the orbit pivot to a reasonable distance from the camera.

        The pivot is pulled in towards the camera along the pick ray when it
        is farther than a few times the camera-to-target distance, keeping
        the orbit comfortable regardless of where the cursor was clicked.
        """
        camera = self.camera
        if camera is None:
            return pivot
        to_pivot = pivot - camera.position
        distance = np.linalg.norm(to_pivot)
        if distance < 1e-9:
            return pivot
        ref = np.linalg.norm(camera.target - camera.position)
        max_distance = max(3.0 * ref, 1.0)
        if distance <= max_distance:
            return pivot
        return camera.position + to_pivot * (max_distance / distance)

    def _update_pan(self, camera: Camera, offset_x: float, offset_y: float):
        """Pans the camera so the scene tracks the mouse 1:1 on screen.

        The camera is moved by the on-screen pixel delta scaled to world
        units (see :meth:`Camera.pan`), so the plane and model translate at
        the speed of the mouse pointer in every view.
        """
        if self._last_pan_offset is None:
            self._last_pan_offset = 0.0, 0.0
        dx = offset_x - self._last_pan_offset[0]
        dy = offset_y - self._last_pan_offset[1]
        camera.pan(-dx, -dy)
        self._last_pan_offset = offset_x, offset_y

    def _get_orbit_delta(self, gesture) -> tuple[float, float] | None:
        """Returns the (dx, dy) since the last orbit step, or None."""
        if not self._is_orbiting or self._rotation_pivot is None:
            return None

        event = gesture.get_last_event()
        if not event:
            return None
        _, x_curr, y_curr = event.get_position()

        if self._last_orbit_pos is None:
            self._last_orbit_pos = x_curr, y_curr
            return None

        prev_x, prev_y = self._last_orbit_pos
        self._last_orbit_pos = x_curr, y_curr
        return x_curr - prev_x, y_curr - prev_y

    def _apply_orbit(
        self,
        camera: Camera,
        pivot: np.ndarray,
        delta_x: float,
        delta_y: float,
    ):
        """Orbits the camera around the given pivot by the drag delta."""
        sensitivity = 0.004

        # A pivot far from the camera makes the camera sweep a huge arc for
        # a fixed angular step, so the view races across the screen.  Scale
        # the sensitivity down with the pivot distance to keep the on-screen
        # rotation rate consistent no matter how far the clicked point is.
        ref = np.linalg.norm(camera.target - camera.position)
        pivot_dist = np.linalg.norm(pivot - camera.position)
        if ref > 1e-9 and pivot_dist > 1e-9:
            sensitivity *= min(ref / pivot_dist, 1.0)

        if camera.is_perspective:
            self._orbit_perspective(
                camera, pivot, delta_x, delta_y, sensitivity
            )
        else:
            self._orbit_orthographic(
                camera, pivot, delta_x, delta_y, sensitivity
            )

    def _orbit_perspective(
        self,
        camera: Camera,
        pivot: np.ndarray,
        delta_x: float,
        delta_y: float,
        sensitivity: float,
    ):
        """Perspective orbit (Z-Up Turntable)."""
        if abs(delta_x) > 1e-6:
            axis_yaw = np.array([0, 0, 1], dtype=np.float64)
            camera.orbit(pivot, axis_yaw, -delta_x * sensitivity)
        if abs(delta_y) > 1e-6:
            forward = camera.target - camera.position
            axis_pitch = np.cross(forward, camera.up)
            if np.linalg.norm(axis_pitch) > 1e-6:
                camera.orbit(pivot, axis_pitch, -delta_y * sensitivity)

    def _orbit_orthographic(
        self,
        camera: Camera,
        pivot: np.ndarray,
        delta_x: float,
        delta_y: float,
        sensitivity: float,
    ):
        """Orthographic orbit (Z-Up Turntable).

        The on-screen rotation rate of an orthographic view scales with the
        orthographic zoom (zooming in does not move the camera, so a fixed
        angular step sweeps the view across proportionally more pixels).  The
        sensitivity is therefore normalised by the visible height so the
        rotation feels the same at every zoom level.
        """
        ref = camera._ortho_ref_distance
        if ref is None:
            ref = np.linalg.norm(camera.target - camera.position)
        if ref < 1e-9:
            return
        scale = camera.get_ortho_height() / ref
        yaw_angle = -delta_x * sensitivity * scale
        pitch_angle = -delta_y * sensitivity * scale

        # Yaw Rotation (around World Z axis)
        if abs(yaw_angle) > 1e-6:
            axis_yaw = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            rot_yaw = rotation_4x4(axis_yaw, yaw_angle)[:3, :3]
            # Apply to position and target vectors
            camera.position = pivot + rot_yaw @ (camera.position - pivot)
            camera.target = pivot + rot_yaw @ (camera.target - pivot)
            camera.up = rot_yaw @ camera.up

        # Pitch Rotation (around Camera's local right axis)
        if abs(pitch_angle) > 1e-6:
            self._apply_ortho_pitch(camera, pivot, pitch_angle)

    def _apply_ortho_pitch(
        self, camera: Camera, pivot: np.ndarray, pitch_angle: float
    ):
        """Applies a single pitch step around the camera's local right axis."""
        # Get camera's state *after* the yaw rotation
        forward_vec = camera.target - camera.position
        world_z_axis = np.array([0.0, 0.0, 1.0])

        # Gimbal Lock Prevention
        norm_fwd = np.linalg.norm(forward_vec)
        if norm_fwd > 1e-6:
            dot_prod = np.dot(forward_vec / norm_fwd, world_z_axis)
            # Stop if looking down and trying to pitch more down
            if (
                dot_prod < -0.999
                and pitch_angle < 0
                or dot_prod > 0.999
                and pitch_angle > 0
            ):
                pitch_angle = 0.0

        if abs(pitch_angle) > 1e-6:
            axis_pitch = np.cross(forward_vec, camera.up)
            if np.linalg.norm(axis_pitch) > 1e-6:
                rot_pitch = rotation_matrix_from_axis_angle(
                    axis_pitch, pitch_angle
                )
                # Apply to position and target vectors
                camera.position = pivot + rot_pitch @ (camera.position - pivot)
                camera.target = pivot + rot_pitch @ (camera.target - pivot)
                camera.up = rot_pitch @ camera.up

    def on_drag_end(self, gesture, offset_x, offset_y):
        """Handles the end of a drag operation."""
        logger.debug(
            "Middle drag end: offset=(%.2f, %.2f)", offset_x, offset_y
        )
        self._clear_drag_state()
        self._request_render()

    def on_z_rotate_begin(self, gesture, x: float, y: float):
        """
        Handles the start of a left-mouse-button drag for Z-axis rotation.
        """
        logger.debug("Z-rotate drag begin at (%.2f, %.2f)", x, y)
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
        self._request_render()

    def on_z_rotate_end(self, gesture, offset_x, offset_y):
        """Handles the end of a Z-axis rotation drag."""
        logger.debug(
            "Z-rotate drag end: offset=(%.2f, %.2f)", offset_x, offset_y
        )
        self._clear_drag_state()
        self._request_render()

    def on_motion(self, controller, x: float, y: float):
        """Stores the current pointer position for scroll zooming."""
        self._mouse_pos = x, y

    def on_motion_leave(self, controller):
        """Clears the stored pointer position when the pointer leaves."""
        self._mouse_pos = None

    def on_scroll(self, controller, dx, dy):
        """Handles the mouse scroll wheel for zooming.

        Zooms towards the point on the floor plane under the mouse cursor:
        the camera is dollied and then translated so that the plane point
        under the cursor stays under the cursor.
        """
        logger.debug("Scroll event: dx=%.2f, dy=%.2f", dx, dy)
        if not self.camera:
            return

        if self._mouse_pos is not None:
            self.zoom_towards_point(*self._mouse_pos, dy)
        else:
            self.camera.dolly(dy)
        self._request_render()

    def zoom_towards_point(self, x: float, y: float, dy: float) -> None:
        """
        Dollies the camera, keeping the plane point under the cursor.

        The camera is dollied along the line of sight (towards the target)
        so the zoom direction and amount are always consistent.  A lateral
        correction then keeps the plane point under the cursor from sliding
        sideways.  The correction's component along the view direction is
        dropped, because keeping a distant plane point under the cursor
        would otherwise pull the camera back and cancel (or reverse) the
        zoom.

        Args:
            x: The screen x coordinate of the anchor point.
            y: The screen y coordinate of the anchor point.
            dy: The scroll delta passed to :meth:`Camera.dolly`.
        """
        camera = self.camera
        if camera is None:
            return

        anchor = self.get_world_coords_on_plane(x, y)
        camera.dolly(dy)
        if anchor is None:
            return
        follow = self.get_world_coords_on_plane(x, y)
        if follow is None:
            return

        shift = anchor - follow
        forward = camera.target - camera.position
        forward_norm = np.linalg.norm(forward)
        if forward_norm < 1e-9:
            return
        forward /= forward_norm
        lateral = shift - forward * np.dot(shift, forward)
        camera.position += lateral
        camera.target += lateral
