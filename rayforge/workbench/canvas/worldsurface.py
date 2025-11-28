import logging
from typing import Tuple
from gi.repository import Graphene, Gdk, Gtk
from ...core.matrix import Matrix
from .canvas import Canvas
from .axis import AxisRenderer


logger = logging.getLogger(__name__)


class WorldSurface(Canvas):
    """
    The WorldSurface provides a generic canvas with a real-world coordinate
    system (in millimeters), a grid, axes, and interactive pan/zoom controls.
    It is the base class for more specific surfaces like the WorkSurface.
    """

    # The minimum allowed zoom level, relative to the "fit-to-view" size
    # (zoom=1.0). 0.1 means you can zoom out until the view is 10% of its
    # "fit" size.
    MIN_ZOOM_FACTOR = 0.1

    # The maximum allowed pixel density when zooming in.
    MAX_PIXELS_PER_MM = 100.0

    def __init__(
        self,
        width_mm: float = 100.0,
        height_mm: float = 100.0,
        y_axis_down: bool = False,
        show_grid: bool = True,
        show_axis: bool = True,
        **kwargs,
    ):
        logger.debug("WorldSurface.__init__ called")
        super().__init__(**kwargs)
        self.grid_size = 1.0  # Set snap grid to 1mm in world coordinates
        self.zoom_level = 1.0
        self.pan_x_mm = 0.0
        self.pan_y_mm = 0.0
        self._last_view_scale_x: float = 0.0
        self._last_view_scale_y: float = 0.0
        self.width_mm = width_mm
        self.height_mm = height_mm

        # The root element is now static and sized in world units (mm).
        self.root.set_size(self.width_mm, self.height_mm)
        self.root.clip = False

        self._axis_renderer = AxisRenderer(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            y_axis_down=y_axis_down,
            show_grid=show_grid,
            show_axis=show_axis,
        )
        self.root.background = 0.8, 0.8, 0.8, 0.1

        # Set theme colors for axis and grid.
        self._update_theme_colors()

        # Add scroll event controller for zoom
        self._scroll_controller = Gtk.EventControllerScroll.new(
            Gtk.EventControllerScrollFlags.VERTICAL
        )
        self._scroll_controller.connect("scroll", self.on_scroll)
        self.add_controller(self._scroll_controller)

        # Add middle click gesture for panning
        self._pan_gesture = Gtk.GestureDrag.new()
        self._pan_gesture.set_button(Gdk.BUTTON_MIDDLE)
        self._pan_gesture.connect("drag-begin", self.on_pan_begin)
        self._pan_gesture.connect("drag-update", self.on_pan_update)
        self._pan_gesture.connect("drag-end", self.on_pan_end)
        self.add_controller(self._pan_gesture)
        self._pan_start = (0.0, 0.0)

        # Add right-click gesture for context menu
        self._context_menu_gesture = Gtk.GestureClick.new()
        self._context_menu_gesture.set_button(Gdk.BUTTON_SECONDARY)
        self._context_menu_gesture.connect(
            "pressed", self.on_right_click_pressed
        )
        self.add_controller(self._context_menu_gesture)

        # This is hacky, but what to do: The EventControllerScroll provides
        # no access to any mouse position, and there is no easy way to
        # get the mouse position in Gtk4. So I have to store it here and
        # track the motion event...
        self._mouse_pos = (0.0, 0.0)

    def set_show_grid(self, show: bool):
        """Sets the visibility of the inner grid lines."""
        self._axis_renderer.show_grid = show
        self.queue_draw()

    def set_show_axis(self, show: bool):
        """Sets the visibility of the outer axis lines and labels."""
        self._axis_renderer.show_axis = show
        self.queue_draw()

    def on_right_click_pressed(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ) -> None:
        """
        Placeholder for handling right-clicks. Subclasses should override this
        to implement context menu logic.
        """
        pass

    def _update_theme_colors(self) -> None:
        """
        Reads the current theme colors from the widget's style context
        and applies them to the AxisRenderer.
        """
        # Get the foreground color for axes and labels
        fg_rgba = self.get_color()
        self._axis_renderer.set_fg_color(
            (fg_rgba.red, fg_rgba.green, fg_rgba.blue, fg_rgba.alpha)
        )

        # Set the separator color for the grid lines
        self._axis_renderer.set_grid_color(
            (
                fg_rgba.red,
                fg_rgba.green,
                fg_rgba.blue,
                fg_rgba.alpha * 0.3,
            )
        )

    def set_pan(self, pan_x_mm: float, pan_y_mm: float) -> None:
        """Sets the pan position in mm and updates the axis importer."""
        self.pan_x_mm = pan_x_mm
        self.pan_y_mm = pan_y_mm
        self._rebuild_view_transform()
        self.queue_draw()

    def set_zoom(self, zoom_level: float) -> None:
        """
        Sets the zoom level and updates the axis importer.
        The caller is responsible for ensuring the zoom_level is clamped.
        """
        self.zoom_level = zoom_level
        self._rebuild_view_transform()
        self.queue_draw()

    def set_size(self, width_mm: float, height_mm: float) -> None:
        """
        Sets the real-world size of the work surface in mm
        and updates related properties.
        """
        self.width_mm = width_mm
        self.height_mm = height_mm
        self.root.set_size(width_mm, height_mm)
        self._axis_renderer.set_width_mm(self.width_mm)
        self._axis_renderer.set_height_mm(self.height_mm)
        self._rebuild_view_transform()
        self.queue_draw()

    def get_size_mm(self) -> Tuple[float, float]:
        """Returns the size of the work surface in mm."""
        return self.width_mm, self.height_mm

    def get_view_scale(self) -> Tuple[float, float]:
        """
        Returns the current effective pixels-per-millimeter scale of the view,
        taking into account the base scale, zoom, and widget size.
        """
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return 1.0, 1.0

        _, _, content_w, content_h = self._axis_renderer.get_content_layout(
            widget_w, widget_h
        )

        base_scale_x = content_w / self.width_mm if self.width_mm > 0 else 1
        base_scale_y = content_h / self.height_mm if self.height_mm > 0 else 1

        return base_scale_x * self.zoom_level, base_scale_y * self.zoom_level

    def on_motion(self, gesture: Gtk.Gesture, x: float, y: float) -> None:
        self._mouse_pos = x, y

        # Let the base canvas handle hover updates and cursor changes.
        super().on_motion(gesture, x, y)

    def on_scroll(
        self, controller: Gtk.EventControllerScroll, dx: float, dy: float
    ) -> None:
        """Handles the scroll event for zoom."""
        logger.debug(f"Scroll event: dx={dx:.2f}, dy={dy:.2f}")
        zoom_speed = 0.1
        # 1. Calculate a desired new zoom level based on scroll direction
        desired_zoom = self.zoom_level * (
            (1 - zoom_speed) if dy > 0 else (1 + zoom_speed)
        )
        # 2. Get the base "fit-to-view" pixel density (for zoom = 1.0)
        base_ppm = self._axis_renderer.get_base_pixels_per_mm(
            self.get_width(), self.get_height()
        )
        if base_ppm <= 0:
            return
        # 3. Calculate the pixel density limits
        min_ppm = base_ppm * self.MIN_ZOOM_FACTOR
        max_ppm = self.MAX_PIXELS_PER_MM

        # 4. Calculate the target density and clamp it within our limits
        clamped_ppm = max(min_ppm, min(base_ppm * desired_zoom, max_ppm))
        # 5. Convert the valid, clamped density back into a final zoom level
        final_zoom = clamped_ppm / base_ppm
        if abs(final_zoom - self.zoom_level) < 1e-9:
            return

        # 6. Calculate pan adjustment to zoom around the mouse cursor
        mouse_x_px, mouse_y_px = self._mouse_pos
        focus_x_mm, focus_y_mm = self._get_world_coords(mouse_x_px, mouse_y_px)
        self.set_zoom(final_zoom)
        new_mouse_x_mm, new_mouse_y_mm = self._get_world_coords(
            mouse_x_px, mouse_y_px
        )
        new_pan_x_mm = self.pan_x_mm + (focus_x_mm - new_mouse_x_mm)
        new_pan_y_mm = self.pan_y_mm + (focus_y_mm - new_mouse_y_mm)
        self.set_pan(new_pan_x_mm, new_pan_y_mm)

    def do_size_allocate(self, width: int, height: int, baseline: int) -> None:
        # Let the parent Canvas/Gtk.DrawingArea do its work first. This will
        # call self.root.set_size() with pixel dimensions, which we will
        # immediately correct.
        super().do_size_allocate(width, height, baseline)

        # Enforce the correct world (mm) dimensions on the root
        # element, overriding the pixel-based sizing from the parent class.
        if (
            self.root.width != self.width_mm
            or self.root.height != self.height_mm
        ):
            self.root.set_size(self.width_mm, self.height_mm)

        # Rebuild the view transform, which depends on the widget's new pixel
        # dimensions to calculate the correct pan/zoom/scale matrix.
        self._rebuild_view_transform()

    def _rebuild_view_transform(self) -> None:
        """
        Constructs the world-to-view transformation matrix.
        """
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return

        content_x, content_y, content_w, content_h = (
            self._axis_renderer.get_content_layout(widget_w, widget_h)
        )

        # Base scale to map mm to the unzoomed content area pixels
        scale_x = content_w / self.width_mm if self.width_mm > 0 else 1
        scale_y = content_h / self.height_mm if self.height_mm > 0 else 1

        # The sequence of transformations is critical and is applied
        # from right-to-left (bottom to top in this code).

        # 5. Final Offset: Translate the transformed content to its
        #    final position within the widget.
        m_offset = Matrix.translation(content_x, content_y)

        # 4. Zoom: Scale the content around its top-left corner (0,0).
        m_zoom = Matrix.scale(self.zoom_level, self.zoom_level)

        # 3. Y-Axis and Pan transformation
        # We combine pan and the y-flip into one matrix. This ensures panning
        # feels correct regardless of the axis orientation.
        pan_transform = Matrix.translation(-self.pan_x_mm, -self.pan_y_mm)

        # The world is ALWAYS Y-up. The view is ALWAYS Y-down.
        # Therefore, we ALWAYS need to flip the Y-axis. This matrix scales
        # the world to pixels and flips it into the view's coordinate system.
        m_scale = Matrix.translation(0, content_h) @ Matrix.scale(
            scale_x, -scale_y
        )

        # Compose final matrix (read operations from bottom to top):
        # Transformation order:
        #   Pan the world
        #   -> Scale&Flip it
        #   -> Zoom it
        #   -> Offset to final position.
        final_transform = m_offset @ m_zoom @ m_scale @ pan_transform

        # Update the base Canvas's view_transform
        self.view_transform = final_transform

        # Check if the effective scale (pixels-per-mm) has changed. Panning
        # does not change the scale, but zooming and resizing the window do.
        # This prevents expensive re-rendering of buffered elements during
        # panning.
        new_scale_x, new_scale_y = self.get_view_scale()
        scale_changed = (
            abs(new_scale_x - self._last_view_scale_x) > 1e-9
            or abs(new_scale_y - self._last_view_scale_y) > 1e-9
        )

        if scale_changed:
            self._last_view_scale_x = new_scale_x
            self._last_view_scale_y = new_scale_y

    def reset_view(self) -> None:
        """
        Resets the view to fit the surface, including a
        full reset of pan and zoom.
        """
        logger.debug("Resetting WorldSurface view.")
        self.set_pan(0.0, 0.0)
        self.set_zoom(1.0)
        self._rebuild_view_transform()
        self.queue_draw()

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        # Update theme colors right before drawing to catch any live changes.
        self._update_theme_colors()

        # Create a Cairo context for the snapshot
        width, height = self.get_width(), self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        # Draw grid and axes first, in pixel space, before any transformations.
        self._axis_renderer.draw_grid_and_labels(
            ctx, self.view_transform, width, height
        )

        # Now, delegate to the base Canvas's snapshot implementation, which
        # will correctly apply the view_transform and render all elements
        # and selection handles.
        super().do_snapshot(snapshot)

    def on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Handles key press events for the work surface."""
        key_name = Gdk.keyval_name(keyval)
        logger.debug(f"Key pressed: key='{key_name}', state={state}")
        if keyval == Gdk.KEY_1:
            # Reset pan and zoom with '1'
            self.reset_view()
            return True  # Event handled

        # Propagate to parent Canvas for its default behavior. The base Canvas
        # handles leaving edit mode on Escape.
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(
        self, gesture: Gtk.GestureDrag, x: float, y: float
    ) -> None:
        logger.debug(f"Pan begin at ({x:.2f}, {y:.2f})")
        self._pan_start = (self.pan_x_mm, self.pan_y_mm)

    def on_pan_update(
        self, gesture: Gtk.GestureDrag, x: float, y: float
    ) -> None:
        # Gtk.GestureDrag.get_offset returns a boolean and populates the
        # provided variables.
        ok, offset_x, offset_y = gesture.get_offset()
        if not ok:
            return

        logger.debug(f"Pan update: offset=({offset_x:.2f}, {offset_y:.2f})")

        # We need to convert the pixel offset into a mm delta. This delta
        # is independent of the pan, so we can calculate it from the scale.
        widget_w, widget_h = self.get_width(), self.get_height()
        if widget_w <= 0 or widget_h <= 0:
            return

        _, _, content_w, content_h = self._axis_renderer.get_content_layout(
            widget_w, widget_h
        )

        base_scale_x = content_w / self.width_mm if self.width_mm > 0 else 1
        base_scale_y = content_h / self.height_mm if self.height_mm > 0 else 1

        delta_x_mm = offset_x / (base_scale_x * self.zoom_level)
        delta_y_mm = offset_y / (base_scale_y * self.zoom_level)

        # The world-to-view transform is always Y-inverting. To make the
        # content follow the mouse ("natural" panning), the logic must be
        # consistent. A rightward drag (positive offset_x) requires a
        # negative adjustment to pan_x. A downward drag (positive offset_y)
        # requires a positive adjustment to pan_y because of the Y-inversion
        # in the transform matrix.
        new_pan_x = self._pan_start[0] - delta_x_mm
        new_pan_y = self._pan_start[1] + delta_y_mm

        self.set_pan(new_pan_x, new_pan_y)

    def on_pan_end(self, gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        logger.debug(f"Pan end at ({x:.2f}, {y:.2f})")
        pass
