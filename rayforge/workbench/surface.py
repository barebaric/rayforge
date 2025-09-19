import logging
from blinker import Signal
from typing import TYPE_CHECKING, Optional, Tuple, cast, Dict, List, Sequence
from gi.repository import Graphene, Gdk, Gtk
from ..core.group import Group
from ..core.layer import Layer
from ..core.workpiece import WorkPiece
from ..core.item import DocItem
from ..core.matrix import Matrix
from ..machine.models.machine import Machine
from ..undo import ChangePropertyCommand
from .canvas import Canvas, CanvasElement
from .axis import AxisRenderer
from .elements.dot import DotElement
from .elements.workpiece import WorkPieceView
from .elements.group import GroupElement
from .elements.camera_image import CameraImageElement
from .elements.layer import LayerElement
from .elements.tab_handle import TabHandleElement
from . import context_menu

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class WorkSurface(Canvas):
    """
    The WorkSurface displays a grid area with WorkPieces and generated Ops
    according to real world dimensions. It is the application-specific
    subclass of the generic Canvas.
    """

    # The minimum allowed zoom level, relative to the "fit-to-view" size
    # (zoom=1.0). 0.1 means you can zoom out until the view is 10% of its
    # "fit" size.
    MIN_ZOOM_FACTOR = 0.1

    # The maximum allowed pixel density when zooming in.
    MAX_PIXELS_PER_MM = 100.0

    def __init__(
        self,
        editor: "DocEditor",
        machine: Optional[Machine],
        cam_visible: bool = False,
        **kwargs,
    ):
        logger.debug("WorkSurface.__init__ called")
        super().__init__(**kwargs)
        self.editor = editor
        self.doc = self.editor.doc
        self.machine = None  # will be assigned by set_machine() below
        self.zoom_level = 1.0
        self.pan_x_mm = 0.0
        self.pan_y_mm = 0.0
        self._last_view_scale_x: float = 0.0
        self._last_view_scale_y: float = 0.0
        self._show_travel_moves = False
        self._workpieces_visible = True
        self.width_mm, self.height_mm = (
            machine.dimensions if machine else (100.0, 100.0)
        )
        self._cam_visible = cam_visible
        self._laser_dot_pos_mm = 0.0, 0.0
        self._transform_start_states: Dict[CanvasElement, dict] = {}
        self.right_click_context: Optional[Dict] = None

        # The root element is now static and sized in world units (mm).
        self.root.set_size(self.width_mm, self.height_mm)
        self.root.clip = False

        y_axis_down = machine.y_axis_down if machine else False
        self._axis_renderer = AxisRenderer(
            width_mm=self.width_mm,
            height_mm=self.height_mm,
            y_axis_down=y_axis_down,
        )
        self.root.background = 0.8, 0.8, 0.8, 0.1

        # Set theme colors for axis and grid.
        self._update_theme_colors()

        # DotElement size is in world units (mm) and is dynamically
        # updated to maintain a constant pixel size on screen.
        self._laser_dot = DotElement(0, 0, 5.0)
        self.root.add(self._laser_dot)

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

        # Signals for clipboard and duplication operations
        self.cut_requested = Signal()
        self.copy_requested = Signal()
        self.paste_requested = Signal()
        self.duplicate_requested = Signal()
        self.aspect_ratio_changed = Signal()
        self.context_changed = Signal()

        # Connect to generic signals from the base Canvas class
        self.move_begin.connect(self._on_any_transform_begin)
        self.resize_begin.connect(self._on_resize_begin)
        self.rotate_begin.connect(self._on_any_transform_begin)
        self.shear_begin.connect(self._on_any_transform_begin)

        # The primary connection for model updates
        self.transform_end.connect(self._on_transform_end)

        self.set_machine(machine)

        # Connect to the history manager's changed signal to sync the view
        # globally, which is necessary for undo/redo actions triggered
        # outside of this widget.
        self.doc.history_manager.changed.connect(self._on_history_changed)

        # --- View State Management ---
        # This property holds the canonical global state for tab visibility.
        self._tabs_globally_visible: bool = True

    def get_global_tab_visibility(self) -> bool:
        """
        Returns the current global visibility state for tab handles. This is
        used by new WorkPieceViews to pull the correct initial state.
        """
        return self._tabs_globally_visible

    def set_global_tab_visibility(self, visible: bool):
        """
        Sets the global visibility for tab handles and propagates the change
        to all existing WorkPieceView elements.
        """
        if self._tabs_globally_visible == visible:
            return  # No change
        self._tabs_globally_visible = visible
        # Propagate the new state to all existing views
        for wp_elem in self.find_by_type(WorkPieceView):
            wp_view = cast(WorkPieceView, wp_elem)
            wp_view.set_tabs_visible_override(visible)

    def on_right_click_pressed(
        self, gesture, n_press: int, x: float, y: float
    ):
        """Handles right-clicks to show the context menu."""
        self.right_click_context = None  # Reset context on each click
        world_x, world_y = self._get_world_coords(x, y)
        hit_elem = self.root.get_elem_hit(world_x, world_y, selectable=True)

        if not hit_elem or hit_elem is self.root:
            self.context_changed.send(self)
            return

        # Determine the context type based on the hit element
        # Case 1: Clicked on a TabHandle
        context_type = None
        if isinstance(hit_elem, TabHandleElement):
            parent_wp_view = cast(WorkPieceView, hit_elem.parent)
            self.right_click_context = {
                "type": "tab",
                "tab_data": hit_elem.data,
                "workpiece": parent_wp_view.data,
            }
        # Case 2: Clicked on a WorkPieceView, check for path proximity
        elif isinstance(hit_elem, WorkPieceView):
            wp_view = cast(WorkPieceView, hit_elem)
            location = wp_view.get_closest_point_on_path(
                world_x, world_y, threshold_px=5.0
            )
            if location:
                self.right_click_context = {
                    "type": "geometry",
                    "workpiece": wp_view.data,
                    "location": location,
                }
            else:
                self.right_click_context = {"type": "item"}
        # Case 3: Clicked on another selectable item (e.g., a Group)
        elif hit_elem.selectable:
            self.right_click_context = {"type": "item"}

        # Notify listeners to update action states *before* showing the menu
        self.context_changed.send(self)

        # Now, call the specific function to show the correct menu
        if self.right_click_context:
            context_type = self.right_click_context["type"]
            if context_type == "item":
                if not hit_elem.selected:
                    self.unselect_all()
                    hit_elem.selected = True
                    self._finalize_selection_state()
                context_menu.show_item_context_menu(self, gesture)
            elif context_type == "geometry":
                context_menu.show_geometry_context_menu(self, gesture)
            elif context_type == "tab":
                context_menu.show_tab_context_menu(self, gesture)

    def _update_theme_colors(self):
        """
        Reads the current theme colors from the widget's style context
        and applies them to the AxisRenderer.
        """
        style_context = self.get_style_context()

        # Get the foreground color for axes and labels
        found, fg_rgba = style_context.lookup_color("view_fg_color")
        if found:
            self._axis_renderer.set_fg_color(
                (fg_rgba.red, fg_rgba.green, fg_rgba.blue, fg_rgba.alpha)
            )

        # Get the separator color for the grid lines
        found, grid_rgba = style_context.lookup_color("separator_color")
        if found:
            self._axis_renderer.set_grid_color(
                (
                    grid_rgba.red,
                    grid_rgba.green,
                    grid_rgba.blue,
                    grid_rgba.alpha,
                )
            )

    def _on_history_changed(self, sender, **kwargs):
        """
        Called when the undo/redo history changes. This handler acts as a
        synchronizer to fix state timing issues. It re-commits the current
        selection state to ensure all listeners are in sync.
        """
        logger.debug(
            f"History changed, synchronizing selection state. Sender: {sender}"
        )
        self._sync_selection_state()
        self.queue_draw()

    def _on_any_transform_begin(
        self,
        sender,
        elements: List[CanvasElement],
        drag_target: Optional[CanvasElement] = None,
        **kwargs,
    ):
        """
        Saves the initial matrix of all transformed elements (including their
        ancestor groups) and the world size of all affected workpieces.
        The 'drag_target' argument is now explicitly accepted from signals
        that provide it (like move_begin).
        """
        logger.debug(
            f"Transform begin for {len(elements)} element(s). "
            f"Drag target: {drag_target}"
        )
        self._transform_start_states.clear()

        # 1. Collect all unique elements and their group ancestors
        items_to_capture = set()
        for element in elements:
            items_to_capture.add(element)
            parent = element.parent
            while isinstance(parent, GroupElement):
                items_to_capture.add(parent)
                parent = parent.parent

        # 2. Store the initial matrix for each captured item
        for element in items_to_capture:
            if isinstance(element.data, DocItem):
                self._transform_start_states[element] = {
                    "matrix": element.data.matrix.copy()
                }

        # 2. Find ALL unique workpieces that will be affected (including
        #    those inside selected groups) and store their initial world size.
        affected_workpieces = set()
        for element in elements:
            if isinstance(element.data, WorkPiece):
                affected_workpieces.add(element.data)
            elif isinstance(element.data, Group):
                affected_workpieces.update(
                    element.data.get_descendants(WorkPiece)
                )

        for wp in affected_workpieces:
            wp_element = self.find_by_data(wp)
            if not wp_element:
                logger.warning(
                    f"Got a transformation for workpiece {wp.name} "
                    "but did not find its element. Skipping."
                )
                continue
            # Store the world size against the element for easy lookup later
            self._transform_start_states.setdefault(wp_element, {})[
                "world_size"
            ] = wp.get_world_transform().get_abs_scale()

    def _on_resize_begin(self, sender, elements: List[CanvasElement]):
        """Handles start of a resize, which may invalidate Ops."""
        logger.debug(
            f"Resize begin for {len(elements)} element(s)."
            " Pausing ops generator."
        )
        # Call the generic transform begin handler.
        # Note: resize_begin signal in canvas.py currently doesn't send
        # drag_target, so this call will pass None for drag_target in
        # _on_any_transform_begin, which is correct.
        self._on_any_transform_begin(sender, elements)
        self.editor.ops_generator.pause()

    def _on_transform_end(self, sender, elements: List[CanvasElement]):
        """
        Finalizes an interactive transform by collecting all matrix changes
        from view elements and creating a single, undoable transaction.
        """
        # Step 1: Collect all elements that may have changed.
        affected_elements = set()
        for element in elements:
            affected_elements.add(element)
            parent = element.parent
            while isinstance(parent, GroupElement):
                affected_elements.add(parent)
                parent = parent.parent

        # Step 2: Create a list of all model changes found.
        changes_to_commit = []
        for element in affected_elements:
            if (
                not isinstance(element.data, DocItem)
                or element not in self._transform_start_states
                or "matrix" not in self._transform_start_states[element]
            ):
                continue

            docitem: DocItem = element.data
            start_matrix = self._transform_start_states[element]["matrix"]
            new_matrix = element.transform

            if start_matrix != new_matrix:
                changes_to_commit.append(
                    (docitem, start_matrix, new_matrix.copy())
                )

        # Step 3: Delegate to the command handler to create the transaction.
        if changes_to_commit:
            self.editor.transform.create_transform_transaction(
                changes_to_commit
            )

        self._transform_start_states.clear()

        # If it was a resize, the ops are now stale. Resume the generator.
        if self._resizing:
            self.editor.ops_generator.resume()

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        """Overrides base to add application-specific layer selection logic."""
        # A left-click should clear any lingering right-click context.
        if gesture.get_button() == Gdk.BUTTON_PRIMARY:
            if self.right_click_context:
                self.right_click_context = None
                self.context_changed.send(self)

        # The base Canvas class handles the conversion from widget (pixel)
        # coordinates to world coordinates. We pass them on directly.
        logger.debug(
            f"Button press: n_press={n_press}, pos=({x:.2f}, {y:.2f})"
        )
        super().on_button_press(gesture, n_press, x, y)

        # After the click, check if the active element dictates a layer change.
        active_elem = self.get_active_element()
        if active_elem and isinstance(active_elem.data, WorkPiece):
            active_layer = active_elem.data.layer
            # If the workpiece's layer is not the document's active layer,
            # create an undoable command to change it.
            if active_layer and active_layer != self.doc.active_layer:
                cmd = ChangePropertyCommand(
                    target=self.doc,
                    property_name="active_layer",
                    new_value=active_layer,
                    name=_("Select Layer"),
                )
                # Using execute() adds it to the undo stack.
                self.doc.history_manager.execute(cmd)

    def set_machine(self, machine: Optional[Machine]):
        """
        Updates the WorkSurface to use a new machine instance. This handles
        disconnecting from the old machine's signals, connecting to the new
        one's, and performing a full reset of the view.
        """
        if self.machine is machine:
            return

        # Disconnect from the old machine's signals
        if self.machine:
            self.machine.changed.disconnect(self._on_machine_changed)

        # Update the machine reference
        self.machine = machine

        # Connect to the new machine's signals
        if self.machine:
            self.machine.changed.connect(self._on_machine_changed)
            self.reset_view()

    def set_pan(self, pan_x_mm: float, pan_y_mm: float):
        """Sets the pan position in mm and updates the axis importer."""
        self.pan_x_mm = pan_x_mm
        self.pan_y_mm = pan_y_mm
        self._rebuild_view_transform()
        self.queue_draw()

    def set_zoom(self, zoom_level: float):
        """
        Sets the zoom level and updates the axis importer.
        The caller is responsible for ensuring the zoom_level is clamped.
        """
        self.zoom_level = zoom_level
        self._rebuild_view_transform()
        self.queue_draw()

    def set_size(self, width_mm: float, height_mm: float):
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

    def on_motion(self, gesture, x: float, y: float):
        self._mouse_pos = x, y

        # Let the base canvas handle hover updates and cursor changes.
        super().on_motion(gesture, x, y)

    def on_scroll(self, controller, dx: float, dy: float):
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

    def do_size_allocate(self, width: int, height: int, baseline: int):
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

    def _rebuild_view_transform(self):
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

            # Propagate the view change to elements that depend on it.
            for elem in self.find_by_type(WorkPieceView):
                wp_view = cast(WorkPieceView, elem)
                wp_view.trigger_update()
                wp_view.update_handle_transforms()

            # Update laser dot size to maintain a constant size in pixels.
            desired_diameter_px = 3.0
            if new_scale_x > 1e-9:
                diameter_mm = desired_diameter_px / new_scale_x
                self._laser_dot.set_size(diameter_mm, diameter_mm)

        self.set_laser_dot_position(
            self._laser_dot_pos_mm[0], self._laser_dot_pos_mm[1]
        )

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers re-rendering."""
        if self._show_travel_moves != show:
            self._show_travel_moves = show
            # Re-render all ops surfaces on all workpiece views
            for elem in self.find_by_type(WorkPieceView):
                assert isinstance(elem, WorkPieceView)
                # Tell the view to re-render its ops.
                # A simple trigger_update won't work if the ops data hasn't
                # changed.
                elem.trigger_ops_rerender()

    def _create_and_add_layer_element(self, layer: "Layer"):
        """Creates a new LayerElement and adds it to the canvas root."""
        logger.debug(f"Adding new LayerElement for '{layer.name}'")
        layer_elem = LayerElement(layer=layer, canvas=self)
        self.root.add(layer_elem)

    def update_from_doc(self):
        """
        Synchronizes the canvas elements with the document model.

        This method ensures that the layers and their contents (workpieces,
        steps) displayed on the canvas perfectly match the state of the
        document's data model. It also reorders the LayerElements to match
        the Z-order of the layers in the document.
        """
        doc = self.doc

        # --- Step 1: Add and Remove LayerElements ---
        doc_layers_set = set(doc.layers)
        current_elements_on_canvas = {
            elem.data: elem for elem in self.find_by_type(LayerElement)
        }

        # Remove elements for layers that are no longer in the doc
        for layer, elem in current_elements_on_canvas.items():
            if layer not in doc_layers_set:
                elem.remove()

        # Add elements for new layers that are not yet on the canvas
        for layer in doc.layers:
            if layer not in current_elements_on_canvas:
                self._create_and_add_layer_element(layer)

        # --- Step 2: Reorder LayerElements for Z-stacking ---
        # The first layer in the list is at the bottom (drawn first).
        # The last layer is at the top (drawn last).
        layer_order_map = {layer: i for i, layer in enumerate(doc.layers)}

        def sort_key(element: CanvasElement):
            """
            Sort key for root's children. Camera at bottom, then dot,
            then layers.
            """
            if isinstance(element, LayerElement):
                # LayerElements are ordered according to the doc.layers list.
                return layer_order_map.get(element.data, len(layer_order_map))
            if isinstance(element, CameraImageElement):
                # Camera images are at the very bottom.
                return -2
            # Other elements (like the laser dot) are above the camera but
            # below layers.
            return -1

        self.root.children.sort(key=sort_key)

        self.queue_draw()

    def set_laser_dot_visible(self, visible=True):
        self._laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm: float, y_mm: float):
        """Sets the laser dot position in real-world mm."""
        self._laser_dot_pos_mm = x_mm, y_mm

        # The dot is a child of self.root, so its coordinates are in the
        # world (mm) space. We want to center it on the given mm coords.
        dot_w_mm = self._laser_dot.width
        dot_h_mm = self._laser_dot.height
        self._laser_dot.set_pos(x_mm - dot_w_mm / 2, y_mm - dot_h_mm / 2)

        self.queue_draw()

    def remove_all(self):
        # Clear all children except the fixed ones
        children_to_remove = [
            c
            for c in self.root.children
            if not isinstance(c, (CameraImageElement, DotElement))
        ]
        for child in children_to_remove:
            child.remove()
        self.queue_draw()

    def find_by_type(self, thetype):
        """
        Search recursively through the root's children
        """
        return self.root.find_by_type(thetype)

    def are_workpieces_visible(self) -> bool:
        """Returns True if the workpiece base images should be visible."""
        return self._workpieces_visible

    def set_workpieces_visible(self, visible=True):
        """
        Sets the visibility of the base image for all workpieces. Ops overlays
        remain visible.
        """
        self._workpieces_visible = visible
        # Find the WorkPieceView elements and toggle their base image
        for wp_elem in self.find_by_type(WorkPieceView):
            cast(WorkPieceView, wp_elem).set_base_image_visible(visible)
        self.queue_draw()

    def set_camera_image_visibility(self, visible: bool):
        self._cam_visible = visible
        for elem in self.find_by_type(CameraImageElement):
            camera_elem = cast(CameraImageElement, elem)
            camera_elem.set_visible(visible and camera_elem.camera.enabled)
        self.queue_draw()

    def _on_machine_changed(self, machine: Optional[Machine]):
        """
        Handles incremental updates from the currently-assigned machine model.
        If core properties like dimensions or axis direction change, it
        performs a full view reset. Otherwise, it syncs other properties like
        cameras.
        """
        logger.debug(
            "Machine changed signal received: "
            f"machine={machine.name if machine else 'None'}"
        )
        if not machine:
            return

        # Check for changes that require a full view reset. A change to either
        # dimensions or y-axis orientation invalidates the current pan, zoom,
        # and all calculated coordinates.
        size_changed = machine.dimensions != (self.width_mm, self.height_mm)
        y_axis_changed = machine.y_axis_down != self._axis_renderer.y_axis_down

        if size_changed or y_axis_changed:
            self.reset_view()
        else:
            # No major reset needed, but other properties like the list of
            # cameras might have changed.
            self._sync_camera_elements(machine)

    def reset_view(self):
        """
        Resets the view to fit the given machine's properties, including a
        full reset of pan, zoom, and size. Also syncs camera elements.
        """
        if not self.machine:
            return
        logger.debug(
            f"Resetting view for machine '{self.machine.name}' "
            f"with dims={self.machine.dimensions} and "
            f"y_down={self.machine.y_axis_down}"
        )
        new_dimensions = self.machine.dimensions
        self.set_size(new_dimensions[0], new_dimensions[1])
        self.set_pan(0.0, 0.0)
        self.set_zoom(1.0)
        self._axis_renderer.set_y_axis_down(self.machine.y_axis_down)
        # _recalculate_sizes must be called after other properties are set,
        # especially after y_axis_down is changed, as it affects all
        # coordinate calculations.
        self._rebuild_view_transform()
        new_ratio = (
            new_dimensions[0] / new_dimensions[1]
            if new_dimensions[1] > 0
            else 1.0
        )
        self.aspect_ratio_changed.send(self, ratio=new_ratio)
        self._sync_camera_elements(self.machine)
        self.queue_draw()

    def _sync_camera_elements(self, machine: Machine):
        """Adds, removes, and updates camera elements on the canvas."""
        # Get current camera elements on the canvas
        current_camera_elements = {
            cast(CameraImageElement, elem).camera: elem
            for elem in self.find_by_type(CameraImageElement)
        }
        cameras_on_canvas = set(current_camera_elements.keys())
        cameras_in_model = set(machine.cameras)
        # If there are no changes, do nothing.
        if cameras_on_canvas == cameras_in_model:
            return

        logger.debug("Syncing camera elements.")

        # Add new camera elements
        for camera in cameras_in_model - cameras_on_canvas:
            elem = CameraImageElement(camera)
            elem.set_visible(self._cam_visible and camera.enabled)
            self.root.insert(0, elem)
            logger.debug(f"Added CameraImageElement for camera {camera.name}")

        # Remove camera elements that no longer exist in the machine
        for camera in cameras_on_canvas - cameras_in_model:
            current_camera_elements[camera].remove()
            logger.debug(
                f"Removed CameraImageElement for camera {camera.name}"
            )

    def do_snapshot(self, snapshot):
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
        self, controller, keyval: int, keycode: int, state: Gdk.ModifierType
    ) -> bool:
        """Handles key press events for the work surface."""
        key_name = Gdk.keyval_name(keyval)
        logger.debug(f"Key pressed: key='{key_name}', state={state}")
        if keyval == Gdk.KEY_1:
            # Reset pan and zoom with '1'
            self.reset_view()
            return True  # Event handled

        elif keyval == Gdk.KEY_Escape:
            # If any elements are selected, unselect them.
            if self.get_selected_elements():
                self.unselect_all()
                return True

        # The base class now expects world coordinates, which this is.
        # However, the key events like arrow keys should not be transformed.
        # We need to handle them here directly.

        is_ctrl = bool(state & Gdk.ModifierType.CONTROL_MASK)
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        # Handle moving workpiece to another layer
        if is_ctrl and (
            keyval == Gdk.KEY_Page_Up or keyval == Gdk.KEY_Page_Down
        ):
            direction = -1 if keyval == Gdk.KEY_Page_Up else 1
            self.editor.layer.move_selected_to_adjacent_layer(self, direction)
            return True

        # Handle clipboard and duplication
        if is_ctrl:
            selected_items = [e.data for e in self.get_selected_elements()]
            if keyval == Gdk.KEY_x:
                if selected_items:
                    self.cut_requested.send(self, items=selected_items)
                    return True
            elif keyval == Gdk.KEY_c:
                if selected_items:
                    self.copy_requested.send(self, items=selected_items)
                    return True
            elif keyval == Gdk.KEY_v:
                self.paste_requested.send(self)
                return True
            elif keyval == Gdk.KEY_d:
                if selected_items:
                    self.duplicate_requested.send(self, items=selected_items)
                    return True
            elif keyval == Gdk.KEY_a:
                self.select_all()
                return True

        move_amount_mm = 1.0
        if is_shift:
            move_amount_mm *= 10
        elif is_ctrl:
            move_amount_mm *= 0.1

        move_x, move_y = 0.0, 0.0
        if keyval == Gdk.KEY_Up:
            move_y = move_amount_mm
        elif keyval == Gdk.KEY_Down:
            move_y = -move_amount_mm
        elif keyval == Gdk.KEY_Left:
            move_x = -move_amount_mm
        elif keyval == Gdk.KEY_Right:
            move_x = move_amount_mm

        if move_x != 0 or move_y != 0:
            selected_items = [
                e.data
                for e in self.get_selected_elements()
                if isinstance(e.data, DocItem)
            ]
            if not selected_items:
                return True  # Consume event but do nothing

            self.editor.transform.nudge_items(selected_items, move_x, move_y)
            return True

        # Propagate to parent Canvas for its default behavior (e.g., Shift/
        # Ctrl)
        return super().on_key_pressed(controller, keyval, keycode, state)

    def on_pan_begin(self, gesture, x: float, y: float):
        logger.debug(f"Pan begin at ({x:.2f}, {y:.2f})")
        self._pan_start = (self.pan_x_mm, self.pan_y_mm)

    def on_pan_update(self, gesture, x: float, y: float):
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

    def on_pan_end(self, gesture, x: float, y: float):
        logger.debug(f"Pan end at ({x:.2f}, {y:.2f})")
        pass

    def get_active_workpiece(self) -> Optional[WorkPiece]:
        active_elem = self.get_active_element()
        if active_elem and isinstance(active_elem.data, WorkPiece):
            return active_elem.data
        return None

    def get_selected_workpieces(self) -> List[WorkPiece]:
        all_wps = []
        for elem in self.get_selected_elements():
            # Check for the element's direct data
            if isinstance(elem.data, WorkPiece):
                all_wps.append(elem.data)
            # If it's a group, get all descendant workpieces from the model
            elif isinstance(elem.data, Group):
                all_wps.extend(elem.data.get_descendants(WorkPiece))
        # Return a unique list
        return list(dict.fromkeys(all_wps))

    def get_selected_items(self) -> Sequence[DocItem]:
        return [
            cast(DocItem, elem.data) for elem in self.get_selected_elements()
        ]

    def get_selected_top_level_items(self) -> List[DocItem]:
        """
        Returns a list of the highest-level selected DocItems.

        This follows a simple, robust algorithm:
        1. For each selected item, find its highest selected ancestor.
        2. Collect these ancestors.
        3. Return the unique list of ancestors.

        This correctly handles all cases, including selecting items inside a
        group. If two workpieces inside a group are selected (and not the
        group itself), this method will correctly return just those two
        workpieces. The business logic for what to do with them belongs
        in the calling code.
        """
        selected_elements = self.get_selected_elements()
        if not selected_elements:
            return []

        # Create a set of the data models for efficient lookup.
        selected_item_data = {
            elem.data
            for elem in selected_elements
            if isinstance(elem.data, DocItem)
        }
        if not selected_item_data:
            return []

        top_level_ancestors = []
        for item in selected_item_data:
            # For each item, walk up its hierarchy to find the highest
            # ancestor that is ALSO in the selection set.
            current = item
            highest_selected_ancestor = item
            while current.parent:
                if current.parent in selected_item_data:
                    highest_selected_ancestor = current.parent
                current = current.parent
            top_level_ancestors.append(highest_selected_ancestor)

        # Return a unique list, preserving order.
        return list(dict.fromkeys(top_level_ancestors))

    def select_all(self):
        """
        Selects all workpieces on all layers.
        """
        for elem in self.root.get_all_children_recursive():
            if isinstance(elem.data, DocItem) and elem.selectable:
                elem.selected = True

        self._finalize_selection_state()

    def select_items(self, items_to_select: Sequence[DocItem]):
        """
        Clears the current selection and selects the canvas elements
        corresponding to the given list of DocItem objects.
        """
        self.unselect_all()
        uids_to_select = {item.uid for item in items_to_select}

        for elem in self.root.get_all_children_recursive():
            if (
                isinstance(elem.data, DocItem)
                and elem.data.uid in uids_to_select
                and elem.selectable
            ):
                elem.selected = True

        self._finalize_selection_state()
