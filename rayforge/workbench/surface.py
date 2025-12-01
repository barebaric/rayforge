import logging
from blinker import Signal
from typing import TYPE_CHECKING, Optional, cast, Dict, List, Sequence
from gi.repository import Gdk, Gtk
from ..camera.controller import CameraController
from ..context import get_context
from ..core.group import Group
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.stock import StockItem
from ..core.workpiece import WorkPiece
from ..machine.models.machine import Machine
from .canvas import WorldSurface, CanvasElement
from .elements.stock import StockElement
from .elements.workpiece import WorkPieceElement
from .elements.group import GroupElement
from .elements.camera_image import CameraImageElement
from .elements.layer import LayerElement
from .elements.tab_handle import TabHandleElement
from .elements.dot import DotElement
from . import context_menu
from .sketcher.editor import SketchEditor
from .sketcher.sketchelement import SketchElement

if TYPE_CHECKING:
    from ..doceditor.editor import DocEditor
    from ..workbench.drag_drop_cmd import DragDropCmd

logger = logging.getLogger(__name__)


class WorkSurface(WorldSurface):
    """
    The WorkSurface displays a grid area with WorkPieces and generated Ops
    according to real world dimensions. It is the application-specific
    subclass of the generic WorldSurface.
    """

    def __init__(
        self,
        editor: "DocEditor",
        parent_window: Gtk.Window,
        machine: Optional[Machine],
        cam_visible: bool = False,
        **kwargs,
    ):
        logger.debug("WorkSurface.__init__ called")
        self.editor = editor
        self.doc = self.editor.doc
        self.machine = None  # will be assigned by set_machine() below
        self._show_travel_moves = False
        self._workpieces_visible = True
        width_mm, height_mm = machine.dimensions if machine else (100.0, 100.0)
        y_axis_down = machine.y_axis_down if machine else False
        self._cam_visible = cam_visible
        self._transform_start_states: Dict[CanvasElement, dict] = {}
        self.right_click_context: Optional[Dict] = None

        # Simulation mode state
        self._simulation_mode = False
        self._simulation_overlay: Optional[CanvasElement] = None

        # Initialize the base WorldSurface with machine dimensions
        super().__init__(
            width_mm=width_mm,
            height_mm=height_mm,
            y_axis_down=y_axis_down,
            **kwargs,
        )

        # The SketchEditor manages sketch editing sessions. It is activated
        # when a SketchElement becomes the edit_context.
        self.sketch_editor = SketchEditor(parent_window)

        # DotElement size is in world units (mm) and is dynamically
        # updated to maintain a constant pixel size on screen.
        self._laser_dot_pos_mm = 0.0, 0.0
        self._laser_dot = DotElement(0, 0, 5.0)
        self.root.add(self._laser_dot)

        # Signals for clipboard and duplication operations
        self.cut_requested = Signal()
        self.copy_requested = Signal()
        self.paste_requested = Signal()
        self.duplicate_requested = Signal()
        self.aspect_ratio_changed = Signal()
        self.context_changed = Signal()
        self.transform_initiated = Signal()

        # Signal to request editing a sketch (handled by MainWindow)
        self.edit_sketch_requested = Signal()
        self.edit_stock_item_requested = Signal()

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

        # Drag-drop command will be initialized by MainWindow after
        # construction
        self.drag_drop_cmd: Optional["DragDropCmd"] = None

    @property
    def show_travel_moves(self) -> bool:
        """Returns True if travel moves should be rendered."""
        return self._show_travel_moves

    def set_laser_dot_visible(self, visible: bool = True) -> None:
        self._laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm: float, y_mm: float) -> None:
        """Sets the laser dot position in real-world mm."""
        self._laser_dot_pos_mm = x_mm, y_mm

        # The dot is a child of self.root, so its coordinates are in the
        # world (mm) space. We want to center it on the given mm coords.
        dot_w_mm = self._laser_dot.width
        dot_h_mm = self._laser_dot.height
        self._laser_dot.set_pos(x_mm - dot_w_mm / 2, y_mm - dot_h_mm / 2)

        self.queue_draw()

    def get_global_tab_visibility(self) -> bool:
        """
        Returns the current global visibility state for tab handles. This is
        used by new WorkPieceElements to pull the correct initial state.
        """
        return self._tabs_globally_visible

    def set_global_tab_visibility(self, visible: bool):
        """
        Sets the global visibility for tab handles and propagates the change
        to all existing WorkPieceElements.
        """
        if self._tabs_globally_visible == visible:
            return  # No change
        self._tabs_globally_visible = visible
        # Propagate the new state to all existing views
        for wp_elem in self.find_by_type(WorkPieceElement):
            wp_view = cast(WorkPieceElement, wp_elem)
            wp_view.set_tabs_visible_override(visible)

    def on_right_click_pressed(
        self, gesture: Gtk.GestureClick, n_press: int, x: float, y: float
    ):
        """
        Handles right-clicks. Dispatches to the SketchEditor if a sketch is
        being edited, otherwise shows the standard WorkSurface context menu.
        """
        if isinstance(self.edit_context, SketchElement):
            self.sketch_editor.handle_right_click(gesture, n_press, x, y)
            return

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
            parent_wp_view = cast(WorkPieceElement, hit_elem.parent)
            self.right_click_context = {
                "type": "tab",
                "tab_data": hit_elem.data,
                "workpiece": parent_wp_view.data,
            }
        # Case 2: Clicked on a WorkPieceElement, check for path proximity
        elif isinstance(hit_elem, WorkPieceElement):
            wp_view = cast(WorkPieceElement, hit_elem)

            # Check if this is a sketch workpiece using the new method
            is_sketch = bool(wp_view.data.sketch_uid)

            # Check path proximity
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
                if is_sketch:
                    self.right_click_context = {"type": "sketch-item"}
                else:
                    self.right_click_context = {"type": "item"}
        # Case 3: Clicked on another selectable item (e.g., a Group)
        elif hit_elem.selectable:
            self.right_click_context = {"type": "item"}

        # Notify listeners to update action states *before* showing the menu
        self.context_changed.send(self)

        # Now, call the specific function to show the menu.
        if self.right_click_context:
            context_type = self.right_click_context["type"]
            if context_type == "item":
                if not hit_elem.selected:
                    self.unselect_all()
                    hit_elem.selected = True
                    self._finalize_selection_state()
                context_menu.show_item_context_menu(self, gesture)
            elif context_type == "sketch-item":
                if not hit_elem.selected:
                    self.unselect_all()
                    hit_elem.selected = True
                    self._finalize_selection_state()
                context_menu.show_sketch_item_context_menu(self, gesture)
            elif context_type == "geometry":
                context_menu.show_geometry_context_menu(self, gesture)
            elif context_type == "tab":
                context_menu.show_tab_context_menu(self, gesture)

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
        self.transform_initiated.send(self)
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
            f"Resize begin for {len(elements)} element(s). Pausing pipeline."
        )
        # Call the generic transform begin handler.
        # Note: resize_begin signal in canvas.py currently doesn't send
        # drag_target, so this call will pass None for drag_target in
        # _on_any_transform_begin, which is correct.
        self._on_any_transform_begin(sender, elements)
        self.editor.pipeline.pause()

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

        # If it was a resize, the ops are now stale. Resume the pipeline.
        if self._resizing:
            self.editor.pipeline.resume()

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        """
        Overrides base to add application-specific layer selection logic,
        handle double-click editing, and manage the SketchEditor lifecycle.
        """
        # A left-click should clear any lingering right-click context.
        if gesture.get_button() == Gdk.BUTTON_PRIMARY:
            if self.right_click_context:
                self.right_click_context = None
                self.context_changed.send(self)

        logger.debug(
            f"Button press: n_press={n_press}, pos=({x:.2f}, {y:.2f})"
        )

        old_context = self.edit_context
        # The base class method handles hit testing and updates
        # self.edit_context
        super().on_button_press(gesture, n_press, x, y)
        new_context = self.edit_context

        # Check for double-click to edit sketches.
        if n_press == 2:
            world_x, world_y = self._get_world_coords(x, y)
            hit_elem = self.root.get_elem_hit(
                world_x, world_y, selectable=True
            )

            if isinstance(hit_elem, WorkPieceElement):
                wp = hit_elem.data
                # The new, correct logic:
                if wp.sketch_uid:
                    self.edit_sketch_requested.send(self, workpiece=wp)
                    return
            elif isinstance(hit_elem, StockElement):
                stock_item = cast(StockItem, hit_elem.data)
                self.edit_stock_item_requested.send(
                    self, stock_item=stock_item
                )
                return

        # Manage SketchEditor activation based on context changes.
        if old_context is not new_context:
            if isinstance(old_context, SketchElement):
                self.sketch_editor.deactivate()
            if isinstance(new_context, SketchElement):
                self.sketch_editor.activate(new_context)

        # After the click, check if the active element dictates a layer change.
        if new_context and isinstance(new_context.data, WorkPiece):
            active_layer = new_context.data.layer
            # If the workpiece's layer is not the document's active layer,
            # create an undoable command to change it.
            if active_layer and active_layer != self.doc.active_layer:
                self.editor.layer.set_active_layer(active_layer)

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

        # Synchronize camera elements to match the new machine. This is called
        # after the machine is set (or cleared) to ensure the view is correct.
        self._sync_camera_elements()

    def _rebuild_view_transform(self):
        """
        Constructs the world-to-view transformation matrix.
        This override propagates view scale changes to WorkPieceElements.
        """
        # Get old scale before rebuilding
        old_scale_x, old_scale_y = self.get_view_scale()

        # Let the base class do the actual transform calculation
        super()._rebuild_view_transform()

        # Check if the effective scale (pixels-per-mm) has changed. Panning
        # does not change the scale, but zooming and resizing the window do.
        # This prevents expensive re-rendering of buffered elements during
        # panning.
        new_scale_x, new_scale_y = self.get_view_scale()
        scale_changed = (
            abs(new_scale_x - old_scale_x) > 1e-9
            or abs(new_scale_y - old_scale_y) > 1e-9
        )

        if scale_changed:
            # Update laser dot size to maintain a constant size in pixels.
            desired_diameter_px = 3.0
            if new_scale_x > 1e-9:
                diameter_mm = desired_diameter_px / new_scale_x
                self._laser_dot.set_size(diameter_mm, diameter_mm)

            # Propagate the view change to elements that depend on it.
            for elem in self.find_by_type(WorkPieceElement):
                wp_view = cast(WorkPieceElement, elem)
                wp_view.trigger_view_update()
                wp_view.update_handle_transforms()

        # Reposition the laser dot after any view change
        self.set_laser_dot_position(
            self._laser_dot_pos_mm[0], self._laser_dot_pos_mm[1]
        )

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers re-rendering."""
        if self._show_travel_moves != show:
            self._show_travel_moves = show
            # Re-render all ops surfaces on all workpiece views
            for elem in self.find_by_type(WorkPieceElement):
                wp_view = cast(WorkPieceElement, elem)
                wp_view.on_travel_visibility_changed()

    def _create_and_add_layer_element(self, layer: "Layer"):
        """Creates a new LayerElement and adds it to the canvas root."""
        logger.debug(f"Adding new LayerElement for '{layer.name}'")
        layer_elem = LayerElement(layer=layer, canvas=self)
        self.root.add(layer_elem)

    def _create_and_add_stock_element(self, stock_item: StockItem):
        """Creates a new StockElement and adds it to the canvas root."""
        logger.debug(f"Adding new StockElement for '{stock_item.name}'")
        stock_elem = StockElement(stock_item=stock_item, canvas=self)
        stock_elem.selectable = stock_elem.visible
        self.root.add(stock_elem)
        child_count = len(self.root.children)
        logger.debug(f"StockElement added, total children: {child_count}")
        # Trigger a redraw to show the new stock element
        self.queue_draw()

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

        # --- Step 1.5: Add and Remove StockElements ---
        doc_stock_items_set = set(doc.stock_items)
        current_stock_elements_on_canvas = {
            elem.data: elem for elem in self.find_by_type(StockElement)
        }

        # Remove elements for stock items that are no longer in the doc
        for stock_item, elem in current_stock_elements_on_canvas.items():
            if stock_item not in doc_stock_items_set:
                elem.remove()

        # Add elements for new stock items that are not yet on the canvas
        for stock_item in doc.stock_items:
            if stock_item not in current_stock_elements_on_canvas:
                self._create_and_add_stock_element(stock_item)

        # --- Step 2: Reorder LayerElements for Z-stacking ---
        # The first layer in the list is at the bottom (drawn first).
        # The last layer is at the top (drawn last).
        layer_order_map = {layer: i for i, layer in enumerate(doc.layers)}

        def sort_key(element: CanvasElement):
            """
            Sort key for root's children. Camera at bottom, then stock,
            then dot, then layers.
            """
            if isinstance(element, LayerElement):
                # LayerElements are ordered according to the doc.layers list.
                # Add a large offset to ensure all layers are above stock
                layer_order = layer_order_map.get(
                    element.data, len(layer_order_map)
                )
                return layer_order + 1000
            if isinstance(element, StockElement):
                # Stock elements are below all layers but above camera images
                return 10
            if isinstance(element, CameraImageElement):
                # Camera images are at the very bottom.
                return -2
            # Other elements (like the laser dot) are above the camera but
            # below stock and layers.
            return -1

        self.root.children.sort(key=sort_key)

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
        # Find the WorkPieceElements and toggle their base image
        for wp_elem in self.find_by_type(WorkPieceElement):
            cast(WorkPieceElement, wp_elem).set_base_image_visible(visible)
        self.queue_draw()

    def set_camera_controllers(self, controllers: List[CameraController]):
        """
        Manages camera elements and their subscriptions based on the
        provided list of live controllers.
        """
        current_elements = {
            cast(CameraImageElement, e).controller: e
            for e in self.find_by_type(CameraImageElement)
        }
        current_controllers = set(current_elements.keys())
        new_controllers = set(controllers)

        # Remove elements for controllers that are no longer active
        for controller in current_controllers - new_controllers:
            element = current_elements[controller]
            element.remove()  # This will disconnect signals
            controller.unsubscribe()
            logger.debug(
                f"Unsubscribed and removed element for camera "
                f"{controller.config.name}"
            )

        # Add elements for new controllers
        for controller in new_controllers - current_controllers:
            element = CameraImageElement(controller)
            element.set_visible(
                self._cam_visible and controller.config.enabled
            )
            self.root.insert(0, element)  # Insert at the bottom of the z-stack
            controller.subscribe()
            logger.debug(
                f"Subscribed and added element for camera "
                f"{controller.config.name}"
            )

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
            # Machine was likely removed or changed to None, clear cameras
            self._sync_camera_elements()
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
            self._sync_camera_elements()

    def reset_view(self):
        """
        Resets the view to fit the given machine's properties, including a
        full reset of pan, zoom, and size.
        """
        if not self.machine:
            # If no machine, reset to a default view
            self.set_size(100.0, 100.0)
            self._axis_renderer.set_y_axis_down(False)
            super().reset_view()
            self.aspect_ratio_changed.send(self, ratio=1.0)
            self._sync_camera_elements()
            return

        logger.debug(
            f"Resetting view for machine '{self.machine.name}' "
            f"with dims={self.machine.dimensions} and "
            f"y_down={self.machine.y_axis_down}"
        )
        new_dimensions = self.machine.dimensions
        self.set_size(new_dimensions[0], new_dimensions[1])
        self._axis_renderer.set_y_axis_down(self.machine.y_axis_down)
        # Call the base class reset which handles pan/zoom
        super().reset_view()
        new_ratio = (
            new_dimensions[0] / new_dimensions[1]
            if new_dimensions[1] > 0
            else 1.0
        )
        self.aspect_ratio_changed.send(self, ratio=new_ratio)
        self._sync_camera_elements()

    def _sync_camera_elements(self):
        """
        Synchronizes the camera elements on the canvas with the cameras
        defined in the current machine model.
        """
        camera_mgr = get_context().camera_mgr
        if not self.machine:
            self.set_camera_controllers([])
            return

        # Get the controller for each camera model in the current machine
        machine_camera_controllers = []
        for camera_model in self.machine.cameras:
            controller = camera_mgr.get_controller(camera_model.device_id)
            if controller:
                machine_camera_controllers.append(controller)
            else:
                logger.warning(
                    "Could not find a live controller for camera "
                    f"with device ID '{camera_model.device_id}'."
                )

        self.set_camera_controllers(machine_camera_controllers)

    def on_key_pressed(
        self,
        controller: Gtk.EventControllerKey,
        keyval: int,
        keycode: int,
        state: Gdk.ModifierType,
    ) -> bool:
        """Handles key press events for the work surface."""
        # First, dispatch to sketch editor if a sketch is active
        if isinstance(self.edit_context, SketchElement):
            if self.sketch_editor.handle_key_press(keyval, keycode, state):
                return True  # Event handled by sketch editor

        # Let the base WorldSurface class handle generic keys (e.g., '1')
        if super().on_key_pressed(controller, keyval, keycode, state):
            return True

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

            self.transform_initiated.send(self)
            self.editor.transform.nudge_items(selected_items, move_x, move_y)
            return True

        return False

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
            elem.data
            for elem in self.get_selected_elements()
            if isinstance(elem.data, DocItem)
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

    def is_simulation_mode(self) -> bool:
        """Returns True if simulation mode is active."""
        return self._simulation_mode

    def set_simulation_mode(
        self, enabled: bool, simulation_overlay: Optional[CanvasElement] = None
    ):
        """
        Enables or disables simulation mode. When enabled:
        - Workpiece selection and transformation remain enabled
        - Zoom and pan gestures remain active
        - Grid and axis render normally
        - Simulation overlay is shown on top
        """
        if self._simulation_mode == enabled:
            return

        self._simulation_mode = enabled

        if enabled:
            # Add simulation overlay if provided
            if simulation_overlay:
                self._simulation_overlay = simulation_overlay
                self.root.add(self._simulation_overlay)
        else:
            # Remove simulation overlay when exiting
            if self._simulation_overlay:
                self._simulation_overlay.remove()
                self._simulation_overlay = None

        self.queue_draw()
