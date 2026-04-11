import logging
from blinker import Signal
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    cast,
    Dict,
    List,
    Sequence,
    Tuple,
)
from gi.repository import Gdk, GLib, Gtk, Graphene
from ...camera.controller import CameraController
from ...context import get_context
from ...core.group import Group
from ...core.item import DocItem
from ...core.layer import Layer
from ...core.stock import StockItem
from ...core.stock_asset import StockAsset
from ...core.workpiece import WorkPiece
from ...machine.models.colors import OpsColorSet
from ...machine.models.machine import Machine
from ...pipeline.artifact import RenderContext
from ...core.color import (
    ColorRGBA,
    ColorSet,
    hex_to_rgba,
    create_lut_from_color,
)
from ..canvas import WorldSurface, Canvas, CanvasElement
from ..shared.gtk_color import GtkColorResolver
from ..shared.keyboard import is_primary_modifier
from .elements.axis_extent_frame import (
    AxisExtentFrameElement,
    WorkareaBackgroundElement,
)
from .elements.nogo_zone import NogoZoneElement
from .elements.camera_image import CameraImageElement
from .elements.dot import DotElement
from .elements.group import GroupElement
from .elements.layer import LayerElement
from .elements.rotary_surface import RotarySurfaceElement
from .elements.stock import StockElement
from .elements.tab_handle import TabHandleElement
from .elements.workpiece import WorkPieceElement
from .elements.work_origin import WorkOriginElement
from .projection import CanvasProjection
from . import context_menu

if TYPE_CHECKING:
    from ...doceditor.editor import DocEditor
    from .drag_drop_cmd import DragDropCmd

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
        self.machine = None  # will be assigned by set_machine() below
        self._show_travel_moves = False
        self._workpieces_visible = True
        self._tracked_axis_extents: Tuple[float, float] = (0.0, 0.0)
        coordinate_space = None
        if machine:
            self._tracked_axis_extents = machine.axis_extents
            # Canvas shows full machine bed, not just workarea
            width_mm, height_mm = (
                float(machine.axis_extents[0]),
                float(machine.axis_extents[1]),
            )
            coordinate_space = machine.get_coordinate_space()
        else:
            width_mm, height_mm = 100.0, 100.0

        self._cam_visible = cam_visible
        self._transform_start_states: Dict[CanvasElement, dict] = {}
        self.right_click_context: Optional[Dict] = None

        # Click-to-zero mode state
        self._click_to_zero_mode = False

        # Ops rendering suppression for lazy ops rendering (Idea 5).
        # During pan/zoom/drag, ops drawing and pipeline context updates
        # are suppressed. They are restored after ~200ms of idle time.
        self._ops_suppressed: bool = False
        self._ops_restore_timer_id: Optional[int] = None

        self._nogo_zone_elements: Dict[str, NogoZoneElement] = {}
        self._nogo_zones_visible = True
        self._projection = CanvasProjection()

        # Initialize the base WorldSurface with machine dimensions
        super().__init__(
            width_mm=width_mm,
            height_mm=height_mm,
            coordinate_space=coordinate_space,
            **kwargs,
        )

        # Prevent GTK from implicitly grabbing focus on click, which can
        # interfere with popover/menu closing logic.
        # We'll manage focus manually.
        self.set_focus_on_click(False)

        # DotElement size is in world units (mm) and is dynamically
        # updated to maintain a constant pixel size on screen.
        self._laser_dot_pos_mm = 0.0, 0.0
        self._laser_dot = DotElement(0, 0, 1.0)
        self.root.add(self._laser_dot)

        # Add the Work Origin visual element
        self._work_origin_element = WorkOriginElement()
        self.root.add(self._work_origin_element)

        # Add the Workarea Background element (gray background for workarea)
        self._workarea_bg_element = WorkareaBackgroundElement()
        self._workarea_bg_element.set_visible(False)
        self.root.add(self._workarea_bg_element)
        # Move to back so it's behind everything
        self.root.children.insert(0, self.root.children.pop())

        # Clear root background since we draw workarea background separately
        self.root.background = (0, 0, 0, 0)

        # Add the Axis Extent Frame element (red frame around machine bed)
        self._extent_frame_element = AxisExtentFrameElement()
        self._extent_frame_element.set_visible(False)
        self.root.add(self._extent_frame_element)

        # Add the Rotary Diameter element (dashed rectangle for cylinder)
        self._rotary_surface_element = RotarySurfaceElement()
        self._rotary_surface_element.set_visible(False)
        self.root.add(self._rotary_surface_element)

        # Signals for clipboard and duplication operations
        self.cut_requested = Signal()
        self.copy_requested = Signal()
        self.paste_requested = Signal()
        self.duplicate_requested = Signal()
        self.aspect_ratio_changed = Signal()
        self.context_changed = Signal()
        self.transform_initiated = Signal()

        # Signal to request editing an item (handled by MainWindow)
        # Sends: (item, action_name)
        self.edit_item_requested = Signal()

        # Signal to set work zero at clicked position
        self.work_zero_requested = Signal()

        # Signal to cancel click-to-zero mode
        self.click_to_zero_cancelled = Signal()

        # Signal for context menu extension - addons can connect to add items
        # Sends: (item, gesture, menu)
        self.context_menu_requested = Signal()

        # Signal emitted when an asset is dropped onto the canvas
        # Sends: (uid, position_mm) where position_mm is (x, y) in world coords
        self.item_dropped = Signal()

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

        self.editor.pipeline.data_stale.connect(self._on_pipeline_data_stale)

        # Connect to view change signals to update pipeline view context
        self.aspect_ratio_changed.connect(self._on_aspect_ratio_changed)

        # Refresh render context when doc structure changes (layer
        # add/remove, workpiece moves between layers) so the view
        # manager always has current layer color data.
        self._connect_doc_structure_signals(self.doc)

        # Reconnect signals when a new document is loaded.
        self.editor.document_changed.connect(self._on_document_changed)

        # --- View State Management ---
        # This property holds the canonical global state for tab visibility.
        self._tabs_globally_visible: bool = True

        # Drag-drop command will be initialized by MainWindow after
        # construction
        self.drag_drop_cmd: Optional["DragDropCmd"] = None

        # Initialize pipeline view context to ensure workpiece artifacts
        # can be rendered immediately when adopted
        self._update_pipeline_view_context()

        get_context().config.changed.connect(self._on_config_changed)

    @property
    def doc(self):
        """Returns the current document from the editor."""
        return self.editor.doc

    @property
    def projection(self) -> CanvasProjection:
        return self._projection

    @projection.setter
    def projection(self, value: CanvasProjection):
        if self._projection != value:
            self._projection = value
            self._update_rotary_surface_element()
            self.queue_draw()

    @property
    def show_travel_moves(self) -> bool:
        """Returns True if travel moves should be rendered."""
        return self._show_travel_moves

    @property
    def ops_suppressed(self) -> bool:
        """Returns True if ops rendering is suppressed during interaction."""
        return self._ops_suppressed

    def _suppress_ops(self):
        """Suppresses ops rendering during pan/zoom/drag interactions."""
        if not self._ops_suppressed:
            logger.debug("Suppressing ops rendering during interaction")
            self._ops_suppressed = True
        if self._ops_restore_timer_id is not None:
            GLib.source_remove(self._ops_restore_timer_id)
            self._ops_restore_timer_id = None

    def _defer_restore_ops(self, delay_ms: int = 200):
        """Schedules deferred ops restoration after an idle period."""
        if self._ops_restore_timer_id is not None:
            GLib.source_remove(self._ops_restore_timer_id)
        self._ops_restore_timer_id = GLib.timeout_add(
            delay_ms, self._restore_ops
        )

    def _restore_ops(self) -> bool:
        """Restores ops rendering after interaction ends."""
        self._ops_restore_timer_id = None
        logger.debug("Restoring ops rendering after idle")
        self._ops_suppressed = False

        self._update_pipeline_view_context()

        ppm_x, _ = self.get_view_scale()
        for elem in self.find_by_type(WorkPieceElement):
            wp_view = cast(WorkPieceElement, elem)
            wp_view.trigger_view_update(ppm_x)

        self.queue_draw()
        return False

    def set_laser_dot_visible(self, visible: bool = True) -> None:
        self._laser_dot.set_visible(visible)
        self.queue_draw()

    def set_laser_dot_position(self, x_mm: float, y_mm: float) -> None:
        """Sets the laser dot position in real-world mm."""
        self._laser_dot_pos_mm = x_mm, y_mm

        # Transform machine coordinates to canvas coordinates (similar to
        # Work Origin logic)
        canvas_x, canvas_y = self._machine_coords_to_canvas(x_mm, y_mm)

        # The dot is a child of self.root, so its coordinates are in the
        # world (mm) space. We want to center it on the given mm coords.
        dot_w_mm = self._laser_dot.width
        dot_h_mm = self._laser_dot.height
        self._laser_dot.set_pos(
            canvas_x - dot_w_mm / 2, canvas_y - dot_h_mm / 2
        )

        self.queue_draw()

    def _machine_coords_to_canvas(
        self, m_x: float, m_y: float
    ) -> tuple[float, float]:
        """
        Convert machine-reported coordinates to canvas world coordinates.

        Machine-reported coordinates come from the controller and may be
        negated based on reverse_x/reverse_y settings. We undo this sign
        flip before transforming to world coordinates.
        """
        if self.machine:
            space = self.machine.get_coordinate_space()
            if space.reverse_x:
                m_x = -m_x
            if space.reverse_y:
                m_y = -m_y
            return space.transform_point_to_world(m_x, m_y, space.extents)
        return m_x, m_y

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
        Handles right-clicks. Shows the standard WorkSurface context menu.
        """
        if self._click_to_zero_mode and n_press == 1:
            self.click_to_zero_cancelled.send(self)
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
                context_menu.show_item_context_menu(
                    self, gesture, item=hit_elem.data
                )
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

    def _on_pipeline_data_stale(self, sender, **kwargs):
        """Clears ops overlays when the pipeline is in manual mode."""
        if self.editor.pipeline.auto_pipeline:
            return
        for elem in self.find_by_type(WorkPieceElement):
            cast(WorkPieceElement, elem).clear_all_ops_caches()
        self.queue_draw()

    def _on_config_changed(self, sender, **kwargs):
        """Re-renders ops when config settings change."""
        self._update_pipeline_view_context()

    def _on_doc_structure_changed(self, sender, **kwargs):
        """Refreshes render context when layers are added/removed."""
        logger.debug(f"_on_doc_structure_changed fired: sender={sender}")
        self._update_pipeline_view_context()

    def _on_document_changed(self, sender, **kwargs):
        """Reconnects doc structure signals when a new doc is loaded."""
        doc = self.doc
        if doc:
            self._connect_doc_structure_signals(doc)
        self._update_pipeline_view_context()

    def _connect_doc_structure_signals(self, doc):
        doc.descendant_added.connect(self._on_doc_structure_changed)
        doc.descendant_removed.connect(self._on_doc_structure_changed)

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
        self._suppress_ops()
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

        self._defer_restore_ops()

    def on_button_press(self, gesture, n_press: int, x: float, y: float):
        """
        Overrides base to add application-specific layer selection logic
        and handle double-click editing.
        """
        logger.debug("WorkSurface.on_button_press fired")

        # Handle click-to-zero mode
        if self._click_to_zero_mode:
            if gesture.get_button() == Gdk.BUTTON_PRIMARY and n_press == 1:
                world_x, world_y = self._get_world_coords(x, y)
                if self.machine:
                    space = self.machine.get_coordinate_space()
                    machine_x, machine_y = space.world_point_to_machine(
                        world_x, world_y
                    )
                else:
                    machine_x, machine_y = world_x, world_y
                self.work_zero_requested.send(self, x=machine_x, y=machine_y)
                return

        # A left-click should clear any lingering right-click context.
        if gesture.get_button() == Gdk.BUTTON_PRIMARY:
            if self.right_click_context:
                self.right_click_context = None
                self.context_changed.send(self)

        logger.debug(
            f"Button press: n_press={n_press}, pos=({x:.2f}, {y:.2f})"
        )

        # The base class method handles hit testing and updates
        # self.edit_context
        super().on_button_press(gesture, n_press, x, y)
        new_context = self.edit_context

        # Check for double-click to edit items.
        if n_press == 2:
            world_x, world_y = self._get_world_coords(x, y)
            hit_elem = self.root.get_elem_hit(
                world_x, world_y, selectable=True
            )

            if isinstance(hit_elem, WorkPieceElement):
                wp = hit_elem.data
                if wp.geometry_provider_uid:
                    asset = self.doc.get_asset_by_uid(wp.geometry_provider_uid)
                    if asset:
                        asset_cls = type(asset)
                        action_name = asset_cls.edit_item_action
                        if action_name:
                            self.edit_item_requested.send(
                                self, item=wp, action_name=action_name
                            )
                            return
            elif isinstance(hit_elem, StockElement):
                stock_item = cast(StockItem, hit_elem.data)
                action_name = StockAsset.edit_item_action
                if action_name:
                    self.edit_item_requested.send(
                        self, item=stock_item, action_name=action_name
                    )
                    return

        # After the click, check if the active element dictates a layer change.
        if new_context and isinstance(new_context.data, WorkPiece):
            active_layer = new_context.data.layer
            # If the workpiece's layer is not the document's active layer,
            # create an undoable command to change it.
            if active_layer and active_layer != self.doc.active_layer:
                self.editor.layer.set_active_layer(active_layer)

    def on_motion(self, gesture: Gtk.Gesture, x: float, y: float) -> None:
        if self._click_to_zero_mode:
            self.set_cursor(Gdk.Cursor.new_from_name("crosshair"))
            return
        super().on_motion(gesture, x, y)

    def on_motion_leave(self, controller: Gtk.EventControllerMotion) -> None:
        if self._click_to_zero_mode:
            self.set_cursor(None)
        super().on_motion_leave(controller)

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
            self.machine.wcs_updated.disconnect(self._on_wcs_updated)
            self.machine.state_changed.disconnect(
                self._on_machine_state_changed
            )

        # Update the machine reference
        self.machine = machine

        # Connect to the new machine's signals
        if self.machine:
            self.machine.changed.connect(self._on_machine_changed)
            self.machine.wcs_updated.connect(self._on_wcs_updated)
            self.machine.state_changed.connect(self._on_machine_state_changed)
            self.reset_view()
            self._on_wcs_updated(self.machine)

        # Synchronize camera elements to match the new machine. This is called
        # after the machine is set (or cleared) to ensure the view is correct.
        self._sync_camera_elements()

    def _on_wcs_updated(self, machine: Machine):
        """Handles updates to the machine's WCS state."""
        space = machine.get_coordinate_space()

        if machine.wcs_origin_is_workarea_origin:
            canvas_x, canvas_y = space.get_workarea_origin_in_machine()
        else:
            wcs_x, wcs_y, _ = machine.get_active_wcs_offset()
            canvas_x, canvas_y = self._machine_coords_to_canvas(wcs_x, wcs_y)

        self._work_origin_element.set_pos(canvas_x, canvas_y)
        self._work_origin_element.set_visible(True)
        self._update_rotary_surface_element()
        self.queue_draw()

    def _on_machine_state_changed(self, machine: Machine, state):
        """Handles machine state changes including position updates."""
        m_pos = state.machine_pos
        if m_pos and all(p is not None for p in m_pos):
            m_x, m_y, _ = m_pos
            self.set_laser_dot_position(m_x, m_y)

    def do_snapshot(self, snapshot: Gtk.Snapshot) -> None:
        self._update_theme_colors()

        width, height = self.get_width(), self.get_height()
        ctx = snapshot.append_cairo(Graphene.Rect().init(0, 0, width, height))

        # Get offset for axis labels (where 0,0 should appear)
        if self.machine:
            space = self.machine.get_coordinate_space()
            wcs_offset = self.machine.get_active_wcs_offset()
            wcs_is_workarea = self.machine.wcs_origin_is_workarea_origin
            origin_offset_mm = space.get_axis_label_origin(
                wcs_offset=wcs_offset,
                wcs_is_workarea_origin=wcs_is_workarea,
            )
        else:
            origin_offset_mm = (0.0, 0.0, 0.0)

        self._axis_renderer.draw_grid_and_labels(
            ctx,
            self.view_transform,
            width,
            height,
            origin_offset_mm=origin_offset_mm,
        )

        Canvas.do_snapshot(self, snapshot)

    def _rebuild_view_transform(self) -> bool:
        """
        Constructs the world-to-view transformation matrix.
        This override propagates view scale changes to WorkPieceElements and
        updates the laser dot to maintain a constant pixel size.
        """
        # Let the base class do the actual transform calculation and tell us
        # if the scale changed.
        scale_changed = super()._rebuild_view_transform()

        logger.debug(
            f"_rebuild_view_transform: scale_changed={scale_changed}, "
            f"view_scale={self.get_view_scale()}"
        )

        if scale_changed:
            # Update laser dot size to maintain a constant size in pixels.
            desired_diameter_px = 6.0
            new_scale_x, _ = self.get_view_scale()
            if new_scale_x > 1e-9:
                diameter_mm = desired_diameter_px / new_scale_x
                self._laser_dot.set_size(diameter_mm, diameter_mm)

            # Skip pipeline context updates and ops re-rendering during
            # interaction. They will be restored after idle via
            # _restore_ops().
            if not self._ops_suppressed:
                logger.debug(
                    "_rebuild_view_transform: Calling "
                    "_update_pipeline_view_context"
                )
                self._update_pipeline_view_context()

                logger.debug(
                    "_rebuild_view_transform: Updating handle transforms "
                    f"for {len(list(self.find_by_type(WorkPieceElement)))} "
                    "WorkPieceElements"
                )
                ppm_x, _ = self.get_view_scale()
                for elem in self.find_by_type(WorkPieceElement):
                    wp_view = cast(WorkPieceElement, elem)
                    wp_view.trigger_view_update(ppm_x)
                    wp_view.update_handle_transforms()
            else:
                logger.debug(
                    "_rebuild_view_transform: ops suppressed, "
                    "skipping pipeline context update"
                )
                for elem in self.find_by_type(WorkPieceElement):
                    cast(WorkPieceElement, elem).update_handle_transforms()

        # Reposition the laser dot after any view change
        self.set_laser_dot_position(
            self._laser_dot_pos_mm[0], self._laser_dot_pos_mm[1]
        )

        return scale_changed

    def on_pan_begin(
        self, gesture: Gtk.GestureDrag, x: float, y: float
    ) -> None:
        self._suppress_ops()
        super().on_pan_begin(gesture, x, y)

    def on_pan_end(self, gesture: Gtk.GestureDrag, x: float, y: float) -> None:
        super().on_pan_end(gesture, x, y)
        self._defer_restore_ops()

    def on_scroll(
        self,
        controller: Gtk.EventControllerScroll,
        dx: float,
        dy: float,
    ) -> None:
        self._suppress_ops()
        super().on_scroll(controller, dx, dy)
        self._defer_restore_ops()

    def set_show_travel_moves(self, show: bool):
        """Sets whether to display travel moves and triggers re-rendering."""
        if self._show_travel_moves != show:
            self._show_travel_moves = show
            # Re-render all ops surfaces on all workpiece views
            for elem in self.find_by_type(WorkPieceElement):
                wp_view = cast(WorkPieceElement, elem)
                wp_view.on_travel_visibility_changed()
            # Update pipeline view context
            self._update_pipeline_view_context()

    def set_click_to_zero_mode(self, active: bool):
        """Sets whether click-to-zero mode is active."""
        self._click_to_zero_mode = active
        if not active:
            self.set_cursor(None)

    def _update_pipeline_view_context(self) -> None:
        """
        Updates the view manager context with the current view settings.

        This method collects all the current view context information
        (pixels per mm, show travel moves, theme colors) and calls
        the view_manager's update_render_context method to trigger
        re-rendering of all cached workpiece views.
        """
        if not self.editor or not self.editor.view_manager:
            logger.debug(
                "_update_pipeline_view_context: No editor or view_manager"
            )
            return

        ppm_x, ppm_y = self.get_view_scale()

        logger.debug(
            f"_update_pipeline_view_context: ppm=({ppm_x:.2f}, "
            f"{ppm_y:.2f}), show_travel_moves={self._show_travel_moves}"
        )

        if ppm_x <= 1e-9 or ppm_y <= 1e-9:
            logger.debug(
                "_update_pipeline_view_context: Scale too small, skipping"
            )
            return

        color_set_dict = self._get_theme_color_dict()
        laser_color_dicts = self._get_laser_color_dicts()
        layer_color_dicts = self._get_layer_color_dicts()
        ops_color_mode = get_context().config.ops_color_mode

        context = RenderContext(
            pixels_per_mm=(ppm_x, ppm_y),
            show_travel_moves=self._show_travel_moves,
            margin_px=5,
            color_set_dict=color_set_dict.to_dict(),
            laser_color_sets=laser_color_dicts,
            layer_color_sets=layer_color_dicts,
            ops_color_mode=ops_color_mode,
        )

        self.editor.view_manager.update_render_context(context)

    def _get_laser_color_dicts(self) -> Dict[str, Dict[str, Any]]:
        """
        Resolve colors for each laser in the machine.

        Returns a dictionary mapping laser UID to its color set dictionary.
        """
        if not self.machine:
            return {}

        theme_colors = self._get_theme_color_dict()
        laser_colors: Dict[str, Dict[str, Any]] = {}

        for laser in self.machine.heads:
            laser_color_set = OpsColorSet.from_laser(laser, theme_colors)
            laser_colors[laser.uid] = laser_color_set.to_color_set().to_dict()

        return laser_colors

    def _get_layer_color_dicts(self) -> Dict[str, Dict[str, Any]]:
        """
        Resolve colors for each layer in the document.

        Returns a dictionary mapping layer UID to its color set dictionary.
        """
        doc = self.doc
        if not doc:
            return {}

        theme_colors = self._get_theme_color_dict()
        layer_colors: Dict[str, Dict[str, Any]] = {}

        for layer in doc.layers:
            cut_rgba = hex_to_rgba(layer.color)
            cut_lut = create_lut_from_color(cut_rgba)
            data = {
                "cut": cut_lut,
                "engrave": cut_lut,
                "travel": theme_colors.get_rgba("travel"),
                "zero_power": theme_colors.get_rgba("zero_power"),
            }
            color_set = ColorSet(_data=data)
            layer_colors[layer.uid] = color_set.to_dict()

        return layer_colors

    def _get_theme_color_dict(self) -> ColorSet:
        """
        Gets the current theme color set as a dictionary.

        This method extracts the color set from the current GTK style
        and converts it to a dictionary format suitable for the pipeline.
        """
        color_resolver = GtkColorResolver(self)

        spec_dict = {
            "cut": ("#ffeeff", "#ff00ff"),
            "engrave": ("#FFFFFF", "#000000"),
            "travel": ("#FF6600", 0.7),
            "zero_power": ("@accent_color", 0.5),
        }

        return color_resolver.resolve(spec_dict)

    def _get_handle_color(self, elem: CanvasElement) -> Optional[ColorRGBA]:
        """Returns the layer color for the element's selection handles."""
        data = getattr(elem, "data", None)
        if data is None:
            return None
        layer = getattr(data, "layer", None)
        if layer is None:
            return None
        return hex_to_rgba(layer.color)

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
            then layers and laser dot on top.
            """
            if isinstance(element, DotElement):
                return float("inf") - 1
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
            if isinstance(element, (CameraImageElement, WorkOriginElement)):
                # Camera images and WCS origin are at the very bottom.
                return -2
            if isinstance(element, AxisExtentFrameElement):
                # Extent frame is above camera but below everything else
                return -1.5
            if isinstance(element, NogoZoneElement):
                return -1.4
            if isinstance(element, RotarySurfaceElement):
                # Rotary surface is above extent frame but below stock
                return -1.25
            # Other elements are above the camera but below stock and layers.
            return -1

        self.root.children.sort(key=sort_key)

        self._update_rotary_surface_element()
        self.queue_draw()

    def remove_all(self):
        # Clear all children except the fixed ones
        children_to_remove = [
            c
            for c in self.root.children
            if not isinstance(
                c,
                (
                    CameraImageElement,
                    DotElement,
                    NogoZoneElement,
                    WorkOriginElement,
                    AxisExtentFrameElement,
                ),
            )
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

    def set_show_nogo_zones(self, visible: bool):
        self._nogo_zones_visible = visible
        for elem in self._nogo_zone_elements.values():
            elem.set_visible(visible and elem.data.enabled)
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
        """
        logger.debug(
            "Machine changed signal received: "
            f"machine={machine.name if machine else 'None'}"
        )
        if not machine:
            self._sync_camera_elements()
            return

        extent_w, extent_h = machine.axis_extents
        extent_changed = (extent_w, extent_h) != self._tracked_axis_extents
        y_axis_changed = machine.y_axis_down != self._axis_renderer.y_axis_down
        x_axis_changed = (
            machine.x_axis_right != self._axis_renderer.x_axis_right
        )
        x_reverse_changed = (
            machine.reverse_x_axis != self._axis_renderer.x_axis_negative
        )
        y_reverse_changed = (
            machine.reverse_y_axis != self._axis_renderer.y_axis_negative
        )

        logger.debug(
            f"_on_machine_changed: extent_changed={extent_changed}, "
            f"extents=({extent_w}, {extent_h}), "
            f"tracked={self._tracked_axis_extents}, "
            f"margins={machine.work_margins}"
        )

        if (
            extent_changed
            or x_axis_changed
            or y_axis_changed
            or x_reverse_changed
            or y_reverse_changed
        ):
            self.reset_view()
        else:
            self._update_extent_frame()
            self._sync_camera_elements()
            self._sync_nogo_zone_elements()
            self._on_wcs_updated(machine)

    def reset_view(self):
        """
        Resets the view to fit the given machine's properties.
        """
        if not self.machine:
            self._tracked_axis_extents = (100, 100)
            self.set_size(100.0, 100.0)
            self._axis_renderer.set_width_mm(100.0)
            self._axis_renderer.set_height_mm(100.0)
            self._axis_renderer.set_margins_mm(0.0, 0.0, 0.0, 0.0)
            self._axis_renderer.set_x_axis_right(False)
            self._axis_renderer.set_y_axis_down(False)
            self._axis_renderer.set_x_axis_negative(False)
            self._axis_renderer.set_y_axis_negative(False)
            super().reset_view()
            self.aspect_ratio_changed.send(self, ratio=1.0)
            self._sync_camera_elements()
            return

        # Canvas shows full machine bed
        width_mm, height_mm = self.machine.axis_extents

        logger.debug(
            f"Resetting view for machine '{self.machine.name}' "
            f"with axis_extents=({width_mm}, {height_mm}), "
            f"x_right={self.machine.x_axis_right}, "
            f"y_down={self.machine.y_axis_down}, "
            f"reverse_x={self.machine.reverse_x_axis}, "
            f"reverse_y={self.machine.reverse_y_axis}"
        )
        self._tracked_axis_extents = self.machine.axis_extents
        self.set_size(float(width_mm), float(height_mm))
        ml, mt, mr, mb = self.machine.work_margins
        self._axis_renderer.set_width_mm(float(width_mm))
        self._axis_renderer.set_height_mm(float(height_mm))
        self._axis_renderer.set_margins_mm(
            float(ml), float(mt), float(mr), float(mb)
        )
        self._axis_renderer.set_x_axis_right(self.machine.x_axis_right)
        self._axis_renderer.set_y_axis_down(self.machine.y_axis_down)
        self._axis_renderer.set_x_axis_negative(self.machine.reverse_x_axis)
        self._axis_renderer.set_y_axis_negative(self.machine.reverse_y_axis)

        space = self.machine.get_coordinate_space()
        self._work_origin_element.set_coordinate_space(space)

        self._update_extent_frame()

        super().reset_view()

        new_ratio = width_mm / height_mm if height_mm > 0 else 1.0
        self.aspect_ratio_changed.send(self, ratio=new_ratio)
        self._sync_camera_elements()
        self._sync_nogo_zone_elements()
        self._on_wcs_updated(self.machine)
        self._update_pipeline_view_context()

    def _update_extent_frame(self):
        """
        Updates the machine extent frame and workarea background.
        """
        if not self.machine:
            self._extent_frame_element.set_visible(False)
            self._workarea_bg_element.set_visible(False)
            return

        extent_w, extent_h = self.machine.axis_extents
        ml, mt, mr, mb = self.machine.work_margins

        logger.debug(
            f"_update_extent_frame: extents=({extent_w}, {extent_h}), "
            f"margins=({ml}, {mt}, {mr}, {mb}), "
            f"canvas_size=({self.width_mm}, {self.height_mm})"
        )

        if (extent_w, extent_h) != (self.width_mm, self.height_mm):
            self.set_size(float(extent_w), float(extent_h))
        self._axis_renderer.set_margins_mm(
            float(ml), float(mt), float(mr), float(mb)
        )

        self._extent_frame_element.set_size(float(extent_w), float(extent_h))
        self._extent_frame_element.set_pos(0.0, 0.0)
        self._extent_frame_element.set_visible(True)

        workarea_w = extent_w - ml - mr
        workarea_h = extent_h - mt - mb
        logger.debug(
            f"_update_extent_frame: setting workarea_bg pos=({ml}, {mb}), "
            f"size=({workarea_w}, {workarea_h})"
        )
        self._workarea_bg_element.set_size(
            float(workarea_w), float(workarea_h)
        )
        self._workarea_bg_element.set_pos(float(ml), float(mb))
        self._workarea_bg_element.set_visible(True)

        self.queue_draw()

    def _sync_nogo_zone_elements(self):
        if not self.machine:
            for elem in self._nogo_zone_elements.values():
                elem.remove()
            self._nogo_zone_elements.clear()
            return

        current_uids = set(self.machine.nogo_zones.keys())
        existing_uids = set(self._nogo_zone_elements.keys())

        for uid in existing_uids - current_uids:
            self._nogo_zone_elements.pop(uid).remove()

        for uid, zone in self.machine.nogo_zones.items():
            if uid in self._nogo_zone_elements:
                elem = self._nogo_zone_elements[uid]
                elem._update_from_zone()
                elem.set_visible(self._nogo_zones_visible and zone.enabled)
            else:
                elem = NogoZoneElement(zone)
                self._nogo_zone_elements[uid] = elem
                self.root.add(elem)

        self.queue_draw()

    def _update_rotary_surface_element(self):
        """
        Updates the rotary diameter indicator element based on the active
        layer's settings and current WCS origin position.
        """
        active_layer = self.doc.active_layer
        rotary_enabled = active_layer.rotary_enabled

        if not rotary_enabled or not self.machine:
            self._rotary_surface_element.set_visible(False)
            return

        rotary_diameter = active_layer.rotary_diameter
        space = self.machine.get_coordinate_space()

        if self.machine.wcs_origin_is_workarea_origin:
            canvas_x, canvas_y = space.get_workarea_origin_in_machine()
        else:
            wcs_x, wcs_y, _ = self.machine.get_active_wcs_offset()
            canvas_x, canvas_y = self._machine_coords_to_canvas(wcs_x, wcs_y)

        self._rotary_surface_element.set_circumferential_axis(
            self._projection.circumferential_axis
        )
        self._rotary_surface_element.set_x_axis_right(
            self.machine.x_axis_right
        )
        self._rotary_surface_element.set_origin(canvas_x, canvas_y)
        bed_width = self.machine.axis_extents[0]
        max_length = bed_width
        default_rm = self.machine.get_default_rotary_module()
        if default_rm:
            max_length = min(max_length, default_rm.max_workpiece_length)
        self._rotary_surface_element.update_for_diameter(
            rotary_diameter, bed_width, max_length
        )
        self._rotary_surface_element.set_visible(True)

    def _on_aspect_ratio_changed(self, sender, **kwargs) -> None:
        """
        Handler for aspect ratio changes (zoom level changes).

        When the zoom level changes, the view context (pixels per mm)
        changes, so we need to update the pipeline view context to trigger
        re-rendering of all cached workpiece views.
        """
        self._update_pipeline_view_context()

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
        # Let the base WorldSurface class handle generic keys (e.g., '1')
        if super().on_key_pressed(controller, keyval, keycode, state):
            return True

        is_primary = is_primary_modifier(state)
        is_shift = bool(state & Gdk.ModifierType.SHIFT_MASK)

        # Handle moving workpiece to another layer
        if is_primary and (
            keyval == Gdk.KEY_Page_Up or keyval == Gdk.KEY_Page_Down
        ):
            direction = -1 if keyval == Gdk.KEY_Page_Up else 1
            self.editor.layer.move_selected_to_adjacent_layer(self, direction)
            return True

        # Handle clipboard and duplication
        if is_primary:
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
        elif is_primary:
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
