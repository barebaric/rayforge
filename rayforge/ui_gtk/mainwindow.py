import asyncio
import logging
import shutil
import webbrowser
from concurrent.futures import Future
from pathlib import Path
from typing import Callable, Coroutine, List, Optional, Tuple
from gettext import gettext as _
from gi.repository import Adw, Gdk, Gio, GLib, Gtk

from .. import __version__
from .. import const
from ..context import get_context
from ..core.asset_registry import asset_type_registry
from ..core.group import Group
from ..core.item import DocItem
from ..core.ops.axis import Axis
from ..core.step_registry import step_registry
from ..core.undo import Command, HistoryManager
from ..core.workpiece import WorkPiece
from ..doceditor.editor import DocEditor
from ..machine.cmd import MachineCmd
from ..machine.driver.driver import DeviceState, DeviceStatus
from ..machine.driver.dummy import NoDeviceDriver
from ..machine.models.machine import Machine
from ..machine.models.zone import check_ops_collides_with_zones
from ..machine.transport import TransportStatus
from ..addon_mgr.update_cmd import UpdateCommand
from ..pipeline.artifact import JobArtifact, JobArtifactHandle
from ..pipeline.encoder.gcode import MachineCodeOpMap
from ..shared.tasker import task_mgr
from ..shared.util.time_format import format_hours_to_hm
from ..updater import AppUpdateChecker
from ..usage import get_usage_tracker
from .about import AboutDialog
from .action_registry import action_registry
from .actions import (
    ActionManager,
    SHORTCUTS,
    action_extension_registry,
)
from .canvas import CanvasElement
from .canvas2d.drag_drop_cmd import DragDropCmd
from .canvas2d.elements.stock import StockElement
from .canvas2d.surface import WorkSurface
from .doceditor import file_dialogs
from .doceditor.bottom_panel import BottomPanel
from .doceditor.import_handler import start_interactive_import
from .doceditor.item_properties import DocItemPropertiesWidget
from .doceditor.missing_features_dialog import MissingFeaturesDialog
from .doceditor.property_providers import register_builtin_providers
from .doceditor.workflow_view import WorkflowView
from .machine.machine_dropdown import MachineDropdown
from .machine.settings_dialog import MachineSettingsDialog
from .main_menu import MainMenu
from .settings.settings_dialog import SettingsWindow
from .project_cmd import ProjectCmd
from .shared.gtk import get_monitor_geometry
from .shared.playback_overlay import PlaybackOverlay
from .shared.progress_bar import ProgressBar
from .shared.usage_consent_dialog import UsageConsentDialog
from .shared.time_estimate_overlay import TimeEstimateOverlay
from .shared.visibility_overlay import VisibilityOverlay
from .sim3d import Canvas3D, initialized as canvas3d_initialized
from .sim3d.camera import ViewDirection
from .sim3d.viewport import ViewportConfig
from .toolbar import MainToolbar
from .view_mode_cmd import ViewModeCmd


logger = logging.getLogger(__name__)


css = """
.right-panel-overlay {
    background-color: transparent;
    border-radius: 8px;
    margin: 6px 12px 12px 6px;
    box-shadow: 0 2px 12px alpha(black, 0.2);
}

.status-message-overlay {
    background-color: @theme_bg_color;
    border-radius: 6px;
    padding: 4px 10px;
    box-shadow: 0 2px 6px alpha(black, 0.15);
}

.in-header-menubar {
    margin-left: 6px;
    box-shadow: none;
}

.in-header-menubar item {
    padding: 6px 12px 6px 12px;
}

.menu separator {
    border-top: 1px solid @borders;
    margin-top: 5px;
    margin-bottom: 5px;
}

.warning-label {
    color: @warning_color;
    font-weight: bold;
}

dropdown.machine-dropdown button {
    padding-top: 2px;
    padding-bottom: 2px;
}
"""


class MainWindow(Adw.ApplicationWindow):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.set_title(const.APP_NAME)
        self._current_machine: Optional[Machine] = None  # For signal handling
        self._last_bottom_panel_height = 200
        self._saved_bottom_panel_visible = False
        self._old_doc = None  # Track previous document for signal reconnection
        self.canvas3d: Optional[Canvas3D] = None
        self._is_syncing_3d = False

        # The ToastOverlay will wrap the main content box
        self.toast_overlay = Adw.ToastOverlay()
        self.set_content(self.toast_overlay)
        # Track active toasts so they can be cleared programmatically
        self._active_toasts: List[Adw.Toast] = []

        # The main content box is now the child of the ToastOverlay
        vbox = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.toast_overlay.set_child(vbox)

        # Create the central document editor. This now owns the Doc and
        # Pipeline.
        context = get_context()
        self.doc_editor = DocEditor(task_mgr, context)
        context.addon_mgr.addon_state_changed.connect(
            self._on_addon_state_changed
        )
        self.machine_cmd = MachineCmd(self.doc_editor)
        self.machine_cmd.job_started.connect(self._on_job_started)

        # Instantiate and connect the UpdateCommand's notification signal
        self.update_cmd = UpdateCommand(task_mgr, context)
        self.update_cmd.notification_requested.connect(
            self._on_editor_notification
        )

        # Instantiate the app version update checker
        self.app_update_checker = AppUpdateChecker(task_mgr, context)
        self.app_update_checker.notification_requested.connect(
            self._on_editor_notification
        )

        # Instantiate UI-specific command handlers
        self.view_cmd = ViewModeCmd(self.doc_editor, self)
        self.project_cmd = ProjectCmd(self, self.doc_editor)

        geometry = get_monitor_geometry()
        if geometry:
            self.set_default_size(
                int(geometry.width * 0.8), int(geometry.height * 0.8)
            )
        else:
            self.set_default_size(1100, 800)

        # HeaderBar with left-aligned menu and centered title
        self.header_bar = Adw.HeaderBar()
        vbox.append(self.header_bar)

        # Create the menu model and the popover menubar
        self.menu_model = MainMenu()
        self.menubar = Gtk.PopoverMenuBar.new_from_model(self.menu_model)
        self.menubar.add_css_class("in-header-menubar")
        self.header_bar.pack_start(self.menubar)

        # Set up Recent Files manager
        self.recent_manager = Gtk.RecentManager.get_default()
        self.recent_manager.connect(
            "changed", self.project_cmd.update_recent_files_menu
        )
        self.project_cmd.update_recent_files_menu()

        # Create and set the centered title widget
        window_title = Adw.WindowTitle(
            title=self.get_title() or "", subtitle=__version__ or ""
        )
        self.header_bar.set_title_widget(window_title)

        # Add machine selector to the header bar (right side)
        self.machine_selector = MachineDropdown()
        self.header_bar.pack_end(self.machine_selector)

        # Create a vertical paned for main content and bottom control panel
        self.vertical_paned = Gtk.Paned(orientation=Gtk.Orientation.VERTICAL)
        self.vertical_paned.set_resize_start_child(True)
        self.vertical_paned.set_resize_end_child(False)
        self.vertical_paned.set_shrink_start_child(False)
        self.vertical_paned.set_shrink_end_child(False)

        self._status_overlay = Gtk.Overlay()
        self._status_overlay.set_child(self.vertical_paned)

        self._status_message_label = Gtk.Label(
            halign=Gtk.Align.END,
            valign=Gtk.Align.END,
            margin_end=12,
            margin_bottom=6,
        )
        self._status_message_label.add_css_class("status-message-overlay")
        self._status_message_label.set_visible(False)
        self._status_overlay.add_overlay(self._status_message_label)

        vbox.append(self._status_overlay)

        # Create a stack for switching between main view and addon pages
        self.main_stack = Gtk.Stack()
        self.main_stack.set_vexpand(True)
        self.main_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_UP_DOWN
        )
        self.vertical_paned.set_start_child(self.main_stack)

        # Create a container for the main UI
        main_ui_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.main_stack.add_named(main_ui_box, "main")

        # Create and add the main toolbar.
        self.toolbar = MainToolbar()
        self._connect_toolbar_signals()
        main_ui_box.append(self.toolbar)

        # Create an overlay so the right panel can float above the canvas.
        self._canvas_overlay = Gtk.Overlay()
        self._canvas_overlay.set_vexpand(True)
        main_ui_box.append(self._canvas_overlay)

        # Apply styles
        display = Gdk.Display.get_default()
        if display:
            provider = Gtk.CssProvider()
            provider.load_from_string(css)
            Gtk.StyleContext.add_provider_for_display(
                display, provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

        # Determine initial machine dimensions for canvases.
        context = get_context()
        config = context.config
        if config.machine:
            viewport = ViewportConfig.from_machine(config.machine)
        else:
            viewport = ViewportConfig.default()

        self.surface = WorkSurface(
            editor=self.doc_editor,
            parent_window=self,
            machine=config.machine,
            cam_visible=True,  # Will be set by action state
        )
        self.surface.set_hexpand(True)

        # Initialize drag-and-drop command for the surface
        self.drag_drop_cmd = DragDropCmd(self, self.surface)
        self.surface.drag_drop_cmd = self.drag_drop_cmd
        self.drag_drop_cmd.setup_drop_targets()

        # Set up action registry before registering actions
        action_registry.set_window(self)
        self.action_registry = action_registry

        # Let addons register action extension handlers before
        # ActionManager.register_actions() invokes setup handlers.
        context.plugin_mgr.hook.register_actions(
            action_registry=action_registry
        )

        # Setup keyboard actions using the new ActionManager.
        self.action_manager = ActionManager(self)
        self.action_manager.register_actions()

        shortcut_controller = Gtk.ShortcutController()
        self.action_manager.register_shortcuts(shortcut_controller)
        self.add_controller(shortcut_controller)

        # Connect document signals
        doc = self.doc_editor.doc
        self._old_doc = doc  # Track initial document for signal reconnection
        self._initialize_document()
        doc.updated.connect(self.on_doc_changed)
        doc.descendant_added.connect(self.on_doc_changed)
        doc.descendant_removed.connect(self.on_doc_changed)
        doc.descendant_updated.connect(self.on_doc_changed)
        doc.active_layer_changed.connect(self._on_active_layer_changed)
        doc.history_manager.changed.connect(self.on_history_changed)

        # Connect editor signals
        self.doc_editor.notification_requested.connect(
            self._on_editor_notification
        )
        self.doc_editor.document_settled.connect(self._on_document_settled)
        self.doc_editor.saved_state_changed.connect(
            self.project_cmd.on_saved_state_changed
        )
        self.doc_editor.document_changed.connect(self._on_document_changed)

        # Create the view stack for 2D and 3D views
        self.view_stack = Gtk.Stack()
        self.view_stack.set_transition_type(
            Gtk.StackTransitionType.SLIDE_LEFT_RIGHT
        )
        self.view_stack.set_margin_start(12)
        self.view_stack.set_hexpand(True)
        self.view_stack.connect(
            "notify::visible-child-name", self._on_view_stack_changed
        )

        # The view stack is the base child of the canvas overlay
        self._canvas_overlay.set_child(self.view_stack)

        # Wrap surface in an overlay to allow preview controls
        self.surface_overlay = Gtk.Overlay()
        self.surface_overlay.set_child(self.surface)
        self._surface_vis_overlay = VisibilityOverlay(
            show_workpiece=True,
            show_camera=bool(
                config.machine
                and any(c.enabled for c in config.machine.cameras)
            ),
            show_tabs=True,
            shortcuts=SHORTCUTS,
        )
        self._surface_vis_overlay.set_margin_end(424)
        self.surface_overlay.add_overlay(self._surface_vis_overlay)
        self._time_estimate_overlay = TimeEstimateOverlay()
        self.surface_overlay.add_overlay(self._time_estimate_overlay)
        self.view_stack.add_named(self.surface_overlay, "2d")

        # Add a click handler to unfocus when clicking the "dead space" of the
        # canvas area. This is the correct place for this handler, as it won't
        # interfere with clicks on the sidebar.
        canvas_click_gesture = Gtk.GestureClick.new()
        canvas_click_gesture.connect(
            "pressed", self._on_canvas_area_click_pressed
        )
        # self.surface_overlay.add_controller(canvas_click_gesture)

        if canvas3d_initialized:
            self._create_canvas3d(context, viewport)

        self._sync_view_toggle_actions()

        # Undo/Redo buttons are now connected to the doc via actions.
        self.toolbar.undo_button.set_history_manager(
            self.doc_editor.history_manager
        )
        self.toolbar.redo_button.set_history_manager(
            self.doc_editor.history_manager
        )

        # Create a vertical paned for the right pane content
        self._right_pane = Gtk.ScrolledWindow()
        self._right_pane.set_policy(
            Gtk.PolicyType.NEVER, Gtk.PolicyType.AUTOMATIC
        )
        self._right_pane.set_vexpand(True)
        self._right_pane.add_css_class("right-panel-overlay")
        self._right_pane.set_halign(Gtk.Align.END)
        self._right_pane.set_valign(Gtk.Align.START)
        self._right_pane.set_propagate_natural_height(True)
        self._canvas_overlay.add_overlay(self._right_pane)

        # Create a vertical box to organize the content within the
        # ScrolledWindow.
        right_pane_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        right_pane_box.set_size_request(400, -1)
        self._right_pane.set_child(right_pane_box)

        # The WorkflowView will be updated when a layer is activated.
        initial_workflow = self.doc_editor.doc.active_layer.workflow
        assert initial_workflow, "Initial active layer must have a workflow"
        self.workflowview = WorkflowView(
            self.doc_editor,
            initial_workflow,
            step_factories=step_registry.get_factories(),
        )
        self.workflowview.set_margin_top(6)
        self.workflowview.set_margin_end(12)
        right_pane_box.append(self.workflowview)

        # Register built-in property providers before creating the widget
        register_builtin_providers()

        # Add the WorkpiecePropertiesWidget
        self.item_props_widget = DocItemPropertiesWidget(
            editor=self.doc_editor
        )
        item_props_container = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.item_props_widget.set_margin_top(6)
        self.item_props_widget.set_margin_end(12)
        item_props_container.append(self.item_props_widget)

        self.item_revealer = Gtk.Revealer()
        self.item_revealer.set_child(item_props_container)
        self.item_revealer.set_reveal_child(False)
        self.item_revealer.set_transition_type(
            Gtk.RevealerTransitionType.SLIDE_UP
        )
        right_pane_box.append(self.item_revealer)

        # Connect signals for item selection and actions
        self.surface.selection_changed.connect(self._on_selection_changed)
        self.surface.elements_deleted.connect(self.on_elements_deleted)
        self.surface.cut_requested.connect(self.on_cut_requested)
        self.surface.copy_requested.connect(self.on_copy_requested)
        self.surface.paste_requested.connect(self.on_paste_requested)
        self.surface.duplicate_requested.connect(self.on_duplicate_requested)
        self.surface.transform_initiated.connect(
            self._on_surface_transform_initiated
        )
        self.surface.transform_end.connect(self._on_surface_transform_end)
        self.surface.work_zero_requested.connect(self._on_work_zero_requested)
        self.surface.click_to_zero_cancelled.connect(
            self._on_click_to_zero_cancelled
        )

        # Connect new signal from WorkSurface for edit item requests
        self.surface.edit_item_requested.connect(self._on_edit_item_requested)

        # Create the control panel
        config = get_context().config
        self.bottom_panel = BottomPanel(
            config.machine, self.doc_editor, self.machine_cmd
        )
        self.bottom_panel.set_size_request(-1, self._last_bottom_panel_height)
        self.bottom_panel.set_visible(True)
        self.vertical_paned.set_end_child(self.bottom_panel)

        self.bottom_panel.gcode_viewer.line_activated.connect(
            self._on_gcode_line_activated
        )

        # Connect edit item requests from the layers tab
        self.bottom_panel.edit_item_requested.connect(
            self._on_edit_item_requested
        )

        config = get_context().config
        if config.bottom_panel:
            self.bottom_panel.from_dict(config.bottom_panel)

        self.bottom_panel.tab_changed.connect(self._on_bottom_tab_changed)
        self.bottom_panel.layout_changed.connect(
            self._on_bottom_layout_changed
        )

        self.bottom_panel.click_to_zero_mode_changed.connect(
            self._on_click_to_zero_mode_changed
        )

        self.bottom_panel.asset_browser.add_asset_requested.connect(
            self.on_add_asset_requested
        )
        self.bottom_panel.asset_browser.asset_activated.connect(
            self.on_asset_activated
        )

        self.bottom_panel.set_get_bounds_callback(self._get_selection_bounds)

        # Connect to position signal to remember user's chosen height
        self.vertical_paned.connect(
            "notify::position", self._on_vertical_pane_position_changed
        )

        # Create and add the progress bar at the bottom of vbox
        self.progress_bar = ProgressBar(task_mgr)
        gesture = Gtk.GestureClick()
        gesture.connect(
            "pressed", lambda *args: self.on_status_bar_clicked(None)
        )
        self.progress_bar.add_controller(gesture)
        vbox.append(self.progress_bar)

        self.doc_editor.pipeline.job_time_updated.connect(
            self._on_job_time_updated
        )

        # Set up config signals.
        config.changed.connect(self.on_config_changed)
        task_mgr.tasks_updated.connect(self.on_running_tasks_changed)
        self.needs_homing = (
            config.machine.home_on_start if config.machine else False
        )

        # Set initial state
        self.on_config_changed(None)

        # Apply saved visibility state
        self._apply_saved_visibility_state()

        # Notify addons that main window is ready
        context.plugin_mgr.hook.main_window_ready(main_window=self)

        # Trigger startup tasks when window is shown
        self.connect("map", self._trigger_startup_tasks)

    def _trigger_startup_tasks(self, widget):
        """
        Runs once when the window is first shown.
        """
        # Disconnect self to ensure it only runs once
        self.disconnect_by_func(self._trigger_startup_tasks)

        # Initialize usage tracking based on saved consent
        config = get_context().config
        if config.has_consented_tracking:
            get_usage_tracker().set_enabled(True)
            get_usage_tracker().track_page_view("/view/2d", "2D View")
        elif config.has_declined_tracking:
            pass  # Explicitly do nothing, respecting the user's choice
        else:
            dialog = UsageConsentDialog(self)
            dialog.present()

        # Trigger the non-blocking check for addon updates
        self.update_cmd.check_for_updates_on_startup()

        # Trigger the non-blocking check for app version updates
        self.app_update_checker.check_on_startup()

    def _on_click_to_zero_mode_changed(self, sender, *, active: bool):
        """Handle click-to-zero mode toggle from control panel."""
        self.surface.set_click_to_zero_mode(active)

    def _on_work_zero_requested(self, sender, *, x: float, y: float):
        """Handle work zero request from canvas click."""
        config = get_context().config
        if not config.machine:
            return

        async def set_zero_func(ctx):
            if config.machine:
                await config.machine.set_work_origin(x, y, 0.0)

        task_mgr.add_coroutine(set_zero_func)
        self.bottom_panel.set_click_to_zero_mode(False)

    def _on_click_to_zero_cancelled(self, sender):
        """Handle click-to-zero mode cancellation."""
        self.bottom_panel.set_click_to_zero_mode(False)

    def _get_selection_bounds(
        self,
    ) -> Optional[Tuple[float, float, float, float]]:
        """
        Get the bounding box of selected items or workarea bounds.

        Returns:
            A tuple (min_x, min_y, max_x, max_y) in world coordinates,
            or None if there is no machine configured.
        """
        selected_elements = self.surface.get_selected_elements()

        if selected_elements:
            workpieces = []
            for elem in selected_elements:
                if isinstance(elem.data, WorkPiece):
                    workpieces.append(elem.data)
                elif isinstance(elem.data, Group):
                    workpieces.extend(elem.data.get_descendants(WorkPiece))

            bboxes = []
            for wp in workpieces:
                bbox = wp.get_geometry_world_bbox()
                if bbox is not None:
                    bboxes.append(bbox)

            if bboxes:
                min_x = min(b[0] for b in bboxes)
                min_y = min(b[1] for b in bboxes)
                max_x = max(b[2] for b in bboxes)
                max_y = max(b[3] for b in bboxes)
                return (min_x, min_y, max_x, max_y)

        config = get_context().config
        machine = config.machine
        if not machine:
            return None

        space = machine.get_coordinate_space()
        workarea_origin_machine = space.get_workarea_origin_in_machine()
        min_x, min_y = space.machine_point_to_world(*workarea_origin_machine)
        workarea_w, workarea_h = space.workarea_size
        max_x = min_x + workarea_w
        max_y = min_y + workarea_h

        return (min_x, min_y, max_x, max_y)

    def _apply_saved_visibility_state(self):
        """
        Applies the saved visibility state for control panel.
        This should be called after actions are registered.
        """
        config = get_context().config

        bottom_panel_action = self.action_manager.get_action(
            "toggle_bottom_panel"
        )
        if (
            bottom_panel_action
            and config.bottom_panel
            and config.bottom_panel.get("visible")
        ):
            bottom_panel_action.change_state(GLib.Variant.new_boolean(True))

    def add_stack_page(self, name: str, widget: Gtk.Widget):
        """Add a page to the main stack.

        This is a public API for addons to add their own pages to the
        main stack (e.g., editor views).

        Args:
            name: The name/identifier for the page
            widget: The widget to add as a page
        """
        self.main_stack.add_named(widget, name)

    def show_stack_page(self, name: str):
        """Switch to a named page in the main stack.

        Args:
            name: The name of the page to show
        """
        self.main_stack.set_visible_child_name(name)

    def remove_stack_page(self, name: str):
        """Remove a page from the main stack.

        Args:
            name: The name of the page to remove
        """
        child = self.main_stack.get_child_by_name(name)
        if child:
            self.main_stack.remove(child)

    def get_stack_page(self, name: str) -> Optional[Gtk.Widget]:
        """Get a page widget from the main stack by name.

        Args:
            name: The name of the page to get

        Returns:
            The widget if found, None otherwise
        """
        return self.main_stack.get_child_by_name(name)

    def open_modal_page(self, name: str):
        """Open a modal page, hiding auxiliary panels.

        This is used for full-screen editor modes (like the sketcher) that
        should hide panels like the control panel.

        Args:
            name: The name of the modal page to show
        """
        self._saved_bottom_panel_visible = self.bottom_panel.get_visible()
        if self._saved_bottom_panel_visible:
            self.bottom_panel.set_visible(False)
        self.main_stack.set_visible_child_name(name)

    def close_modal_page(self):
        """Close the current modal page and return to main view.

        Restores the visibility of auxiliary panels that were hidden.
        """
        if self._saved_bottom_panel_visible:
            self.bottom_panel.set_visible(True)
        self.main_stack.set_visible_child_name("main")

    def on_add_child(self, sender):
        """Handler for adding a new stock item."""
        self.doc_editor.stock.add_stock()

    def on_add_asset_requested(self, sender, *, type_name: str):
        """Handler for add asset requests, dispatches via action lookup."""
        asset_cls = asset_type_registry.get(type_name)
        if asset_cls and asset_cls.add_action:
            action = self.action_manager.get_action(asset_cls.add_action)
            if action:
                action.activate(None)

    def on_asset_activated(self, sender, *, asset):
        """Handler for asset activation, dispatches via action lookup."""
        asset_cls = type(asset)
        if asset_cls.activate_action:
            action = self.action_manager.get_action(asset_cls.activate_action)
            if action:
                action.activate(GLib.Variant.new_string(asset.uid))

    def _on_edit_item_requested(self, sender, *, item, action_name: str):
        """Signal handler for edit item requests from the surface."""
        action = self.action_manager.get_action(action_name)
        if action:
            action.activate(GLib.Variant.new_string(item.uid))

    def load_project(self, file_path: Path):
        """Public method to load a project from a given path."""
        self.project_cmd.load_project(file_path)

    def _update_macros_menu(self, *args):
        """Rebuilds the dynamic 'Macros' menu."""
        config = get_context().config
        if not config.machine:
            self.menu_model.update_macros_menu([])
            return

        macros = sorted(
            config.machine.macros.values(), key=lambda m: m.name.lower()
        )
        enabled_macros = [m for m in macros if m.enabled]
        self.menu_model.update_macros_menu(enabled_macros)

    def on_execute_macro(self, action: Gio.SimpleAction, param: GLib.Variant):
        """Handler for the 'execute-macro' action."""
        config = get_context().config
        if not config.machine:
            return
        macro_uid = param.get_string()
        logger.info(f"Executing macro: {macro_uid}")
        self.machine_cmd.execute_macro_by_uid(config.machine, macro_uid)

    def _on_job_started(self, sender):
        logger.debug("Job started")
        self.machine_selector.update_eta(None)
        self._update_actions_and_ui()

    def _on_addon_state_changed(self, sender, addon_name):
        """Handle addon enable/disable to refresh action handlers."""
        action_extension_registry.invoke_setup_handlers(self.action_manager)
        self.action_manager.update_action_states()

    def _on_job_progress_updated(self, metrics: dict):
        """Callback for when job progress is updated."""
        eta_seconds = metrics.get("eta_seconds")
        self.machine_selector.update_eta(eta_seconds)

    def _on_job_finished(self, sender):
        """Handles the completion of a machine job."""
        logger.debug("Job finished")
        self.machine_selector.update_eta(None)

    def _on_job_future_done(self, future: Future):
        """Callback for when the job submission task completes or fails."""
        try:
            # Check for exceptions during job assembly or submission.
            future.result()
        except Exception as e:
            logger.error(f"Job submission failed: {e}", exc_info=True)
            # If the submission failed, the driver's 'job_finished' signal
            # will never fire, so we must stop the live view here to prevent
            # the UI from getting stuck.
            self.machine_selector.update_eta(None)

        # Ensure UI is updated (e.g. Cancel button disabled, others enabled)
        self._update_actions_and_ui()

    def _on_bottom_tab_changed(self, sender, *, name: str):
        if name == "gcode":
            self.refresh_previews()
        self._save_bottom_panel()

    def _on_bottom_layout_changed(self, sender):
        self._save_bottom_panel()

    def _save_bottom_panel(self):
        get_context().config.set_bottom_panel(self.bottom_panel.to_dict())

    def _on_gcode_line_activated(self, sender, *, line_number: int):
        """
        Handles the user activating a line in the G-code previewer.
        Syncs the highlight and the 3D playback slider.
        """
        # 1. Update the visual highlight to match the cursor, no scroll.
        self.bottom_panel.gcode_viewer.highlight_line(
            line_number, use_align=False
        )

        # 2. If 3D playback is active, sync the slider.
        op_map = self.bottom_panel.gcode_viewer.op_map
        if op_map and line_number in op_map.machine_code_to_op:
            op_index = op_map.machine_code_to_op[line_number]
            self._is_syncing_3d = True
            self._canvas3d_playback.set_playback_position(op_index)
            if self.canvas3d:
                self.canvas3d.queue_render()
            self._is_syncing_3d = False

    def _on_3d_playback_step_changed(self, sender, *, ops_index: int):
        """
        Handles the 3D playback slider changing. Syncs the G-code viewer
        highlight to the corresponding line.
        """
        if self._is_syncing_3d:
            return
        self.bottom_panel.gcode_viewer.highlight_op(ops_index)

    def _on_vertical_pane_position_changed(self, paned, param):
        position = paned.get_position()
        full_height = paned.get_height()
        panel_height = full_height - position
        if panel_height > 1:
            self._last_bottom_panel_height = panel_height

    def _on_surface_transform_initiated(self, sender):
        pass

    def _on_view_stack_changed(self, stack: Gtk.Stack, param):
        """Handles logic when switching between 2D and 3D views."""
        child_name = stack.get_visible_child_name()
        if child_name == "3d":
            self._update_3d_view_content()
        self._update_actions_and_ui()

    def _update_3d_view_content(self):
        """
        Updates the 3D canvas by delegating to its internal update method.
        This is now a fast, non-blocking operation.
        """
        if not self.canvas3d:
            return
        if self.canvas3d.has_stale_job():
            self.refresh_previews()
        self.canvas3d.update_scene_from_doc()

    def _update_gcode_preview(
        self, gcode_string: Optional[str], op_map: Optional[MachineCodeOpMap]
    ):
        """Updates the G-code preview panel from a pre-generated string."""
        if gcode_string is None:
            self.bottom_panel.gcode_viewer.clear()
            return

        self.bottom_panel.gcode_viewer.set_gcode(gcode_string)
        if op_map:
            self.bottom_panel.gcode_viewer.set_op_map(op_map)

    def on_show_3d_view(
        self, action: Gio.SimpleAction, value: Optional[GLib.Variant]
    ):
        """Delegates the view switching logic to the command module."""
        self.view_cmd.toggle_3d_view(action, value)

    def on_show_workpieces_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_workpieces_visible(is_visible)
        action.set_state(value)
        config = get_context().config
        config.canvas_view.show_workpieces = is_visible
        config.changed.send(config)

    def on_toggle_camera_view_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_camera_image_visibility(is_visible)
        action.set_state(value)
        config = get_context().config
        config.canvas_view.show_camera = is_visible
        config.changed.send(config)

    def on_toggle_travel_view_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_show_travel_moves(is_visible)
        if self.canvas3d is not None:
            self.canvas3d.set_show_travel_moves(is_visible)
        action.set_state(value)
        config = get_context().config
        config.canvas_view.show_travel_lines = is_visible
        config.changed.send(config)

    def on_show_nogo_zones_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        self.surface.set_show_nogo_zones(is_visible)
        if self.canvas3d is not None:
            self.canvas3d.set_show_nogo_zones(is_visible)
        action.set_state(value)
        config = get_context().config
        config.canvas_view.show_nogo_zones = is_visible
        config.changed.send(config)

    def on_show_models_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        if self.canvas3d is not None:
            self.canvas3d.set_show_models(is_visible)
        action.set_state(value)
        config = get_context().config
        config.canvas_view.show_models = is_visible
        config.changed.send(config)

    def on_view_top(self, action, param):
        """Action handler to set the 3D view to top-down."""
        self.view_cmd.set_view(ViewDirection.TOP, self.canvas3d)

    def on_view_front(self, action, param):
        """Action handler to set the 3D view to front."""
        self.view_cmd.set_view(ViewDirection.FRONT, self.canvas3d)

    def on_view_right(self, action, param):
        """Action handler to set the 3D view to right."""
        self.view_cmd.set_view(ViewDirection.RIGHT, self.canvas3d)

    def on_view_left(self, action, param):
        """Action handler to set the 3D view to left."""
        self.view_cmd.set_view(ViewDirection.LEFT, self.canvas3d)

    def on_view_back(self, action, param):
        """Action handler to set the 3D view to back."""
        self.view_cmd.set_view(ViewDirection.BACK, self.canvas3d)

    def on_view_iso(self, action, param):
        """Action handler to set the 3D view to isometric."""
        self.view_cmd.set_view(ViewDirection.ISO, self.canvas3d)

    def on_view_perspective_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """Handles state changes for the perspective view action."""
        self.view_cmd.toggle_perspective(self.canvas3d, action, value)

    def _initialize_document(self):
        """
        Adds required initial state to a new document, such as default
        steps to workpiece layers.
        """
        self.doc_editor.step.initialize_default_steps()

    def _sync_view_toggle_actions(self):
        """
        Re-triggers each persisted view toggle action so that both the
        canvas surfaces and the overlay buttons reflect the persisted
        config values at startup.
        """
        am = self.action_manager
        cv = get_context().config.canvas_view

        am.get_action("show_workpieces").set_state(
            GLib.Variant.new_boolean(not cv.show_workpieces)
        )
        self.on_show_workpieces_state_change(
            am.get_action("show_workpieces"),
            GLib.Variant.new_boolean(cv.show_workpieces),
        )

        am.get_action("toggle_camera_view").set_state(
            GLib.Variant.new_boolean(not cv.show_camera)
        )
        self.on_toggle_camera_view_state_change(
            am.get_action("toggle_camera_view"),
            GLib.Variant.new_boolean(cv.show_camera),
        )

        am.get_action("toggle_travel_view").set_state(
            GLib.Variant.new_boolean(not cv.show_travel_lines)
        )
        self.on_toggle_travel_view_state_change(
            am.get_action("toggle_travel_view"),
            GLib.Variant.new_boolean(cv.show_travel_lines),
        )

        am.get_action("show_nogo_zones").set_state(
            GLib.Variant.new_boolean(not cv.show_nogo_zones)
        )
        self.on_show_nogo_zones_state_change(
            am.get_action("show_nogo_zones"),
            GLib.Variant.new_boolean(cv.show_nogo_zones),
        )

        am.get_action("show_models").set_state(
            GLib.Variant.new_boolean(not cv.show_models)
        )
        self.on_show_models_state_change(
            am.get_action("show_models"),
            GLib.Variant.new_boolean(cv.show_models),
        )

        am.get_action("show_tabs").set_state(
            GLib.Variant.new_boolean(not cv.show_tabs)
        )
        am.on_show_tabs_state_change(
            am.get_action("show_tabs"),
            GLib.Variant.new_boolean(cv.show_tabs),
        )

        am.get_action("view_toggle_perspective").set_state(
            GLib.Variant.new_boolean(not cv.perspective_mode)
        )
        self.on_view_perspective_state_change(
            am.get_action("view_toggle_perspective"),
            GLib.Variant.new_boolean(cv.perspective_mode),
        )

    def _connect_toolbar_signals(self):
        """Connects signals from the MainToolbar to their handlers.
        Most buttons are connected via Gio.Actions. Only view-state toggles
        and special widgets are connected here.
        """
        self.toolbar.machine_warning_clicked.connect(
            self.on_machine_warning_clicked
        )
        self.machine_selector.machine_selected.connect(
            self.on_machine_selected_by_selector
        )

    def on_zero_here_clicked(self, action, param):
        """Handler for 'zero-here' action."""
        config = get_context().config
        if not config.machine:
            return

        # 'param' is likely "all" string from the action setup
        axes_to_zero = Axis.X | Axis.Y | Axis.Z

        async def zero_func(ctx):
            # Explicitly check again to satisfy type checker
            if config.machine:
                await config.machine.set_work_origin_here(axes_to_zero)

        # Launch async zeroing
        task_mgr.add_coroutine(zero_func)

    def _on_canvas_area_click_pressed(self, gesture, n_press, x, y):
        """
        Handler for clicks on the canvas overlay area (the 'dead space').
        It unfocuses any other widget and gives focus to the surface for
        keyboard shortcuts.
        """
        logger.debug("Clicked on canvas area dead space, focusing surface.")
        self.surface.grab_focus()

    def on_machine_selected_by_selector(self, sender, *, machine: Machine):
        """
        Handles the 'machine_selected' signal from the MachineSelector widget,
        delegating the logic to the MachineManager.
        """
        context = get_context()
        context.machine_mgr.set_active_machine(machine)

    def _on_machine_status_changed(self, machine: Machine, state: DeviceState):
        """Called when the active machine's state changes."""
        config = get_context().config
        if self.needs_homing and config.machine and config.machine.driver:
            if state.status == DeviceStatus.IDLE:
                self.needs_homing = False
                driver = config.machine.driver
                task_mgr.add_coroutine(lambda ctx: driver.home())
        self._update_actions_and_ui()

    def _on_connection_status_changed(
        self,
        machine: Machine,
        status: TransportStatus,
        message: Optional[str] = None,
    ):
        """Called when the active machine's connection status changes."""
        if (
            status == TransportStatus.CONNECTED
            and machine.clear_alarm_on_connect
            and machine.device_state.status == DeviceStatus.ALARM
        ):
            logger.info(
                "Machine connected in ALARM state. Auto-clearing alarm."
            )
            self.machine_cmd.clear_alarm(machine)
        self._update_actions_and_ui()

    def _on_machine_hours_changed(self, sender, **kwargs):
        """
        Called when machine hours change. Checks for maintenance notifications.
        """
        due_counters = sender.consume_due_notifications()
        for counter in due_counters:
            msg = _(
                "Maintenance Alert: {name} has reached its limit "
                "({curr} / {limit})"
            ).format(
                name=counter.name,
                curr=format_hours_to_hm(counter.value),
                limit=format_hours_to_hm(counter.notify_at),
            )
            self._on_editor_notification(
                self,
                msg,
                persistent=True,
                action_label=_("View Counters"),
                action_callback=lambda: self._open_machine_hours_dialog(),
            )

    def _open_machine_hours_dialog(self):
        """Opens the machine settings dialog on the Hours page."""
        config = get_context().config
        if not config.machine:
            return
        dialog = MachineSettingsDialog(
            machine=config.machine,
            transient_for=self,
            initial_page="hours",
        )
        dialog.present()

    def on_history_changed(
        self, history_manager: HistoryManager, command: Command
    ):
        self._update_actions_and_ui()
        # After undo/redo, the document state may have changed in ways
        # that require a full UI sync (e.g., layer visibility).
        self.on_doc_changed(self.doc_editor.doc)
        self._update_macros_menu()

    def on_doc_changed(self, sender, **kwargs):
        # Synchronize UI elements that depend on the document model
        self.surface.update_from_doc()
        doc = self.doc_editor.doc
        if doc.active_layer and doc.active_layer.workflow:
            self.workflowview.set_workflow(doc.active_layer.workflow)

        # Sync the selectability of stock items based on active layer
        self._sync_element_selectability()

        # Update button sensitivity and other state
        self._update_actions_and_ui()

    def _sync_element_selectability(self):
        """
        Updates the 'selectable' property of StockElements on the canvas
        based on which layer is currently active and their visibility.
        """
        # Find all StockElement instances currently on the canvas
        for element in self.surface.find_by_type(StockElement):
            # Stock items are only selectable when they are visible
            element.selectable = element.visible

    def _on_active_layer_changed(self, sender):
        """
        Handles activation of a new layer. Updates the workflow view and
        resets the paste counter.
        """
        logger.debug("Active layer changed, updating UI.")
        # Reset the paste counter to ensure the next paste is in-place.
        self.doc_editor.edit.reset_paste_counter()

        # Get the newly activated layer from the document
        activated_layer = self.doc_editor.doc.active_layer
        has_workflow = activated_layer.workflow is not None

        # Show/hide the workflow view based on the layer type
        self.workflowview.set_visible(has_workflow)

        if has_workflow:
            # For regular layers, update the workflow view with the
            # new workflow
            self.workflowview.set_workflow(activated_layer.workflow)

    def _on_document_changed(self, sender):
        """
        Handles when a new document is set on the DocEditor.
        Reconnects signal handlers to the new document and updates the UI.
        """
        new_doc = self.doc_editor.doc

        # Disconnect from old document signals if they were connected
        # We need to track the old doc to disconnect properly
        if self._old_doc is not None:
            self._old_doc.updated.disconnect(self.on_doc_changed)
            self._old_doc.descendant_added.disconnect(self.on_doc_changed)
            self._old_doc.descendant_removed.disconnect(self.on_doc_changed)
            self._old_doc.descendant_updated.disconnect(self.on_doc_changed)
            self._old_doc.active_layer_changed.disconnect(
                self._on_active_layer_changed
            )
            self._old_doc.history_manager.changed.disconnect(
                self.on_history_changed
            )

        # Connect to new document's signals
        new_doc.updated.connect(self.on_doc_changed)
        new_doc.descendant_added.connect(self.on_doc_changed)
        new_doc.descendant_removed.connect(self.on_doc_changed)
        new_doc.descendant_updated.connect(self.on_doc_changed)
        new_doc.active_layer_changed.connect(self._on_active_layer_changed)
        new_doc.history_manager.changed.connect(self.on_history_changed)

        # Store reference to current doc for future disconnection
        self._old_doc = new_doc

        # Update Undo/Redo buttons to listen to the new history manager
        self.toolbar.undo_button.set_history_manager(new_doc.history_manager)
        self.toolbar.redo_button.set_history_manager(new_doc.history_manager)

        # Update child views to point to the new document
        self.bottom_panel.set_doc(new_doc)

        # Initialize new document
        self._initialize_document()

        # Check for missing producer types and show dialog if needed
        missing_types = new_doc.missing_producer_types
        if missing_types:
            dialog = MissingFeaturesDialog(self, missing_types)
            dialog.present()

        # Trigger update to sync UI with new document
        self.on_doc_changed(new_doc)

        # Update the UI with the new document's content
        self.on_doc_changed(new_doc)

    def _on_editor_notification(
        self,
        sender,
        message: str,
        persistent: bool = False,
        action_label: Optional[str] = None,
        action_callback: Optional[Callable] = None,
    ):
        """
        Shows a toast when requested by the DocEditor.
        If 'persistent' is True, the toast will have a dismiss button and
        remain visible until closed.
        If 'action_label' and 'action_callback' are provided, a button
        will be added to the toast that triggers the callback.
        """
        toast = Adw.Toast.new(message)
        if persistent:
            toast.set_timeout(0)  # 0 = persistent
            toast.set_priority(Adw.ToastPriority.HIGH)

        if action_label and action_callback:
            toast.set_button_label(action_label)
            # Connecting directly to 'button-clicked' is the simplest way
            # to handle a callback without defining a GAction.
            toast.connect("button-clicked", lambda t: action_callback())

        self._add_toast(toast)

    def _add_toast(self, toast: Adw.Toast):
        """Helper to add a toast to the overlay and track it."""
        self._active_toasts.append(toast)
        # Connect to dismissed signal to clean up our reference
        toast.connect("dismissed", self._on_toast_dismissed)
        self.toast_overlay.add_toast(toast)

    def _on_toast_dismissed(self, toast):
        """Removes the toast from the tracking list when dismissed."""
        if toast in self._active_toasts:
            self._active_toasts.remove(toast)

    def _on_surface_transform_end(self, sender, *args, **kwargs):
        """Clears all active toasts from the toast overlay."""
        logger.debug("Clearing all toasts from overlay.")

        # Iterate over a copy of the list because dismiss() triggers removal
        for toast in list(self._active_toasts):
            toast.dismiss()

    def _on_assembly_for_preview_finished(
        self,
        handle: Optional[JobArtifactHandle],
        error: Optional[Exception],
    ):
        """Callback for when the job assembly for previews is complete."""
        if error:
            logger.error(
                "Failed to aggregate ops for preview",
                exc_info=error,
            )
            # Release handle on error if it exists
            if handle:
                self.doc_editor.pipeline.artifact_manager.release_handle(
                    handle
                )
            handle = None

        # Schedule the UI update on the main thread, passing the handle.
        # The handle will be released in the main thread callback.
        GLib.idle_add(self._on_previews_ready, handle)

    def _on_previews_ready(self, handle: Optional[JobArtifactHandle]):
        """
        Main-thread callback to distribute assembled Ops to all consumers.
        This method is responsible for releasing the artifact handle.
        """
        artifact_manager = self.doc_editor.pipeline.artifact_manager

        with artifact_manager.checkout_handle(handle) as final_artifact:
            if final_artifact is None:
                if handle is None:
                    self._update_gcode_preview(None, None)
                    return

                logger.warning("Final artifact is None, not a JobArtifact")
                return

            assert isinstance(final_artifact, JobArtifact)

            # 2. Update G-code Preview
            is_gcode_visible = self.bottom_panel.is_item_visible("gcode")
            is_3d_visible = self.view_stack.get_visible_child_name() == "3d"

            if final_artifact and (is_gcode_visible or is_3d_visible):
                self._update_gcode_preview(
                    final_artifact.machine_code, final_artifact.op_map
                )
            else:
                self._update_gcode_preview(None, None)

        return GLib.SOURCE_REMOVE

    def refresh_previews(self):
        """
        Public method to trigger a refresh of all data previews, like the
        simulator and G-code view.
        """
        if get_context().exit_after_settle:
            return

        is_gcode_visible = self.bottom_panel.is_item_visible("gcode")
        is_3d_visible = self.view_stack.get_visible_child_name() == "3d"

        if not is_gcode_visible and not is_3d_visible:
            return

        config = get_context().config
        if not config.machine:
            # Pass None to clear previews if no machine is configured
            self._on_previews_ready(None)
            return

        # Try to use existing job artifact first
        existing_handle = self.doc_editor.pipeline.get_existing_job_handle()
        if existing_handle is not None:
            # Use existing artifact without regenerating
            self._on_previews_ready(existing_handle)
        else:
            # No existing artifact, trigger generation
            self.doc_editor.file.assemble_job_in_background(
                when_done=self._on_assembly_for_preview_finished
            )

    def _refresh_gcode_preview(self, sender=None, **kwargs):
        """Refresh G-code preview when machine settings change."""
        if self.bottom_panel.is_item_visible("gcode"):
            self.refresh_previews()

    def _create_canvas3d(self, context, viewport: ViewportConfig):
        """
        Creates a Canvas3D instance and adds it to the view stack.
        """
        self.canvas3d = Canvas3D(
            context,
            self.doc_editor,
            viewport=viewport,
        )
        self._canvas3d_overlay = Gtk.Overlay()
        self._canvas3d_overlay.set_child(self.canvas3d)
        self._canvas3d_vis_overlay = VisibilityOverlay(
            show_workpiece=False,
            show_models=True,
            shortcuts=SHORTCUTS,
        )
        self._canvas3d_vis_overlay.set_margin_end(424)
        self._canvas3d_overlay.add_overlay(self._canvas3d_vis_overlay)
        self._canvas3d_playback = PlaybackOverlay()
        self.canvas3d.set_playback_overlay(self._canvas3d_playback)
        self._canvas3d_overlay.add_overlay(self._canvas3d_playback)
        self._canvas3d_playback.step_changed.connect(
            self._on_3d_playback_step_changed
        )
        self.view_stack.add_named(self._canvas3d_overlay, "3d")

    def _on_document_settled(self, sender):
        """
        Called when all background processing is complete. This is the main
        hook for refreshing previews that depend on the final assembled job.
        """
        self.refresh_previews()

    def _on_selection_changed(
        self,
        sender,
        elements: List[CanvasElement],
        active_element: Optional[CanvasElement],
    ):
        """Handles the 'selection-changed' signal from the WorkSurface."""
        # Get all selected DocItems (WorkPieces, Groups, etc.)
        selected_items = [
            elem.data for elem in elements if isinstance(elem.data, DocItem)
        ]

        # Get the primary active item from the signal payload
        active_item = (
            active_element.data
            if active_element and isinstance(active_element.data, DocItem)
            else None
        )

        # Reorder the list to put the active one first, if it exists
        if active_item and active_item in selected_items:
            selected_items.remove(active_item)
            selected_items.insert(0, active_item)

        self.item_props_widget.set_items(selected_items)
        self.item_revealer.set_reveal_child(bool(selected_items))
        self.bottom_panel.update_position_menu_sensitivity()
        self._update_actions_and_ui()

    def on_config_changed(self, sender, **kwargs):
        config = get_context().config
        machine_changed = config.machine is not self._current_machine

        if machine_changed:
            self._on_machine_signals_changed(config)
            self._update_canvas3d(config.machine)

        # Update the control panel to use the new machine
        self.bottom_panel.set_machine(config.machine, self.machine_cmd)

        # Update the main WorkSurface to use the new size
        self.surface.set_machine(config.machine)

        # Show/hide camera toggle based on whether machine has cameras
        has_cameras = bool(
            config.machine and any(c.enabled for c in config.machine.cameras)
        )
        self._surface_vis_overlay.set_camera_visible(has_cameras)

        self.surface.update_from_doc()
        self._update_macros_menu()

        # Check for any pending notifications from the new machine immediately
        if self._current_machine:
            self._on_machine_hours_changed(self._current_machine.machine_hours)

        self._update_actions_and_ui()
        self.apply_theme()

    def _on_machine_signals_changed(self, config):
        # Disconnect from the previously active machine, if any
        if self._current_machine:
            self._current_machine.state_changed.disconnect(
                self._on_machine_status_changed
            )
            self._current_machine.connection_status_changed.disconnect(
                self._on_connection_status_changed
            )
            self._current_machine.job_finished.disconnect(
                self._on_job_finished
            )
            self._current_machine.changed.disconnect(self._update_macros_menu)
            self._current_machine.changed.disconnect(
                self._refresh_gcode_preview
            )
            self._current_machine.machine_hours.changed.disconnect(
                self._on_machine_hours_changed
            )

        self._current_machine = config.machine

        # Connect to the new active machine's signals
        if self._current_machine:
            self._current_machine.state_changed.connect(
                self._on_machine_status_changed
            )
            self._current_machine.connection_status_changed.connect(
                self._on_connection_status_changed
            )
            self._current_machine.job_finished.connect(self._on_job_finished)
            self._current_machine.changed.connect(self._update_macros_menu)
            self._current_machine.changed.connect(self._refresh_gcode_preview)
            self._current_machine.machine_hours.changed.connect(
                self._on_machine_hours_changed
            )

    def _update_canvas3d(self, new_machine):
        if self.canvas3d is None:
            return
        if new_machine:
            viewport = ViewportConfig.from_machine(new_machine)
        else:
            viewport = ViewportConfig.default()
        self.canvas3d.set_machine(viewport=viewport)

    def apply_theme(self):
        """Reads the theme from config and applies it to the UI."""
        style_manager = Adw.StyleManager.get_default()
        config = get_context().config
        if config.theme == "light":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_LIGHT)
        elif config.theme == "dark":
            style_manager.set_color_scheme(Adw.ColorScheme.FORCE_DARK)
        else:  # "system" or any other invalid value
            style_manager.set_color_scheme(Adw.ColorScheme.DEFAULT)

    def on_running_tasks_changed(self, sender, tasks, progress):
        self._update_actions_and_ui()
        self._update_status_message(tasks)

    def _update_status_message(self, tasks):
        if not tasks:
            self._status_message_label.set_visible(False)
            return

        oldest_task = tasks[0]
        message = oldest_task.get_message()
        status_text = message if message is not None else ""

        if status_text and len(tasks) > 1:
            status_text += _(" (+{tasks} more)").format(tasks=len(tasks) - 1)
        elif len(tasks) > 1:
            status_text = _("{tasks} tasks").format(tasks=len(tasks))

        self._status_message_label.set_text(status_text)
        self._status_message_label.set_visible(bool(status_text))

    def _update_actions_and_ui(self):
        config = get_context().config
        active_machine = config.machine
        am = self.action_manager
        doc = self.doc_editor.doc

        if not active_machine:
            am.get_action("export").set_enabled(False)
            am.get_action("machine-settings").set_enabled(False)
            am.get_action("machine-home").set_enabled(False)
            am.get_action("machine-frame").set_enabled(False)
            am.get_action("machine-send").set_enabled(False)
            am.get_action("machine-hold").set_enabled(False)
            am.get_action("machine-cancel").set_enabled(False)
            am.get_action("machine-clear-alarm").set_enabled(False)
            am.get_action("execute-macro").set_enabled(False)
            am.get_action("zero-here").set_enabled(False)

            self.toolbar.export_button.set_tooltip_text(
                _("Select a machine to enable G-code export")
            )
            self.toolbar.machine_warning_box.set_visible(False)
            self.surface.set_laser_dot_visible(False)
        else:
            device_status = active_machine.device_state.status
            conn_status = active_machine.connection_status
            state = active_machine.device_state
            active_driver = active_machine.driver
            is_dummy = isinstance(active_driver, NoDeviceDriver)

            can_export = doc.has_result() and not task_mgr.has_tasks()
            am.get_action("export").set_enabled(can_export)
            export_tooltip = _("Generate G-code")
            if task_mgr.has_tasks():
                export_tooltip = _(
                    "Cannot export while other tasks are running"
                )
            elif not doc.has_workpiece():
                export_tooltip = _("Add a workpiece to enable export")
            elif not doc.has_result():
                export_tooltip = _(
                    "Add or enable a processing step to enable export"
                )
            self.toolbar.export_button.set_tooltip_text(export_tooltip)

            if active_driver and active_driver.state.error:
                self.toolbar.set_machine_warning(
                    active_driver.state.error.title,
                    active_driver.state.error.code,
                    active_driver.state.error.description,
                )
                self.toolbar.machine_warning_box.set_visible(True)
            else:
                self.toolbar.machine_warning_box.set_visible(False)
            am.get_action("machine-settings").set_enabled(True)

            # A job/task is running if the machine is not idle or a UI task is
            # active.
            machine_processing = (
                conn_status == TransportStatus.CONNECTED
                and device_status != DeviceStatus.IDLE
            )

            is_job_or_task_active = (
                machine_processing
                or task_mgr.has_tasks()
                or self.machine_cmd.is_job_running
            )

            am.get_action("machine-home").set_enabled(
                not is_job_or_task_active
            )

            can_frame = (
                active_machine.can_frame()
                and doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("machine-frame").set_enabled(can_frame)
            if not active_machine.can_frame():
                self.toolbar.frame_button.set_tooltip_text(
                    _("Configure frame power to enable")
                )
            else:
                self.toolbar.frame_button.set_tooltip_text(
                    _("Cycle laser head around the occupied area")
                )

            send_sensitive = (
                not isinstance(active_driver, NoDeviceDriver)
                and (active_driver and not active_driver.state.error)
                and conn_status == TransportStatus.CONNECTED
                and doc.has_result()
                and not is_job_or_task_active
            )
            am.get_action("machine-send").set_enabled(send_sensitive)
            self.toolbar.send_button.set_tooltip_text(_("Send to machine"))

            hold_sensitive = device_status in (
                DeviceStatus.RUN,
                DeviceStatus.HOLD,
                DeviceStatus.CYCLE,
            )
            is_holding = device_status == DeviceStatus.HOLD
            am.get_action("machine-hold").set_enabled(hold_sensitive)
            am.get_action("machine-hold").set_state(
                GLib.Variant.new_boolean(is_holding)
            )
            if is_holding:
                self.toolbar.hold_button.set_child(self.toolbar.hold_on_icon)
                self.toolbar.hold_button.set_tooltip_text(_("Resume machine"))
            else:
                self.toolbar.hold_button.set_child(self.toolbar.hold_off_icon)
                self.toolbar.hold_button.set_tooltip_text(_("Pause machine"))

            cancel_sensitive = conn_status == TransportStatus.CONNECTED
            am.get_action("machine-cancel").set_enabled(cancel_sensitive)

            clear_alarm_sensitive = bool(
                device_status == DeviceStatus.ALARM
                or (active_driver and active_driver.state.error)
            )
            am.get_action("machine-clear-alarm").set_enabled(
                clear_alarm_sensitive
            )
            if clear_alarm_sensitive:
                self.toolbar.clear_alarm_button.add_css_class(
                    "suggested-action"
                )
            else:
                self.toolbar.clear_alarm_button.remove_css_class(
                    "suggested-action"
                )

            # Update focus button sensitivity
            head = active_machine.get_default_head()
            can_focus = (
                head
                and head.focus_power_percent > 0
                and not is_job_or_task_active
            )
            am.get_action("toggle-focus").set_enabled(can_focus)

            connected = conn_status == TransportStatus.CONNECTED
            self.surface.set_laser_dot_visible(connected)
            if state and connected:
                x, y = state.machine_pos[:2]
                if x is not None and y is not None:
                    self.surface.set_laser_dot_position(x, y)

            # Set macro action sensitivity
            can_run_macros = connected and not is_job_or_task_active
            am.get_action("execute-macro").set_enabled(can_run_macros)

            # WCS UI
            is_g53 = (
                active_machine.active_wcs == active_machine.machine_space_wcs
            )

            # Allow zeroing if connected OR if it's the dummy driver
            can_zero = (
                (connected or is_dummy)
                and not is_g53
                and not is_job_or_task_active
            )
            am.get_action("zero-here").set_enabled(can_zero)

        # Update actions that don't depend on the machine state
        selected_elements = self.surface.get_selected_elements()
        has_selection = len(selected_elements) > 0

        am.get_action("undo").set_enabled(
            self.doc_editor.history_manager.can_undo()
        )
        am.get_action("redo").set_enabled(
            self.doc_editor.history_manager.can_redo()
        )
        am.get_action("cut").set_enabled(has_selection)
        am.get_action("copy").set_enabled(has_selection)
        am.get_action("paste").set_enabled(self.doc_editor.edit.can_paste())
        am.get_action("select_all").set_enabled(doc.has_workpiece())
        am.get_action("duplicate").set_enabled(has_selection)
        am.get_action("remove").set_enabled(has_selection)
        am.get_action("clear").set_enabled(doc.has_workpiece())

        # Update sensitivity for Grouping actions
        can_group = len(selected_elements) >= 2
        am.get_action("group").set_enabled(can_group)

        can_ungroup = any(
            isinstance(elem.data, Group) for elem in selected_elements
        )
        am.get_action("ungroup").set_enabled(can_ungroup)

        # Update sensitivity for Layer actions
        can_move_layers = has_selection and len(doc.layers) > 1
        am.get_action("layer-move-up").set_enabled(can_move_layers)
        am.get_action("layer-move-down").set_enabled(can_move_layers)

        # Update sensitivity for 3D view actions
        is_3d_view_active = self.view_stack.get_visible_child_name() == "3d"
        can_show_3d = is_3d_view_active or canvas3d_initialized
        am.get_action("show_3d_view").set_enabled(can_show_3d)
        am.get_action("view_top").set_enabled(is_3d_view_active)
        am.get_action("view_front").set_enabled(is_3d_view_active)
        am.get_action("view_iso").set_enabled(is_3d_view_active)
        am.get_action("view_toggle_perspective").set_enabled(is_3d_view_active)

        # Update sensitivity for Arrangement actions
        can_distribute = len(self.surface.get_selected_workpieces()) >= 2
        am.get_action("align-h-center").set_enabled(has_selection)
        am.get_action("align-v-center").set_enabled(has_selection)
        am.get_action("align-left").set_enabled(has_selection)
        am.get_action("align-right").set_enabled(has_selection)
        am.get_action("align-top").set_enabled(has_selection)
        am.get_action("align-bottom").set_enabled(has_selection)
        am.get_action("spread-h").set_enabled(can_distribute)
        am.get_action("spread-v").set_enabled(can_distribute)
        self.toolbar.arrange_menu_button.set_sensitive(has_selection)

        # Update sensitivity for Tab buttons
        show_tabs_action = am.get_action("show_tabs")
        has_any_tabs = any(wp.tabs for wp in doc.all_workpieces)
        show_tabs_action.set_enabled(has_any_tabs)

    def on_machine_warning_clicked(self, sender):
        """Opens the machine settings dialog for the current machine."""
        config = get_context().config
        if not config.machine:
            return
        dialog = MachineSettingsDialog(
            machine=config.machine,
            transient_for=self,
        )
        dialog.present()

    def on_status_bar_clicked(self, sender):
        action = self.action_manager.get_action("toggle_bottom_panel")
        state = action.get_state()
        if state:
            new_state = not state.get_boolean()
            action.change_state(GLib.Variant.new_boolean(new_state))
        else:
            action.change_state(GLib.Variant.new_boolean(True))

    def on_toggle_bottom_panel_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        action.set_state(value)

        if is_visible:
            self.bottom_panel.set_visible(True)
            full_height = self.vertical_paned.get_height()
            self.vertical_paned.set_position(
                full_height - self._last_bottom_panel_height
            )
            get_usage_tracker().track_page_view(
                "/bottom-panel/open", "Bottom Panel Opened"
            )
        else:
            self.bottom_panel.set_visible(False)

        self._save_bottom_panel()

    def on_toggle_right_panel_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        is_visible = value.get_boolean()
        action.set_state(value)
        self._right_pane.set_visible(is_visible)
        get_context().config.set_right_panel_visible(is_visible)

    def _on_dialog_notification(self, sender, message: str = ""):
        """Shows a toast when requested by a child dialog."""
        toast = Adw.Toast.new(message)
        self._add_toast(toast)

    def on_quit_action(self, action, parameter):
        self.close()

    def do_close_request(self):
        """
        Handles the 'close-request' signal to check for unsaved changes.
        For GTK signals, returning True PREVENTS the default handler from
        running (i.e., stops the close). Returning False allows it.
        """
        if self.doc_editor.is_saved:
            return False  # Allow the window to close

        self.project_cmd.show_unsaved_changes_dialog(
            self._on_close_request_dialog_response
        )
        return True  # Prevent the window from closing until user responds

    def _on_close_request_dialog_response(self, response):
        """Callback for unsaved changes dialog in do_close_request."""
        if response == "cancel":
            return  # Do nothing, window remains open.

        if response == "discard":
            self.destroy()
            return

        if response == "save":
            self.project_cmd.on_save_project(None, None)
            if self.doc_editor.is_saved:
                self.destroy()

    def on_menu_import(self, action, param=None):
        start_interactive_import(self, self.doc_editor)

    def on_open_clicked(self, sender):
        self.on_menu_import(sender)

    def on_clear_clicked(self, action, param):
        self.doc_editor.edit.clear_all_items()

    def on_recalculate_clicked(self, action, param):
        self.doc_editor.pipeline.recalculate()

    def on_force_recalculate_clicked(self, action, param):
        self.doc_editor.pipeline.recalculate(force=True)

    def _check_nogo_zones_and_proceed(self, proceed_callback):
        config = get_context().config
        machine = config.machine
        if not machine:
            proceed_callback()
            return

        enabled_zones = {
            k: v for k, v in machine.nogo_zones.items() if v.enabled
        }
        if not enabled_zones:
            proceed_callback()
            return

        existing = self.doc_editor.pipeline.get_existing_job_handle()
        if existing is not None:
            artifact_manager = self.doc_editor.pipeline.artifact_manager
            try:
                with artifact_manager.checkout_handle(existing) as artifact:
                    if isinstance(artifact, JobArtifact):
                        if check_ops_collides_with_zones(
                            artifact.ops, enabled_zones
                        ):
                            self._show_nogo_zone_warning(proceed_callback)
                            return
            except Exception:
                logger.warning("Failed to check no-go zones", exc_info=True)
            proceed_callback()
            return

        def _on_artifact_ready(handle, error):
            if error or not handle:
                proceed_callback()
                return
            try:
                artifact_manager = self.doc_editor.pipeline.artifact_manager
                with artifact_manager.checkout_handle(handle) as artifact:
                    if isinstance(artifact, JobArtifact):
                        if check_ops_collides_with_zones(
                            artifact.ops, enabled_zones
                        ):
                            self._show_nogo_zone_warning(proceed_callback)
                            return
            except Exception:
                logger.warning("Failed to check no-go zones", exc_info=True)
            proceed_callback()

        self.doc_editor.file.assemble_job_in_background(
            when_done=_on_artifact_ready
        )

    def _show_nogo_zone_warning(self, proceed_callback):
        dialog = Adw.MessageDialog(
            transient_for=self,
            heading=_("No-Go Zone Collision"),
            body=_(
                "The toolpath enters one or more enabled no-go zones. "
                "Proceeding may cause damage to your machine or "
                "workpiece."
            ),
        )
        dialog.add_response("cancel", _("_Cancel"))
        dialog.add_response("proceed", _("_Proceed"))
        dialog.set_default_response("cancel")
        dialog.set_close_response("cancel")
        dialog.set_response_appearance(
            "proceed", Adw.ResponseAppearance.DESTRUCTIVE
        )

        def on_response(dialog, response_id):
            if response_id == "proceed":
                proceed_callback()

        dialog.connect("response", on_response)
        dialog.present()

    def on_export_clicked(self, action, param=None):
        def _proceed():
            initial_name = None
            if self.doc_editor.file_path:
                initial_name = f"{self.doc_editor.file_path.stem}.gcode"
            file_dialogs.show_export_gcode_dialog(
                self, self._on_save_dialog_response, initial_name
            )

        self._check_nogo_zones_and_proceed(_proceed)

    def on_export_document_clicked(self, action, param=None):
        initial_name = "document.svg"
        if self.doc_editor.file_path:
            initial_name = f"{self.doc_editor.file_path.stem}.svg"
        file_dialogs.show_export_document_dialog(
            self, self._on_export_document_response, initial_name
        )

    def on_export_object_clicked(self, action, param=None):
        selected = self.surface.get_selected_workpieces()
        if len(selected) == 1:
            file_dialogs.show_export_object_dialog(
                self, self._on_export_object_response, selected[0]
            )
        else:
            self._on_editor_notification(
                self, _("Please select a single object to export.")
            )

    def _on_export_object_response(self, dialog, result, user_data):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())

            selected = self.surface.get_selected_workpieces()
            if len(selected) != 1:
                return

            self.doc_editor.file.export_object_to_path(file_path, selected[0])

        except GLib.Error as e:
            logger.error(f"Error exporting object: {e.message}")

    def _on_export_document_response(self, dialog, result, user_data):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())
        except GLib.Error as e:
            logger.error(f"Error exporting document: {e.message}")
            return

        self.doc_editor.file.export_document_to_path(file_path)

    def _on_save_dialog_response(self, dialog, result, user_data):
        try:
            file = dialog.save_finish(result)
            if not file:
                return
            file_path = Path(file.get_path())
        except GLib.Error as e:
            logger.error(f"Error saving file: {e.message}")
            return

        # This is now a non-blocking call.
        self.doc_editor.file.export_gcode_to_path(file_path)

    def on_home_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return

        # Disable focus mode when homing
        focus_action = self.action_manager.get_action("toggle-focus")
        focus_state = focus_action.get_state()
        if focus_state and focus_state.get_boolean():
            focus_action.change_state(GLib.Variant.new_boolean(False))

        self.machine_cmd.home(config.machine)

    def _run_machine_job(self, job_coroutine: Coroutine):
        """
        Wraps a machine job coroutine in an asyncio.Task and handles
        its completion or failure.
        """
        fut = asyncio.run_coroutine_threadsafe(job_coroutine, task_mgr.loop)
        # Add a callback to handle the result (or exception) of the task
        fut.add_done_callback(self._on_job_future_done)

    def on_frame_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return

        # Disable focus mode when framing
        focus_action = self.action_manager.get_action("toggle-focus")
        focus_state = focus_action.get_state()
        if focus_state and focus_state.get_boolean():
            focus_action.change_state(GLib.Variant.new_boolean(False))

        # Get the coroutine object for the framing job
        job_coro = self.machine_cmd.frame_job(
            config.machine, on_progress=self._on_job_progress_updated
        )
        # Run the job using the helper
        self._run_machine_job(job_coro)

    def on_send_clicked(self, action, param):
        config = get_context().config
        machine = config.machine
        if not machine:
            return

        def _proceed():
            focus_action = self.action_manager.get_action("toggle-focus")
            focus_state = focus_action.get_state()
            if focus_state and focus_state.get_boolean():
                focus_action.change_state(GLib.Variant.new_boolean(False))

            job_coro = self.machine_cmd.send_job(
                machine,
                on_progress=self._on_job_progress_updated,
            )
            self._run_machine_job(job_coro)

        self._check_nogo_zones_and_proceed(_proceed)

    def on_hold_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'hold' action.
        This is the correct handler for a stateful action.
        """
        config = get_context().config
        if not config.machine:
            return
        is_requesting_hold = value.get_boolean()
        self.machine_cmd.set_hold(config.machine, is_requesting_hold)
        action.set_state(value)

    def on_cancel_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return
        self.machine_cmd.cancel_job(config.machine)

    def on_clear_alarm_clicked(self, action, param):
        config = get_context().config
        if not config.machine:
            return
        self.machine_cmd.clear_alarm(config.machine)

    def on_toggle_focus_state_change(
        self, action: Gio.SimpleAction, value: GLib.Variant
    ):
        """
        Handles the 'change-state' signal for the 'toggle-focus' action.
        This toggles the laser focus mode on/off.
        """
        config = get_context().config
        if not config.machine:
            return

        is_focus_on = value.get_boolean()
        head = config.machine.get_default_head()

        if is_focus_on:
            self.machine_cmd.set_focus_power(head, head.focus_power_percent)
        else:
            self.machine_cmd.set_focus_power(head, 0)
        action.set_state(value)

        # Update the toolbar button icon
        if is_focus_on:
            self.toolbar.focus_button.set_child(self.toolbar.focus_off_icon)
        else:
            self.toolbar.focus_button.set_child(self.toolbar.focus_on_icon)

    def on_elements_deleted(self, sender, elements: List[CanvasElement]):
        """Handles the deletion signal from the WorkSurface."""
        items_to_delete = [
            elem.data for elem in elements if isinstance(elem.data, DocItem)
        ]
        if items_to_delete:
            self.doc_editor.edit.remove_items(
                items_to_delete, "Delete item(s)"
            )

    def on_cut_requested(self, sender, items: List[DocItem]):
        """Handles the 'cut-requested' signal from the WorkSurface."""
        self.doc_editor.edit.cut_items(items)
        self._update_actions_and_ui()

    def on_copy_requested(self, sender, items: List[DocItem]):
        """
        Handles the 'copy-requested' signal from the WorkSurface.
        """
        self.doc_editor.edit.copy_items(items)
        self._update_actions_and_ui()

    def on_paste_requested(self, sender, *args):
        """
        Handles the 'paste-requested' signal from the WorkSurface.
        Checks for image data on system clipboard first, then falls back
        to workpiece paste.
        """
        # Priority 1: Check if system clipboard contains image data
        if self.drag_drop_cmd.handle_clipboard_paste():
            return

        # Priority 2: Standard workpiece paste
        newly_pasted = self.doc_editor.edit.paste_items()
        if newly_pasted:
            self.surface.select_items(newly_pasted)
        self._update_actions_and_ui()

    def on_select_all(self, action, param):
        """
        Selects all top-level items (workpieces and groups) in the document.
        """
        doc = self.doc_editor.doc
        items_to_select = doc.get_top_level_items()
        if items_to_select:
            self.surface.select_items(items_to_select)

    def on_duplicate_requested(self, sender, items: List[DocItem]):
        """
        Handles the 'duplicate-requested' signal from the WorkSurface.
        """
        newly_duplicated = self.doc_editor.edit.duplicate_items(items)
        if newly_duplicated:
            self.surface.select_items(newly_duplicated)

    def on_menu_cut(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            self.doc_editor.edit.cut_items(list(selection))
            self._update_actions_and_ui()

    def on_menu_copy(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            self.doc_editor.edit.copy_items(list(selection))
            self._update_actions_and_ui()

    def on_menu_duplicate(self, action, param):
        selection = self.surface.get_selected_items()
        if selection:
            newly_duplicated = self.doc_editor.edit.duplicate_items(
                list(selection)
            )
            self.surface.select_items(newly_duplicated)

    def on_menu_remove(self, action, param):
        items = self.surface.get_selected_items()
        if items:
            self.doc_editor.edit.remove_items(list(items))

    def show_about_dialog(self, action, param):
        dialog = AboutDialog(transient_for=self)
        dialog.present()

    def on_donate_clicked(self, action, param):
        webbrowser.open("https://www.patreon.com/c/knipknap")

    def on_save_debug_log(self, action, param):
        archive_path = get_context().debug_dump_manager.create_dump_archive()

        if not archive_path:
            self._on_editor_notification(
                self, _("Failed to create debug archive.")
            )
            return

        dialog = Gtk.FileDialog.new()
        dialog.set_title(_("Save Debug Log"))
        dialog.set_initial_name(archive_path.name)

        def save_callback(dialog, result):
            try:
                destination_file = dialog.save_finish(result)
                if destination_file:
                    destination_path = Path(destination_file.get_path())
                    shutil.move(archive_path, destination_path)
                    self._on_editor_notification(
                        self,
                        _("Debug log saved to {path}").format(
                            path=destination_path.name
                        ),
                    )
            except GLib.Error as e:
                if not e.matches(
                    Gio.io_error_quark(), Gio.IOErrorEnum.CANCELLED
                ):
                    self._on_editor_notification(
                        self,
                        _("Error saving file: {msg}").format(msg=e.message),
                    )
            except Exception as e:
                self._on_editor_notification(
                    self,
                    _("An unexpected error occurred: {error}").format(error=e),
                )
            finally:
                if archive_path.exists():
                    archive_path.unlink()

        dialog.save(self, None, save_callback)

    def show_settings(self, action, param):
        dialog = SettingsWindow(transient_for=self)
        dialog.present()
        dialog.connect("close-request", self._on_settings_dialog_closed)

    def show_machine_settings(self, action, param):
        """Opens the machine settings dialog for the current machine."""
        config = get_context().config
        if not config.machine:
            return
        dialog = MachineSettingsDialog(
            machine=config.machine,
            transient_for=self,
        )
        dialog.present()

    def _on_settings_dialog_closed(self, dialog):
        logger.debug("Settings dialog closed")
        self.surface.grab_focus()  # re-enables keyboard shortcuts

    def _on_job_time_updated(self, sender, *, total_seconds):
        self._time_estimate_overlay.set_estimated_time(total_seconds)
