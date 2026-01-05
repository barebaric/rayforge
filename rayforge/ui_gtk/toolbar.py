import logging
from gi.repository import Gtk
from blinker import Signal
from .icons import get_icon
from .shared.undo_button import UndoButton, RedoButton
from .machine.machine_selector import MachineSelector
from .shared.splitbutton import SplitMenuButton
from .canvas3d import initialized as canvas3d_initialized

logger = logging.getLogger(__name__)


class MainToolbar(Gtk.Box):
    """
    The main application toolbar.
    Connects its buttons to Gio.Actions for centralized control.
    """

    def __init__(self, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, **kwargs
        )
        # Signals for View-State controls (not app actions)
        self.machine_warning_clicked = Signal()

        self.set_margin_bottom(2)
        self.set_margin_top(2)
        self.set_margin_start(12)
        self.set_margin_end(12)

        # File related buttons (import, export)
        open_button = Gtk.Button(child=get_icon("open-symbolic"))
        open_button.set_tooltip_text(_("Import image"))
        open_button.set_action_name("win.import")
        self.append(open_button)

        self.export_button = Gtk.Button(child=get_icon("save-symbolic"))
        self.export_button.set_tooltip_text(_("Generate G-code"))
        self.export_button.set_action_name("win.export")
        self.append(self.export_button)

        # Sketch related buttons
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        edit_sketch_button = Gtk.Button(child=get_icon("sketch-edit-symbolic"))
        edit_sketch_button.set_tooltip_text(_("Edit Sketch"))
        edit_sketch_button.set_action_name("win.edit_sketch")
        self.append(edit_sketch_button)

        # Undo/Redo Buttons
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.undo_button = UndoButton()
        self.undo_button.set_tooltip_text(_("Undo"))
        self.undo_button.set_action_name("win.undo")
        self.append(self.undo_button)

        self.redo_button = RedoButton()
        self.redo_button.set_tooltip_text(_("Redo"))
        self.redo_button.set_action_name("win.redo")
        self.append(self.redo_button)

        # Visibility controls
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        # The visibility button is a ToggleButton linked to a stateful action.
        # It manages its own icon state by listening to its "toggled" signal.
        self.visibility_on_icon = get_icon("visibility-on-symbolic")
        self.visibility_off_icon = get_icon("visibility-off-symbolic")
        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_active(True)
        self.visibility_button.set_child(self.visibility_on_icon)
        self.visibility_button.set_tooltip_text(
            _("Toggle workpiece visibility")
        )
        self.visibility_button.set_action_name("win.show_workpieces")
        self.visibility_button.connect("toggled", self._on_visibility_toggled)
        self.append(self.visibility_button)

        self.camera_visibility_on_icon = get_icon("camera-on-symbolic")
        self.camera_visibility_off_icon = get_icon("camera-off-symbolic")
        self.camera_visibility_button = Gtk.ToggleButton()
        self.camera_visibility_button.set_active(True)
        self.camera_visibility_button.set_child(self.camera_visibility_on_icon)
        self.camera_visibility_button.set_tooltip_text(
            _("Toggle camera image visibility")
        )
        self.camera_visibility_button.set_action_name("win.toggle_camera_view")
        self.append(self.camera_visibility_button)

        self.show_travel_button = Gtk.ToggleButton()
        self.show_travel_button.set_child(get_icon("travel-path-symbolic"))
        self.show_travel_button.set_active(False)
        self.show_travel_button.set_tooltip_text(
            _("Toggle travel move visibility")
        )
        self.show_travel_button.set_action_name("win.toggle_travel_view")
        self.append(self.show_travel_button)

        # Add a button to open the 3D preview window.
        view_3d_button = Gtk.ToggleButton(child=get_icon("3d-symbolic"))
        view_3d_button.set_action_name("win.show_3d_view")
        view_3d_button.set_sensitive(canvas3d_initialized)
        if not canvas3d_initialized:
            view_3d_button.set_tooltip_text(
                _("3D view disabled (missing dependencies like PyOpenGL)")
            )
        else:
            view_3d_button.set_tooltip_text(_("Show 3D Preview"))
        self.append(view_3d_button)

        # Add a button to toggle execution simulation
        self.simulate_button = Gtk.ToggleButton()
        self.simulate_button.set_child(get_icon("play-arrow-symbolic"))
        self.simulate_button.set_active(False)
        self.simulate_button.set_tooltip_text(_("Toggle execution simulation"))
        self.simulate_button.set_action_name("win.simulate_mode")
        self.append(self.simulate_button)

        # Add a button for the G-code Preview
        self.gcode_preview_button = Gtk.ToggleButton()
        self.gcode_preview_button.set_child(get_icon("gcode-symbolic"))
        self.gcode_preview_button.set_active(False)
        self.gcode_preview_button.set_tooltip_text(_("Toggle G-code Preview"))
        self.gcode_preview_button.set_action_name("win.toggle_gcode_preview")
        self.append(self.gcode_preview_button)

        # Add a button to toggle tab visibility.
        self.show_tabs_button = Gtk.ToggleButton()
        self.show_tabs_button.set_child(get_icon("tabs-visible-symbolic"))
        self.show_tabs_button.set_active(True)
        self.show_tabs_button.set_tooltip_text(_("Toggle tab visibility"))
        self.show_tabs_button.set_action_name("win.show_tabs")
        self.append(self.show_tabs_button)

        # Arrangement buttons (Consolidated Dropdown)
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        arrange_actions = [
            (
                _("Center Horizontally"),
                "align-horizontal-center-symbolic",
                "win.align-h-center",
            ),
            (
                _("Center Vertically"),
                "align-vertical-center-symbolic",
                "win.align-v-center",
            ),
            (_("Align Left"), "align-left-symbolic", "win.align-left"),
            (_("Align Right"), "align-right-symbolic", "win.align-right"),
            (_("Align Top"), "align-top-symbolic", "win.align-top"),
            (_("Align Bottom"), "align-bottom-symbolic", "win.align-bottom"),
            (
                _("Spread Horizontally"),
                "distribute-horizontal-symbolic",
                "win.spread-h",
            ),
            (
                _("Spread Vertically"),
                "distribute-vertical-symbolic",
                "win.spread-v",
            ),
            (
                _("Flip Horizontal"),
                "flip-horizontal-symbolic",
                "win.flip-horizontal",
            ),
            (
                _("Flip Vertical"),
                "flip-vertical-symbolic",
                "win.flip-vertical",
            ),
            (
                _("Auto Layout (pack workpieces)"),
                "auto-layout-symbolic",
                "win.layout-pixel-perfect",
            ),
        ]
        self.arrange_menu_button = SplitMenuButton(actions=arrange_actions)
        self.arrange_menu_button.set_tooltip_text(_("Arrange selection"))
        self.append(self.arrange_menu_button)

        # Tabbing buttons (Split Dropdown)
        tab_actions = [
            (
                _("Add Equidistant Tabs…"),
                "tabs-equidistant-symbolic",
                "win.add-tabs-equidistant",
            ),
            (
                _("Add Cardinal Tabs (N,S,E,W)"),
                "compass-symbolic",
                "win.add-tabs-cardinal",
            ),
        ]
        self.tab_menu_button = SplitMenuButton(actions=tab_actions)
        self.tab_menu_button.set_tooltip_text(_("Add Tabs to selection"))
        self.append(self.tab_menu_button)

        # Control buttons: home, send, pause, stop
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.home_button = Gtk.Button(child=get_icon("home-symbolic"))
        self.home_button.set_tooltip_text(_("Home the machine"))
        self.home_button.set_action_name("win.machine-home")
        self.append(self.home_button)

        self.frame_button = Gtk.Button(child=get_icon("frame-symbolic"))
        self.frame_button.set_tooltip_text(
            _("Cycle laser head around the occupied area")
        )
        self.frame_button.set_action_name("win.machine-frame")
        self.append(self.frame_button)

        self.send_button = Gtk.Button(child=get_icon("send-symbolic"))
        self.send_button.set_tooltip_text(_("Send to machine"))
        self.send_button.set_action_name("win.machine-send")
        self.append(self.send_button)

        self.hold_on_icon = get_icon("play-arrow-symbolic")
        self.hold_off_icon = get_icon("pause-symbolic")
        self.hold_button = Gtk.ToggleButton()
        self.hold_button.set_child(self.hold_off_icon)
        self.hold_button.set_tooltip_text(_("Pause machine"))
        self.hold_button.set_action_name("win.machine-hold")
        self.append(self.hold_button)

        self.cancel_button = Gtk.Button(child=get_icon("stop-symbolic"))
        self.cancel_button.set_tooltip_text(_("Cancel running job"))
        self.cancel_button.set_action_name("win.machine-cancel")
        self.append(self.cancel_button)

        self.clear_alarm_button = Gtk.Button(
            child=get_icon("clear-alarm-symbolic")
        )
        self.clear_alarm_button.set_tooltip_text(
            _("Clear machine alarm (unlock)")
        )
        self.clear_alarm_button.set_action_name("win.machine-clear-alarm")
        self.append(self.clear_alarm_button)

        self.jog_button = Gtk.Button(child=get_icon("jog-symbolic"))
        self.jog_button.set_tooltip_text(_("Manual jog control"))
        self.jog_button.set_action_name("win.machine-jog")
        self.append(self.jog_button)

        self.focus_on_icon = get_icon("laser-on-symbolic")
        self.focus_off_icon = get_icon("laser-off-symbolic")
        self.focus_button = Gtk.ToggleButton()
        self.focus_button.set_child(self.focus_on_icon)
        self.focus_button.set_tooltip_text(_("Toggle focus laser"))
        self.focus_button.set_action_name("win.toggle-focus")
        self.focus_button.connect("toggled", self._on_focus_toggled)
        self.append(self.focus_button)

        # Add spacer to push machine selector to the right
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.append(spacer)

        # Add clickable warning for misconfigured machine
        self.machine_warning_box = Gtk.Box(spacing=6)
        self.machine_warning_box.set_margin_end(12)
        warning_icon = get_icon("warning-symbolic")
        warning_label = Gtk.Label(label=_("Machine not fully configured"))
        warning_label.add_css_class("warning-label")
        self.machine_warning_box.append(warning_icon)
        self.machine_warning_box.append(warning_label)
        self.machine_warning_box.set_tooltip_text(
            _("Machine driver is missing required settings. Click to edit.")
        )
        self.machine_warning_box.set_visible(False)
        warning_click = Gtk.GestureClick.new()
        warning_click.connect(
            "pressed", lambda *_: self.machine_warning_clicked.send(self)
        )
        self.machine_warning_box.add_controller(warning_click)
        self.append(self.machine_warning_box)

        # Add machine selector dropdown
        self.machine_selector = MachineSelector()
        self.append(self.machine_selector)

    def _on_visibility_toggled(self, button: Gtk.ToggleButton):
        """Callback to update the visibility icon when the button's
        state changes for any reason (user click or action state change)."""
        if button.get_active():
            button.set_child(self.visibility_on_icon)
        else:
            button.set_child(self.visibility_off_icon)

    def _on_focus_toggled(self, button: Gtk.ToggleButton):
        """Callback to update the focus icon when the button's
        state changes for any reason (user click or action state change)."""
        if button.get_active():
            button.set_child(self.focus_off_icon)
        else:
            button.set_child(self.focus_on_icon)
