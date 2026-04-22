import logging
from gettext import gettext as _
from gi.repository import Gdk, Gtk
from blinker import Signal
from .action_registry import action_registry
from .icons import get_icon
from .shared.splitbutton import SplitMenuButton
from .shared.undo_button import UndoButton, RedoButton
from .sim3d import initialized as canvas3d_initialized

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

        # File related buttons (open, save, import, export)
        self.open_button = Gtk.Button(child=get_icon("open-symbolic"))
        self.open_button.set_tooltip_text(_("Open Project"))
        self.open_button.set_action_name("win.open")
        self.append(self.open_button)

        self.save_button = Gtk.Button(child=get_icon("save-symbolic"))
        self.save_button.set_tooltip_text(_("Save"))
        self.save_button.set_action_name("win.save")
        self.append(self.save_button)

        self.save_as_button = Gtk.Button(child=get_icon("save-as-symbolic"))
        self.save_as_button.set_tooltip_text(_("Save As..."))
        self.save_as_button.set_action_name("win.save-as")
        self.append(self.save_as_button)

        open_button = Gtk.Button(child=get_icon("download-symbolic"))
        open_button.set_tooltip_text(_("Import image"))
        open_button.set_action_name("win.import")
        self.append(open_button)

        self.export_button = Gtk.Button(child=get_icon("export-symbolic"))
        self.export_button.set_tooltip_text(_("Generate G-code"))
        self.export_button.set_action_name("win.export")
        self.append(self.export_button)

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

        self.recalculate_button = Gtk.Button(
            child=get_icon("view-refresh-symbolic"),
        )
        self.recalculate_button.set_tooltip_text(
            _("Recalculate (Shift+Click to force)")
        )
        self.recalculate_button.connect(
            "clicked", self._on_recalculate_clicked
        )
        recalc_gesture = Gtk.GestureClick.new()
        recalc_gesture.set_propagation_phase(Gtk.PropagationPhase.CAPTURE)
        recalc_gesture.connect("pressed", self._on_recalculate_pressed)
        self.recalculate_button.add_controller(recalc_gesture)
        self.append(self.recalculate_button)
        self._recalculate_force = False

        # Add a button to toggle the control panel.
        self.bottom_panel_button = Gtk.ToggleButton()
        self.bottom_panel_button.set_child(get_icon("jog-symbolic"))
        self.bottom_panel_button.set_active(False)
        self.bottom_panel_button.set_tooltip_text(_("Toggle bottom panel"))
        self.bottom_panel_button.set_action_name("win.toggle_bottom_panel")
        self.append(self.bottom_panel_button)

        # Arrangement buttons (Consolidated Dropdown)
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.arrange_actions = self._build_arrange_actions()
        self.arrange_menu_button = SplitMenuButton(
            actions=self.arrange_actions
        )
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

        self.focus_on_icon = get_icon("laser-on-symbolic")
        self.focus_off_icon = get_icon("laser-off-symbolic")
        self.focus_button = Gtk.ToggleButton()
        self.focus_button.set_child(self.focus_on_icon)
        self.focus_button.set_tooltip_text(_("Toggle focus laser"))
        self.focus_button.set_action_name("win.toggle-focus")
        self.focus_button.connect("toggled", self._on_focus_toggled)
        self.append(self.focus_button)

        # Add clickable warning for misconfigured machine
        self.machine_warning_box = Gtk.Box(spacing=6)
        self.machine_warning_box.set_margin_end(12)
        warning_icon = get_icon("warning-symbolic")
        self.warning_label = Gtk.Label(label=_("Machine not fully configured"))
        self.warning_label.add_css_class("warning-label")
        self.machine_warning_box.append(warning_icon)
        self.machine_warning_box.append(self.warning_label)
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

        # Connect to action registry changes for dynamic toolbar updates
        action_registry.changed.connect(self._on_action_registry_changed)

    def _on_recalculate_pressed(self, gesture, n_press, x, y):
        self._recalculate_force = bool(
            gesture.get_current_event_state() & Gdk.ModifierType.SHIFT_MASK
        )

    def _on_recalculate_clicked(self, button):
        force = self._recalculate_force
        self._recalculate_force = False
        action_name = "win.force-recalculate" if force else "win.recalculate"
        self.activate_action(action_name, None)

    def _build_arrange_actions(self):
        """Build the list of arrange actions including registered layouts."""
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
        ]
        for info in action_registry.get_toolbar_items("arrange"):
            if info.label:
                icon = info.icon_name or "auto-layout-symbolic"
                arrange_actions.append(
                    (
                        info.label,
                        icon,
                        f"win.{info.action_name}",
                    )
                )
        return arrange_actions

    def _on_action_registry_changed(self, sender):
        """Handle action registry changes by refreshing arrange menu."""
        self.arrange_actions = self._build_arrange_actions()
        self.arrange_menu_button.update_actions(self.arrange_actions)

    def _on_focus_toggled(self, button: Gtk.ToggleButton):
        """Callback to update the focus icon when the button's
        state changes for any reason (user click or action state change)."""
        if button.get_active():
            button.set_child(self.focus_off_icon)
        else:
            button.set_child(self.focus_on_icon)

    def set_machine_warning(
        self, error_title: str, error_code: int, error_description: str
    ):
        """
        Update the machine warning label with title, code and description.
        """
        self.warning_label.set_label(f"{error_title} ({error_code})")
        self.machine_warning_box.set_tooltip_text(error_description)
