import logging
from gi.repository import Gtk  # type: ignore
from blinker import Signal
from .icons import get_icon
from .undo.ui.undo_button import UndoButton, RedoButton
from .machine.ui.machine_selector import MachineSelector
from .splitbutton import SplitMenuButton

logger = logging.getLogger(__name__)


class MainToolbar(Gtk.Box):
    """
    The main application toolbar.
    """

    def __init__(self, **kwargs):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, **kwargs
        )

        self.open_clicked = Signal()
        self.export_clicked = Signal()
        self.clear_clicked = Signal()
        self.visibility_toggled = Signal()
        self.camera_visibility_toggled = Signal()
        self.show_travel_toggled = Signal()
        self.center_horizontally_clicked = Signal()
        self.center_vertically_clicked = Signal()
        self.align_left_clicked = Signal()
        self.align_right_clicked = Signal()
        self.align_top_clicked = Signal()
        self.align_bottom_clicked = Signal()
        self.spread_horizontally_clicked = Signal()
        self.spread_vertically_clicked = Signal()
        self.home_clicked = Signal()
        self.frame_clicked = Signal()
        self.send_clicked = Signal()
        self.hold_toggled = Signal()
        self.cancel_clicked = Signal()
        self.machine_warning_clicked = Signal()

        self.set_margin_bottom(2)
        self.set_margin_top(2)
        self.set_margin_start(12)
        self.set_margin_end(12)

        # --- Import and export icons ---
        open_button = Gtk.Button()
        open_button.set_child(get_icon("document-open-symbolic"))
        open_button.set_tooltip_text(_("Import image"))
        open_button.connect("clicked", lambda _: self.open_clicked.send(self))
        self.append(open_button)

        self.export_button = Gtk.Button()
        self.export_button.set_child(get_icon("document-save-symbolic"))
        self.export_button.set_tooltip_text(_("Generate G-code"))
        self.export_button.connect(
            "clicked", lambda _: self.export_clicked.send(self)
        )
        self.append(self.export_button)

        # --- Undo/Redo Buttons ---
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.undo_button = UndoButton()
        self.append(self.undo_button)

        self.redo_button = RedoButton()
        self.append(self.redo_button)

        # --- Clear and visibility ---
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        clear_button = Gtk.Button()
        clear_button.set_child(get_icon("edit-clear-all-symbolic"))
        clear_button.set_tooltip_text(_("Remove all workpieces"))
        clear_button.connect(
            "clicked", lambda _: self.clear_clicked.send(self)
        )
        self.append(clear_button)

        self.visibility_on_icon = get_icon("stock-eye-symbolic")
        self.visibility_off_icon = get_icon("eye-not-looking-symbolic")
        self.visibility_button = Gtk.ToggleButton()
        self.visibility_button.set_active(True)
        self.visibility_button.set_child(self.visibility_on_icon)
        self.visibility_button.set_tooltip_text(
            _("Toggle workpiece visibility")
        )
        self.visibility_button.connect(
            "toggled",
            lambda btn: self.visibility_toggled.send(
                self, active=btn.get_active()
            ),
        )
        self.append(self.visibility_button)

        # --- Camera Image Visibility Toggle Button ---
        self.camera_visibility_on_icon = get_icon("camera-app-symbolic")
        self.camera_visibility_off_icon = get_icon("camera-disabled-symbolic")
        self.camera_visibility_button = Gtk.ToggleButton()
        self.camera_visibility_button.set_active(True)
        self.camera_visibility_button.set_child(self.camera_visibility_on_icon)
        self.camera_visibility_button.set_tooltip_text(
            _("Toggle camera image visibility")
        )
        self.camera_visibility_button.connect(
            "toggled",
            lambda btn: self.camera_visibility_toggled.send(
                self, active=btn.get_active()
            ),
        )
        self.append(self.camera_visibility_button)

        # --- Show Travel Moves Toggle Button ---
        self.show_travel_button = Gtk.ToggleButton()
        self.show_travel_button.set_child(get_icon("function-linear-symbolic"))
        self.show_travel_button.set_active(False)
        self.show_travel_button.set_tooltip_text(
            _("Toggle travel move visibility")
        )
        self.show_travel_button.connect(
            "toggled",
            lambda btn: self.show_travel_toggled.send(
                self, active=btn.get_active()
            ),
        )
        self.append(self.show_travel_button)

        # --- Align buttons ---
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.align_h_center_button = Gtk.Button()
        self.align_h_center_button.set_child(
            get_icon("align-horizontal-center-symbolic")
        )
        self.align_h_center_button.set_tooltip_text(_("Center Horizontally"))
        self.align_h_center_button.connect(
            "clicked", lambda _: self.center_horizontally_clicked.send(self)
        )
        self.append(self.align_h_center_button)

        self.align_v_center_button = Gtk.Button()
        self.align_v_center_button.set_child(
            get_icon("align-vertical-center-symbolic")
        )
        self.align_v_center_button.set_tooltip_text(_("Center Vertically"))
        self.align_v_center_button.connect(
            "clicked", lambda _: self.center_vertically_clicked.send(self)
        )
        self.append(self.align_v_center_button)

        # --- Align Edge buttons (Split Dropdown) ---
        align_actions = [
            (_("Align Left"), "align-left-symbolic", self.align_left_clicked),
            (
                _("Align Right"),
                "align-right-symbolic",
                self.align_right_clicked,
            ),
            (_("Align Top"), "align-top-symbolic", self.align_top_clicked),
            (
                _("Align Bottom"),
                "align-bottom-symbolic",
                self.align_bottom_clicked,
            ),
        ]
        self.align_menu_button = SplitMenuButton(actions=align_actions)
        self.append(self.align_menu_button)

        # --- Distribute buttons (Split Dropdown) ---
        distribute_actions = [
            (
                _("Spread Horizontally"),
                "distribute-horizontal-symbolic",
                self.spread_horizontally_clicked,
            ),
            (
                _("Spread Vertically"),
                "distribute-vertical-symbolic",
                self.spread_vertically_clicked,
            ),
        ]
        self.distribute_menu_button = SplitMenuButton(
            actions=distribute_actions
        )
        self.append(self.distribute_menu_button)

        # --- Control buttons: home, send, pause, stop ---
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.home_button = Gtk.Button()
        self.home_button.set_child(get_icon("go-home-symbolic"))
        self.home_button.set_tooltip_text(_("Home the machine"))
        self.home_button.connect(
            "clicked", lambda _: self.home_clicked.send(self)
        )
        self.append(self.home_button)

        self.frame_button = Gtk.Button()
        self.frame_button.set_child(get_icon("edit-select-all-symbolic"))
        self.frame_button.set_tooltip_text(
            _("Cycle laser head around the occupied area")
        )
        self.frame_button.connect(
            "clicked", lambda _: self.frame_clicked.send(self)
        )
        self.append(self.frame_button)

        self.send_button = Gtk.Button()
        self.send_button.set_child(get_icon("document-send-symbolic"))
        self.send_button.set_tooltip_text(_("Send to machine"))
        self.send_button.connect(
            "clicked", lambda _: self.send_clicked.send(self)
        )
        self.append(self.send_button)

        self.hold_on_icon = get_icon("pause-symbolic")
        self.hold_off_icon = get_icon("pause-symbolic")
        self.hold_button = Gtk.ToggleButton()
        self.hold_button.set_child(self.hold_off_icon)
        self.hold_button.set_tooltip_text(_("Pause machine"))
        self.hold_button.connect(
            "toggled",
            lambda btn: self.hold_toggled.send(self, active=btn.get_active()),
        )
        self.append(self.hold_button)

        self.cancel_button = Gtk.Button()
        self.cancel_button.set_child(get_icon("process-stop-symbolic"))
        self.cancel_button.set_tooltip_text(_("Cancel running job"))
        self.cancel_button.connect(
            "clicked", lambda _: self.cancel_clicked.send(self)
        )
        self.append(self.cancel_button)

        # --- Add spacer to push machine selector to the right ---
        spacer = Gtk.Box()
        spacer.set_hexpand(True)
        self.append(spacer)

        # --- Add clickable warning for misconfigured machine ---
        self.machine_warning_box = Gtk.Box(spacing=6)
        self.machine_warning_box.set_margin_end(12)
        warning_icon = Gtk.Image.new_from_icon_name("dialog-warning-symbolic")
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

        # --- Add machine selector dropdown ---
        self.machine_selector = MachineSelector()
        self.append(self.machine_selector)
