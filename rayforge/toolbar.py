import logging
from gi.repository import Gtk  # type: ignore
from .icons import get_icon
from .undo.ui.undo_button import UndoButton, RedoButton
from .machine.ui.machine_selector import MachineSelector


logger = logging.getLogger(__name__)


class MainToolbar(Gtk.Box):
    """
    The main application toolbar, containing buttons for file operations,
    editing, alignment, and machine control.
    """

    def __init__(self, *, action_handler, **kwargs):
        """
        Initializes the MainToolbar.

        :param action_handler: An object that will handle all the button
                               click events and actions. It is expected to
                               have methods like `on_open_clicked`,
                               `on_export_clicked`, etc.
        """
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL, spacing=6, **kwargs
        )

        self.action_handler = action_handler

        self.set_margin_bottom(2)
        self.set_margin_top(2)
        self.set_margin_start(12)
        self.set_margin_end(12)

        # --- Import and export icons ---
        open_button = Gtk.Button()
        open_button.set_child(get_icon("document-open-symbolic"))
        open_button.set_tooltip_text(_("Import image"))
        open_button.connect("clicked", self.action_handler.on_open_clicked)
        self.append(open_button)

        self.export_button = Gtk.Button()
        self.export_button.set_child(get_icon("document-save-symbolic"))
        self.export_button.set_tooltip_text(_("Generate G-code"))
        self.export_button.connect(
            "clicked", self.action_handler.on_export_clicked
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
        clear_button.connect("clicked", self.action_handler.on_clear_clicked)
        self.append(clear_button)

        self.visibility_on_icon = get_icon("stock-eye-symbolic")
        self.visibility_off_icon = get_icon("eye-not-looking-symbolic")
        visibility_button = Gtk.ToggleButton()
        visibility_button.set_active(True)
        visibility_button.set_child(self.visibility_on_icon)
        visibility_button.set_tooltip_text(_("Toggle workpiece visibility"))
        visibility_button.connect(
            "clicked", self.action_handler.on_button_visibility_clicked
        )
        self.append(visibility_button)

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
            "toggled", self.action_handler.on_camera_image_visibility_toggled
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
            "toggled", self.action_handler.on_show_travel_toggled
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
            "clicked", self.action_handler.on_align_h_center_clicked
        )
        self.append(self.align_h_center_button)

        self.align_v_center_button = Gtk.Button()
        self.align_v_center_button.set_child(
            get_icon("align-vertical-center-symbolic")
        )
        self.align_v_center_button.set_tooltip_text(_("Center Vertically"))
        self.align_v_center_button.connect(
            "clicked", self.action_handler.on_align_v_center_clicked
        )
        self.append(self.align_v_center_button)

        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.align_left_button = Gtk.Button()
        self.align_left_button.set_child(get_icon("align-left-symbolic"))
        self.align_left_button.set_tooltip_text(_("Align Left"))
        self.align_left_button.connect(
            "clicked", self.action_handler.on_align_left_clicked
        )
        self.append(self.align_left_button)

        self.align_right_button = Gtk.Button()
        self.align_right_button.set_child(get_icon("align-right-symbolic"))
        self.align_right_button.set_tooltip_text(_("Align Right"))
        self.align_right_button.connect(
            "clicked", self.action_handler.on_align_right_clicked
        )
        self.append(self.align_right_button)

        self.align_top_button = Gtk.Button()
        self.align_top_button.set_child(get_icon("align-top-symbolic"))
        self.align_top_button.set_tooltip_text(_("Align Top"))
        self.align_top_button.connect(
            "clicked", self.action_handler.on_align_top_clicked
        )
        self.append(self.align_top_button)

        self.align_bottom_button = Gtk.Button()
        self.align_bottom_button.set_child(get_icon("align-bottom-symbolic"))
        self.align_bottom_button.set_tooltip_text(_("Align Bottom"))
        self.align_bottom_button.connect(
            "clicked", self.action_handler.on_align_bottom_clicked
        )
        self.append(self.align_bottom_button)

        # --- Control buttons: home, send, pause, stop ---
        sep = Gtk.Separator(orientation=Gtk.Orientation.VERTICAL)
        self.append(sep)

        self.home_button = Gtk.Button()
        self.home_button.set_child(get_icon("go-home-symbolic"))
        self.home_button.set_tooltip_text(_("Home the machine"))
        self.home_button.connect(
            "clicked", self.action_handler.on_home_clicked
        )
        self.append(self.home_button)

        self.frame_button = Gtk.Button()
        self.frame_button.set_child(get_icon("edit-select-all-symbolic"))
        self.frame_button.set_tooltip_text(
            _("Cycle laser head around the occupied area")
        )
        self.frame_button.connect(
            "clicked", self.action_handler.on_frame_clicked
        )
        self.append(self.frame_button)

        self.send_button = Gtk.Button()
        self.send_button.set_child(get_icon("document-send-symbolic"))
        self.send_button.set_tooltip_text(_("Send to machine"))
        self.send_button.connect(
            "clicked", self.action_handler.on_send_clicked
        )
        self.append(self.send_button)

        self.hold_on_icon = get_icon("pause-symbolic")
        self.hold_off_icon = get_icon("pause-symbolic")
        self.hold_button = Gtk.ToggleButton()
        self.hold_button.set_child(self.hold_off_icon)
        self.hold_button.set_tooltip_text(_("Pause machine"))
        self.hold_button.connect(
            "clicked", self.action_handler.on_hold_clicked
        )
        self.append(self.hold_button)

        self.cancel_button = Gtk.Button()
        self.cancel_button.set_child(get_icon("process-stop-symbolic"))
        self.cancel_button.set_tooltip_text(_("Cancel running job"))
        self.cancel_button.connect(
            "clicked", self.action_handler.on_cancel_clicked
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
            "pressed", self.action_handler.on_machine_warning_clicked
        )
        self.machine_warning_box.add_controller(warning_click)
        self.append(self.machine_warning_box)

        # --- Add machine selector dropdown ---
        self.machine_selector = MachineSelector()
        self.machine_selector.machine_selected.connect(
            self.action_handler.on_machine_selected_by_selector
        )
        self.append(self.machine_selector)
