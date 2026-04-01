from gettext import gettext as _
from gi.repository import Gtk

from ..icons import get_icon
from .gtk import apply_css


css = """
.visibility-overlay {
    background-color: alpha(@theme_bg_color, 0.75);
    border-radius: 6px;
    padding: 3px;
}
.visibility-overlay button {
    min-width: 28px;
    min-height: 28px;
    padding: 0;
}
"""


class VisibilityOverlay(Gtk.Box):
    """
    A row of visibility toggle buttons meant to be placed as an overlay
    on top of a canvas widget.
    """

    def __init__(
        self,
        show_workpiece=True,
        show_camera=False,
        show_models=False,
        **kwargs,
    ):
        super().__init__(
            orientation=Gtk.Orientation.HORIZONTAL,
            spacing=2,
            **kwargs,
        )
        apply_css(css)
        self.add_css_class("visibility-overlay")
        self.set_halign(Gtk.Align.END)
        self.set_valign(Gtk.Align.START)
        self.set_margin_top(6)
        self.set_margin_end(6)

        if show_workpiece:
            self._vis_on_icon = get_icon("visibility-on-symbolic")
            self._vis_off_icon = get_icon("visibility-off-symbolic")
            self.workpiece_button = Gtk.ToggleButton()
            self.workpiece_button.set_active(True)
            self.workpiece_button.set_child(self._vis_on_icon)
            self.workpiece_button.set_tooltip_text(
                _("Toggle workpiece visibility")
            )
            self.workpiece_button.set_action_name("win.show_workpieces")
            self.workpiece_button.connect(
                "toggled", self._on_workpiece_toggled
            )
            self.append(self.workpiece_button)

        self._cam_on_icon = get_icon("camera-on-symbolic")
        self._cam_off_icon = get_icon("camera-off-symbolic")
        self.camera_button = Gtk.ToggleButton()
        self.camera_button.set_active(True)
        self.camera_button.set_child(self._cam_on_icon)
        self.camera_button.set_tooltip_text(
            _("Toggle camera image visibility")
        )
        self.camera_button.set_action_name("win.toggle_camera_view")
        self.camera_button.connect("toggled", self._on_camera_toggled)
        self.append(self.camera_button)
        self.camera_button.set_visible(show_camera)

        if show_models:
            self.models_button = Gtk.ToggleButton()
            self.models_button.set_child(get_icon("model-symbolic"))
            self.models_button.set_active(True)
            self.models_button.set_tooltip_text(
                _("Toggle 3D model visibility")
            )
            self.models_button.set_action_name("win.show_models")
            self.append(self.models_button)

        self.travel_button = Gtk.ToggleButton()
        self.travel_button.set_child(get_icon("travel-path-symbolic"))
        self.travel_button.set_active(False)
        self.travel_button.set_tooltip_text(_("Toggle travel move visibility"))
        self.travel_button.set_action_name("win.toggle_travel_view")
        self.append(self.travel_button)

        self.nogo_button = Gtk.ToggleButton()
        self.nogo_button.set_child(get_icon("block-symbolic"))
        self.nogo_button.set_active(True)
        self.nogo_button.set_tooltip_text(_("Toggle no-go zone visibility"))
        self.nogo_button.set_action_name("win.show_nogo_zones")
        self.append(self.nogo_button)

    def set_camera_visible(self, visible: bool):
        self.camera_button.set_visible(visible)

    def _on_workpiece_toggled(self, button):
        if button.get_active():
            button.set_child(self._vis_on_icon)
        else:
            button.set_child(self._vis_off_icon)

    def _on_camera_toggled(self, button):
        if button.get_active():
            button.set_child(self._cam_on_icon)
        else:
            button.set_child(self._cam_off_icon)
