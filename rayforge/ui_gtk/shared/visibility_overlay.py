from gettext import gettext as _

from gi.repository import Gdk, Gtk

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
        show_grid=False,
        show_tabs=False,
        show_ops_underlay=False,
        show_stock=False,
        show_workpiece_image=False,
        show_nogo_zones=True,
        shortcuts=None,
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
        self._shortcuts = shortcuts or {}

        self._vis_on_icon = get_icon("visibility-on-symbolic")
        self._vis_off_icon = get_icon("visibility-off-symbolic")

        if show_workpiece:
            self.workpiece_button = Gtk.ToggleButton()
            self.workpiece_button.set_active(True)
            self.workpiece_button.set_child(self._vis_on_icon)
            self.workpiece_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle workpiece visibility"),
                    "win.show_workpieces",
                )
            )
            self.workpiece_button.set_action_name("win.show_workpieces")
            self.workpiece_button.connect(
                "toggled", self._on_workpiece_toggled
            )
            self.append(self.workpiece_button)

        self.workpiece_image_button = None
        if show_workpiece_image:
            self.workpiece_image_button = Gtk.ToggleButton()
            self.workpiece_image_button.set_active(True)
            self.workpiece_image_button.set_child(self._vis_on_icon)
            self.workpiece_image_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle workpiece image visibility"),
                    "win.show_workpiece_image",
                )
            )
            self.workpiece_image_button.set_action_name(
                "win.show_workpiece_image"
            )
            self.workpiece_image_button.connect(
                "toggled", self._on_workpiece_toggled
            )
            self.append(self.workpiece_image_button)

        self.stock_button: Gtk.ToggleButton | None = None
        if show_stock:
            self.stock_button = Gtk.ToggleButton()
            self.stock_button.set_child(get_icon("stock-symbolic"))
            self.stock_button.set_active(True)
            self.stock_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle stock visibility"), "win.show_stock"
                )
            )
            self.stock_button.set_action_name("win.show_stock")
            self.append(self.stock_button)

        if show_tabs:
            self.tabs_button = Gtk.ToggleButton()
            self.tabs_button.set_child(get_icon("tabs-visible-symbolic"))
            self.tabs_button.set_active(True)
            self.tabs_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle tab visibility"), "win.show_tabs"
                )
            )
            self.tabs_button.set_action_name("win.show_tabs")
            self.append(self.tabs_button)

        self._cam_on_icon = get_icon("camera-on-symbolic")
        self._cam_off_icon = get_icon("camera-off-symbolic")
        self.camera_button = Gtk.ToggleButton()
        self.camera_button.set_active(True)
        self.camera_button.set_child(self._cam_on_icon)
        self.camera_button.set_tooltip_text(
            self._format_tooltip(
                _("Toggle camera image visibility"),
                "win.toggle_camera_view",
            )
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
                self._format_tooltip(
                    _("Toggle 3D model visibility"), "win.show_models"
                )
            )
            self.models_button.set_action_name("win.show_models")
            self.append(self.models_button)

        if show_grid:
            self.grid_button = Gtk.ToggleButton()
            self.grid_button.set_child(get_icon("sketch-grid-symbolic"))
            self.grid_button.set_active(True)
            self.grid_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle grid visibility"), "win.show_grid"
                )
            )
            self.grid_button.set_action_name("win.show_grid")
            self.append(self.grid_button)

        self.underlay_button: Gtk.ToggleButton | None = None
        if show_ops_underlay:
            self.underlay_button = Gtk.ToggleButton()
            self.underlay_button.set_child(get_icon("ops-underlay-symbolic"))
            self.underlay_button.set_active(True)
            self.underlay_button.set_tooltip_text(
                self._format_tooltip(
                    _("Toggle ops underlay visibility"),
                    "win.show_ops_underlay",
                )
            )
            self.underlay_button.set_action_name("win.show_ops_underlay")
            self.append(self.underlay_button)

        self.travel_button = Gtk.ToggleButton()
        self.travel_button.set_child(get_icon("travel-path-symbolic"))
        self.travel_button.set_active(False)
        self.travel_button.set_tooltip_text(
            self._format_tooltip(
                _("Toggle travel move visibility"),
                "win.toggle_travel_view",
            )
        )
        self.travel_button.set_action_name("win.toggle_travel_view")
        self.append(self.travel_button)

        self.nogo_button = Gtk.ToggleButton()
        self.nogo_button.set_child(get_icon("block-symbolic"))
        self.nogo_button.set_active(True)
        self.nogo_button.set_tooltip_text(
            self._format_tooltip(
                _("Toggle no-go zone visibility"), "win.show_nogo_zones"
            )
        )
        self.nogo_button.set_action_name("win.show_nogo_zones")
        self.nogo_button.set_visible(show_nogo_zones)
        self.append(self.nogo_button)

    def set_camera_visible(self, visible: bool):
        self.camera_button.set_visible(visible)

    def set_nogo_visible(self, visible: bool):
        self.nogo_button.set_visible(visible)

    def set_stock_present(self, present: bool) -> None:
        """Shows the stock toggle only when a stock exists in the doc.

        The stock and ops-underlay toggles are mutually exclusive: a
        burned stock replaces the floating ops quads, so showing the
        underlay toggle next to a stock is redundant.
        """
        if self.stock_button is not None:
            self.stock_button.set_visible(present)
        self.set_ops_underlay_visible(not present)

    def set_ops_underlay_visible(self, visible: bool) -> None:
        """Shows/hides the ops-underlay toggle button."""
        if self.underlay_button is not None:
            self.underlay_button.set_visible(visible)

    def _format_tooltip(self, text, action_name):
        if action_name in self._shortcuts:
            shortcut_str = self._shortcuts[action_name]
            trigger = Gtk.ShortcutTrigger.parse_string(shortcut_str)
            if trigger is None:
                return text
            display = Gdk.Display.get_default()
            if display is not None:
                label = trigger.to_label(display)
                if label is not None:
                    return f"{text} ({label})"
        return text

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
