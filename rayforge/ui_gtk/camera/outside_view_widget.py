import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...camera.controller import CameraController
from ...camera.models.camera import OUTSIDE_VIEW_MAX_MARGIN_MM, Camera
from ..shared.pref_rows import LengthSpinRow
from ..shared.slider import create_slider_row

logger = logging.getLogger(__name__)


class CameraOutsideViewGroup(Adw.PreferencesGroup):
    """Settings for showing the camera image around the workspace."""

    def __init__(self, **kwargs):
        super().__init__(
            title=_("View Outside Workspace"),
            description=_(
                "Show the camera image in a margin around the work area "
                "to help position workpieces. Requires image alignment."
            ),
            **kwargs,
        )
        self._camera: Camera | None = None
        self._updating_ui: bool = False

        self.enabled_row = Adw.ActionRow(
            title=_("Show Outside Workspace"),
            subtitle=_("Draw the camera image around the work area"),
        )
        self.enabled_switch = Gtk.Switch()
        self.enabled_switch.set_valign(Gtk.Align.CENTER)
        self.enabled_switch.connect("notify::active", self._on_enabled_changed)
        self.enabled_row.add_suffix(self.enabled_switch)
        self.enabled_row.set_activatable_widget(self.enabled_switch)
        self.add(self.enabled_row)

        self.margin_row = LengthSpinRow(
            _("Margin"),
            _("Size of the visible area on each side of the work area"),
            lower=0.0,
            upper=OUTSIDE_VIEW_MAX_MARGIN_MM,
            step_increment=1.0,
            digits=0,
            numeric=True,
        )
        self.margin_row.value_changed.connect(self._on_margin_changed)
        self.add(self.margin_row)

        self.transparency_adjustment = Gtk.Adjustment(
            value=0.0,
            lower=0.0,
            upper=1.0,
            step_increment=0.01,
            page_increment=0.1,
        )
        self.transparency_row, self.transparency_scale = create_slider_row(
            title=_("Transparency"),
            subtitle=_("Transparency outside the work area"),
            adjustment=self.transparency_adjustment,
            digits=2,
            on_value_changed=self._on_transparency_changed,
        )
        self.add(self.transparency_row)

        self.set_controller(None)

    def set_controller(self, controller: CameraController | None):
        if self._camera:
            self._camera.changed.disconnect(self._on_camera_changed)

        self._camera = controller.config if controller else None

        if self._camera:
            self._camera.changed.connect(self._on_camera_changed)
            self.set_sensitive(True)
        else:
            self.set_sensitive(False)
        self.update_ui()

    def update_ui(self):
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            camera = self._camera
            enabled = bool(camera and camera.outside_view_enabled)
            self.enabled_switch.set_active(enabled)
            self.margin_row.set_sensitive(enabled)
            self.transparency_row.set_sensitive(enabled)
            if camera:
                self.margin_row.set_value_in_base_units(
                    camera.outside_view_margin_mm
                )
                self.transparency_adjustment.set_value(
                    camera.outside_view_transparency
                )
            self._update_alignment_hint()
        finally:
            self._updating_ui = False

    def _update_alignment_hint(self):
        camera = self._camera
        if camera and not camera.has_alignment:
            self.enabled_row.set_subtitle(
                _("Not shown until the camera image is aligned")
            )
        else:
            self.enabled_row.set_subtitle(
                _("Draw the camera image around the work area")
            )

    def _on_camera_changed(self, camera, *args):
        self.update_ui()

    def _on_enabled_changed(self, switch, _pspec):
        if self._updating_ui or not self._camera:
            return
        self._camera.outside_view_enabled = switch.get_active()

    def _on_margin_changed(self, row):
        if self._updating_ui or not self._camera:
            return
        self._camera.outside_view_margin_mm = row.get_value_in_base_units()

    def _on_transparency_changed(self, scale):
        if self._updating_ui or not self._camera:
            return
        self._camera.outside_view_transparency = scale.get_value()
