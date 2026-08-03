"""Reusable manual lens-distortion coefficient controls.

Composed by both :class:`LensCalibrationDialog` and the camera
wizard's manual lens-calibration page so the two stay in sync.
"""

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

logger = logging.getLogger(__name__)

_DISTORTION_FIELDS = [
    ("distortion_k1", _("Radial 1 (k1)"), _("First order radial distortion")),
    ("distortion_k2", _("Radial 2 (k2)"), _("Second order radial distortion")),
    ("distortion_k3", _("Radial 3 (k3)"), _("Third order radial distortion")),
    (
        "distortion_p1",
        _("Tangential 1 (p1)"),
        _("First order tangential distortion"),
    ),
    (
        "distortion_p2",
        _("Tangential 2 (p2)"),
        _("Second order tangential distortion"),
    ),
]


class LensCalibrationWidget(Gtk.Box):
    """Preferences group for entering lens-distortion coefficients."""

    def __init__(self, camera, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)
        self.camera = camera
        self._distortion_rows = {}
        self._updating_ui = False

        group = Adw.PreferencesGroup(
            title=_("Lens Calibration"),
            description=_(
                "Correct lens distortion for straighter lines. "
                "Adjust the coefficients manually."
            ),
        )
        self.append(group)

        for key, title, subtitle in _DISTORTION_FIELDS:
            row = self._create_spin_row(
                title, subtitle, getattr(self.camera, key), key
            )
            self._distortion_rows[key] = row
            group.add(row)

        self.camera.settings_changed.connect(self._on_camera_settings_changed)

    def _create_spin_row(
        self, title: str, subtitle: str, value: float, config_key: str
    ) -> Adw.SpinRow:
        row = Adw.SpinRow(
            title=title,
            subtitle=subtitle,
            adjustment=Gtk.Adjustment(
                value=value,
                lower=-10.0,
                upper=10.0,
                step_increment=0.001,
                page_increment=0.01,
            ),
            digits=4,
            numeric=True,
        )
        row.connect(
            "notify::value",
            self._on_distortion_value_changed,
            config_key,
        )
        return row

    def _on_camera_settings_changed(self, camera) -> None:
        if self._updating_ui:
            return
        self._updating_ui = True
        try:
            for key, row in self._distortion_rows.items():
                row.set_value(getattr(camera, key))
        finally:
            self._updating_ui = False

    def _on_distortion_value_changed(
        self, spin_row: Adw.SpinRow, pspec, config_key: str
    ) -> None:
        if self._updating_ui:
            return
        setattr(self.camera, config_key, spin_row.get_value())

    def stop(self) -> None:
        self.camera.settings_changed.disconnect(
            self._on_camera_settings_changed
        )


__all__ = ["LensCalibrationWidget"]
