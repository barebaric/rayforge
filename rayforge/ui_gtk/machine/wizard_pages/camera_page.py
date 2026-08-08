"""Step 10 — Camera setup (optional).

Collects which V4L devices the user wants attached to this machine
and launches the per-device camera wizard for the full setup
(detection, image settings, lens calibration, world alignment):

* Detect V4L devices
* Pick a camera + resolution
* Image settings (WB, brightness, contrast, denoise, transparency)
* Calibrate lens (Charuco frames, OpenCV solvePnP, de-distortion) —
  optional
* Align image ↔ world point pairs

Here we only collect which V4L devices the user wants enabled; the
detailed per-device setup runs on demand. If the user opts out, no
`cameras` entry is written and the existing machine-level default
applies.
"""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw

from ....camera.models.camera import Camera
from ....camera.v4l import display_name, get_sorted_by_id_paths
from ....machine.device.profile import DeviceProfile
from . import WizardPage, _makePreferencesGroup


class CameraPage(WizardPage):
    step_number = 10
    title = _("Cameras")
    subtitle = _(
        "Optional. Configure any cameras you want to use for "
        "preview and alignment."
    )

    def __init__(self, wizard, **kwargs):
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.cameras_group = _makePreferencesGroup(
            title=_("Cameras"),
            description=_(
                "Set up cameras now or do it later from machine "
                "settings. The wizard records which V4L devices you "
                "mark as 'enabled'; detailed lens calibration is "
                "performed on the camera settings page."
            ),
        )
        self.content.append(self.cameras_group)

        # Each row is a 2-tuple of (substr_for_by_id_path, switch_row).
        self._device_id_for_row: dict[int, str] = {}
        # Holds both real SwitchRows and the single empty-state
        # ActionRow shown when no cameras are detected.
        self._switch_rows: list[Adw.PreferencesRow] = []
        # Detection is deferred to `enter()` so we re-scan each time
        # the page is shown (USB cameras may have been plugged in
        # since the wizard was opened).

        self.set_ready(True)

    def selected_device_ids(self) -> list[str]:
        """Device IDs the user has enabled on this page."""
        ids: list[str] = []
        for row in self._switch_rows:
            if not isinstance(row, Adw.SwitchRow):
                continue
            if not row.get_active():
                continue
            by_id = self._device_id_for_row.get(id(row))
            if by_id:
                ids.append(by_id)
        return ids

    def enter(self, profile: DeviceProfile) -> None:
        # Clear previous rows
        for row in self._switch_rows:
            self.cameras_group.remove(row)
        self._switch_rows.clear()
        self._device_id_for_row.clear()

        try:
            by_id_paths = get_sorted_by_id_paths()
        except Exception:
            by_id_paths = []

        if not by_id_paths:
            empty = Adw.ActionRow(
                title=_("No cameras detected"),
                subtitle=_("You can add cameras later from machine settings."),
            )
            self.cameras_group.add(empty)
            self._switch_rows.append(empty)
            return

        already_added = {
            (c.get("device_id"))
            for c in (profile.machine_config.cameras or [])
            if c.get("device_id")
        }

        for by_id in by_id_paths:
            row = Adw.SwitchRow(
                title=display_name(by_id),
                subtitle=by_id,
            )
            row.set_active(by_id in already_added)
            self.cameras_group.add(row)
            self._switch_rows.append(row)
            self._device_id_for_row[id(row)] = by_id

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        selected: list[dict[str, Any]] = []
        for by_id in self.selected_device_ids():
            cam = Camera(name=display_name(by_id), device_id=by_id)
            cam.enabled = True
            selected.append(cam.to_dict())
        profile.machine_config.cameras = selected or None
        return True


__all__ = ["CameraPage"]
