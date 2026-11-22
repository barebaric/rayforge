"""Step 10 — Camera setup (optional).

Collects which cameras the user wants attached to this machine and
launches the per-device camera wizard for the full local setup
(detection, image settings, lens calibration, world alignment):

* Detect V4L devices
* Pick a camera + resolution
* Image settings (WB, brightness, contrast, denoise, transparency)
* Calibrate lens (Charuco frames, OpenCV solvePnP, de-distortion) —
  optional
* Align image ↔ world point pairs

For a remote machine, cameras advertised by the device host are
listed beside the local ones. Their stream URLs are network cameras,
so no local calibration workflow is launched for them.

Here we only collect which cameras the user wants enabled; the
detailed per-device setup runs on demand. If the user opts out, no
`cameras` entry is written and the existing machine-level default
applies.
"""

from gettext import gettext as _
from typing import Any

from gi.repository import Adw

from ....camera.models.camera import Camera
from ....camera.remote import (
    RemoteCamera,
    fetch_remote_cameras,
    is_remote_driver_args,
)
from ....camera.v4l import display_name, get_sorted_by_id_paths
from ....machine.device.profile import DeviceProfile
from ....shared.tasker import task_mgr
from . import WizardPage, _makePreferencesGroup


class CameraPage(WizardPage):
    step_number = 11
    title = _("Cameras")
    subtitle = _(
        "Optional. Configure any cameras you want to use for "
        "preview and alignment."
    )

    def __init__(self, wizard, **kwargs):
        self._remote_cameras: list[RemoteCamera] = []
        self._remote_switch_rows: dict[
            int, tuple[Adw.SwitchRow, RemoteCamera]
        ] = {}
        self._remote_status_row: Adw.ActionRow | None = None
        self._fetch_generation = 0
        super().__init__(wizard, **kwargs)

    def build_ui(self) -> None:
        self.cameras_group = _makePreferencesGroup(
            title=_("Cameras"),
            description=_(
                "Set up cameras now or do it later from machine "
                "settings. The wizard records which cameras you mark "
                "as 'enabled'; detailed lens calibration is performed "
                "on the camera settings page."
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

    def selected_remote_cameras(self) -> list[RemoteCamera]:
        """Daemon-advertised cameras the user has enabled."""
        return [
            camera
            for row, camera in self._remote_switch_rows.values()
            if row.get_active()
        ]

    def enter(self, profile: DeviceProfile) -> None:
        self._clear_rows()
        self._fetch_generation += 1
        generation = self._fetch_generation
        self._remote_cameras = []
        self._remote_switch_rows = {}

        try:
            by_id_paths = get_sorted_by_id_paths()
        except OSError:
            by_id_paths = []

        existing_device_ids = {
            c.get("device_id")
            for c in (profile.machine_config.cameras or [])
            if c.get("device_id")
        }
        for by_id in by_id_paths:
            row = Adw.SwitchRow(
                title=display_name(by_id),
                subtitle=by_id,
            )
            row.set_active(by_id in existing_device_ids)
            self.cameras_group.add(row)
            self._switch_rows.append(row)
            self._device_id_for_row[id(row)] = by_id

        driver_args = profile.machine_config.driver_args or {}
        if is_remote_driver_args(driver_args):
            self._start_remote_fetch(driver_args, profile, generation)
        elif not self._switch_rows:
            self._add_empty_state()

    def _clear_rows(self) -> None:
        for row in self._switch_rows:
            self.cameras_group.remove(row)
        self._switch_rows.clear()
        self._device_id_for_row.clear()
        if self._remote_status_row is not None:
            self.cameras_group.remove(self._remote_status_row)
            self._remote_status_row = None

    def _add_empty_state(self) -> None:
        empty = Adw.ActionRow(
            title=_("No cameras detected"),
            subtitle=_("You can add cameras later from machine settings."),
        )
        self.cameras_group.add(empty)
        self._switch_rows.append(empty)

    def _start_remote_fetch(
        self,
        driver_args: dict[str, Any],
        profile: DeviceProfile,
        generation: int,
    ) -> None:
        existing_uris = {
            c.get("source_config", {}).get("uri")
            for c in (profile.machine_config.cameras or [])
            if isinstance(c.get("source_config"), dict)
        }
        existing_uris.discard(None)
        self._remote_status_row = Adw.ActionRow(
            title=_("Looking for cameras on the device host…")
        )
        self.cameras_group.add(self._remote_status_row)

        async def _coroutine(exec_ctx: Any) -> list[RemoteCamera]:
            return await fetch_remote_cameras(driver_args)

        task_mgr.add_coroutine(
            _coroutine,
            key=f"camera-page-remote-{id(self)}",
            when_done=lambda task: self._on_remote_fetched(
                task, generation, existing_uris
            ),
        )

    def _on_remote_fetched(
        self, task: Any, generation: int, existing_uris: set[Any]
    ) -> None:
        def _update() -> None:
            if generation != self._fetch_generation:
                return
            if self._remote_status_row is not None:
                self.cameras_group.remove(self._remote_status_row)
                self._remote_status_row = None
            cameras: list[RemoteCamera] = []
            if task.get_status() == "completed":
                cameras = task.result()
            self._remote_cameras = cameras
            for camera in cameras:
                row = Adw.SwitchRow(
                    title=camera.name,
                    subtitle=camera.uri,
                )
                row.set_active(camera.uri in existing_uris)
                self.cameras_group.add(row)
                self._switch_rows.append(row)
                self._remote_switch_rows[id(row)] = (row, camera)
            if not self._switch_rows:
                self._add_empty_state()

        task_mgr.schedule_on_main_thread(_update)

    def apply_to_profile(self, profile: DeviceProfile) -> bool:
        selected: list[dict[str, Any]] = []
        for by_id in self.selected_device_ids():
            cam = Camera(name=display_name(by_id), device_id=by_id)
            cam.enabled = True
            selected.append(cam.to_dict())
        for remote in self.selected_remote_cameras():
            cam = Camera(
                name=remote.name,
                source_type=remote.source_type,
                source_config={"uri": remote.uri},
            )
            cam.enabled = True
            selected.append(cam.to_dict())
        profile.machine_config.cameras = selected or None
        return True


__all__ = ["CameraPage"]
