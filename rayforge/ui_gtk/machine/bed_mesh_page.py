"""Machine-settings page for probing and inspecting the bed mesh."""

from __future__ import annotations

import logging
from gettext import gettext as _

from gi.repository import Adw, Gtk

from ...machine.bed_probe import (
    BedProbeAborted,
    BedProbeError,
    probe_bed_mesh,
)
from ...machine.models.bed_mesh import BedMesh
from ...machine.models.machine import Machine
from ...shared.tasker import task_mgr
from ..shared.pref_rows.base import SpinRow
from ..shared.pref_rows.length_spin_row import LengthSpinRow
from ..shared.preferences_page import TrackedPreferencesPage
from .bed_mesh_view import BedMeshView

logger = logging.getLogger(__name__)

_PROBE_TASK_KEY = "bed-mesh-probe"

DEFAULT_FEED_RATE = 100.0
DEFAULT_MAX_TRAVEL = 10.0
DEFAULT_SAFE_Z = 5.0


def _estimate_duration(points: int, feed: float, travel: float) -> str:
    """Rough probe-cycle duration: plunge out and back at *feed*."""
    if points == 0 or feed <= 0:
        return "—"
    seconds = points * (2.0 * travel / (feed / 60.0))
    if seconds >= 90:
        return _("about {minutes:.0f} min").format(minutes=seconds / 60.0)
    return _("about {seconds:.0f} s").format(seconds=seconds)


class _ViewRow(Adw.PreferencesRow):
    """A bare preferences row hosting an arbitrary widget."""

    def __init__(self, child: Gtk.Widget):
        super().__init__()
        self.set_child(child)


class BedMeshPage(TrackedPreferencesPage):
    """Probe the bed surface on a grid and inspect the height map."""

    def __init__(self, machine: Machine, **kwargs):
        super().__init__(**kwargs)
        self.machine = machine
        self._heights: list[list[float | None]] = []
        self._initializing = True

        work = machine.work_area
        self._x0 = work[0]
        self._y0 = work[1]
        self._width = work[2]
        self._height = work[3]

        grid_group = Adw.PreferencesGroup(
            title=_("Probe Grid"),
            description=_(
                "The area of the bed to probe and the grid density."
            ),
        )
        self.add(grid_group)

        self.x0_row = LengthSpinRow(
            _("X origin"),
            _("Left edge of the probed area"),
            lower=0,
            upper=100000,
            value_in_base=self._x0,
        )
        self.x0_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.x0_row)

        self.y0_row = LengthSpinRow(
            _("Y origin"),
            _("Bottom edge of the probed area"),
            lower=0,
            upper=100000,
            value_in_base=self._y0,
        )
        self.y0_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.y0_row)

        self.width_row = LengthSpinRow(
            _("Width"),
            None,
            lower=1,
            upper=100000,
            value_in_base=self._width,
        )
        self.width_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.width_row)

        self.height_row = LengthSpinRow(
            _("Height"),
            None,
            lower=1,
            upper=100000,
            value_in_base=self._height,
        )
        self.height_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.height_row)

        self.columns_row = SpinRow(
            _("Columns"),
            _("Grid points along X"),
            lower=2,
            upper=101,
            step_increment=1,
        )
        self.columns_row.set_value(9)
        self.columns_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.columns_row)

        self.rows_row = SpinRow(
            _("Rows"),
            _("Grid points along Y"),
            lower=2,
            upper=101,
            step_increment=1,
        )
        self.rows_row.set_value(9)
        self.rows_row.value_changed.connect(self._on_grid_changed)
        grid_group.add(self.rows_row)

        self.points_row = Adw.ActionRow(title=_("Probe Points"))
        grid_group.add(self.points_row)

        probe_group = Adw.PreferencesGroup(
            title=_("Probing"),
            description=_(
                "Make sure the probe (or the laser tip) can reach the "
                "bed at every grid point."
            ),
        )
        self.add(probe_group)

        self.feed_row = SpinRow(
            _("Feed Rate (mm/min)"),
            _("Speed of the probing move"),
            lower=1,
            upper=10000,
            step_increment=10,
        )
        self.feed_row.set_value(DEFAULT_FEED_RATE)
        probe_group.add(self.feed_row)

        self.travel_row = LengthSpinRow(
            _("Maximum Travel"),
            _("How far down to search for the surface at each point"),
            lower=0.1,
            upper=100,
            value_in_base=DEFAULT_MAX_TRAVEL,
        )
        probe_group.add(self.travel_row)

        self.safe_z_row = LengthSpinRow(
            _("Safe Z"),
            _("Z height traveled at between probe points"),
            lower=0.1,
            upper=1000,
            value_in_base=DEFAULT_SAFE_Z,
        )
        probe_group.add(self.safe_z_row)

        self.status_row = Adw.ActionRow(title=_("Status"), subtitle=_("Idle"))
        probe_group.add(self.status_row)

        self.start_button = Gtk.Button(label=_("Start Probing"))
        self.start_button.add_css_class("suggested-action")
        self.start_button.connect("clicked", lambda *_: self._start())
        self.start_button.set_valign(Gtk.Align.CENTER)
        self.status_row.add_suffix(self.start_button)

        self.stop_button = Gtk.Button(label=_("Stop"))
        self.stop_button.add_css_class("destructive-action")
        self.stop_button.connect("clicked", lambda *_: self._stop())
        self.stop_button.set_valign(Gtk.Align.CENTER)
        self.stop_button.set_visible(False)
        self.status_row.add_suffix(self.stop_button)

        mesh_group = Adw.PreferencesGroup(
            title=_("Current Mesh"),
        )
        self.add(mesh_group)

        self.mesh_info_row = Adw.ActionRow(title=_("No mesh probed yet"))
        mesh_group.add(self.mesh_info_row)

        clear_button = Gtk.Button(
            child=Gtk.Image.new_from_icon_name("edit-delete-symbolic")
        )
        clear_button.set_tooltip_text(_("Delete Mesh"))
        clear_button.add_css_class("flat")
        clear_button.connect("clicked", lambda *_: self._clear_mesh())
        clear_button.set_valign(Gtk.Align.CENTER)
        self.mesh_info_row.add_suffix(clear_button)

        self.view = BedMeshView(hexpand=True, vexpand=True)
        view_row = _ViewRow(self.view)
        mesh_group.add(view_row)

        self._update_points_row()
        self._show_current_mesh()
        self._initializing = False

    # ── Grid bookkeeping ──────────────────────────────────────────

    def _grid_params(self) -> dict:
        x0 = self.x0_row.get_value_in_base_units()
        y0 = self.y0_row.get_value_in_base_units()
        width = self.width_row.get_value_in_base_units()
        height = self.height_row.get_value_in_base_units()
        nx = int(self.columns_row.get_value())
        ny = int(self.rows_row.get_value())
        dx = width / (nx - 1) if nx > 1 else 0.0
        dy = height / (ny - 1) if ny > 1 else 0.0
        return {"x0": x0, "y0": y0, "dx": dx, "dy": dy, "nx": nx, "ny": ny}

    def _on_grid_changed(self, _row):
        if not self._initializing:
            self._update_points_row()

    def _update_points_row(self):
        p = self._grid_params()
        estimate = _estimate_duration(
            p["nx"] * p["ny"],
            self.feed_row.get_value(),
            self.travel_row.get_value_in_base_units(),
        )
        self.points_row.set_subtitle(
            _("{} points · {}").format(p["nx"] * p["ny"], estimate)
        )

    # ── Current mesh ──────────────────────────────────────────────

    def _show_current_mesh(self):
        mesh = self.machine.bed_mesh
        if mesh is None:
            self.mesh_info_row.set_title(_("No mesh probed yet"))
            self.mesh_info_row.set_subtitle("")
            self._heights = []
            return
        zmin, zmax = mesh.z_range
        self.mesh_info_row.set_title(
            _("{}×{} grid, {} mm spacing").format(mesh.nx, mesh.ny, mesh.dx)
        )
        self.mesh_info_row.set_subtitle(
            _("Z range {min:.2f} … {max:.2f} mm (probed {date})").format(
                min=zmin, max=zmax, date=mesh.probed_at[:10]
            )
        )
        self._heights = [
            [mesh.heights[j * mesh.nx + i] for i in range(mesh.nx)]
            for j in range(mesh.ny)
        ]
        self.view.update_mesh(
            mesh.x0, mesh.y0, mesh.dx, mesh.dy, self._heights
        )

    def _clear_mesh(self):
        self.machine.clear_bed_mesh()
        self._show_current_mesh()

    # ── Probing ───────────────────────────────────────────────────

    def _set_probing(self, probing: bool, status: str | None = None):
        self.start_button.set_visible(not probing)
        self.stop_button.set_visible(probing)
        self.start_button.set_sensitive(not probing)
        for row in (
            self.x0_row,
            self.y0_row,
            self.width_row,
            self.height_row,
            self.columns_row,
            self.rows_row,
        ):
            row.set_sensitive(not probing)
        if status is not None:
            self.status_row.set_subtitle(status)

    def _start(self):
        machine = self.machine
        if not machine.is_connected():
            self.status_row.set_subtitle(_("Not connected"))
            return
        driver_cls = type(machine.driver)
        if not driver_cls.supports_probing:
            self.status_row.set_subtitle(
                _("The current driver does not support probing")
            )
            return

        params = self._grid_params()
        feed = int(self.feed_row.get_value())
        travel = self.travel_row.get_value_in_base_units()
        safe_z = self.safe_z_row.get_value_in_base_units()
        driver = machine.driver

        self._heights = [[None] * params["nx"] for _ in range(params["ny"])]
        self._set_probing(True, _("Probing…"))
        self.view.update_mesh(
            params["x0"],
            params["y0"],
            params["dx"],
            params["dy"],
            self._heights,
        )

        def on_progress(done, total, x, y, z):
            j, i = divmod(done - 1, params["nx"])
            if j % 2:
                i = params["nx"] - 1 - i
            self._heights[j][i] = z
            task_mgr.schedule_on_main_thread(
                self._progress_tick, done, total, x, y, z
            )

        async def coroutine(exec_ctx):
            return await probe_bed_mesh(
                driver,
                x0=params["x0"],
                y0=params["y0"],
                dx=params["dx"],
                dy=params["dy"],
                nx=params["nx"],
                ny=params["ny"],
                feed_rate_mm_min=feed,
                max_travel_mm=travel,
                safe_z_mm=safe_z,
                on_progress=on_progress,
                should_abort=exec_ctx.is_cancelled,
            )

        task_mgr.add_coroutine(
            coroutine,
            key=(machine.id, _PROBE_TASK_KEY),
            when_done=self._on_probe_done,
        )

    def _progress_tick(self, done, total, x, y, z):
        self.status_row.set_subtitle(
            _("{done}/{total}: {x:.0f}, {y:.0f} → {z:.2f} mm").format(
                done=done, total=total, x=x, y=y, z=z
            )
        )
        p = self._grid_params()
        self.view.update_mesh(
            p["x0"], p["y0"], p["dx"], p["dy"], self._heights
        )

    def _stop(self):
        task_mgr.cancel_task((self.machine.id, _PROBE_TASK_KEY))

    def _on_probe_done(self, task):
        def update():
            try:
                heights = task.result()
            except BedProbeAborted:
                self._set_probing(False, _("Probing cancelled"))
                return
            except (BedProbeError, OSError, RuntimeError) as exc:
                self._set_probing(False, str(exc))
                return
            if task.get_status() != "completed":
                self._set_probing(False, _("Probing cancelled"))
                return

            params = self._grid_params()
            mesh = BedMesh()
            mesh.set_params(
                x0=params["x0"],
                y0=params["y0"],
                dx=params["dx"],
                dy=params["dy"],
                nx=params["nx"],
                ny=params["ny"],
                heights=[z for row in heights for z in row],
                probe_params={
                    "feed_mm_min": self.feed_row.get_value(),
                    "max_travel_mm": (
                        self.travel_row.get_value_in_base_units()
                    ),
                    "safe_z_mm": self.safe_z_row.get_value_in_base_units(),
                },
            )
            self.machine.set_bed_mesh(mesh)
            self._set_probing(False, _("Mesh saved"))
            self._show_current_mesh()

        task_mgr.schedule_on_main_thread(update)
