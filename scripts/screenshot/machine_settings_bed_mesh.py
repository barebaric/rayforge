"""Screenshot: Machine settings - Bed Mesh page.

The page is only visible on machines with a Z axis, so the script
stages one on the active test machine first (and seeds a demo mesh so
the 3D view shows a probed surface). ``restore_config`` reverts every
staged change afterwards.
"""

import logging
import time

from gi.repository import Gtk
from utils import (
    get_target,
    open_machine_settings,
    restore_config,
    run_on_main_thread,
    take_screenshot,
)

from rayforge.context import get_context
from rayforge.machine.models.bed_mesh import BedMesh
from rayforge.uiscript import app, win

logger = logging.getLogger(__name__)
PAGE = "bed-mesh"
TARGET = get_target(f"machine-settings:{PAGE}")


def _stage_mesh(machine) -> None:
    machine.set_has_z_axis(True)
    mesh = BedMesh()
    cols, rows = 7, 5
    width, height = 180.0, 120.0
    heights = [
        1.6 * ((x - 1.1) ** 2 + (y - 0.7) ** 2)
        for y in (y / (rows - 1) for y in range(rows))
        for x in (x / (cols - 1) for x in range(cols))
    ]
    mesh.set_params(
        x0=10.0,
        y0=10.0,
        dx=width / (cols - 1),
        dy=height / (rows - 1),
        nx=cols,
        ny=rows,
        heights=heights,
        probe_params={"feed_mm_min": 100.0},
    )
    machine.set_bed_mesh(mesh)


@restore_config
def main():
    time.sleep(0.25)

    def _stage():
        machine = get_context().config.machine
        if machine:
            _stage_mesh(machine)

    run_on_main_thread(_stage)
    time.sleep(0.25)

    dialog = open_machine_settings(win, PAGE)
    time.sleep(0.5)

    def _scroll_to_mesh_view():
        page = dialog.content_stack.get_child_by_name(PAGE)
        scrolled = page
        while scrolled is not None and not isinstance(
            scrolled, Gtk.ScrolledWindow
        ):
            scrolled = scrolled.get_first_child()
        if scrolled is None:
            logger.warning("could not find the page scrolled window")
            return
        adj = scrolled.get_vadjustment()
        adj.set_value(adj.get_upper())

    run_on_main_thread(_scroll_to_mesh_view)
    time.sleep(0.5)
    take_screenshot()
    time.sleep(0.25)
    app.quit_idle()


main()
