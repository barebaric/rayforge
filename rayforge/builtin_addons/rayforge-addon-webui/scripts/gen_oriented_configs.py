"""
Generates one rayforge config directory per machine-orientation variant
so the web UI can be verified against every origin corner, axis
reversal, panel rotation and workarea-origin mode.

Usage:

    python gen_oriented_configs.py /tmp/rf-orient

Each variant becomes a full config directory with a single machine
(<variant>/machines/<variant>.yaml), bootable with:

    rayforge serve --config /tmp/rf-orient/<variant> --port 8150

All variants use the NoDeviceDriver placeholder, so servers boot
without hardware.
"""

import argparse
import sys
from pathlib import Path
from typing import Any, cast

from rayforge.machine.device.schema_migration import CURRENT_SCHEMA_VERSION
from rayforge.machine.models.machine import Machine, Origin
from rayforge.machine.models.machine_panel import PanelOrientation
from rayforge.shared.util.atomic import atomic_write_yaml

DRIVER = "NoDeviceDriver"


def _machine_dict(context, variant: dict) -> dict:
    machine = Machine(context)
    machine.name = variant["name"]
    machine.driver_name = DRIVER
    machine.placeholder = False
    machine.schema_version = CURRENT_SCHEMA_VERSION
    machine.set_origin(variant["origin"])
    machine.set_reverse_x_axis(variant["reverse_x"])
    machine.set_reverse_y_axis(variant["reverse_y"])
    machine.set_panel_orientation(variant["orientation"])
    machine.wcs_origin_is_workarea_origin = variant["wcs_is_workarea"]
    return machine.to_dict(include_frozen_dialect=False)


VARIANTS = [
    {
        "key": "bottom_left",
        "name": "Origin Bottom Left",
        "origin": Origin.BOTTOM_LEFT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "top_left",
        "name": "Origin Top Left",
        "origin": Origin.TOP_LEFT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "top_right",
        "name": "Origin Top Right",
        "origin": Origin.TOP_RIGHT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "bottom_right",
        "name": "Origin Bottom Right",
        "origin": Origin.BOTTOM_RIGHT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "top_right_reverse_x",
        "name": "Origin Top Right + Reverse X",
        "origin": Origin.TOP_RIGHT,
        "reverse_x": True,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "bottom_left_reverse_y",
        "name": "Origin Bottom Left + Reverse Y",
        "origin": Origin.BOTTOM_LEFT,
        "reverse_x": False,
        "reverse_y": True,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": False,
    },
    {
        "key": "bottom_left_rotated_left",
        "name": "Origin Bottom Left, Panel Rotated Left",
        "origin": Origin.BOTTOM_LEFT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.ROTATED_LEFT,
        "wcs_is_workarea": False,
    },
    {
        "key": "top_left_rotated_right",
        "name": "Origin Top Left, Panel Rotated Right",
        "origin": Origin.TOP_LEFT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.ROTATED_RIGHT,
        "wcs_is_workarea": False,
    },
    {
        "key": "bottom_left_workarea_origin",
        "name": "Origin Bottom Left, Workarea Origin",
        "origin": Origin.BOTTOM_LEFT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": True,
    },
    {
        "key": "top_right_workarea_origin",
        "name": "Origin Top Right, Workarea Origin",
        "origin": Origin.TOP_RIGHT,
        "reverse_x": False,
        "reverse_y": False,
        "orientation": PanelOrientation.NATIVE,
        "wcs_is_workarea": True,
    },
]


def generate(output_dir: Path) -> None:
    from rayforge.shared import tasker
    from rayforge.shared.tasker.manager import TaskManager

    def _direct_scheduler(callback, *args, **kwargs):
        callback(*args, **kwargs)

    task_manager = TaskManager(main_thread_scheduler=_direct_scheduler)
    proxy = cast(Any, tasker.task_mgr)
    proxy._instance = task_manager
    try:
        from rayforge.context import get_context

        context = get_context()
        header = (
            f"{'variant':<28} {'origin':<12} {'rev_x':<5} "
            f"{'rev_y':<5} {'panel':<14} {'wcs=workarea'}"
        )
        print(header)
        print("-" * len(header))
        for variant in VARIANTS:
            config_dir = output_dir / variant["key"]
            machines_dir = config_dir / "machines"
            machines_dir.mkdir(parents=True, exist_ok=True)
            data = _machine_dict(context, variant)
            machine_file = machines_dir / f"{variant['key']}.yaml"
            atomic_write_yaml(machine_file, data)
            print(
                f"{variant['key']:<28} {variant['origin'].value:<12} "
                f"{variant['reverse_x']!s:<5} "
                f"{variant['reverse_y']!s:<5} "
                f"{variant['orientation'].value:<14} "
                f"{variant['wcs_is_workarea']}"
            )
    finally:
        proxy._instance = None
        task_manager.shutdown()
    print(
        f"\nBoot one with:\n  rayforge serve "
        f"--config {output_dir}/<variant> --port 8150"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="gen_oriented_configs",
        description=(
            "Write one rayforge config dir per machine-orientation "
            "variant for web UI verification."
        ),
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory that receives one config dir per variant.",
    )
    args = parser.parse_args(argv)
    generate(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
