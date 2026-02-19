#!/usr/bin/env python3
"""Screenshot CLI for Rayforge."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent

APP_SETTINGS_PAGES = [
    "general",
    "machines",
    "materials",
    "recipes",
    "packages",
]
MACHINE_SETTINGS_PAGES = [
    "general",
    "hardware",
    "advanced",
    "gcode",
    "hooks-macros",
    "device",
    "laser",
    "camera",
    "maintenance",
]
RECIPE_PAGES = ["general", "applicability", "settings"]
STEP_TYPES = [
    "contour",
    "engrave",
    "frame-outline",
    "shrink-wrap",
]

HELP_TEXT = """
Available Targets:
  main                          All main window screenshots
  main:standard                 Main window (standard mode)
  main:simulation               Main window (simulation mode)
  main:3d                       Main window (3D view)
  app-settings                  All app settings pages
  app-settings:general          App settings - General
  app-settings:machines         App settings - Machines
  app-settings:materials        App settings - Materials
  app-settings:recipes          App settings - Recipes
  app-settings:packages         App settings - Packages
  machine-settings              All machine settings pages
  machine-settings:general      Machine settings - General
  machine-settings:hardware     Machine settings - Hardware
  machine-settings:advanced     Machine settings - Advanced
  machine-settings:gcode        Machine settings - G-code
  machine-settings:hooks-macros Machine settings - Hooks & Macros
  machine-settings:device       Machine settings - Device
  machine-settings:laser        Machine settings - Laser
  machine-settings:camera       Machine settings - Camera
  machine-settings:maintenance  Machine settings - Maintenance
  step-settings                 All step settings screenshots
  step-settings:contour         Step settings - Contour
  step-settings:engrave         Step settings - Engrave
  step-settings:frame-outline   Step settings - Frame Outline
  step-settings:shrink-wrap     Step settings - Shrink Wrap
  recipe-editor                 Recipe editor - General page
  recipe-editor:general         Recipe editor - General
  recipe-editor:applicability   Recipe editor - Applicability
  recipe-editor:settings        Recipe editor - Settings
  material-test                 Material test grid dialog
"""


def get_script_path(name: str) -> Path:
    return SCRIPTS_DIR / f"{name}.py"


def run_rayforge(script_name: str, env: dict | None = None):
    cmd = [
        "pixi",
        "run",
        "rayforge",
        "--uiscript",
        str(get_script_path(script_name)),
    ]
    print(f"Running: {' '.join(cmd)}")
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    return subprocess.run(cmd, env=run_env).returncode


def main():
    parser = argparse.ArgumentParser(
        description="Take screenshots for Rayforge documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=HELP_TEXT,
    )
    parser.add_argument("target", help="Screenshot target")

    args = parser.parse_args()

    if ":" in args.target:
        base, page = args.target.split(":", 1)
    else:
        base, page = args.target, None

    if base == "main":
        if page == "standard":
            return run_rayforge("main_standard")
        elif page == "simulation":
            return run_rayforge("main_simulation")
        elif page == "3d":
            return run_rayforge("main_3d")
        elif page is None:
            for script in ["main_standard", "main_simulation", "main_3d"]:
                result = run_rayforge(script)
                if result != 0:
                    return result
            return 0
        else:
            print(f"Unknown main subtype: {page}")
            print("Options: standard, simulation, 3d")
            return 1

    elif base == "app-settings":
        if page:
            if page not in APP_SETTINGS_PAGES:
                print(f"Unknown app-settings page: {page}")
                print(f"Options: {', '.join(APP_SETTINGS_PAGES)}")
                return 1
            return run_rayforge(f"app_settings_{page}")
        else:
            for p in APP_SETTINGS_PAGES:
                result = run_rayforge(f"app_settings_{p}")
                if result != 0:
                    return result
            return 0

    elif base == "machine-settings":
        if page:
            if page not in MACHINE_SETTINGS_PAGES:
                print(f"Unknown machine-settings page: {page}")
                print(f"Options: {', '.join(MACHINE_SETTINGS_PAGES)}")
                return 1
            return run_rayforge(f"machine_settings_{page}")
        else:
            for p in MACHINE_SETTINGS_PAGES:
                result = run_rayforge(f"machine_settings_{p}")
                if result != 0:
                    return result
            return 0

    elif base == "step-settings":
        if page:
            if page not in STEP_TYPES:
                print(f"Unknown step-settings type: {page}")
                print(f"Options: {', '.join(STEP_TYPES)}")
                return 1
            return run_rayforge("step_settings", env={"STEP_TYPE": page})
        else:
            for step_type in STEP_TYPES:
                result = run_rayforge(
                    "step_settings", env={"STEP_TYPE": step_type}
                )
                if result != 0:
                    return result
            return 0

    elif base == "recipe-editor":
        if page:
            if page not in RECIPE_PAGES:
                print(f"Unknown recipe-editor page: {page}")
                print(f"Options: {', '.join(RECIPE_PAGES)}")
                return 1
            return run_rayforge(f"recipe_editor_{page}")
        return run_rayforge("recipe_editor_general")

    elif base == "material-test":
        return run_rayforge("material_test")

    else:
        print(f"Unknown target: {base}")
        print(
            "Options: main, app-settings, machine-settings, step-settings, "
            "recipe-editor, material-test"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
