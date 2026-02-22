#!/usr/bin/env python3
"""Screenshot CLI for Rayforge."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
TEST_CONFIG_DIR = PROJECT_ROOT / "tests" / "config"

TARGETS = {
    "main:standard": "main_standard",
    "main:simulation": "main_simulation",
    "main:3d": "main_3d",
    "app-settings:general": "app_settings_general",
    "app-settings:machines": "app_settings_machines",
    "app-settings:materials": "app_settings_materials",
    "app-settings:recipes": "app_settings_recipes",
    "app-settings:packages": "app_settings_packages",
    "machine-settings:general": "machine_settings_general",
    "machine-settings:hardware": "machine_settings_hardware",
    "machine-settings:advanced": "machine_settings_advanced",
    "machine-settings:gcode": "machine_settings_gcode",
    "machine-settings:hooks-macros": "machine_settings_hooks-macros",
    "machine-settings:device": "machine_settings_device",
    "machine-settings:laser": "machine_settings_laser",
    "machine-settings:camera": "machine_settings_camera",
    "machine-settings:maintenance": "machine_settings_maintenance",
    "step-settings:contour:general": "step_settings",
    "step-settings:contour:post": "step_settings",
    "step-settings:engrave:general:constant_power": "step_settings",
    "step-settings:engrave:general:dither": "step_settings",
    "step-settings:engrave:general:multi_pass": "step_settings",
    "step-settings:engrave:general:variable": "step_settings",
    "step-settings:engrave:post": "step_settings",
    "step-settings:frame-outline:general": "step_settings",
    "step-settings:frame-outline:post": "step_settings",
    "step-settings:shrink-wrap:general": "step_settings",
    "step-settings:shrink-wrap:post": "step_settings",
    "recipe-editor:general": "recipe_editor_general",
    "recipe-editor:applicability": "recipe_editor_applicability",
    "recipe-editor:settings": "recipe_editor_settings",
    "material-test": "material_test",
    "control-panel": "control_panel",
    "import-dialog": "import_dialog",
}


def get_matching_targets(target: str) -> list[str]:
    """Find all leaf targets that match the given prefix.

    A leaf target is one with no children.
    """
    children = [t for t in TARGETS if t.startswith(target + ":")]
    if children:
        leaves = [
            t
            for t in children
            if not any(other.startswith(t + ":") for other in TARGETS)
        ]
        return leaves
    if target in TARGETS:
        return [target]
    return []


def run_script(script_name: str, target: str) -> int:
    cmd = [
        "pixi",
        "run",
        "rayforge",
        "--config",
        str(TEST_CONFIG_DIR),
        "--uiscript",
        str(SCRIPTS_DIR / f"{script_name}.py"),
    ]
    print(f"Running: {' '.join(cmd)} (TARGET={target})")
    env = os.environ.copy()
    env["TARGET"] = target
    return subprocess.run(cmd, env=env).returncode


def generate_help_text() -> str:
    lines = ["Available leaf targets:"]
    for target in sorted(TARGETS.keys()):
        lines.append(f"  {target}")
    lines.append("")
    lines.append("Useful prefixes (match all leaves under):")
    lines.append("  main, app-settings, machine-settings, step-settings")
    lines.append("  step-settings:engrave, step-settings:engrave:general")
    lines.append("  recipe-editor, etc.")
    lines.append("")
    lines.append("Use 'all' to run everything")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Take screenshots for Rayforge documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_text(),
    )
    parser.add_argument("target", help="Screenshot target")
    args = parser.parse_args()

    if args.target == "all":
        targets = list(TARGETS.keys())
    else:
        targets = get_matching_targets(args.target)

    if not targets:
        print(f"No targets match: {args.target}")
        return 1

    for target in targets:
        script = TARGETS[target]
        result = run_script(script, target)
        if result != 0:
            return result

    return 0


if __name__ == "__main__":
    sys.exit(main())
