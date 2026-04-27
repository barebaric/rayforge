#!/usr/bin/env python3
"""Screenshot CLI for Rayforge."""

import argparse
import fnmatch
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
TEST_CONFIG_DIR = PROJECT_ROOT / "tests" / "config"

TARGETS = {
    "main:standard": "main_standard",
    "main:3d": "main_3d",
    "main:3d-rotary": "main_3d_rotary",
    "app-settings:general": "app_settings_general",
    "app-settings:machines": "app_settings_machines",
    "app-settings:materials": "app_settings_materials",
    "app-settings:recipes": "app_settings_recipes",
    "app-settings:addons": "app_settings_addons",
    "app-settings:ai": "app_settings_ai",
    "addon:ai-workpiece-generator": "ai_workpiece_generator",
    "addon:print-and-cut:pick": "print_and_cut",
    "addon:print-and-cut:jog": "print_and_cut",
    "addon:print-and-cut:apply": "print_and_cut",
    "bottom-panel:console": "bottom_panel",
    "bottom-panel:layers": "bottom_panel",
    "import-dialog": "import_dialog",
    "machine-settings:general": "machine_settings_general",
    "machine-settings:hardware": "machine_settings_hardware",
    "machine-settings:advanced": "machine_settings_advanced",
    "machine-settings:gcode": "machine_settings_gcode",
    "machine-settings:hooks-macros": "machine_settings_hooks-macros",
    "machine-settings:device": "machine_settings_device",
    "machine-settings:laser": "machine_settings_laser",
    "machine-settings:rotary-module": "machine_settings_rotary_module",
    "machine-settings:camera": "machine_settings_camera",
    "machine-settings:maintenance": "machine_settings_maintenance",
    "machine-settings:nogo-zones": "machine_settings_nogo_zones",
    "material-test": "material_test",
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
}


def get_matching_targets(target: str) -> list[str]:
    """Find all leaf targets that match the given target spec.

    Supports glob patterns (e.g. "step-settings*post") and prefix
    matching (e.g. "step-settings" matches all leaves under it).
    A leaf target is one with no children.
    """
    if any(c in target for c in "*?["):
        matches = [t for t in TARGETS if fnmatch.fnmatch(t, target)]
        if not matches:
            matches = [
                t
                for t in TARGETS
                if fnmatch.fnmatch(t, target.replace("*", ":*"))
            ]
        return matches
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
    with tempfile.TemporaryDirectory(prefix="rayforge-screenshot-") as tmpdir:
        shutil.copytree(TEST_CONFIG_DIR, tmpdir, dirs_exist_ok=True)
        cmd = [
            "pixi",
            "run",
            "rayforge",
            "--config",
            tmpdir,
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
    lines.append(
        "  main, app-settings, machine-settings, step-settings, bottom-panel"
    )
    lines.append(
        "  addon, step-settings:engrave, step-settings:engrave:general"
    )
    lines.append("  recipe-editor, etc.")
    lines.append("")
    lines.append("Use 'all' to run everything")
    lines.append("")
    lines.append("Glob patterns are supported (e.g. 'step-settings*post')")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Take screenshots for Rayforge documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_text(),
    )
    parser.add_argument("target", help="Screenshot target")
    args = parser.parse_args()
    target: str = args.target

    if target == "all":
        targets = list(TARGETS.keys())
    else:
        targets = get_matching_targets(target)

    if not targets:
        print(f"No targets match: {target}")
        return 1

    for target in targets:
        script = TARGETS[target]
        result = run_script(script, target)
        if result != 0:
            return result

    return 0


if __name__ == "__main__":
    sys.exit(main())
