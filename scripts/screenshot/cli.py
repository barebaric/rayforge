#!/usr/bin/env python3
"""Screenshot CLI for Rayforge."""

import argparse
import fnmatch
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from xml.sax.saxutils import escape

SCRIPTS_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPTS_DIR.parent.parent
TEST_CONFIG_DIR = PROJECT_ROOT / "tests" / "config"

TARGETS = {
    "addon:ai-workpiece-generator": "ai_workpiece_generator",
    "addon:deepnest": "deepnest",
    "addon:print-and-cut:pick": "print_and_cut",
    "addon:print-and-cut:jog": "print_and_cut",
    "addon:print-and-cut:apply": "print_and_cut",
    "addon:projector-mode": "projector_mode",
    "app-settings:general": "app_settings_general",
    "app-settings:machines": "app_settings_machines",
    "app-settings:machines:add": "add_machine_dialog",
    "app-settings:materials": "app_settings_materials",
    "app-settings:recipes": "app_settings_recipes",
    "app-settings:addons": "app_settings_addons",
    "app-settings:ai": "app_settings_ai",
    "bottom-panel:console": "bottom_panel",
    "bottom-panel:layers": "bottom_panel",
    "config-wizard:ai-lookup": "config_wizard",
    "config-wizard:ai-provider": "config_wizard",
    "config-wizard:camera": "config_wizard",
    "config-wizard:controller": "config_wizard",
    "config-wizard:connect": "config_wizard",
    "config-wizard:permissions": "config_wizard",
    "config-wizard:probe": "config_wizard",
    "config-wizard:profile": "config_wizard",
    "config-wizard:hardware": "config_wizard",
    "config-wizard:head": "config_wizard",
    "config-wizard:review": "config_wizard",
    "config-wizard:rotary": "config_wizard",
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
    "machine-settings:camera:image-settings": "machine_settings_camera",
    "machine-settings:camera:lens-calibration": "machine_settings_camera",
    "machine-settings:camera:lens-calibration:wizard-card": (
        "machine_settings_camera"
    ),
    "machine-settings:camera:lens-calibration:wizard-capture": (
        "machine_settings_camera"
    ),
    "machine-settings:camera:image-alignment": "machine_settings_camera",
    "machine-settings:maintenance": "machine_settings_maintenance",
    "machine-settings:nogo-zones": "machine_settings_nogo_zones",
    "main:standard": "main_standard",
    "main:3d": "main_3d",
    "main:3d-bee": "main_3d_bee",
    "main:3d-rotary": "main_3d_rotary",
    "main:array:grid": "array_grid",
    "main:array:point-rotation": "array_point_rotation",
    "main:array:circular": "array_circular",
    "material-test": "material_test",
    "operations:wavefront": "wavefront",
    "recipe-editor:general": "recipe_editor_general",
    "recipe-editor:applicability": "recipe_editor_applicability",
    "recipe-editor:laser": "recipe_editor_settings",
    "recipe-editor:step-settings": "recipe_editor_settings",
    "recipe-editor:post-processing": "recipe_editor_settings",
    "sanity-check": "sanity_check",
    "step-settings:contour:general": "step_settings",
    "step-settings:contour:laser": "step_settings",
    "step-settings:contour:post": "step_settings",
    "step-settings:engrave:general:constant_power": "step_settings",
    "step-settings:engrave:general:dither": "step_settings",
    "step-settings:engrave:general:multi_pass": "step_settings",
    "step-settings:engrave:general:variable": "step_settings",
    "step-settings:engrave:laser": "step_settings",
    "step-settings:engrave:post": "step_settings",
    "step-settings:frame-outline:general": "step_settings",
    "step-settings:frame-outline:laser": "step_settings",
    "step-settings:frame-outline:post": "step_settings",
    "step-settings:shrink-wrap:general": "step_settings",
    "step-settings:shrink-wrap:laser": "step_settings",
    "step-settings:shrink-wrap:post": "step_settings",
    "step-settings:wavefront:general": "step_settings",
    "step-settings:wavefront:laser": "step_settings",
    "step-settings:wavefront:post": "step_settings",
}


XFT_RESOURCES = [
    "Xft.antialias: 1",
    "Xft.hinting: 1",
    "Xft.hintstyle: hintslight",
    "Xft.rgba: none",
    "Xft.dpi: 96",
]

FONT_CONFIG_TEMPLATE = """<?xml version="1.0"?>
<!DOCTYPE fontconfig SYSTEM "fonts.dtd">
<fontconfig>
  <include ignore_missing="yes">/etc/fonts/fonts.conf</include>
  <match target="pattern">
    <test qual="any" name="family"><string>Cantarell</string></test>
    <edit name="family" mode="prepend" binding="same">
      <string>{family}</string>
    </edit>
  </match>
  <match target="pattern">
    <test qual="any" name="family"><string>sans-serif</string></test>
    <edit name="family" mode="prepend" binding="same">
      <string>{family}</string>
    </edit>
  </match>
  <match target="font">
    <edit name="antialias" mode="assign"><bool>true</bool></edit>
    <edit name="hinting" mode="assign"><bool>true</bool></edit>
    <edit name="hintstyle" mode="assign"><const>hintslight</const></edit>
    <edit name="rgba" mode="assign"><const>none</const></edit>
  </match>
</fontconfig>
"""


def get_desktop_font_family() -> str | None:
    """Return the desktop default font family, if it can be read.

    Under Xvfb no XSETTINGS daemon publishes the desktop font, so
    GTK would fall back to a different family and render the
    screenshots with different text metrics.
    """
    for prog in ("gsettings", "/usr/bin/gsettings"):
        try:
            result = subprocess.run(
                [
                    prog,
                    "get",
                    "org.gnome.desktop.interface",
                    "font-name",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode != 0:
            continue
        value = result.stdout.strip().strip("'\"")
        parts = value.rsplit(" ", 1)
        if len(parts) == 2 and parts[0]:
            return parts[0]
    return None


def write_font_config(family: str) -> str:
    """Write a fontconfig file mapping GTK's fallback families to
    the desktop font, with the desktop rendering settings."""
    fd, path = tempfile.mkstemp(prefix="rayforge-fontconfig-")
    with os.fdopen(fd, "w") as f:
        f.write(FONT_CONFIG_TEMPLATE.format(family=escape(family)))
    return path


class XvfbSession:
    """A temporary Xvfb server providing a headless display.

    Screenshots captured under Xvfb are independent of the
    developer's desktop (resolution, scale, theme, window focus).
    RAYFORGE_XVFB tells the capture helpers inside the app to skip
    desktop-only tools such as gnome-screenshot. GSK_RENDERER=cairo
    is required because Xvfb has no GLX: GTK's GL renderer blocks
    forever in frame uploads with drivers such as NVIDIA's.
    """

    SCREEN = "2560x1700x24"
    DISPLAY_RANGE = range(99, 200)
    STARTUP_TIMEOUT = 5.0
    WM_STARTUP_TIMEOUT = 20.0

    def __init__(self) -> None:
        self.process: subprocess.Popen | None = None
        self.wm_process: subprocess.Popen | None = None
        self._bus_address: str | None = None
        self._bus_address_file: str | None = None
        self._font_config: str | None = None

    @staticmethod
    def is_available() -> bool:
        return shutil.which("Xvfb") is not None

    @property
    def env(self) -> dict[str, str]:
        env = {
            "DISPLAY": f":{self.number}",
            "RAYFORGE_XVFB": "1",
            "GSK_RENDERER": "cairo",
        }
        if self._font_config:
            env["FONTCONFIG_FILE"] = self._font_config
        if (
            self.wm_process is not None
            and self.wm_process.poll() is None
            and self._bus_address
        ):
            env["DBUS_SESSION_BUS_ADDRESS"] = self._bus_address
            env["RAYFORGE_XVFB_WM"] = "1"
        return env

    def start(self) -> bool:
        for number in self.DISPLAY_RANGE:
            if self._claim_display(number):
                self._start_wm()
                self._configure_fonts()
                return True
        return False

    def _start_wm(self) -> None:
        """Run GNOME Shell as a window manager for the display.

        Together with the private D-Bus session this turns the
        Xvfb display into an invisible but fully functional GNOME
        session: dialogs are placed by the WM and gnome-screenshot
        captures window buffers with real shadows and alpha. The
        developer's own desktop session is never touched.
        """
        if (
            shutil.which("gnome-shell") is None
            or shutil.which("dbus-run-session") is None
        ):
            return
        fd, self._bus_address_file = tempfile.mkstemp(prefix="rayforge-bus-")
        os.close(fd)
        inner = (
            "gsettings set org.gnome.desktop.interface "
            "enable-animations false; "
            f'printf "%s" "$DBUS_SESSION_BUS_ADDRESS" > '
            f"{self._bus_address_file}; "
            "exec gnome-shell --x11"
        )
        self.wm_process = subprocess.Popen(
            ["dbus-run-session", "--", "bash", "-c", inner],
            env={**os.environ, **self.env},
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        if self._wait_for_wm():
            self._read_bus_address()

    def _read_bus_address(self) -> None:
        try:
            address = Path(self._bus_address_file).read_text().strip()
        except OSError:
            address = ""
        if address:
            self._bus_address = address
        else:
            print("Warning: private session bus address not found")

    def _wait_for_wm(self) -> bool:
        deadline = time.monotonic() + self.WM_STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.wm_process.poll() is not None:
                print("Warning: GNOME Shell exited during startup")
                return False
            check = subprocess.run(
                ["xprop", "-root", "_NET_SUPPORTING_WM_CHECK"],
                capture_output=True,
                text=True,
                check=False,
                env={**os.environ, **self.env},
            )
            if check.returncode == 0 and "not found" not in check.stdout:
                print("Running GNOME Shell as window manager")
                return True
            time.sleep(0.5)
        print("Warning: window manager did not register in time")
        return False

    def _configure_fonts(self) -> None:
        """Mirror the desktop's font and Xft settings on the
        headless display."""
        family = get_desktop_font_family()
        if family:
            self._font_config = write_font_config(family)
        if shutil.which("xrdb") is not None:
            subprocess.run(
                ["xrdb", "-merge"],
                input="\n".join(XFT_RESOURCES),
                text=True,
                capture_output=True,
                check=False,
                env={**os.environ, **self.env},
            )

    def _claim_display(self, number: int) -> bool:
        if self._display_in_use(number):
            return False
        self.process = subprocess.Popen(
            [
                "Xvfb",
                f":{number}",
                "-screen",
                "0",
                self.SCREEN,
                "-nolisten",
                "tcp",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if self._wait_until_ready(number):
            self.number = number
            return True
        self.stop()
        return False

    @staticmethod
    def _display_in_use(number: int) -> bool:
        lock = Path(f"/tmp/.X{number}-lock")
        socket = Path(f"/tmp/.X11-unix/X{number}")
        return lock.exists() or socket.exists()

    def _wait_until_ready(self, number: int) -> bool:
        socket = Path(f"/tmp/.X11-unix/X{number}")
        deadline = time.monotonic() + self.STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                return False
            if socket.exists():
                return True
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        self._stop_wm()
        if self.process is None:
            return
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        self.process = None
        self.number = None
        if self._bus_address_file:
            try:
                os.unlink(self._bus_address_file)
            except OSError:
                pass
            self._bus_address_file = None
        self._bus_address = None
        if self._font_config:
            try:
                os.unlink(self._font_config)
            except OSError:
                pass
            self._font_config = None

    def _stop_wm(self) -> None:
        if self.wm_process is None:
            return
        try:
            group = os.getpgid(self.wm_process.pid)
            os.killpg(group, signal.SIGTERM)
            self.wm_process.wait(timeout=5)
        except (
            ProcessLookupError,
            PermissionError,
            subprocess.TimeoutExpired,
        ):
            try:
                group = os.getpgid(self.wm_process.pid)
                os.killpg(group, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        self.wm_process = None


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
    if target in TARGETS:
        return [target]
    children = [t for t in TARGETS if t.startswith(target + ":")]
    if children:
        return [
            t
            for t in children
            if not any(other.startswith(t + ":") for other in TARGETS)
        ]
    return []


def run_script(script_name: str, target: str, base_env: dict[str, str]) -> int:
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
        env = dict(base_env)
        env["TARGET"] = target
        # Force the isolated test config even if the --config argument
        # were ever dropped or parsed by a wrapping command, so screenshots
        # never depend on the developer's personal machine configuration.
        env["RAYFORGE_CONFIG_DIR"] = tmpdir
        return subprocess.run(cmd, env=env, check=False).returncode


def suggest_targets(target: str) -> list[str]:
    """Suggest close matches for a non-matching target.

    1. Find targets sharing the longest prefix with the given target.
    2. Within those, find fuzzy (substring) matches on the last component.
    3. If no fuzzy matches, return all leaves under the longest prefix.
    """
    # Find the longest prefix that exists in TARGETS
    parts = target.split(":")
    best_prefix = ""
    for i in range(len(parts), 0, -1):
        prefix = ":".join(parts[:i])
        if (
            any(t.startswith(prefix + ":") for t in TARGETS)
            or prefix in TARGETS
        ):
            best_prefix = prefix
            break

    if not best_prefix:
        return []

    # Collect leaves under this prefix
    leaves = [
        t
        for t in TARGETS
        if t.startswith(best_prefix + ":")
        and not any(other.startswith(t + ":") for other in TARGETS)
    ]
    if not leaves and best_prefix in TARGETS:
        leaves = [best_prefix]

    # Try fuzzy match on the last component
    last = parts[-1].lower()
    fuzzy = [t for t in leaves if last in t.split(":")[-1].lower()]

    return fuzzy if fuzzy else leaves


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
    lines.append("")
    lines.append("Use 'all' to run everything")
    lines.append("")
    lines.append("Glob patterns are supported (e.g. 'step-settings*post')")
    return "\n".join(lines)


def setup_display(no_xvfb: bool) -> XvfbSession | None:
    """Start Xvfb for the run when available, else keep the desktop.

    Returns the started session, or None when the run should use the
    developer's desktop display.
    """
    if no_xvfb or not XvfbSession.is_available():
        return None
    session = XvfbSession()
    if session.start():
        print(f"Running under Xvfb on display :{session.number}")
        return session
    print("Warning: failed to start Xvfb, using the desktop display")
    return None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Take screenshots for Rayforge documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=generate_help_text(),
    )
    parser.add_argument("target", help="Screenshot target")
    parser.add_argument(
        "--no-xvfb",
        action="store_true",
        help="Run on the desktop display even if Xvfb is available",
    )
    args = parser.parse_args()
    target: str = args.target

    if target == "all":
        targets = list(TARGETS.keys())
    else:
        targets = get_matching_targets(target)

    if not targets:
        print(f"No targets match: {target}")
        suggestions = suggest_targets(target)
        if suggestions:
            print("\nDid you mean one of these?")
            for s in suggestions:
                print(f"  {s}")
        return 1

    base_env = os.environ.copy()
    session = setup_display(args.no_xvfb)
    if session is not None:
        base_env.update(session.env)
    try:
        for target in targets:
            script = TARGETS[target]
            result = run_script(script, target, base_env)
            if result != 0:
                return result
    finally:
        if session is not None:
            session.stop()

    return 0


if __name__ == "__main__":
    sys.exit(main())
