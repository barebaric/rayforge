"""
Shared utilities for screenshot scripts.

These scripts are designed to be run via `rayforge --uiscript`.
Scripts run in a background thread, so UI operations use
GLib.idle_add for thread safety.
"""

import atexit
import ctypes
import functools
import logging
import os
import re
import subprocess
import tempfile
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from threading import Event
from typing import (
    TYPE_CHECKING,
    Any,
    Optional,
    TypeVar,
)

import numpy as np
import pyvips
from gi.repository import Adw, GdkX11, GLib, Gtk
from PIL import Image, ImageDraw, ImageFilter

from rayforge.config import CONFIG_DIR
from rayforge.context import get_context
from rayforge.core.recipe import Recipe
from rayforge.core.step_registry import step_registry
from rayforge.doceditor.array import ArrayMode
from rayforge.ui_gtk.array_dialog import (
    CircularArrayDialog,
    GridArrayDialog,
    PointRotationArrayDialog,
)
from rayforge.ui_gtk.doceditor.recipes import AddEditRecipeDialog
from rayforge.ui_gtk.doceditor.step_settings.dialog import StepSettingsDialog
from rayforge.ui_gtk.machine.settings_dialog import MachineSettingsDialog
from rayforge.ui_gtk.settings.settings_dialog import SettingsWindow

if TYPE_CHECKING:
    from rayforge.core.step import Step
    from rayforge.ui_gtk.array_dialog import _BaseArrayDialog
    from rayforge.ui_gtk.mainwindow import MainWindow

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "website" / "static" / "screenshots"
TESTS_DIR = PROJECT_ROOT / "tests"

SCREENSHOT_TOOLS = [
    (["import", "-window", "root", "-silent"], "ImageMagick import"),
    (["gnome-screenshot", "-f"], "gnome-screenshot"),
]

T = TypeVar("T")


def _under_xvfb() -> bool:
    return bool(os.environ.get("RAYFORGE_XVFB"))


def _wm_active() -> bool:
    """True when the Xvfb session runs its own window manager.

    The session then provides a private D-Bus and a compositor, so
    ``gnome-screenshot`` captures the virtual display's windows
    with real shadows and alpha, exactly like on the desktop.
    """
    return bool(os.environ.get("RAYFORGE_XVFB_WM"))


def uses_synthetic_decorations() -> bool:
    """True if window captures get synthetic decorations drawn in.

    On bare Xvfb without a window manager there is nothing to draw
    a title bar or drop shadow, so the capture pipeline paints its
    own frame around the window content.
    """
    return _under_xvfb() and not _wm_active()


def get_screenshot_tools() -> list[tuple[list[str], str]]:
    """
    Return the capture tools usable on the current display.

    ImageMagick's ``import`` is tried first: it scrapes the root
    window directly, without GNOME Shell's flash and camera shutter
    sound, which the Xvfb session would otherwise route to the
    developer's desktop audio. ``gnome-screenshot`` stays as a
    fallback. Without a window manager on the display (plain Xvfb)
    gnome-screenshot would reach the developer's desktop shell and
    screenshot that, so only X11-native tools may be used there.
    """
    if uses_synthetic_decorations():
        return [t for t in SCREENSHOT_TOOLS if t[1] != "gnome-screenshot"]
    return SCREENSHOT_TOOLS


def capture_full_screen() -> Image.Image | None:
    """
    Capture the entire screen via the first succeeding tool.

    The returned image's (0, 0) is the root origin and matches
    absolute window geometries as reported by xwininfo.
    """
    for cmd_args, _tool_name in get_screenshot_tools():
        with tempfile.TemporaryDirectory(prefix="rayforge-shot-") as tmp:
            temp_path = Path(tmp) / "capture.png"
            result = subprocess.run(
                [*cmd_args, str(temp_path)],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                try:
                    return Image.open(temp_path)
                except (OSError, ValueError) as e:
                    logger.error(f"Failed to open capture: {e}")
                    return None
    logger.error("Failed to take screenshot with available tools")
    return None


def _get_xid_pid(xid: str) -> int | None:
    result = subprocess.run(
        ["xprop", "-id", str(xid), "_NET_WM_PID"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"=\s*(\d+)", result.stdout)
    return int(match.group(1)) if match else None


def _get_frame_extents(xid: str) -> tuple[int, int, int, int]:
    """Return the CSD shadow border as (left, right, top, bottom).

    GTK draws its drop shadow inside the X window but outside the
    visible content; _GTK_FRAME_EXTENTS advertises that border.
    """
    result = subprocess.run(
        ["xprop", "-id", str(xid), "_GTK_FRAME_EXTENTS"],
        capture_output=True,
        text=True,
        check=False,
    )
    match = re.search(r"=\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)", result.stdout)
    if not match:
        return (0, 0, 0, 0)
    return (
        int(match.group(1)),
        int(match.group(2)),
        int(match.group(3)),
        int(match.group(4)),
    )


def get_app_window_box() -> tuple[int, int, int, int] | None:
    """
    Locate the app's topmost toplevel window on the X server.

    Xvfb runs without a window manager, so there is no 'active
    window' for capture tools to grab. Direct children of the root
    window belonging to this process are reported by xwininfo in
    stacking order; the first match is the window a desktop
    capture would have shown. The GTK shadow border is excluded,
    so the box tightly encloses the visible content. Returns
    (x, y, width, height).
    """
    result = subprocess.run(
        ["xwininfo", "-root", "-tree"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.error(f"xwininfo failed: {result.stderr}")
        return None
    pattern = re.compile(
        r'^ {5}(0x[0-9a-f]+) "[^"]*".*?(\d+)x(\d+)([+-]\d+)([+-]\d+)',
        re.MULTILINE,
    )
    own_pid = os.getpid()
    for match in pattern.finditer(result.stdout):
        xid = match.group(1)
        if _get_xid_pid(xid) != own_pid:
            continue
        x = int(match.group(4))
        y = int(match.group(5))
        w = int(match.group(2))
        h = int(match.group(3))
        left, right, top, bottom = _get_frame_extents(xid)
        w -= left + right
        h -= top + bottom
        if w <= 0 or h <= 0:
            logger.warning("Invalid window geometry after frame extents")
            return None
        box = (x + left, y + top, w, h)
        logger.info(f"App window content: x={box[0]} y={box[1]} w={w} h={h}")
        return box
    logger.warning("Could not locate the app window on the root window")
    return None


def get_toplevel_window_box() -> tuple[int, int, int, int] | None:
    """
    Locate the app's topmost toplevel window on a managed display.

    _NET_CLIENT_LIST_STACKING lists managed windows bottom to top;
    the last entry belonging to this process is the window a
    window-mode capture would have picked, while remaining
    independent of the pointer position. The box encloses the
    window content, inside the CSD shadow margins advertised via
    _GTK_FRAME_EXTENTS. Returns (x, y, width, height).
    """
    result = subprocess.run(
        ["xprop", "-root", "_NET_CLIENT_LIST_STACKING"],
        capture_output=True,
        text=True,
        check=False,
    )
    own_pid = os.getpid()
    topmost_xid = None
    for xid in re.findall(r"0x[0-9a-f]+", result.stdout):
        if _get_xid_pid(xid) == own_pid:
            topmost_xid = int(xid, 16)
    if topmost_xid is None:
        logger.warning("No app window found in the stacking order")
        return None
    geometry = get_xid_geometry(topmost_xid)
    if geometry is None:
        return None
    wx, wy, ww, wh = geometry
    left, right, top, bottom = _get_frame_extents(topmost_xid)
    w = ww - left - right
    h = wh - top - bottom
    if w <= 0 or h <= 0:
        logger.warning("Invalid window geometry after frame extents")
        return None
    return (wx + left, wy + top, w, h)


# Window framing applied to Xvfb captures, approximating what a
# desktop compositor would draw around the window.
DECOR_MARGIN_PX = 24
DECOR_RADIUS_PX = 12
DECOR_BLUR_PX = 12
DECOR_OFFSET_Y_PX = 6
DECOR_SHADOW_ALPHA = 110


def _add_window_decorations(img: Image.Image) -> Image.Image:
    """Frame a content screenshot with rounded corners and a soft
    semi-transparent drop shadow."""
    margin = DECOR_MARGIN_PX
    width, height = img.size
    size = (width + 2 * margin, height + 2 * margin)
    canvas = Image.new("RGBA", size, (0, 0, 0, 0))

    rect = (
        margin,
        margin + DECOR_OFFSET_Y_PX,
        margin + width,
        margin + height + DECOR_OFFSET_Y_PX,
    )
    shadow = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        rect, radius=DECOR_RADIUS_PX, fill=(0, 0, 0, DECOR_SHADOW_ALPHA)
    )
    alpha = shadow.getchannel("A").filter(
        ImageFilter.GaussianBlur(DECOR_BLUR_PX)
    )
    shadow.putalpha(alpha)
    canvas.alpha_composite(shadow)

    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=DECOR_RADIUS_PX,
        fill=255,
    )
    canvas.paste(img.convert("RGBA"), (margin, margin), mask)
    return canvas


def capture_app_window() -> Image.Image | None:
    """Capture the app's topmost toplevel window as an image.

    The full screen is captured and cropped to the window's content
    box, so neither the pointer position nor which window the
    capture tool considers active can displace the captured
    content, and the desktop behind the window cannot bleed into
    the transparent CSD shadow margins. A synthetic frame with
    rounded corners and a drop shadow is drawn around the content,
    which also masks the corners GTK rounds off.
    """
    img = capture_full_screen()
    if img is None:
        return None
    if uses_synthetic_decorations():
        box = get_app_window_box()
    else:
        box = get_toplevel_window_box()
    if box is None:
        return None
    x, y, w, h = box
    return _add_window_decorations(img.crop((x, y, x + w, y + h)))


# ---------------------------------------------------------------------------
# X11 helpers (Xvfb runs without a window manager)
# ---------------------------------------------------------------------------
_x11_lib: Optional["ctypes.CDLL"] = None
_x11_display = None


def _load_x11() -> tuple["ctypes.CDLL", int] | None:
    """Load libX11 and open a display connection (cached)."""
    global _x11_lib, _x11_display
    if _x11_lib is None:
        try:
            lib = ctypes.CDLL("libX11.so.6")
        except OSError:
            return None
        lib.XOpenDisplay.restype = ctypes.c_void_p
        lib.XOpenDisplay.argtypes = [ctypes.c_char_p]
        lib.XMoveWindow.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.XFlush.argtypes = [ctypes.c_void_p]
        lib.XDefaultRootWindow.restype = ctypes.c_ulong
        lib.XDefaultRootWindow.argtypes = [ctypes.c_void_p]
        lib.XGetGeometry.restype = ctypes.c_int
        lib.XGetGeometry.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(ctypes.c_ulong),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
            ctypes.POINTER(ctypes.c_uint),
        ]
        lib.XWarpPointer.restype = ctypes.c_int
        lib.XWarpPointer.argtypes = [
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.c_ulong,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_int,
            ctypes.c_int,
        ]
        _x11_lib = lib
    if _x11_display is None:
        _x11_display = _x11_lib.XOpenDisplay(None)
    if not _x11_display:
        return None
    return _x11_lib, _x11_display


def move_xid(xid: int, x: int, y: int) -> None:
    """Move an X window to the given position."""
    loaded = _load_x11()
    if loaded is None:
        return
    lib, display = loaded
    lib.XMoveWindow(display, xid, x, y)
    lib.XFlush(display)


def get_root_window_size() -> tuple[int, int] | None:
    """Return the dimensions of the display's root window."""
    loaded = _load_x11()
    if loaded is None:
        return None
    lib, display = loaded
    root = lib.XDefaultRootWindow(display)
    root_id = ctypes.c_ulong()
    x = ctypes.c_int()
    y = ctypes.c_int()
    width = ctypes.c_uint()
    height = ctypes.c_uint()
    border = ctypes.c_uint()
    depth = ctypes.c_uint()
    ok = lib.XGetGeometry(
        display,
        root,
        ctypes.byref(root_id),
        ctypes.byref(x),
        ctypes.byref(y),
        ctypes.byref(width),
        ctypes.byref(height),
        ctypes.byref(border),
        ctypes.byref(depth),
    )
    if not ok:
        return None
    return width.value, height.value


def park_mouse_pointer() -> None:
    """Park the pointer in the screen's bottom-right corner.

    The pointer starts out at the screen center of the headless
    session, inside the app window, and a pointer resting on a
    widget makes GTK pop up a tooltip after the hover timeout --
    right around when the capture fires. The corner holds no app
    or shell UI, and a tooltip that is already showing hides when
    the pointer leaves. Never runs on the developer's desktop
    display.
    """
    if not _under_xvfb():
        return
    size = get_root_window_size()
    loaded = _load_x11()
    if size is None or loaded is None:
        return
    lib, display = loaded
    root = lib.XDefaultRootWindow(display)
    lib.XWarpPointer(display, 0, root, 0, 0, 0, 0, size[0] - 2, size[1] - 2)
    lib.XFlush(display)


def _get_win_xid(win: "MainWindow") -> int | None:
    """Return the X11 window id of the main window, if any."""
    surface = win.get_surface()
    if isinstance(surface, GdkX11.X11Surface):
        return surface.get_xid()
    return None


def get_xid_geometry(xid: int) -> tuple[int, int, int, int] | None:
    """Return (x, y, width, height) of an X window via xwininfo."""
    result = subprocess.run(
        ["xwininfo", "-id", str(xid)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.debug(f"xwininfo failed: {result.stderr}")
        return None
    logger.debug(f"xwininfo output:\n{result.stdout}")
    info: dict[str, int] = {}
    for line in result.stdout.splitlines():
        stripped = line.strip()
        for key in (
            "Absolute upper-left X",
            "Absolute upper-left Y",
            "Width",
            "Height",
        ):
            if stripped.startswith(key):
                info[key] = int(stripped.split(":")[-1].strip())
    wx = info.get("Absolute upper-left X", 0)
    wy = info.get("Absolute upper-left Y", 0)
    ww = info.get("Width", 0)
    wh = info.get("Height", 0)
    if ww == 0 or wh == 0:
        return None
    return (wx, wy, ww, wh)


def _center_transient_on_parent(win: Gtk.Window) -> None:
    """Center a transient dialog on its parent window."""
    parent = win.get_transient_for()
    if parent is None:
        return
    dsurface = win.get_surface()
    psurface = parent.get_surface()
    if not isinstance(dsurface, GdkX11.X11Surface):
        return
    if not isinstance(psurface, GdkX11.X11Surface):
        return
    geometry = get_xid_geometry(psurface.get_xid())
    if geometry is None:
        return
    px, py, pw, ph = geometry
    dw = dsurface.get_width()
    dh = dsurface.get_height()
    x = max(0, int(px + (pw - dw) / 2))
    y = max(0, int(py + (ph - dh) / 2))
    logger.info(f"Placing dialog '{win.get_title()}' at {x},{y}")
    move_xid(dsurface.get_xid(), x, y)


def _center_when_mapped(win: Gtk.Window, attempts: int) -> bool:
    try:
        if win.get_surface() is None and attempts < 50:
            return GLib.SOURCE_CONTINUE
        _center_transient_on_parent(win)
    except Exception:
        logger.exception("Dialog placement failed")
    return GLib.SOURCE_REMOVE


def install_dialog_placement() -> None:
    """Present transient dialogs centered on their parent.

    Xvfb runs no window manager, so dialogs would otherwise map at
    the top-left corner instead of where a desktop WM places them.
    """
    original_present = Gtk.Window.present

    def present(self, *args, **kwargs):
        result = original_present(self, *args, **kwargs)
        if self.get_transient_for() is not None:
            GLib.idle_add(_center_when_mapped, self, 0)
        return result

    Gtk.Window.present = present


def get_desktop_font_name() -> str | None:
    """Return the desktop default font name (e.g. 'Ubuntu Sans 11')."""
    for prog in ("gsettings", "/usr/bin/gsettings"):
        try:
            result = subprocess.run(
                [prog, "get", "org.gnome.desktop.interface", "font-name"],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError:
            continue
        if result.returncode == 0:
            value = result.stdout.strip().strip("'\"")
            if value:
                return value
    return None


def _apply_gtk_font() -> None:
    """Publish the desktop font via GTK settings.

    The private Xvfb session has no XSettings provider, so GTK
    would fall back to its built-in default font, which renders
    smaller than the desktop's.
    """
    name = get_desktop_font_name()
    if not name:
        return
    settings = Gtk.Settings.get_default()
    if settings is None:
        return
    before = settings.get_property("gtk-font-name")
    if before != name:
        logger.info(f"Overriding GTK font {before!r} with {name!r}")
        settings.set_property("gtk-font-name", name)


def _snapshot_config_dir() -> dict[str, bytes]:
    """Return a byte-exact copy of every file in the config dir."""
    return {
        str(path.relative_to(CONFIG_DIR)): path.read_bytes()
        for path in CONFIG_DIR.rglob("*")
        if path.is_file()
    }


def _write_config_dir(files: dict[str, bytes]) -> None:
    """Restore the config dir to the given snapshot of file contents."""
    for path in CONFIG_DIR.rglob("*"):
        if path.is_file():
            rel = str(path.relative_to(CONFIG_DIR))
            if rel not in files:
                path.unlink()
                logger.debug(f"Removed generated config file: {rel}")
    for rel, content in files.items():
        path = CONFIG_DIR / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def _noop_save() -> None:
    pass


def _noop_save_machine(machine) -> None:
    pass


def _suppress_config_writes() -> None:
    """
    Stop Rayforge from persisting configuration for this process.

    The config manager auto-saves on every config change, the machine
    manager auto-saves on every machine change, and the app saves the
    config once more during shutdown. Both save entry points are
    neutered here, so staging machine settings for a screenshot can
    never reach the disk -- not during the run, not from late driver
    events, and not from the shutdown sequence.
    """
    try:
        context = get_context()
        context.config_mgr.save = _noop_save
        context.machine_mgr.save_machine = _noop_save_machine
        logger.info("Config persistence suppressed for this run")
    except Exception as e:  # noqa: BLE001 - must not break scripts
        logger.warning(f"Could not suppress config writes: {e}")


def restore_config(func: Callable) -> Callable:
    """
    Guarantee the configuration is set back after a screenshot script.

    Screenshot scripts routinely stage machine settings -- switching
    the active machine, changing the WCS, adding cameras, toggling the
    theme -- and Rayforge persists every one of those changes
    immediately. When such a script runs against the shared test
    configuration, those modifications leak into tests/config and
    pollute subsequent runs and commits.

    Two layers of protection are applied for the duration of the
    wrapped function:

    * The config and machine managers' save entry points are replaced
      with no-ops, so nothing can persist through them -- neither
      during the run nor from the app shutdown sequence afterwards.
    * Every file in the configuration directory is snapshotted before
      the run and restored byte-for-byte when it exits, even on error.
      This catches writers outside those two managers, such as dialect
      migrations or machines removed without going through the savers.
      A final restore is registered with atexit so it wins over any
      straggler regardless of thread scheduling.
    """

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        files = _snapshot_config_dir()
        _suppress_config_writes()
        atexit.register(_write_config_dir, files)
        try:
            return func(*args, **kwargs)
        finally:
            _write_config_dir(files)

    return wrapper


def get_target(default: str) -> str:
    """Return the screenshot target, defaulting to ``default``."""
    return os.environ.setdefault("TARGET", default)


def target_to_filename(target: str) -> str:
    """Map a target to its output filename (1:1).

    Targets use ``:`` as a separator (e.g. ``machine-settings:camera``);
    filenames use ``-`` (e.g. ``machine-settings-camera.webp``).

    """
    return target.replace(":", "-") + ".webp"


def get_output_name() -> str:
    """Return the output filename for the current TARGET env var."""
    target = os.environ.get("TARGET")
    if not target:
        raise RuntimeError(
            "TARGET environment variable is not set; run via "
            "'pixi run screenshot <target>'"
        )
    return target_to_filename(target)


WEBP_QUALITY = 90
WEBP_EFFORT = 6


def _encode_webp(img: Image.Image) -> bytes:
    """Encode a PIL image to lossless WebP.

    Encoding is deterministic: identical pixels always produce
    identical files.
    """
    with tempfile.TemporaryDirectory(prefix="rayforge-webp-") as tmp:
        src = Path(tmp) / "input.png"
        img.save(src, format="PNG", compress_level=9)
        image = pyvips.Image.new_from_file(str(src))
        image = image.colourspace("srgb")
        return image.webpsave_buffer(
            lossless=True, Q=WEBP_QUALITY, effort=WEBP_EFFORT
        )


def _images_visually_equal(
    img1: Image.Image,
    img2: Image.Image,
    threshold: int = 5,
    max_different: float = 0.001,
) -> bool:
    """
    Compare two images using a perceptual heuristic.

    Rendered captures carry sub-perceptual pixel noise between runs
    (anti-aliasing, composited shadows), so byte comparison would
    rewrite visually identical screenshots on every run.

    Args:
        img1: First image to compare.
        img2: Second image to compare.
        threshold: Minimum per-channel difference to count as changed
            (0-255).
        max_different: Maximum fraction of pixels that can differ
            (0.0-1.0).

    Returns:
        True if images are visually equal within tolerance.
    """
    arr1 = np.array(img1.convert("RGBA"))
    arr2 = np.array(img2.convert("RGBA"))

    diff = np.abs(arr1.astype(int) - arr2.astype(int))
    significant_diff = np.any(diff > threshold, axis=-1)
    different_pixels = np.sum(significant_diff)
    total_pixels = arr1.shape[0] * arr1.shape[1]

    return different_pixels / total_pixels <= max_different


def _save_webp_deterministic(img: Image.Image, output_path: Path) -> bool:
    """
    Save an image as lossless WebP, only writing if content changed.

    The existing file is decoded — losslessly, so it holds the
    previous capture's exact pixels — and compared perceptually
    against the new capture. Only sub-perceptual render noise
    between runs is tolerated; visually identical screenshots are
    left untouched and identical pixels always produce identical
    files.
    """
    if output_path.exists():
        try:
            existing = Image.open(output_path)
            if existing.size == img.size and _images_visually_equal(
                existing, img
            ):
                logger.info(f"Screenshot unchanged: {output_path}")
                return True
        except (OSError, ValueError) as e:
            logger.debug(f"Comparison failed: {e}")

    output_path.write_bytes(_encode_webp(img))
    logger.info(f"Screenshot saved to {output_path}")
    return True


def run_on_main_thread(func: Callable[[], T], timeout: float = 10.0) -> T:
    """
    Run a function on the main GTK thread and wait for completion.
    """
    result: list[T] = []
    exception: list[Exception | None] = [None]
    done = Event()

    def wrapper() -> bool:
        try:
            result.append(func())
        except Exception as e:  # noqa: BLE001 - arbitrary main-thread callback
            exception[0] = e
        finally:
            done.set()
        return GLib.SOURCE_REMOVE

    GLib.idle_add(wrapper)
    if done.wait(timeout=timeout):
        if exception[0]:
            raise exception[0]
        return result[0]
    raise TimeoutError(f"Function did not complete within {timeout}s")


def take_screenshot(output_name: str | None = None) -> bool:
    """
    Take a screenshot of the app's topmost toplevel window.

    Args:
        output_name: Filename (saved to OUTPUT_DIR). Defaults to the
            filename derived from the TARGET environment variable.

    Returns:
        True if screenshot was saved successfully.
    """
    if output_name is None:
        output_name = get_output_name()
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    park_mouse_pointer()
    time.sleep(0.5)

    try:
        img = capture_app_window()
        if img is None:
            logger.error("Failed to take screenshot with available tools")
            return False
        _save_webp_deterministic(img, output_path)
        logger.info(f"Screenshot saved to {output_path}")
        return True
    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to process screenshot: {e}")
        return False


def take_window_screenshot(
    win: "MainWindow", output_name: str | None = None
) -> bool:
    """
    Take a screenshot of the main window including any open non-modal
    dialogs.  Captures the full screen and crops to the main-window
    geometry obtained from ``xwininfo``.

    Args:
        win: The main window.
        output_name: Filename (saved to OUTPUT_DIR). Defaults to the
            filename derived from the TARGET environment variable.
    """
    if output_name is None:
        output_name = get_output_name()
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    park_mouse_pointer()
    time.sleep(0.5)

    xid = run_on_main_thread(lambda: _get_win_xid(win))
    if xid is None:
        logger.error("Could not get X11 window id")
        return False

    geometry = get_xid_geometry(xid)
    if geometry is None:
        logger.error("Could not parse window geometry from xwininfo")
        return False
    wx, wy, ww, wh = geometry

    logger.info(f"Window geometry: x={wx} y={wy} w={ww} h={wh}")

    def _get_gtk_size():
        return win.get_width(), win.get_height()

    gtk_w, gtk_h = run_on_main_thread(_get_gtk_size)
    shadow_x = (ww - gtk_w) // 2
    shadow_y = (wh - gtk_h) // 2
    logger.info(
        f"GTK size: {gtk_w}x{gtk_h}, shadow offset: {shadow_x},{shadow_y}"
    )

    img = capture_full_screen()
    if img is None:
        return False

    try:
        if uses_synthetic_decorations():
            # No window manager: the captured geometry is the bare
            # client area; crop to it and draw the frame ourselves.
            shadow_x = (ww - gtk_w) // 2
            shadow_y = (wh - gtk_h) // 2
            crop_box = (
                wx + shadow_x,
                wy + shadow_y,
                wx + shadow_x + gtk_w,
                wy + shadow_y + gtk_h,
            )
        else:
            # With a window manager the geometry spans the full
            # client-side-decoration surface; crop to the content
            # inside the CSD shadow margins so the desktop behind
            # the window cannot bleed into the output.
            left, right, top, bottom = _get_frame_extents(xid)
            crop_box = (
                wx + left,
                wy + top,
                wx + ww - right,
                wy + wh - bottom,
            )
        cropped = _add_window_decorations(img.crop(crop_box))
        _save_webp_deterministic(cropped, output_path)
        logger.info(f"Window screenshot saved to {output_path}")
        return True
    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to process screenshot: {e}")
        return False


def take_cropped_screenshot(
    output_name: str | None = None,
    *,
    from_bottom: int | None = None,
    from_top: int | None = None,
    from_left: int | None = None,
    from_right: int | None = None,
) -> bool:
    """
    Takes a screenshot of the app's topmost toplevel window and
    crops margins off it.

    Args:
        output_name: Filename (saved to OUTPUT_DIR). Defaults to the
            filename derived from the TARGET environment variable.
        from_bottom: Crop this many pixels from the bottom.
        from_top: Crop this many pixels from the top.
        from_left: Crop this many pixels from the left.
        from_right: Crop this many pixels from the right.

    Returns:
        True if screenshot was saved successfully.
    """
    if output_name is None:
        output_name = get_output_name()
    output_path = OUTPUT_DIR / output_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    park_mouse_pointer()
    time.sleep(0.5)

    try:
        img = capture_app_window()
        if img is None:
            logger.error("Failed to take screenshot with available tools")
            return False
        width, height = img.size

        left = from_left or 0
        top = from_top or 0
        right = width - (from_right or 0)
        bottom = height - (from_bottom or 0)

        cropped = img.crop((left, top, right, bottom))
        _save_webp_deterministic(cropped, output_path)
        logger.info(f"Cropped screenshot saved to {output_path}")
        return True
    except (OSError, ValueError, TypeError) as e:
        logger.error(f"Failed to crop screenshot: {e}")
        return False


def wait_for_settled(win: "MainWindow", timeout: float = 30.0) -> bool:
    """
    Wait for the document to finish processing.

    Returns:
        True if settled within timeout.
    """
    return win.doc_editor.wait_until_settled_sync(timeout=timeout)


def wait_for_3d_rendered(win: "MainWindow", timeout: float = 15.0) -> bool:
    """
    Wait for the 3D canvas to finish compiling and rendering the scene.

    Waits until:
    - The canvas has a compiled artifact
    - All GL dirty flags have been consumed by a render frame
    - No scene preparation task is in flight

    Returns:
        True if the scene is rendered within timeout.
    """
    start = time.time()

    while time.time() - start < timeout:
        canvas = run_on_main_thread(lambda: win.canvas3d)
        if canvas is None:
            time.sleep(0.1)
            continue

        ready = run_on_main_thread(lambda c=canvas: c.scene_is_ready())
        if ready:
            time.sleep(0.3)
            logger.info("3D scene is compiled and rendered")
            return True

        time.sleep(0.1)

    logger.warning("3D scene did not render within timeout")
    return False


def load_project(win: "MainWindow", project_name: str) -> None:
    """Load a project file from the tests/assets directory."""
    project_path = TESTS_DIR / "assets" / project_name
    if not project_path.exists():
        raise FileNotFoundError(f"Project not found: {project_path}")

    def _load() -> None:
        win.doc_editor.file.load_project_from_path(project_path)

    run_on_main_thread(_load)
    logger.info(f"Loaded project: {project_name}")


def set_window_size(
    win: "MainWindow", width: int, height: int, timeout: float = 1.0
) -> bool:
    """
    Force the window to a specific size, handling maximized state.

    Args:
        win: The main window.
        width: Desired width in pixels.
        height: Desired height in pixels.
        timeout: Time to wait for size to be applied.

    Returns:
        True if size was successfully applied.
    """

    def _set_size() -> None:
        if win.is_maximized():
            win.unmaximize()
        win.set_default_size(width, height)
        win.set_size_request(width, height)

    run_on_main_thread(_set_size)

    actual_width = 0
    actual_height = 0
    applied = False
    start = time.time()
    while time.time() - start < timeout:
        actual_width = run_on_main_thread(lambda: win.get_width())
        actual_height = run_on_main_thread(lambda: win.get_height())
        if actual_width == width and actual_height == height:
            applied = True
            break
        time.sleep(0.1)

    if applied:
        logger.info(f"Window size set to {width}x{height}")
    else:
        logger.warning(
            f"Window size not applied (expected {width}x{height}, "
            f"got {actual_width}x{actual_height})"
        )

    # Anchor the window near the origin so its full client-side
    # decoration surface (shadow margins included) always sits fully
    # on screen, at a deterministic position, no matter how the
    # window manager placed or anchored it before.
    xid = run_on_main_thread(lambda: _get_win_xid(win))
    if xid is not None:
        move_xid(xid, 16, 16)

    return applied


def show_panel(
    win: "MainWindow", panel_name: str, visible: bool = True
) -> None:
    """
    Show or hide a UI panel.

    Args:
        panel_name: Action name (e.g., "toggle_bottom_panel").
        visible: True to show, False to hide.
    """

    def _show() -> None:
        action = win.action_manager.get_action(panel_name)
        action.change_state(GLib.Variant.new_boolean(visible))

    run_on_main_thread(_show)


def show_bottom_tab(win: "MainWindow", tab_name: str) -> None:
    """Switch the bottom panel to a given tab (e.g. 'console' or 'gcode')."""

    def _switch() -> None:
        area = win.bottom_panel.dock_layout.find_item_area(tab_name)
        if area is not None:
            area.set_active_item(tab_name)

    run_on_main_thread(_switch)


def hide_panel(win: "MainWindow", panel_name: str) -> None:
    """Hide a UI panel."""
    show_panel(win, panel_name, visible=False)


def get_panel_state(win: "MainWindow", panel_name: str) -> bool:
    """Get the current visibility state of a panel."""

    def get_state() -> bool:
        action = win.action_manager.get_action(panel_name)
        state = action.get_state()
        if state is None:
            return False
        return state.get_boolean()

    return run_on_main_thread(get_state)


def save_panel_states(
    win: "MainWindow", panel_names: list[str]
) -> dict[str, bool]:
    """Save the current state of multiple panels."""
    return {name: get_panel_state(win, name) for name in panel_names}


def restore_panel_states(win: "MainWindow", states: dict[str, bool]) -> None:
    """Restore panel states from a saved dictionary."""
    for name, visible in states.items():
        show_panel(win, name, visible)


def open_machine_settings(
    win: "MainWindow", page: str = "general"
) -> "MachineSettingsDialog":
    """Open machine settings dialog on the specified page."""

    def _open() -> "MachineSettingsDialog":
        config = get_context().config
        machine = config.machine
        if not machine:
            raise ValueError("No machine configured")
        dialog = MachineSettingsDialog(
            machine=machine,
            transient_for=win,
            initial_page=page,
        )
        dialog.present()
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info(f"Opened machine settings on page: {page}")
    return dialog


def open_app_settings(
    win: "MainWindow", page: str = "general"
) -> "SettingsWindow":
    """Open app settings dialog on the specified page."""

    def _open() -> "SettingsWindow":
        dialog = SettingsWindow(initial_page=page)
        dialog.set_transient_for(win)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info(f"Opened app settings on page: {page}")
    return dialog


def open_step_settings(
    win: "MainWindow", step_index: int = 0, page: str = "step-settings"
) -> "StepSettingsDialog":
    """Open step settings dialog for the step at the given index."""
    step = get_step_by_index(win, step_index)
    if not step:
        raise ValueError(f"Step at index {step_index} not found")

    def _open() -> "StepSettingsDialog":
        dialog = StepSettingsDialog(
            editor=win.doc_editor,
            step=step,
            transient_for=win,
        )
        dialog.set_default_size(600, 900)
        dialog.present()
        dialog.set_initial_page(page)
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info(f"Opened step settings for: {step.name} on page: {page}")
    return dialog


def get_step_by_index(win: "MainWindow", index: int) -> Optional["Step"]:
    """Get a step by its index across all layers."""

    def _get() -> Optional["Step"]:
        step_index = index
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                if step_index < len(layer.workflow.steps):
                    return layer.workflow.steps[step_index]
                step_index -= len(layer.workflow.steps)
        return None

    return run_on_main_thread(_get)


def get_all_steps(win: "MainWindow") -> list["Step"]:
    """Get all steps across all layers."""

    def _get() -> list["Step"]:
        steps: list[Step] = []
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                steps.extend(layer.workflow.steps)
        return steps

    return run_on_main_thread(_get)


def get_step_types(win: "MainWindow") -> list[str]:
    """Get all unique step types (typelabels) in the document."""

    def _get() -> list[str]:
        types: set = set()
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for step in layer.workflow.steps:
                    types.add(step.typelabel.lower().replace(" ", "-"))
        return sorted(types)

    return run_on_main_thread(_get)


def find_step_by_type(
    win: "MainWindow", step_type: str
) -> tuple[Optional["Step"], int]:
    """Find first step matching the given type."""

    def _find() -> tuple[Optional["Step"], int]:
        normalized = step_type.lower().replace(" ", "-")
        for layer in win.doc_editor.doc.layers:
            if layer.workflow and layer.workflow.steps:
                for i, step in enumerate(layer.workflow.steps):
                    if step.typelabel.lower().replace(" ", "-") == normalized:
                        return step, i
        return None, -1

    return run_on_main_thread(_find)


def open_recipe_editor(
    win: "MainWindow",
    page: str = "general",
    *,
    step_type: str | None = None,
    settings_page: int = 0,
) -> "AddEditRecipeDialog":
    """Open recipe editor dialog from app settings.

    Args:
        page: Which tab to activate ("general", "applicability",
            "settings", or "post-processing"). For "settings",
            ``settings_page`` selects which of the dynamic settings
            pages to show. "post-processing" requires ``step_type`` so
            the tab exists.
        step_type: Optional step class name to target. Selecting a step
            type (e.g. a laser step) splits the settings into inherited
            and step-specific pages and enables the post-processing tab.
        settings_page: Index into the dynamic settings pages to activate
            when ``page`` is "settings".
    """

    settings_dialog = open_app_settings(win, "recipes")
    time.sleep(0.5)

    recipe = Recipe(name="3mm Plywood Cut")
    recipe.description = "A recipe for cutting 3mm plywood with a diode laser"
    if step_type:
        recipe.target_step_types = [step_type]

    def _open() -> "AddEditRecipeDialog":
        dialog = AddEditRecipeDialog(
            parent=settings_dialog,
            recipe=recipe,
        )
        dialog.set_default_size(700, 800)
        dialog.present()

        if page == "general":
            dialog._tab_buttons["general"].set_active(True)
        elif page == "applicability":
            dialog._tab_buttons["applicability"].set_active(True)
        elif page == "settings" and dialog._settings_pages:
            names = list(dialog._settings_pages)
            index = min(settings_page, len(names) - 1)
            dialog._tab_buttons[names[index]].set_active(True)
        elif page == "post-processing":
            button = dialog._tab_buttons.get("post-processing")
            if button is not None:
                button.set_active(True)
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info(f"Opened recipe editor on page: {page}")
    return dialog


def open_material_test(win: "MainWindow") -> "StepSettingsDialog":
    """Open material test grid dialog."""

    def _open() -> "StepSettingsDialog":
        context = win.doc_editor.context
        step_cls = step_registry.get("MaterialTestStep")
        assert step_cls is not None
        step = step_cls.create(context)
        step.name = "Material Test Grid"
        dialog = StepSettingsDialog(
            editor=win.doc_editor,
            step=step,
            transient_for=win,
        )
        dialog.set_initial_page("step-settings")
        dialog.set_default_size(600, 900)
        dialog.present()
        return dialog

    dialog = run_on_main_thread(_open)
    logger.info("Opened material test grid dialog")
    return dialog


def open_array_dialog(
    win: "MainWindow", mode: str = "grid"
) -> "_BaseArrayDialog":
    """Open an array dialog for the current selection."""

    mode_map = {
        "grid": (ArrayMode.GRID, GridArrayDialog),
        "point_rotation": (
            ArrayMode.POINT_ROTATION,
            PointRotationArrayDialog,
        ),
        "circular": (ArrayMode.CIRCULAR, CircularArrayDialog),
    }
    _array_mode, cls = mode_map[mode]

    def _open():
        items = list(win.surface.get_selected_items())
        if not items:
            raise ValueError("No items selected")
        dialog = cls(win, win.doc_editor, win.surface, items)
        dialog.present()
        return dialog

    return run_on_main_thread(_open)


def clear_window_subtitle(win: "MainWindow") -> None:
    """
    Clear the version subtitle from the main window for deterministic
    screenshots.
    """

    def _clear() -> None:
        title_widget = win.header_bar.get_title_widget()
        if isinstance(title_widget, Adw.WindowTitle):
            title_widget.set_subtitle("")

    run_on_main_thread(_clear)


def seek_3d_playback(win: "MainWindow", fraction: float) -> None:
    """
    Seek the 3D playback to the given fraction (0.0 to 1.0).
    """

    def _seek() -> None:
        win._canvas3d_playback.seek_to_fraction(fraction)

    run_on_main_thread(_seek)
    time.sleep(0.3)


@contextmanager
def wcs(win: "MainWindow", wcs_name: str):
    """
    Context manager to temporarily switch the active WCS for a screenshot.

    Restores the original WCS on exit.
    """
    machine = win.doc_editor.context.machine
    assert machine is not None
    original = run_on_main_thread(lambda: machine.active_wcs)

    def _switch():
        machine.set_active_wcs(wcs_name)

    run_on_main_thread(_switch)
    try:
        yield
    finally:
        run_on_main_thread(lambda: machine.set_active_wcs(original))


def _bootstrap_xvfb_session() -> None:
    """Apply session fixes once all helpers are defined.

    Runs when the harness is used inside the composited Xvfb
    session: park the pointer so no widget is ever hovered (the
    Xvfb pointer starts on top of the window, and a tooltip shown
    under a modal grab survives even a later pointer warp), place
    unmanaged dialogs ourselves when no window manager is present,
    and publish the desktop font, since the private session has no
    XSettings provider.
    """
    if not _under_xvfb():
        return
    park_mouse_pointer()
    if not _wm_active():
        install_dialog_placement()
    run_on_main_thread(_apply_gtk_font)


_bootstrap_xvfb_session()
