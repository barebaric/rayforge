"""OS-level permission checks for hardware the wizard needs.

The unified machine configuration wizard shows a preliminary
"permissions" page whenever one of the hardware categories it relies
on — serial ports for controllers, V4L devices for cameras — is
present on the system but not accessible to the current user. The
checks are intentionally cheap and side-effect free: they only inspect
the filesystem, never open a device.

Platform coverage mirrors the remediation matrix documented in the
troubleshooting guide (docs/troubleshooting/connection.md):

* **Linux, regular install** — USB serial nodes and V4L devices are
  owned by a restricted group: ``dialout`` on Debian/Fedora, ``uucp``
  on Arch and friends, ``video`` for cameras. Rather than guessing,
  the owning group is read straight off the device node and the
  generated ``usermod`` command names exactly that group, followed by
  a ``groups | grep …`` verification command.
* **Linux, Snap install** — confinement adds its own layer: the
  ``serial-port`` interface must be connected (and hotplug enabled)
  before any port can be opened, and the ``camera`` interface for
  V4L access. Group membership may additionally be required, so the
  derived group is mentioned in the note.
* **Windows / macOS** — neither serial ports nor cameras are
  mediated by filesystem permissions, so there is nothing this check
  can meaningfully detect. Driver problems (CH340 on macOS, Device
  Manager warnings on Windows) and ports held by other applications
  surface as connection errors instead.

Each check returns a :class:`PermissionIssue` describing what is wrong
and what the user can do about it. :func:`check_permissions`
aggregates every category in a stable order so the wizard can render
one section per problem.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from gettext import gettext as _

from ...camera.v4l import get_sorted_by_id_paths

logger = logging.getLogger(__name__)

# Groups used when the owning group of a device node cannot be
# determined (unusual udev setups, exotic filesystems).
_DIALOUT_FALLBACK = "dialout"
_VIDEO_FALLBACK = "video"


@dataclass(frozen=True)
class PermissionIssue:
    """A single permission problem the user can act on.

    Attributes:
        category: Short stable id (``"serial"`` / ``"camera"``) used
            by the wizard to group issues and pick an icon.
        title: Human-readable heading, e.g. "Serial Port Access".
        summary: One-line explanation of what is wrong.
        commands: Ordered shell commands to fix the problem — first
            the remedy, then a verification step. Copied individually
            from the wizard page with a single click each.
        note: Optional extra paragraph shown below the commands, e.g.
            "Log out and back in for the change to take effect."
    """

    category: str
    title: str
    summary: str
    commands: list[str] = field(default_factory=list)
    note: str | None = None


def _is_snap() -> bool:
    return "SNAP" in os.environ


def _snap_name() -> str:
    return os.environ.get("SNAP_NAME", "rayforge")


def _device_group_name(paths: list[str]) -> str | None:
    """The owning group name shared by the given device nodes.

    Follows symlinks, so persistent by-id paths resolve to their
    ``/dev/videoN`` target. Returns ``None`` when the group cannot be
    determined (node vanished, unknown gid, non-Unix platform).
    """
    try:
        import grp
    except ImportError:
        return None
    for path in paths:
        try:
            return grp.getgrgid(os.stat(path).st_gid).gr_name
        except (OSError, KeyError):
            continue
    return None


# ----- serial -----------------------------------------------------------


def _list_usb_serial_ports() -> list[str]:
    """USB serial ports visible to the system, ignoring access rights.

    Imported lazily so the module stays importable on systems without
    pyserial (e.g. minimal CI images).
    """
    try:
        from ...machine.transport.serial import SerialTransport

        return [
            p for p in SerialTransport.list_usb_ports() if os.path.exists(p)
        ]
    except Exception:
        logger.debug("Could not list USB serial ports", exc_info=True)
        return []


def check_serial_permissions() -> PermissionIssue | None:
    """Detect a systemic serial-port access problem (Linux only).

    Returns ``None`` when access looks fine: either no ports are
    visible (the discover page already handles "nothing plugged in"),
    or at least one port is readable+writable by the current user.
    Only reports an issue when ports exist yet none can be opened.
    """
    if not sys.platform.startswith("linux"):
        return None

    ports = _list_usb_serial_ports()
    if not ports:
        return None
    blocked = [p for p in ports if not os.access(p, os.R_OK | os.W_OK)]
    if not blocked:
        return None

    group = _device_group_name(blocked) or _DIALOUT_FALLBACK

    if _is_snap():
        snap = _snap_name()
        return PermissionIssue(
            category="serial",
            title=_("Serial Port Access"),
            summary=_(
                "Serial ports were detected, but this Snap is not "
                "allowed to open them. Connect the serial-port "
                "interface so the wizard can talk to your machine."
            ),
            commands=[
                "sudo snap set system experimental.hotplug=true",
                f"sudo snap connect {snap}:serial-port",
                f"snap connections {snap} | grep serial-port",
            ],
            note=_(
                "On Debian-based distributions you must also add your "
                "user to the '{group}' group, then log out and back "
                "in, even under Snap."
            ).format(group=group),
        )

    return PermissionIssue(
        category="serial",
        title=_("Serial Port Access"),
        summary=_(
            "Serial ports were detected, but your user cannot open "
            "them. Add yourself to the '{group}' group — the owner "
            "group of {device} on this system — so the wizard can "
            "communicate with your machine."
        ).format(group=group, device=blocked[0]),
        commands=[
            f"sudo usermod -a -G {group} $USER",
            f"groups | grep {group}",
        ],
        note=_(
            "Log out and log back in (or reboot) for the new group "
            "membership to take effect."
        ),
    )


# ----- camera -----------------------------------------------------------


def _list_v4l_devices() -> list[str]:
    """Persistent V4L device paths, ignoring access rights."""
    try:
        return [p for p in get_sorted_by_id_paths() if os.path.exists(p)]
    except Exception:
        logger.debug("Could not list V4L devices", exc_info=True)
        return []


def check_camera_permissions() -> PermissionIssue | None:
    """Detect a V4L camera access problem (Linux only).

    Returns ``None`` when cameras are either absent (the camera page
    already handles "no cameras") or at least one is readable by the
    current user. Only reports an issue when cameras exist but none
    are accessible.
    """
    if not sys.platform.startswith("linux"):
        return None

    devices = _list_v4l_devices()
    if not devices:
        return None
    blocked = [p for p in devices if not os.access(p, os.R_OK | os.W_OK)]
    if not blocked:
        return None

    group = _device_group_name(blocked) or _VIDEO_FALLBACK

    if _is_snap():
        snap = _snap_name()
        return PermissionIssue(
            category="camera",
            title=_("Camera Access"),
            summary=_(
                "Cameras were detected, but this Snap is not allowed "
                "to open them. Connect the camera interface so the "
                "wizard can preview and calibrate."
            ),
            commands=[
                f"sudo snap connect {snap}:camera",
                f"snap connections {snap} | grep camera",
            ],
        )

    return PermissionIssue(
        category="camera",
        title=_("Camera Access"),
        summary=_(
            "Cameras were detected, but your user cannot open them. "
            "Add yourself to the '{group}' group — the owner group of "
            "{device} on this system — so the wizard can preview and "
            "calibrate."
        ).format(group=group, device=blocked[0]),
        commands=[
            f"sudo usermod -a -G {group} $USER",
            f"groups | grep {group}",
        ],
        note=_(
            "Log out and log back in (or reboot) for the new group "
            "membership to take effect."
        ),
    )


# ----- aggregate --------------------------------------------------------


def check_permissions() -> list[PermissionIssue]:
    """Run every hardware permission check, in a stable order.

    Returns the list of issues found (possibly empty). The wizard
    shows a preliminary page iff this list is non-empty.
    """
    issues: list[PermissionIssue] = []
    serial = check_serial_permissions()
    if serial is not None:
        issues.append(serial)
    camera = check_camera_permissions()
    if camera is not None:
        issues.append(camera)
    return issues


__all__ = [
    "PermissionIssue",
    "check_camera_permissions",
    "check_permissions",
    "check_serial_permissions",
]
