"""
LightBurn .lbdev device profile importer.

Provides parsing and conversion of LightBurn device profile files
into Rayforge :class:`DeviceProfile` objects.  The mapping is
best-effort; users are warned that some configuration may need
manual adjustment.
"""

import json
import logging
from dataclasses import dataclass
from gettext import gettext as _
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ...machine.models.machine import Origin

from .profile import (
    DeviceMeta,
    DeviceProfile,
    MachineConfig,
)

logger = logging.getLogger(__name__)

_DRIVER_MAP: Dict[str, Optional[str]] = {
    "Serial": "GrblSerialDriver",
    "Network": "GrblNetworkDriver",
    "Ruida": "RuidaDriver",
    "EZCAD": None,
    "LaserCAD": None,
    "LightBurn": None,
    "Galvo": None,
    "Custom": None,
}

_CUT_ORIGIN_MAP: Dict[int, Optional[str]] = {
    0: None,
    1: "top_left",
    2: "bottom_left",
}

_DEFAULT_GRBL_DIALECT: Dict[str, Any] = {
    "laser_on": "M4 S{power:.0f}",
    "laser_off": "M5",
    "focus_laser_on": "M3 S{power:.0f}",
    "tool_change": "T{tool_number}",
    "set_speed": "",
    "travel_move": "G0{x_cmd}{y_cmd}{z_cmd}{extra_cmd}",
    "linear_move": "G1{x_cmd}{y_cmd}{z_cmd}{extra_cmd}{f_command}",
    "arc_cw": "G2{x_cmd}{y_cmd}{z_cmd}{extra_cmd} I{i} J{j}{f_command}",
    "arc_ccw": "G3{x_cmd}{y_cmd}{z_cmd}{extra_cmd} I{i} J{j}{f_command}",
    "bezier_cubic": "",
    "air_assist_on": "M8",
    "air_assist_off": "M9",
    "home_all": "$H",
    "home_axis": "$H{axis_letter}",
    "move_to": "$J=G90 G21 F{speed} X{x} Y{y}",
    "jog": "$J=G91 G21 F{speed}",
    "clear_alarm": "$X",
    "set_wcs_offset": "G10 L2 P{p_num} X{x} Y{y} Z{z}",
    "probe_cycle": "G38.2 {axis_letter}{max_travel} F{feed_rate}",
    "dwell": "G4 P{seconds:.3f}",
    "preamble": ["G21 ;Set units to mm", "G90 ;Absolute positioning"],
    "postscript": [
        "M5 ;Ensure laser is off",
        "G0 X0 Y0 ;Return to origin",
    ],
    "inject_wcs_after_preamble": True,
    "omit_unchanged_coords": True,
}


@dataclass
class ImportSummary:
    """
    Human-readable summary of values imported from a LightBurn profile.

    Displayed to the user so they can verify what was captured and
    what may need manual configuration.
    """

    name: str = ""
    axis_extents: Optional[Tuple[float, float]] = None
    driver: Optional[str] = None
    driver_args: Optional[Dict[str, Any]] = None
    home_on_start: Optional[bool] = None
    max_travel_speed: Optional[int] = None
    origin: Optional[str] = None
    mirror_x: Optional[bool] = None
    mirror_y: Optional[bool] = None

    def to_lines(self) -> List[str]:
        """Return a bulleted list of human-readable summary lines."""
        lines: List[str] = []
        if self.name:
            lines.append(f"\u2022 Device name: {self.name}")
        if self.axis_extents:
            w, h = self.axis_extents
            lines.append(f"\u2022 Work area: {w:.0f} \u00d7 {h:.0f} mm")
        if self.driver:
            lines.append(f"\u2022 Driver: {self.driver}")
        if self.driver_args and "baudrate" in self.driver_args:
            lines.append(f"\u2022 Baud rate: {self.driver_args['baudrate']}")
        if self.home_on_start is not None:
            lines.append(f"\u2022 Home on start: {self.home_on_start}")
        if self.max_travel_speed is not None:
            lines.append(
                f"\u2022 Max travel speed: {self.max_travel_speed} mm/min"
            )
        if self.origin:
            lines.append(f"\u2022 Origin: {self.origin}")
        if self.mirror_x is not None:
            lines.append(f"\u2022 Mirror X: {self.mirror_x}")
        if self.mirror_y is not None:
            lines.append(f"\u2022 Mirror Y: {self.mirror_y}")
        if not lines:
            lines.append(_("(no fields mapped)"))
        return lines

    def to_items(self) -> List[Tuple[str, str]]:
        """Return ``(field_label, value)`` pairs for table display."""
        items: List[Tuple[str, str]] = []
        if self.name:
            items.append((_("Device name"), self.name))
        if self.axis_extents:
            w, h = self.axis_extents
            items.append((_("Work area"), f"{w:.0f} \u00d7 {h:.0f} mm"))
        if self.driver:
            items.append((_("Driver"), self.driver))
        if self.driver_args and "baudrate" in self.driver_args:
            items.append((_("Baud rate"), str(self.driver_args["baudrate"])))
        if self.home_on_start is not None:
            items.append((_("Home on start"), str(self.home_on_start)))
        if self.max_travel_speed is not None:
            items.append(
                (_("Max travel speed"), f"{self.max_travel_speed} mm/min")
            )
        if self.origin:
            items.append((_("Origin"), self.origin))
        if self.mirror_x is not None:
            items.append((_("Mirror X"), str(self.mirror_x)))
        if self.mirror_y is not None:
            items.append((_("Mirror Y"), str(self.mirror_y)))
        return items


def parse_lbdev(path: Path) -> Dict[str, Any]:
    """Parse a LightBurn .lbdev JSON file and return the first device."""
    if not path.exists():
        raise FileNotFoundError(f"LightBurn profile not found: {path}")
    try:
        with open(path, "r", encoding="utf-8-sig") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid LightBurn profile JSON: {e}")

    if not isinstance(data, dict) or "DeviceList" not in data:
        raise ValueError("LightBurn profile missing 'DeviceList' key")
    device_list = data["DeviceList"]
    if not device_list:
        raise ValueError("LightBurn profile has empty DeviceList")
    return device_list[0]


def _map_driver(device_type: str) -> Optional[str]:
    return _DRIVER_MAP.get(device_type)


def _map_origin(cut_origin: Optional[int]) -> Optional[str]:
    if cut_origin is None:
        return None
    return _CUT_ORIGIN_MAP.get(cut_origin)


def convert_to_profile(
    lbdev_path: Path,
) -> Tuple[DeviceProfile, ImportSummary]:
    """
    Parse a LightBurn ``.lbdev`` file and produce a
    :class:`DeviceProfile` and :class:`ImportSummary`.

    The returned ``DeviceProfile`` has no ``source_dir`` set.  Use
    :meth:`DeviceProfileManager.install_from_lbdev` to persist it
    to the user devices directory.
    """
    device_data = parse_lbdev(lbdev_path)
    meta, machine_config, summary = _convert(device_data)

    profile = DeviceProfile(
        meta=meta,
        machine_config=machine_config,
        dialect_config=_DEFAULT_GRBL_DIALECT,
    )

    return profile, summary


def _convert(
    device_data: Dict[str, Any],
) -> Tuple[DeviceMeta, "MachineConfig", "ImportSummary"]:
    """
    Convert a parsed LightBurn device dict to Rayforge models.
    """
    name = device_data.get("DisplayName") or device_data.get("Name", "Unknown")
    meta = DeviceMeta(
        name=name,
    )
    summary = ImportSummary(name=name)

    kwargs: Dict[str, Any] = {}

    device_type = device_data.get("Type", "")
    driver_name = _map_driver(device_type)
    if driver_name:
        kwargs["driver"] = driver_name
        summary.driver = driver_name

    settings = device_data.get("Settings", {})

    baud_rate = settings.get("BaudRate")
    if baud_rate and driver_name == "GrblSerialDriver":
        kwargs["driver_args"] = {"baudrate": str(baud_rate)}
        summary.driver_args = {"baudrate": str(baud_rate)}

    width = device_data.get("Width")
    height = device_data.get("Height")
    if width is not None and height is not None:
        kwargs["axis_extents"] = (float(width), float(height))
        summary.axis_extents = (float(width), float(height))

    home = device_data.get("HomeOnStartup")
    if home is not None:
        kwargs["home_on_start"] = bool(home)
        summary.home_on_start = bool(home)

    rapid_speed = settings.get("Sim_RapidSpeed")
    max_speed_x = settings.get("Sim_MaxSpeedX")
    if rapid_speed is not None:
        kwargs["max_travel_speed"] = int(rapid_speed)
        summary.max_travel_speed = int(rapid_speed)
    elif max_speed_x is not None:
        kwargs["max_travel_speed"] = int(max_speed_x)
        summary.max_travel_speed = int(max_speed_x)

    cut_origin = settings.get("CutOrigin")
    mapped_origin = _map_origin(cut_origin)
    if mapped_origin:
        kwargs["origin"] = Origin(mapped_origin)
        summary.origin = mapped_origin

    mirror_x = device_data.get("MirrorX")
    mirror_y = device_data.get("MirrorY")
    if mirror_x is not None:
        summary.mirror_x = bool(mirror_x)
    if mirror_y is not None:
        summary.mirror_y = bool(mirror_y)

    return meta, MachineConfig(**kwargs), summary
