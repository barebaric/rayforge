"""
Color rules: map SVG colors to step types at import time.

A :class:`ColorPreset` assigns a step class (e.g. ``ContourStep``,
``EngraveStep``, or any addon-provided step) to a normalized hex color.
The SVG importer resolves these during vectorization and the resulting
layer is given the corresponding step type, after which the regular
recipe matching applies its settings.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from .. import config as rf_config
from .color import normalize_color

logger = logging.getLogger(__name__)


@dataclass
class ColorPreset:
    """
    A single color rule: a color maps to a step class name.

    The step class name refers to a class registered in
    ``step_registry``. It may reference a step that is currently
    unregistered (e.g. its addon is uninstalled); importing tolerates
    this by falling back to the default behavior.
    """

    color: str
    step_type: str
    label: str = ""
    uid: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Serializes the preset to a dictionary suitable for YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ColorPreset":
        """Deserializes a preset from a dictionary."""
        return cls(
            color=data.get("color", ""),
            step_type=data.get("step_type", ""),
            label=data.get("label", ""),
            uid=data.get("uid", str(uuid.uuid4())),
        )


class ColorPresetManager:
    """
    Manages loading and saving ColorPreset objects from a directory.

    Presets are keyed by normalized color; a color maps to at most one
    preset. The YAML file is only created when a preset is first saved.
    """

    def __init__(self, base_dir: Path):
        self.base_dir = base_dir
        self._presets_by_color: dict[str, ColorPreset] = {}
        self.load()

    @property
    def _presets_file(self) -> Path:
        return self.base_dir / "color_presets.yaml"

    def load(self) -> None:
        """Loads all presets from the presets file."""
        self._presets_by_color.clear()
        if not self._presets_file.exists():
            return
        try:
            with open(self._presets_file, "r") as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load color presets: {e}")
            return
        if not isinstance(data, dict):
            logger.warning("Color presets file has an invalid structure.")
            return
        for color, entry in data.items():
            if not isinstance(entry, dict):
                continue
            preset = ColorPreset.from_dict(entry)
            preset.color = color
            self._presets_by_color[color] = preset
        logger.debug(f"Loaded {len(self._presets_by_color)} color presets.")

    def _save(self) -> None:
        """Persists all presets to the presets file."""
        self.base_dir.mkdir(parents=True, exist_ok=True)
        data = {
            color: preset.to_dict()
            for color, preset in self._presets_by_color.items()
        }
        try:
            with open(self._presets_file, "w") as f:
                yaml.safe_dump(data, f, sort_keys=True)
        except Exception as e:
            logger.error(f"Failed to save color presets: {e}")

    def add_preset(self, preset: ColorPreset) -> None:
        """
        Adds or replaces a preset for its color.

        The color is normalized before storage. Replacing an existing
        preset for the same color preserves nothing from the old one.

        Args:
            preset: The preset to store.
        """
        normalized = normalize_color(preset.color)
        if not normalized:
            logger.warning(
                f"Not adding color preset: unreadable color '{preset.color}'."
            )
            return
        preset.color = normalized
        self._presets_by_color[normalized] = preset
        self._save()

    def delete_preset(self, color: str) -> bool:
        """
        Deletes the preset for a given color.

        Args:
            color: The color to remove (normalized before lookup).

        Returns:
            True if a preset was removed, False otherwise.
        """
        normalized = normalize_color(color)
        if normalized and normalized in self._presets_by_color:
            del self._presets_by_color[normalized]
            self._save()
            return True
        return False

    def get_preset(self, color: str) -> Optional[ColorPreset]:
        """
        Returns the preset for a color, or None if none matches.

        Args:
            color: The color to look up (normalized before lookup).

        Returns:
            The matching preset, or None.
        """
        normalized = normalize_color(color)
        if not normalized:
            return None
        return self._presets_by_color.get(normalized)

    def all_presets(self) -> list[ColorPreset]:
        """Returns a list of all stored presets."""
        return list(self._presets_by_color.values())


_color_preset_mgr_instance: Optional[ColorPresetManager] = None


def get_color_preset_mgr() -> ColorPresetManager:
    """
    Returns the process-wide ColorPresetManager, creating it on first use.

    The manager is backed by the user color presets directory and is
    shared between the import pipeline and the settings UI.
    """
    global _color_preset_mgr_instance
    if _color_preset_mgr_instance is None:
        _color_preset_mgr_instance = ColorPresetManager(
            rf_config.USER_COLOR_PRESETS_DIR
        )
    return _color_preset_mgr_instance


def reset_color_preset_mgr() -> None:
    """
    Resets the process-wide ColorPresetManager singleton.

    Intended for tests that need a fresh manager against a different
    directory.
    """
    global _color_preset_mgr_instance
    _color_preset_mgr_instance = None
