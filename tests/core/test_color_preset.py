"""Tests for the color preset (color rule) system."""

from pathlib import Path

import pytest
import yaml

from rayforge import config
from rayforge.core.color_preset import (
    ColorPreset,
    ColorPresetManager,
    get_color_preset_mgr,
    reset_color_preset_mgr,
)


class TestColorPreset:
    def test_roundtrip_dict(self):
        preset = ColorPreset(
            color="#e34c4c",
            step_type="ContourStep",
            label="Cut Red",
            uid="abc",
        )
        data = preset.to_dict()
        restored = ColorPreset.from_dict(data)
        assert restored == preset

    def test_from_dict_generates_uid(self):
        preset = ColorPreset.from_dict({"color": "#e34c4c"})
        assert preset.uid


class TestColorPresetManager:
    @pytest.fixture
    def presets_dir(self, tmp_path: Path) -> Path:
        return tmp_path / "color_presets"

    def test_creation_empty(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        assert manager.all_presets() == []
        # No file should be created until a preset is saved.
        assert not presets_dir.exists()

    def test_add_and_get(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(
            ColorPreset(color="#e34c4c", step_type="ContourStep")
        )
        preset = manager.get_preset("#E34C4C")
        assert preset is not None
        assert preset.step_type == "ContourStep"
        # Color is normalized on store.
        assert preset.color == "#e34c4c"

    def test_add_normalizes_color(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(
            ColorPreset(color="#E34C4C", step_type="EngraveStep")
        )
        preset = manager.get_preset("#e34c4c")
        assert preset is not None
        assert preset.step_type == "EngraveStep"

    def test_add_invalid_color_ignored(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(ColorPreset(color="not-a-color", step_type="Cut"))
        assert manager.all_presets() == []

    def test_add_replaces_existing_color(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(
            ColorPreset(color="#e34c4c", step_type="ContourStep")
        )
        manager.add_preset(
            ColorPreset(color="#e34c4c", step_type="EngraveStep")
        )
        assert len(manager.all_presets()) == 1
        preset = manager.get_preset("#e34c4c")
        assert preset is not None
        assert preset.step_type == "EngraveStep"

    def test_delete(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(
            ColorPreset(color="#e34c4c", step_type="ContourStep")
        )
        assert manager.delete_preset("#E34C4C") is True
        assert manager.get_preset("#e34c4c") is None
        assert manager.all_presets() == []

    def test_delete_unknown_returns_false(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        assert manager.delete_preset("#ffffff") is False

    def test_persist_roundtrip(self, presets_dir: Path):
        manager = ColorPresetManager(presets_dir)
        manager.add_preset(
            ColorPreset(
                color="#e34c4c", step_type="ContourStep", label="Cut Red"
            )
        )
        manager.add_preset(
            ColorPreset(color="#00ff00", step_type="EngraveStep")
        )

        reloaded = ColorPresetManager(presets_dir)
        assert len(reloaded.all_presets()) == 2
        red = reloaded.get_preset("#e34c4c")
        assert red is not None
        assert red.label == "Cut Red"
        green = reloaded.get_preset("#00ff00")
        assert green is not None
        assert green.step_type == "EngraveStep"

    def test_load_ignores_invalid_entries(self, presets_dir: Path):
        presets_dir.mkdir(parents=True)
        (presets_dir / "color_presets.yaml").write_text(
            yaml.safe_dump({"#e34c4c": {"step_type": "ContourStep"}})
        )
        manager = ColorPresetManager(presets_dir)
        assert len(manager.all_presets()) == 1


class TestColorPresetManagerSingleton:
    def test_get_and_reset(self, tmp_path: Path, monkeypatch):
        reset_color_preset_mgr()
        try:
            monkeypatch.setattr(
                config, "USER_COLOR_PRESETS_DIR", tmp_path / "color_presets"
            )
            manager = get_color_preset_mgr()
            assert isinstance(manager, ColorPresetManager)
            assert get_color_preset_mgr() is manager
        finally:
            reset_color_preset_mgr()
