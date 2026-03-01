import pytest
import yaml
from rayforge.core.addon_config import (
    AddonConfig,
    AddonState,
    AddonConfigEntry,
)


class TestAddonConfigEntry:
    """Tests for AddonConfigEntry class."""

    def test_to_dict_with_version(self):
        entry = AddonConfigEntry(state="enabled", version="1.0.0")
        assert entry.to_dict() == {"state": "enabled", "version": "1.0.0"}

    def test_to_dict_without_version(self):
        entry = AddonConfigEntry(state="disabled", version=None)
        assert entry.to_dict() == {"state": "disabled", "version": None}

    def test_from_dict_with_all_fields(self):
        data = {"state": "disabled", "version": "2.0.0"}
        entry = AddonConfigEntry.from_dict(data)
        assert entry.state == "disabled"
        assert entry.version == "2.0.0"

    def test_from_dict_with_missing_state(self):
        data = {"version": "1.0.0"}
        entry = AddonConfigEntry.from_dict(data)
        assert entry.state == AddonState.ENABLED
        assert entry.version == "1.0.0"

    def test_from_dict_with_none_state(self):
        data = {"state": None, "version": "1.0.0"}
        entry = AddonConfigEntry.from_dict(data)
        assert entry.state == AddonState.ENABLED


class TestAddonConfig:
    """Tests for AddonConfig class."""

    def test_init_creates_config_file_path(self, tmp_path):
        config = AddonConfig(tmp_path)
        assert config.config_file == tmp_path / "addons.yaml"
        assert config._entries == {}

    def test_load_missing_file_uses_defaults(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.load()
        assert config._entries == {}

    def test_load_reads_existing_file(self, tmp_path):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("addon1:\n  state: disabled\n  version: null\n")

        config = AddonConfig(tmp_path)
        config.load()
        assert "addon1" in config._entries
        assert config._entries["addon1"].state == "disabled"

    def test_load_handles_invalid_yaml(self, tmp_path, caplog):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("not: valid: yaml: ::")

        config = AddonConfig(tmp_path)
        config.load()
        assert config._entries == {}
        assert "Failed to load addon config" in caplog.text

    def test_load_handles_non_dict(self, tmp_path):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("- addon1\n- addon2\n")

        config = AddonConfig(tmp_path)
        config.load()
        assert config._entries == {}

    def test_load_handles_legacy_string_format(self, tmp_path):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("addon1: disabled\n")

        config = AddonConfig(tmp_path)
        config.load()
        assert "addon1" in config._entries
        assert config._entries["addon1"].state == "disabled"
        assert config._entries["addon1"].version is None

    def test_save_creates_file(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(
            state="disabled", version="1.0.0"
        )
        config.save()

        assert config.config_file.exists()
        data = yaml.safe_load(config.config_file.read_text())
        assert data == {"addon1": {"state": "disabled", "version": "1.0.0"}}

    def test_save_creates_parent_directory(self, tmp_path):
        config_dir = tmp_path / "subdir"
        config = AddonConfig(config_dir)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        config.save()

        assert config_dir.exists()
        assert config.config_file.exists()

    def test_save_deletes_file_when_empty(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        config.save()
        assert config.config_file.exists()

        config._entries = {}
        config.save()
        assert not config.config_file.exists()

    def test_get_state_returns_enabled_by_default(self, tmp_path):
        config = AddonConfig(tmp_path)
        assert config.get_state("unknown-addon") == AddonState.ENABLED

    def test_get_state_returns_saved_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        assert config.get_state("addon1") == AddonState.DISABLED

    def test_get_state_handles_invalid_state(self, tmp_path, caplog):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="invalid")
        result = config.get_state("addon1")
        assert result == AddonState.ENABLED
        assert "Invalid state" in caplog.text

    def test_set_state_updates_and_saves(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.set_state("addon1", AddonState.DISABLED)

        assert config._entries["addon1"].state == "disabled"
        assert config.config_file.exists()
        data = yaml.safe_load(config.config_file.read_text())
        assert "addon1" in data

    def test_set_state_rejects_invalid_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        with pytest.raises(ValueError, match="Invalid addon state"):
            config.set_state("addon1", "invalid")

    def test_get_version(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(
            state="enabled", version="1.2.3"
        )
        assert config.get_version("addon1") == "1.2.3"
        assert config.get_version("unknown") is None

    def test_set_version(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.set_version("addon1", "2.0.0")

        assert config._entries["addon1"].version == "2.0.0"
        data = yaml.safe_load(config.config_file.read_text())
        assert data["addon1"]["version"] == "2.0.0"

    def test_set_entry(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.set_entry("addon1", "disabled", "1.0.0")

        assert config._entries["addon1"].state == "disabled"
        assert config._entries["addon1"].version == "1.0.0"

    def test_remove_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        config._entries["addon2"] = AddonConfigEntry(state="enabled")
        config.remove_state("addon1")

        assert "addon1" not in config._entries
        assert "addon2" in config._entries
        data = yaml.safe_load(config.config_file.read_text())
        assert "addon2" in data

    def test_remove_state_deletes_file_when_empty(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        config.remove_state("addon1")

        assert config._entries == {}
        assert not config.config_file.exists()

    def test_remove_state_nonexistent(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._entries["addon1"] = AddonConfigEntry(state="disabled")
        config.remove_state("nonexistent")

        assert "addon1" in config._entries


class TestAddonState:
    """Tests for AddonState constants."""

    def test_enabled_value(self):
        assert AddonState.ENABLED == "enabled"

    def test_disabled_value(self):
        assert AddonState.DISABLED == "disabled"
