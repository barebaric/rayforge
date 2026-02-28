import pytest
import yaml
from rayforge.core.addon_config import AddonConfig, AddonState


class TestAddonConfig:
    """Tests for AddonConfig class."""

    def test_init_creates_config_file_path(self, tmp_path):
        config = AddonConfig(tmp_path)
        assert config.config_file == tmp_path / "addons.yaml"
        assert config._states == {}

    def test_load_missing_file_uses_defaults(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.load()
        assert config._states == {}

    def test_load_reads_existing_file(self, tmp_path):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("addon1: disabled\n")

        config = AddonConfig(tmp_path)
        config.load()
        assert config._states == {"addon1": "disabled"}

    def test_load_handles_invalid_yaml(self, tmp_path, caplog):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("not: valid: yaml: ::")

        config = AddonConfig(tmp_path)
        config.load()
        assert config._states == {}
        assert "Failed to load addon config" in caplog.text

    def test_load_handles_non_dict(self, tmp_path):
        config_file = tmp_path / "addons.yaml"
        config_file.write_text("- addon1\n- addon2\n")

        config = AddonConfig(tmp_path)
        config.load()
        assert config._states == {}

    def test_save_creates_file(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled"}
        config.save()

        assert config.config_file.exists()
        data = yaml.safe_load(config.config_file.read_text())
        assert data == {"addon1": "disabled"}

    def test_save_creates_parent_directory(self, tmp_path):
        config_dir = tmp_path / "subdir"
        config = AddonConfig(config_dir)
        config._states = {"addon1": "disabled"}
        config.save()

        assert config_dir.exists()
        assert config.config_file.exists()

    def test_save_deletes_file_when_empty(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled"}
        config.save()
        assert config.config_file.exists()

        config._states = {}
        config.save()
        assert not config.config_file.exists()

    def test_get_state_returns_enabled_by_default(self, tmp_path):
        config = AddonConfig(tmp_path)
        assert config.get_state("unknown-addon") == AddonState.ENABLED

    def test_get_state_returns_saved_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled"}
        assert config.get_state("addon1") == AddonState.DISABLED

    def test_get_state_handles_invalid_state(self, tmp_path, caplog):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "invalid"}
        result = config.get_state("addon1")
        assert result == AddonState.ENABLED
        assert "Invalid state" in caplog.text

    def test_set_state_updates_and_saves(self, tmp_path):
        config = AddonConfig(tmp_path)
        config.set_state("addon1", AddonState.DISABLED)

        assert config._states["addon1"] == "disabled"
        assert config.config_file.exists()
        data = yaml.safe_load(config.config_file.read_text())
        assert data == {"addon1": "disabled"}

    def test_set_state_rejects_invalid_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        with pytest.raises(ValueError, match="Invalid addon state"):
            config.set_state("addon1", "invalid")

    def test_remove_state(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled", "addon2": "enabled"}
        config.remove_state("addon1")

        assert "addon1" not in config._states
        assert config._states == {"addon2": "enabled"}
        data = yaml.safe_load(config.config_file.read_text())
        assert data == {"addon2": "enabled"}

    def test_remove_state_deletes_file_when_empty(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled"}
        config.remove_state("addon1")

        assert config._states == {}
        assert not config.config_file.exists()

    def test_remove_state_nonexistent(self, tmp_path):
        config = AddonConfig(tmp_path)
        config._states = {"addon1": "disabled"}
        config.remove_state("nonexistent")

        assert config._states == {"addon1": "disabled"}


class TestAddonState:
    """Tests for AddonState constants."""

    def test_enabled_value(self):
        assert AddonState.ENABLED == "enabled"

    def test_disabled_value(self):
        assert AddonState.DISABLED == "disabled"
