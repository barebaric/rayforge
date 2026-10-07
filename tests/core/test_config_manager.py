"""Tests for ConfigManager persistence behavior."""

from pathlib import Path

import yaml
from blinker import Signal

from rayforge.core.config import ConfigManager


class _FakeMachineManager:
    """Minimal stand-in for the machine manager."""

    def __init__(self):
        self.machine_removed = Signal()

    def get_machine_by_id(self, machine_id):
        return None


def _make_manager(path: Path) -> ConfigManager:
    return ConfigManager(path, _FakeMachineManager())


def test_load_keeps_healthy_file_in_place(tmp_path):
    """A readable config file must not be moved aside on load."""
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"setup_completed": True}))

    manager = _make_manager(path)

    assert manager.config.setup_completed is True
    assert path.exists()
    assert not (tmp_path / "config.yaml.corrupt").exists()


def test_load_recovers_from_backup_and_quarantines_primary(tmp_path):
    """An unreadable config file falls back to the backup, and the
    damaged file is moved aside so it cannot shadow it again."""
    path = tmp_path / "config.yaml"
    path.write_text("{this is not yaml: [")
    (tmp_path / "config.yaml.bak").write_text(
        yaml.safe_dump({"setup_completed": True})
    )

    manager = _make_manager(path)

    assert manager.config.setup_completed is True
    assert not path.exists()
    assert (tmp_path / "config.yaml.corrupt").exists()


def test_save_writes_recovered_settings_back(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("{this is not yaml: [")
    (tmp_path / "config.yaml.bak").write_text(
        yaml.safe_dump({"setup_completed": True})
    )
    manager = _make_manager(path)

    manager.save()

    assert yaml.safe_load(path.read_text())["setup_completed"] is True


def test_save_keeps_backup_of_previous_version(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump({"setup_completed": False}))
    manager = _make_manager(path)

    manager.config.set_setup_completed(True)

    assert yaml.safe_load(path.read_text())["setup_completed"] is True
    backup = tmp_path / "config.yaml.bak"
    assert yaml.safe_load(backup.read_text())["setup_completed"] is False
