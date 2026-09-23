import yaml

from rayforge.shared.util.atomic import (
    atomic_write_yaml,
    load_yaml_with_backup,
)


def test_atomic_write_creates_file(tmp_path):
    path = tmp_path / "data.yaml"
    atomic_write_yaml(path, {"a": 1})
    assert path.exists()
    assert yaml.safe_load(path.read_text()) == {"a": 1}


def test_atomic_write_leaves_no_tmp_file(tmp_path):
    path = tmp_path / "data.yaml"
    atomic_write_yaml(path, {"a": 1})
    assert not (tmp_path / "data.yaml.tmp").exists()


def test_atomic_write_keeps_backup_of_previous_version(tmp_path):
    path = tmp_path / "data.yaml"
    atomic_write_yaml(path, {"version": 1})
    atomic_write_yaml(path, {"version": 2})
    assert yaml.safe_load(path.read_text()) == {"version": 2}
    backup = tmp_path / "data.yaml.bak"
    assert backup.exists()
    assert yaml.safe_load(backup.read_text()) == {"version": 1}


def test_atomic_write_serialization_error_leaves_original_intact(tmp_path):
    class Unserializable:
        pass

    path = tmp_path / "data.yaml"
    atomic_write_yaml(path, {"version": 1})
    try:
        atomic_write_yaml(path, {"bad": Unserializable()})
    except yaml.YAMLError:
        pass
    assert yaml.safe_load(path.read_text()) == {"version": 1}
    assert not (tmp_path / "data.yaml.tmp").exists()


def test_load_returns_primary_content(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("value: 42\n")
    data, recovered = load_yaml_with_backup(path)
    assert data == {"value": 42}
    assert recovered is False


def test_load_falls_back_to_backup_when_primary_is_corrupt(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("{this is not yaml: [")
    backup = tmp_path / "data.yaml.bak"
    backup.write_text("recovered: true\n")
    data, recovered = load_yaml_with_backup(path)
    assert data == {"recovered": True}
    assert recovered is True


def test_load_raises_without_backup_when_primary_is_corrupt(tmp_path):
    path = tmp_path / "data.yaml"
    path.write_text("{this is not yaml: [")
    try:
        load_yaml_with_backup(path)
    except yaml.YAMLError:
        pass
    else:
        raise AssertionError("expected YAMLError")


def test_load_raises_file_not_found_without_any_file(tmp_path):
    path = tmp_path / "missing.yaml"
    try:
        load_yaml_with_backup(path)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("expected FileNotFoundError")
