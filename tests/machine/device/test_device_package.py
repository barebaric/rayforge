import pytest
import zipfile
from pathlib import Path

import yaml

from rayforge.machine.device.package import (
    DevicePackage,
    export_machine_to_dir,
    CURRENT_API_VERSION,
    DIALECT_FILENAME,
    MANIFEST_FILENAME,
)
from rayforge.machine.models.machine import Origin
from rayforge.machine.device.manager import DevicePackageManager


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


GRBL_DIALECT_YAML = {
    "laser_on": "M4 S{power:.0f}",
    "laser_off": "M5",
    "focus_laser_on": "M4 S{power:.0f}",
    "tool_change": "T{tool_number}",
    "set_speed": "",
    "travel_move": "G0{x_cmd}{y_cmd}{z_cmd}",
    "linear_move": "G1{x_cmd}{y_cmd}{z_cmd}{f_command}",
    "arc_cw": "G2{x_cmd}{y_cmd}{z_cmd} I{i} J{j}{f_command}",
    "arc_ccw": "G3{x_cmd}{y_cmd}{z_cmd} I{i} J{j}{f_command}",
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
    "preamble": ["G21 ;Set units to mm", "G90 ;Absolute positioning"],
    "postscript": ["M5 ;Ensure laser is off", "G0 X0 Y0 ;Return to origin"],
    "inject_wcs_after_preamble": True,
    "omit_unchanged_coords": True,
}


def _write_grbl_dialect(device_dir: Path, overrides=None):
    data = dict(GRBL_DIALECT_YAML)
    if overrides:
        data.update(overrides)
    _write_yaml(device_dir / DIALECT_FILENAME, data)


def _make_device(
    tmp_path: Path,
    name: str = "Test Device",
    subdir: str = "test-device",
    machine_extra=None,
    dialect_overrides=None,
) -> Path:
    device_dir = tmp_path / subdir
    machine = {
        "driver": "GrblSerialDriver",
        **(machine_extra or {}),
    }
    _write_yaml(
        device_dir / "device.yaml",
        {
            "api_version": CURRENT_API_VERSION,
            "device": {"name": name},
            "machine": machine,
        },
    )
    _write_grbl_dialect(device_dir, dialect_overrides)
    return device_dir


class TestDevicePackageLoad:
    def test_load_minimal(self, tmp_path):
        device_dir = _make_device(tmp_path)
        pkg = DevicePackage.from_path(device_dir)

        assert isinstance(pkg, DevicePackage)
        assert pkg.meta.name == "Test Device"
        assert pkg.meta.api_version == CURRENT_API_VERSION
        assert pkg.machine_config.driver == "GrblSerialDriver"
        assert pkg.dialect_config is not None
        assert pkg.dialect_config["laser_on"] == "M4 S{power:.0f}"
        assert pkg.source_dir == device_dir

    def test_load_full_meta(self, tmp_path):
        device_dir = tmp_path / "full-device"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {
                    "name": "OMTech K40+",
                    "vendor": "OMTech",
                    "model": "K40-40W",
                    "description": "40W CO2 laser",
                },
                "machine": {
                    "driver": "GrblSerialDriver",
                    "axis_extents": [300, 200],
                    "origin": "top_left",
                    "supports_arcs": True,
                    "supports_curves": False,
                    "max_travel_speed": 3000,
                    "max_cut_speed": 18000,
                    "gcode_precision": 3,
                },
            },
        )
        _write_grbl_dialect(device_dir)

        pkg = DevicePackage.from_path(device_dir)
        assert pkg.meta.vendor == "OMTech"
        assert pkg.meta.model == "K40-40W"
        assert pkg.meta.description == "40W CO2 laser"
        assert pkg.machine_config.axis_extents == (300, 200)
        assert pkg.machine_config.origin == Origin("top_left")
        assert pkg.machine_config.supports_arcs is True

    def test_load_with_custom_dialect(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Custom Dialect Device",
            dialect_overrides={
                "laser_on": "M3 S{power:.0f}",
                "laser_off": "M5",
            },
        )
        pkg = DevicePackage.from_path(device_dir)
        assert pkg.dialect_config["laser_on"] == "M3 S{power:.0f}"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            DevicePackage.from_path(tmp_path / "nonexistent")

    def test_missing_name_raises(self, tmp_path):
        device_dir = tmp_path / "no-name"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {},
                "machine": {"driver": "GrblSerialDriver"},
            },
        )
        _write_grbl_dialect(device_dir)
        with pytest.raises(ValueError, match="Missing 'device.name'"):
            DevicePackage.from_path(device_dir)

    def test_unsupported_api_version_raises(self, tmp_path):
        device_dir = tmp_path / "bad-version"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 99,
                "device": {"name": "Bad"},
                "machine": {},
            },
        )
        _write_grbl_dialect(device_dir)
        with pytest.raises(ValueError, match="Unsupported api_version"):
            DevicePackage.from_path(device_dir)

    def test_missing_dialect_file_raises(self, tmp_path):
        device_dir = tmp_path / "no-dialect"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "No Dialect"},
                "machine": {"driver": "GrblSerialDriver"},
            },
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            DevicePackage.from_path(device_dir)

    def test_missing_dialect_fields_raises(self, tmp_path):
        device_dir = tmp_path / "partial-dialect"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "Partial Dialect"},
                "machine": {"driver": "GrblSerialDriver"},
            },
        )
        _write_yaml(
            device_dir / DIALECT_FILENAME,
            {"laser_on": "M4 S{power:.0f}"},
        )
        with pytest.raises(ValueError, match="Missing required dialect"):
            DevicePackage.from_path(device_dir)

    def test_invalid_origin_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Origin",
            subdir="bad-origin",
            machine_extra={"origin": "middle_center"},
        )
        with pytest.raises(ValueError, match="Invalid origin"):
            DevicePackage.from_path(device_dir)

    def test_invalid_axis_extents_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Extents",
            subdir="bad-extents",
            machine_extra={"axis_extents": [300]},
        )
        with pytest.raises(ValueError, match="axis_extents"):
            DevicePackage.from_path(device_dir)

    def test_invalid_work_margins_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Margins",
            subdir="bad-margins",
            machine_extra={"work_margins": [5, 5]},
        )
        with pytest.raises(ValueError, match="work_margins"):
            DevicePackage.from_path(device_dir)

    def test_non_mapping_manifest_raises(self, tmp_path):
        device_dir = tmp_path / "bad-manifest"
        device_dir.mkdir(parents=True)
        (device_dir / "device.yaml").write_text("just a string")
        with pytest.raises(ValueError, match="not a mapping"):
            DevicePackage.from_path(device_dir)

    def test_non_mapping_machine_section_raises(self, tmp_path):
        device_dir = tmp_path / "bad-machine"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "Bad Machine"},
                "machine": "not a dict",
            },
        )
        _write_grbl_dialect(device_dir)
        with pytest.raises(ValueError, match="Invalid 'machine'"):
            DevicePackage.from_path(device_dir)

    def test_load_with_heads(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Multi Head",
            subdir="multi-head",
            machine_extra={
                "heads": [
                    {
                        "max_power": 1000,
                        "frame_power_percent": 1.0,
                        "spot_size_mm": [0.1, 0.1],
                    }
                ],
            },
        )
        pkg = DevicePackage.from_path(device_dir)
        assert pkg.machine_config.heads is not None
        assert len(pkg.machine_config.heads) == 1
        assert pkg.machine_config.heads[0]["max_power"] == 1000

    def test_load_with_hookmacros(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Hooks",
            subdir="hooks",
            machine_extra={
                "hookmacros": [
                    {
                        "trigger": "JOB_START",
                        "commands": ["M3", "G4 P0.5"],
                    }
                ],
            },
        )
        pkg = DevicePackage.from_path(device_dir)
        assert pkg.machine_config.hookmacros is not None
        assert len(pkg.machine_config.hookmacros) == 1


class TestDevicePackageManager:
    def test_discover_empty_dir(self, tmp_path):
        mgr = DevicePackageManager([tmp_path / "empty"])
        packages = mgr.discover()
        assert packages == []

    def test_discover_single_package(self, tmp_path):
        _make_device(tmp_path, name="Device A")
        mgr = DevicePackageManager([tmp_path])
        packages = mgr.discover()
        assert len(packages) == 1
        assert packages[0].name == "Device A"

    def test_discover_multiple_packages(self, tmp_path):
        _make_device(tmp_path, name="Device A", subdir="dev-a")
        _make_device(
            tmp_path,
            name="Device B",
            subdir="dev-b",
            machine_extra={"driver": "GrblNetworkDriver"},
        )
        mgr = DevicePackageManager([tmp_path])
        packages = mgr.discover()
        assert len(packages) == 2
        names = {p.name for p in packages}
        assert names == {"Device A", "Device B"}

    def test_user_overrides_builtin(self, tmp_path):
        builtin = tmp_path / "builtin"
        user = tmp_path / "user"
        _make_device(builtin, name="Same Name", subdir="dev")
        _make_device(
            user,
            name="Same Name",
            subdir="dev",
            machine_extra={"driver": "GrblNetworkDriver"},
        )

        mgr = DevicePackageManager([builtin, user])
        packages = mgr.discover()
        assert len(packages) == 1
        assert packages[0].machine_config.driver == "GrblNetworkDriver"

    def test_get_by_name(self, tmp_path):
        _make_device(tmp_path, name="FindMe")
        mgr = DevicePackageManager([tmp_path])
        mgr.discover()
        pkg = mgr.get("FindMe")
        assert pkg is not None
        assert pkg.name == "FindMe"

    def test_get_missing_returns_none(self, tmp_path):
        mgr = DevicePackageManager([tmp_path])
        mgr.discover()
        assert mgr.get("NonExistent") is None

    def test_load_errors_on_bad_package(self, tmp_path):
        bad_dir = tmp_path / "bad-pkg"
        bad_dir.mkdir()
        (bad_dir / "device.yaml").write_text("not a dict")

        mgr = DevicePackageManager([tmp_path])
        mgr.discover()
        errors = mgr.get_load_errors()
        assert len(errors) == 1

    def test_load_package_directly(self, tmp_path):
        device_dir = _make_device(tmp_path, name="Direct")
        mgr = DevicePackageManager()
        pkg = mgr.load_package(device_dir)
        assert pkg.name == "Direct"
        assert mgr.get("Direct") is pkg

    def test_add_source_dir(self, tmp_path):
        mgr = DevicePackageManager()
        mgr.add_source_dir(tmp_path)
        mgr.add_source_dir(tmp_path)
        assert len(mgr.source_dirs) == 1

    def test_sorted_alphabetically(self, tmp_path):
        _make_device(tmp_path, name="Zebra Device", subdir="z-device")
        _make_device(tmp_path, name="Alpha Device", subdir="a-device")
        mgr = DevicePackageManager([tmp_path])
        packages = mgr.discover()
        assert packages[0].name == "Alpha Device"
        assert packages[1].name == "Zebra Device"

    def test_skip_non_directory_entries(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a device")
        _make_device(tmp_path, name="Valid Device")
        mgr = DevicePackageManager([tmp_path])
        packages = mgr.discover()
        assert len(packages) == 1

    def test_skip_dirs_without_manifest(self, tmp_path):
        (tmp_path / "empty-dir").mkdir()
        _make_device(tmp_path, name="Valid Device")
        mgr = DevicePackageManager([tmp_path])
        packages = mgr.discover()
        assert len(packages) == 1

    def test_nonexistent_source_dir_ignored(self, tmp_path):
        mgr = DevicePackageManager([tmp_path / "does_not_exist"])
        packages = mgr.discover()
        assert packages == []


def _make_zip_from_device(
    tmp_path: Path,
    name: str = "Zipped Device",
    subdir: str = "zipped-device",
    machine_extra=None,
    dialect_overrides=None,
    flat: bool = False,
) -> Path:
    device_dir = tmp_path / subdir
    _make_device(
        tmp_path,
        name=name,
        subdir=subdir,
        machine_extra=machine_extra,
        dialect_overrides=dialect_overrides,
    )
    zip_path = tmp_path / (subdir + ".rfdevice.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in sorted(device_dir.rglob("*")):
            if fp.is_file():
                arcname = fp.name if flat else str(fp.relative_to(tmp_path))
                zf.write(fp, arcname)
    return zip_path


class TestInstallFromZip:
    def test_install_flat_zip(self, tmp_path):
        zip_path = _make_zip_from_device(
            tmp_path, flat=True, subdir="flat-dev"
        )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DevicePackageManager(install_dir=install_dir)

        pkg = mgr.install_from_zip(zip_path)

        assert pkg.meta.name == "Zipped Device"
        assert pkg.machine_config.driver == "GrblSerialDriver"
        assert mgr.get("Zipped Device") is pkg
        assert (install_dir / "Zipped Device" / "device.yaml").exists()

    def test_install_nested_zip(self, tmp_path):
        zip_path = _make_zip_from_device(
            tmp_path, flat=False, subdir="nested-dev"
        )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DevicePackageManager(install_dir=install_dir)

        pkg = mgr.install_from_zip(zip_path)

        assert pkg.meta.name == "Zipped Device"
        assert (install_dir / "Zipped Device" / "device.yaml").exists()

    def test_install_missing_zip_raises(self, tmp_path):
        mgr = DevicePackageManager(install_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            mgr.install_from_zip(tmp_path / "missing.zip")

    def test_install_no_install_dir_raises(self, tmp_path):
        zip_path = _make_zip_from_device(tmp_path, subdir="nodev")
        mgr = DevicePackageManager()
        with pytest.raises(RuntimeError, match="install directory"):
            mgr.install_from_zip(zip_path)

    def test_install_bad_api_version_raises(self, tmp_path):
        device_dir = tmp_path / "bad-ver"
        _write_yaml(
            device_dir / MANIFEST_FILENAME,
            {
                "api_version": 99,
                "device": {"name": "BadVer"},
                "machine": {"driver": "GrblSerialDriver"},
            },
        )
        _write_grbl_dialect(device_dir)
        zip_path = tmp_path / "bad-ver.zip"
        with zipfile.ZipFile(zip_path, "w") as zf:
            for fp in sorted(device_dir.rglob("*")):
                if fp.is_file():
                    zf.write(fp, fp.name)

        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DevicePackageManager(install_dir=install_dir)
        with pytest.raises(ValueError, match="Unsupported api_version"):
            mgr.install_from_zip(zip_path)

    def test_install_overwrites_existing(self, tmp_path):
        zip_path = _make_zip_from_device(
            tmp_path,
            name="Overwrite",
            subdir="overwrite-dev",
        )
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DevicePackageManager(install_dir=install_dir)
        mgr.install_from_zip(zip_path)
        assert mgr.get("Overwrite") is not None

        zip_path2 = _make_zip_from_device(
            tmp_path,
            name="Overwrite",
            subdir="overwrite-dev-v2",
            machine_extra={"driver": "GrblNetworkDriver"},
        )
        mgr.install_from_zip(zip_path2)
        pkg = mgr.get("Overwrite")
        assert pkg is not None
        assert pkg.machine_config.driver == "GrblNetworkDriver"


class TestExportToZip:
    def test_export_creates_zip(self, tmp_path):
        _make_device(tmp_path, name="Export Me")
        mgr = DevicePackageManager([tmp_path])
        mgr.discover()
        pkg = mgr.get("Export Me")
        assert pkg is not None

        dest = tmp_path / "output"
        zip_path = mgr.export_to_zip(pkg, dest)

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"
        with zipfile.ZipFile(zip_path, "r") as zf:
            names = zf.namelist()
            assert MANIFEST_FILENAME in names
            assert DIALECT_FILENAME in names

    def test_export_no_source_dir_raises(self, tmp_path):
        from rayforge.machine.device.package import (
            DeviceMeta,
            MachineConfig,
        )

        pkg = DevicePackage(
            meta=DeviceMeta(name="X"),
            machine_config=MachineConfig(),
            dialect_config={},
        )
        mgr = DevicePackageManager()
        with pytest.raises(ValueError, match="no source directory"):
            mgr.export_to_zip(pkg, tmp_path)

    def test_export_roundtrip(self, tmp_path):
        _make_device(
            tmp_path,
            name="Roundtrip",
            machine_extra={"axis_extents": [400, 300]},
        )
        mgr = DevicePackageManager([tmp_path])
        mgr.discover()
        pkg = mgr.get("Roundtrip")
        assert pkg is not None

        dest = tmp_path / "output"
        zip_path = mgr.export_to_zip(pkg, dest)

        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr2 = DevicePackageManager(install_dir=install_dir)
        pkg2 = mgr2.install_from_zip(zip_path)

        assert pkg2.meta.name == "Roundtrip"
        assert pkg2.machine_config.axis_extents == (400, 300)


class TestExportMachine:
    def _make_mock_machine(self, name, **overrides):
        from unittest.mock import MagicMock
        from rayforge.machine.models.dialect import GcodeDialect

        machine = MagicMock()
        machine.name = name
        machine.driver_name = "GrblSerialDriver"
        machine.driver_args = {}
        machine.axis_extents = (300.0, 200.0)
        machine.origin = Origin.BOTTOM_LEFT
        machine.supports_arcs = True
        machine.supports_curves = False
        machine.max_travel_speed = 5000
        machine.max_cut_speed = 1000
        machine.gcode_precision = 3
        machine.home_on_start = False
        machine.rotary_enabled_default = False
        machine.work_margins = (0, 0, 0, 0)
        machine.soft_limits = None
        machine.heads = []
        machine.rotary_modules = {}
        machine.hookmacros = {}
        machine.nogo_zones = {}
        machine.dialect = GcodeDialect.from_dict({"label": name})
        for k, v in overrides.items():
            setattr(machine, k, v)
        return machine

    def test_export_machine_basic(self, tmp_path):
        machine = self._make_mock_machine("Test Export")

        dest = tmp_path / "output"
        mgr = DevicePackageManager(install_dir=tmp_path / "inst")
        zip_path = mgr.export_machine(machine, dest)

        assert zip_path.exists()
        assert zip_path.name.endswith(".rfdevice.zip")

        with zipfile.ZipFile(zip_path, "r") as zf:
            with zf.open(MANIFEST_FILENAME) as f:
                data = yaml.safe_load(f)
            assert data["device"]["name"] == "Test Export"
            assert data["machine"]["driver"] == "GrblSerialDriver"

    def test_export_machine_to_dir(self, tmp_path):
        machine = self._make_mock_machine("Dir Export")

        dest_dir = tmp_path / "pkg"
        pkg = export_machine_to_dir(machine, dest_dir)

        assert pkg.meta.name == "Dir Export"
        assert (dest_dir / MANIFEST_FILENAME).exists()
        assert (dest_dir / DIALECT_FILENAME).exists()
