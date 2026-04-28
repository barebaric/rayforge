import asyncio
import pytest
import zipfile
from pathlib import Path

import yaml

from rayforge.machine.device.manager import DeviceProfileManager
from rayforge.machine.device.profile import (
    DeviceProfile,
    export_machine_to_dir,
    CURRENT_API_VERSION,
    DIALECT_FILENAME,
    MANIFEST_FILENAME,
)
from rayforge.machine.models.laser import LaserType
from rayforge.machine.models.machine import Origin
from rayforge.shared.tasker.manager import TaskManager


async def _wait_for_tasks(task_mgr: TaskManager):
    if await asyncio.to_thread(task_mgr.wait_until_settled, 2000):
        return
    pytest.fail("Task manager did not become idle in time.")


def _write_yaml(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


GRBL_DIALECT_YAML = {
    "laser_on": "M4 S{power:.0f}",
    "laser_off": "M5",
    "focus_laser_on": "M3 S{power:.0f}",
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


class TestDeviceProfileLoad:
    def test_load_minimal(self, tmp_path):
        device_dir = _make_device(tmp_path)
        pkg = DeviceProfile.from_path(device_dir)

        assert isinstance(pkg, DeviceProfile)
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

        pkg = DeviceProfile.from_path(device_dir)
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
        pkg = DeviceProfile.from_path(device_dir)
        assert pkg.dialect_config["laser_on"] == "M3 S{power:.0f}"

    def test_missing_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            DeviceProfile.from_path(tmp_path / "nonexistent")

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
            DeviceProfile.from_path(device_dir)

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
            DeviceProfile.from_path(device_dir)

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
            DeviceProfile.from_path(device_dir)

    def test_missing_dialect_ok_for_non_gcode_driver(self, tmp_path):
        device_dir = tmp_path / "ruida-no-dialect"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "Ruida Device"},
                "machine": {"driver": "RuidaDriver"},
            },
        )
        pkg = DeviceProfile.from_path(device_dir)
        assert pkg.machine_config.driver == "RuidaDriver"
        assert pkg.dialect_config == {}

    def test_dialect_optional_when_driver_absent(self, tmp_path):
        device_dir = tmp_path / "no-driver-no-dialect"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "No Driver"},
                "machine": {},
            },
        )
        with pytest.raises(FileNotFoundError, match="not found"):
            DeviceProfile.from_path(device_dir)

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
            DeviceProfile.from_path(device_dir)

    def test_invalid_origin_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Origin",
            subdir="bad-origin",
            machine_extra={"origin": "middle_center"},
        )
        with pytest.raises(ValueError, match="Invalid origin"):
            DeviceProfile.from_path(device_dir)

    def test_invalid_axis_extents_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Extents",
            subdir="bad-extents",
            machine_extra={"axis_extents": [300]},
        )
        with pytest.raises(ValueError, match="axis_extents"):
            DeviceProfile.from_path(device_dir)

    def test_invalid_work_margins_raises(self, tmp_path):
        device_dir = _make_device(
            tmp_path,
            name="Bad Margins",
            subdir="bad-margins",
            machine_extra={"work_margins": [5, 5]},
        )
        with pytest.raises(ValueError, match="work_margins"):
            DeviceProfile.from_path(device_dir)

    def test_non_mapping_manifest_raises(self, tmp_path):
        device_dir = tmp_path / "bad-manifest"
        device_dir.mkdir(parents=True)
        (device_dir / "device.yaml").write_text("just a string")
        with pytest.raises(ValueError, match="not a mapping"):
            DeviceProfile.from_path(device_dir)

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
            DeviceProfile.from_path(device_dir)

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
        pkg = DeviceProfile.from_path(device_dir)
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
        pkg = DeviceProfile.from_path(device_dir)
        assert pkg.machine_config.hookmacros is not None
        assert len(pkg.machine_config.hookmacros) == 1


class TestDeviceProfileManager:
    def test_discover_empty_dir(self, tmp_path):
        mgr = DeviceProfileManager([tmp_path / "empty"])
        profiles = mgr.discover()
        assert profiles == []

    def test_discover_single_profile(self, tmp_path):
        _make_device(tmp_path, name="Device A")
        mgr = DeviceProfileManager([tmp_path])
        profiles = mgr.discover()
        assert len(profiles) == 1
        assert profiles[0].name == "Device A"

    def test_discover_multiple_profiles(self, tmp_path):
        _make_device(tmp_path, name="Device A", subdir="dev-a")
        _make_device(
            tmp_path,
            name="Device B",
            subdir="dev-b",
            machine_extra={"driver": "GrblNetworkDriver"},
        )
        mgr = DeviceProfileManager([tmp_path])
        profiles = mgr.discover()
        assert len(profiles) == 2
        names = {p.name for p in profiles}
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

        mgr = DeviceProfileManager([builtin, user])
        profiles = mgr.discover()
        assert len(profiles) == 1
        assert profiles[0].machine_config.driver == "GrblNetworkDriver"

    def test_get_by_name(self, tmp_path):
        _make_device(tmp_path, name="FindMe")
        mgr = DeviceProfileManager([tmp_path])
        mgr.discover()
        pkg = mgr.get("FindMe")
        assert pkg is not None
        assert pkg.name == "FindMe"

    def test_get_missing_returns_none(self, tmp_path):
        mgr = DeviceProfileManager([tmp_path])
        mgr.discover()
        assert mgr.get("NonExistent") is None

    def test_load_errors_on_bad_profile(self, tmp_path):
        bad_dir = tmp_path / "bad-profile"
        bad_dir.mkdir()
        (bad_dir / "device.yaml").write_text("not a dict")

        mgr = DeviceProfileManager([tmp_path])
        mgr.discover()
        errors = mgr.get_load_errors()
        assert len(errors) == 1

    def test_load_profile_directly(self, tmp_path):
        device_dir = _make_device(tmp_path, name="Direct")
        mgr = DeviceProfileManager()
        pkg = mgr.load_profile(device_dir)
        assert pkg.name == "Direct"
        assert mgr.get("Direct") is pkg

    def test_add_source_dir(self, tmp_path):
        mgr = DeviceProfileManager()
        mgr.add_source_dir(tmp_path)
        mgr.add_source_dir(tmp_path)
        assert len(mgr.source_dirs) == 1

    def test_sorted_alphabetically(self, tmp_path):
        _make_device(tmp_path, name="Zebra Device", subdir="z-device")
        _make_device(tmp_path, name="Alpha Device", subdir="a-device")
        mgr = DeviceProfileManager([tmp_path])
        profiles = mgr.discover()
        assert profiles[0].name == "Alpha Device"
        assert profiles[1].name == "Zebra Device"

    def test_skip_non_directory_entries(self, tmp_path):
        (tmp_path / "readme.txt").write_text("not a device")
        _make_device(tmp_path, name="Valid Device")
        mgr = DeviceProfileManager([tmp_path])
        profiles = mgr.discover()
        assert len(profiles) == 1

    def test_skip_dirs_without_manifest(self, tmp_path):
        (tmp_path / "empty-dir").mkdir()
        _make_device(tmp_path, name="Valid Device")
        mgr = DeviceProfileManager([tmp_path])
        profiles = mgr.discover()
        assert len(profiles) == 1

    def test_nonexistent_source_dir_ignored(self, tmp_path):
        mgr = DeviceProfileManager([tmp_path / "does_not_exist"])
        profiles = mgr.discover()
        assert profiles == []


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
        mgr = DeviceProfileManager(install_dir=install_dir)

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
        mgr = DeviceProfileManager(install_dir=install_dir)

        pkg = mgr.install_from_zip(zip_path)

        assert pkg.meta.name == "Zipped Device"
        assert (install_dir / "Zipped Device" / "device.yaml").exists()

    def test_install_missing_zip_raises(self, tmp_path):
        mgr = DeviceProfileManager(install_dir=tmp_path)
        with pytest.raises(FileNotFoundError, match="not found"):
            mgr.install_from_zip(tmp_path / "missing.zip")

    def test_install_no_install_dir_raises(self, tmp_path):
        zip_path = _make_zip_from_device(tmp_path, subdir="nodev")
        mgr = DeviceProfileManager()
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
        mgr = DeviceProfileManager(install_dir=install_dir)
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
        mgr = DeviceProfileManager(install_dir=install_dir)
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
        mgr = DeviceProfileManager([tmp_path])
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
        from rayforge.machine.device.profile import (
            DeviceMeta,
            MachineConfig,
        )

        pkg = DeviceProfile(
            meta=DeviceMeta(name="X"),
            machine_config=MachineConfig(),
            dialect_config={},
        )
        mgr = DeviceProfileManager()
        with pytest.raises(ValueError, match="no source directory"):
            mgr.export_to_zip(pkg, tmp_path)

    def test_export_roundtrip(self, tmp_path):
        _make_device(
            tmp_path,
            name="Roundtrip",
            machine_extra={"axis_extents": [400, 300]},
        )
        mgr = DeviceProfileManager([tmp_path])
        mgr.discover()
        pkg = mgr.get("Roundtrip")
        assert pkg is not None

        dest = tmp_path / "output"
        zip_path = mgr.export_to_zip(pkg, dest)

        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr2 = DeviceProfileManager(install_dir=install_dir)
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

    def _make_mock_ruida_machine(self, name):
        return self._make_mock_machine(
            name,
            driver_name="RuidaDriver",
            dialect=None,
        )

    def test_export_machine_basic(self, tmp_path):
        machine = self._make_mock_machine("Test Export")

        dest = tmp_path / "output"
        mgr = DeviceProfileManager(install_dir=tmp_path / "inst")
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

    def test_export_ruida_machine_no_dialect(self, tmp_path):
        machine = self._make_mock_ruida_machine("Ruida Export")

        dest_dir = tmp_path / "pkg"
        pkg = export_machine_to_dir(machine, dest_dir)

        assert pkg.meta.name == "Ruida Export"
        assert (dest_dir / MANIFEST_FILENAME).exists()
        assert not (dest_dir / DIALECT_FILENAME).exists()


class TestCreateMachine:
    @pytest.mark.asyncio
    async def test_create_machine_gcode_has_dialect(
        self, tmp_path, lite_context, task_mgr
    ):
        device_dir = _make_device(tmp_path, name="GRBL Test")
        pkg = DeviceProfile.from_path(device_dir)
        m = pkg.create_machine(lite_context)
        await _wait_for_tasks(task_mgr)

        assert m.dialect_uid is not None
        assert m.dialect is not None

    @pytest.mark.asyncio
    async def test_create_machine_ruida_no_dialect(
        self, tmp_path, lite_context, task_mgr
    ):
        device_dir = tmp_path / "ruida-device"
        _write_yaml(
            device_dir / "device.yaml",
            {
                "api_version": 1,
                "device": {"name": "Ruida Test"},
                "machine": {"driver": "RuidaDriver"},
            },
        )
        pkg = DeviceProfile.from_path(device_dir)
        m = pkg.create_machine(lite_context)
        await _wait_for_tasks(task_mgr)

        assert m.dialect_uid is None
        assert m.dialect is None
        assert m.driver_name == "RuidaDriver"

    @pytest.mark.asyncio
    async def test_co2_profile_loads_pwm_fields(
        self, tmp_path, lite_context, task_mgr
    ):
        device_dir = _make_device(
            tmp_path,
            name="CO2 PWM Test",
            subdir="co2-pwm",
            machine_extra={
                "heads": [
                    {
                        "max_power": 1000,
                        "laser_type": "co2",
                        "pwm_frequency": 1000,
                        "max_pwm_frequency": 5000,
                        "pulse_width": 50,
                        "min_pulse_width": 5,
                        "max_pulse_width": 500,
                    }
                ],
            },
        )
        pkg = DeviceProfile.from_path(device_dir)
        m = pkg.create_machine(lite_context)
        await _wait_for_tasks(task_mgr)

        assert len(m.heads) == 1
        head = m.heads[0]
        assert head.laser_type == LaserType.CO2
        assert head.pwm_frequency == 1000
        assert head.max_pwm_frequency == 5000
        assert head.pulse_width == 50
        assert head.min_pulse_width == 5
        assert head.max_pulse_width == 500

    @pytest.mark.asyncio
    async def test_diode_profile_defaults_zero_pwm(
        self, tmp_path, lite_context, task_mgr
    ):
        device_dir = _make_device(
            tmp_path,
            name="Diode Test",
            subdir="diode-test",
            machine_extra={
                "heads": [
                    {
                        "max_power": 1000,
                    }
                ],
            },
        )
        pkg = DeviceProfile.from_path(device_dir)
        m = pkg.create_machine(lite_context)
        await _wait_for_tasks(task_mgr)

        head = m.heads[0]
        assert head.laser_type == LaserType.DIODE
