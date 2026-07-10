import json
from pathlib import Path
from typing import Optional

import pytest
import yaml

from rayforge.machine.device.lightburn_importer import (
    _DEFAULT_GRBL_DIALECT,
    ImportSummary,
    _map_driver,
    _map_origin,
    convert_to_profile,
    parse_lbdev,
)
from rayforge.machine.device.manager import (
    DIALECT_FILENAME,
    MANIFEST_FILENAME,
    DeviceProfileManager,
)
from rayforge.machine.models.machine import Origin

ASSETS_DIR = Path(__file__).parent / "assets"


def _make_lbdev(
    path: Path,
    overrides: Optional[dict] = None,
    settings_overrides: Optional[dict] = None,
) -> Path:
    """Write a minimal .lbdev file and return its path."""
    device = {
        "DisplayName": "Test Laser",
        "Name": "GRBL",
        "Width": 400,
        "Height": 390,
        "HomeOnStartup": False,
        "MirrorX": False,
        "MirrorY": False,
        "Type": "Serial",
        "Settings": {
            "BaudRate": 115200,
            "Sim_RapidSpeed": 400,
            "Sim_MaxSpeedX": 500,
            "Sim_MaxSpeedY": 400,
            "CutOrigin": 2,
        },
    }
    if overrides:
        device.update(overrides)
    if settings_overrides:
        device["Settings"].update(settings_overrides)

    data = {"DeviceList": [device]}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


class TestParseLBDev:
    def test_parse_minimal(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "test.lbdev")
        result = parse_lbdev(lbdev)
        assert result["DisplayName"] == "Test Laser"
        assert result["Width"] == 400
        assert result["Height"] == 390
        assert result["Type"] == "Serial"
        assert result["Settings"]["BaudRate"] == 115200

    def test_parse_missing_file(self):
        with pytest.raises(FileNotFoundError):
            parse_lbdev(Path("/nonexistent/no.lbdev"))

    def test_parse_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("not json")
        with pytest.raises(ValueError, match="Invalid LightBurn profile"):
            parse_lbdev(p)

    def test_parse_no_devicelist(self, tmp_path):
        p = tmp_path / "no-dl.lbdev"
        p.write_text(json.dumps({"foo": "bar"}))
        with pytest.raises(ValueError, match="missing 'DeviceList'"):
            parse_lbdev(p)

    def test_parse_empty_devicelist(self, tmp_path):
        p = tmp_path / "empty.lbdev"
        p.write_text(json.dumps({"DeviceList": []}))
        with pytest.raises(ValueError, match="empty DeviceList"):
            parse_lbdev(p)

    def test_parse_bom_encoding(self, tmp_path):
        """UTF-8 BOM should be handled gracefully."""
        p = tmp_path / "bom.lbdev"
        raw = json.dumps(
            {"DeviceList": [{"DisplayName": "BOM Test", "Type": "Serial"}]}
        )
        p.write_bytes(b"\xef\xbb\xbf" + raw.encode("utf-8"))
        result = parse_lbdev(p)
        assert result["DisplayName"] == "BOM Test"


class TestMapDriver:
    def test_serial(self):
        assert _map_driver("Serial") == "GrblSerialDriver"

    def test_network(self):
        assert _map_driver("Network") == "GrblNetworkDriver"

    def test_ruida(self):
        assert _map_driver("Ruida") == "RuidaDriver"

    def test_unsupported(self):
        assert _map_driver("Galvo") is None

    def test_unknown(self):
        assert _map_driver("FooBar") is None

    def test_empty(self):
        assert _map_driver("") is None


class TestMapOrigin:
    def test_origin_0(self):
        assert _map_origin(0) is None

    def test_origin_1(self):
        assert _map_origin(1) == "top_left"

    def test_origin_2(self):
        assert _map_origin(2) == "bottom_left"

    def test_none(self):
        assert _map_origin(None) is None


class TestImportSummary:
    def test_empty_summary(self):
        s = ImportSummary()
        lines = s.to_lines()
        assert len(lines) == 1
        assert "no fields" in lines[0]

    def test_full_summary(self, tmp_path):
        s = ImportSummary(
            name="Test Laser",
            axis_extents=(400.0, 390.0),
            driver="GrblSerialDriver",
            driver_args={"baudrate": "115200"},
            home_on_start=False,
            max_travel_speed=400,
            origin="bottom_left",
            mirror_x=False,
            mirror_y=False,
        )
        lines = s.to_lines()
        assert any("Test Laser" in line for line in lines)
        assert any("400" in line and "390" in line for line in lines)
        assert any("GrblSerialDriver" in line for line in lines)
        assert any("115200" in line for line in lines)
        assert any("bottom_left" in line for line in lines)

    def test_to_items_empty(self):
        items = ImportSummary().to_items()
        assert items == []

    def test_to_items_full(self):
        s = ImportSummary(
            name="My Laser",
            axis_extents=(300.0, 200.0),
            driver="GrblSerialDriver",
            driver_args={"baudrate": "115200"},
            home_on_start=True,
            max_travel_speed=500,
            origin="top_left",
            mirror_x=False,
            mirror_y=True,
        )
        items = s.to_items()
        pairs = dict(items)
        assert pairs.get("Device name") == "My Laser"
        assert pairs.get("Work area") == "300 × 200 mm"
        assert pairs.get("Driver") == "GrblSerialDriver"
        assert pairs.get("Baud rate") == "115200"
        assert pairs.get("Home on start") == "True"
        assert pairs.get("Max travel speed") == "500 mm/min"
        assert pairs.get("Origin") == "top_left"
        assert pairs.get("Mirror X") == "False"
        assert pairs.get("Mirror Y") == "True"


class TestConvertToProfile:
    def test_convert_minimal(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "minimal.lbdev")
        profile, summary = convert_to_profile(lbdev)

        assert profile.meta.name == "Test Laser"
        assert profile.machine_config.driver == "GrblSerialDriver"
        assert profile.machine_config.driver_args == {"baudrate": "115200"}
        assert profile.machine_config.axis_extents == (400.0, 390.0)
        assert profile.machine_config.home_on_start is False
        assert profile.machine_config.max_travel_speed == 400
        assert profile.machine_config.origin == Origin("bottom_left")
        assert profile.dialect_config == _DEFAULT_GRBL_DIALECT

        assert summary.name == "Test Laser"
        assert summary.axis_extents == (400.0, 390.0)
        assert summary.driver == "GrblSerialDriver"
        assert summary.driver_args == {"baudrate": "115200"}

    def test_convert_uses_display_name(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "display.lbdev",
            {"DisplayName": "My Laser", "Name": "GRBL"},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.meta.name == "My Laser"

    def test_convert_falls_back_to_name(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "fallback.lbdev",
            {"DisplayName": "", "Name": "Fallback"},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.meta.name == "Fallback"

    def test_convert_network_driver(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "net.lbdev",
            {"Type": "Network"},
        )
        profile, summary = convert_to_profile(lbdev)
        assert profile.machine_config.driver == "GrblNetworkDriver"
        assert summary.driver == "GrblNetworkDriver"

    def test_convert_ruida_driver(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "ruida.lbdev",
            {"Type": "Ruida"},
        )
        profile, summary = convert_to_profile(lbdev)
        assert profile.machine_config.driver == "RuidaDriver"
        assert summary.driver == "RuidaDriver"
        assert profile.machine_config.driver_args is None

    def test_convert_unknown_driver(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "unknown.lbdev",
            {"Type": "Galvo"},
        )
        profile, summary = convert_to_profile(lbdev)
        assert profile.machine_config.driver is None
        assert summary.driver is None

    def test_convert_no_baudrate_for_network(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "net-baud.lbdev",
            {"Type": "Network"},
            {"BaudRate": 115200},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.machine_config.driver_args is None

    def test_convert_no_rapid_speed_uses_max_speed_x(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "nospeed.lbdev",
            settings_overrides={"Sim_RapidSpeed": None},
        )
        profile, summary = convert_to_profile(lbdev)
        assert profile.machine_config.max_travel_speed == 500
        assert summary.max_travel_speed == 500

    def test_convert_no_speeds_at_all(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "nospeed.lbdev",
            settings_overrides={
                "Sim_RapidSpeed": None,
                "Sim_MaxSpeedX": None,
            },
        )
        profile, summary = convert_to_profile(lbdev)
        assert profile.machine_config.max_travel_speed is None
        assert summary.max_travel_speed is None

    def test_convert_origin_0(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "origin0.lbdev",
            settings_overrides={"CutOrigin": 0},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.machine_config.origin is None

    def test_convert_no_dimensions(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "nodims.lbdev",
            {"Width": None, "Height": None},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.machine_config.axis_extents is None

    def test_convert_no_home(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "nohome.lbdev",
            {"HomeOnStartup": None},
        )
        profile, _ = convert_to_profile(lbdev)
        assert profile.machine_config.home_on_start is None

    def test_convert_mirror_on(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "mirror.lbdev",
            {"MirrorX": True, "MirrorY": True},
        )
        _, summary = convert_to_profile(lbdev)
        assert summary.mirror_x is True
        assert summary.mirror_y is True

    def test_convert_camera_data(self, tmp_path):
        lbdev = _make_lbdev(
            tmp_path / "camera.lbdev",
            settings_overrides={
                "cameraMatrix": [1000, 0, 0, 0, 1000, 0, 320, 240, 1],
                "distortionMatrix": [0.1, -0.05, 0.01, 0.02, 0.001],
                "cameraIsFisheye": False,
                "isHeadCamera": False,
                "mapScale": 0.75,
            },
        )
        profile, summary = convert_to_profile(lbdev)
        assert summary.camera_calibration is True
        assert profile.machine_config.cameras is not None
        assert len(profile.machine_config.cameras) == 1

        cam = profile.machine_config.cameras[0]
        assert cam["camera_matrix_fx"] == 1000.0
        assert cam["camera_matrix_fy"] == 1000.0
        assert cam["camera_matrix_cx"] == 320.0
        assert cam["camera_matrix_cy"] == 240.0
        assert cam["distortion_k1"] == 0.1
        assert cam["distortion_k2"] == -0.05
        assert cam["distortion_p1"] == 0.01
        assert cam["distortion_p2"] == 0.02
        assert cam["distortion_k3"] == 0.001
        assert cam["camera_is_fisheye"] is False
        assert cam["is_head_camera"] is False
        assert cam["map_scale"] == 0.75

    def test_convert_no_camera(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "nocam.lbdev")
        profile, summary = convert_to_profile(lbdev)
        assert summary.camera_calibration is False
        assert profile.machine_config.cameras is None

    def test_convert_partial_camera(self, tmp_path):
        """cameraMatrix without distortionMatrix still gets captured."""
        lbdev = _make_lbdev(
            tmp_path / "partial.lbdev",
            settings_overrides={
                "cameraMatrix": [2000, 0, 0, 0, 2000, 0, 640, 480, 1],
            },
        )
        profile, summary = convert_to_profile(lbdev)
        assert summary.camera_calibration is True
        assert profile.machine_config.cameras is not None
        assert profile.machine_config.cameras[0]["camera_matrix_fx"] == 2000.0
        assert profile.machine_config.cameras[0].get("distortion_k1") is None


class TestInstallFromLBDev:
    def test_install_basic(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "basic.lbdev")
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DeviceProfileManager(install_dir=install_dir)

        profile, summary = mgr.install_from_lbdev(lbdev)

        assert profile.meta.name == "Test Laser"
        assert profile.machine_config.driver == "GrblSerialDriver"
        assert profile.machine_config.axis_extents == (400.0, 390.0)
        assert profile.source_dir == install_dir / "Test Laser"
        assert mgr.get("Test Laser") is profile

        manifest = install_dir / "Test Laser" / MANIFEST_FILENAME
        assert manifest.exists()
        with open(manifest) as f:
            data = yaml.safe_load(f)
        assert data["device"]["name"] == "Test Laser"
        assert data["machine"]["driver"] == "GrblSerialDriver"
        assert data["machine"]["axis_extents"] == [400.0, 390.0]

        dialect = install_dir / "Test Laser" / DIALECT_FILENAME
        assert dialect.exists()

        assert summary.name == "Test Laser"
        assert summary.driver == "GrblSerialDriver"

    def test_install_overwrites_existing(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "v1.lbdev")
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DeviceProfileManager(install_dir=install_dir)

        mgr.install_from_lbdev(lbdev)
        assert mgr.get("Test Laser") is not None

        lbdev_v2 = _make_lbdev(
            tmp_path / "v2.lbdev",
            {"DisplayName": "Test Laser"},
            {"Sim_RapidSpeed": 999},
        )
        profile2, _ = mgr.install_from_lbdev(lbdev_v2)
        assert profile2.machine_config.max_travel_speed == 999

    def test_install_missing_file_raises(self, tmp_path):
        mgr = DeviceProfileManager(install_dir=tmp_path)
        with pytest.raises(FileNotFoundError):
            mgr.install_from_lbdev(tmp_path / "missing.lbdev")

    def test_install_no_install_dir_raises(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "x.lbdev")
        mgr = DeviceProfileManager()
        with pytest.raises(RuntimeError, match="install directory"):
            mgr.install_from_lbdev(lbdev)

    def test_install_invalid_file_raises(self, tmp_path):
        p = tmp_path / "bad.lbdev"
        p.write_text("{bad")
        mgr = DeviceProfileManager(install_dir=tmp_path)
        with pytest.raises(ValueError, match="Invalid LightBurn"):
            mgr.install_from_lbdev(p)

    def test_loaded_profile_is_usable(self, tmp_path):
        lbdev = _make_lbdev(tmp_path / "usable.lbdev")
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DeviceProfileManager(install_dir=install_dir)

        profile, _ = mgr.install_from_lbdev(lbdev)

        assert profile.meta.name == "Test Laser"
        assert profile.machine_config.driver == "GrblSerialDriver"
        assert profile.machine_config.axis_extents == (400.0, 390.0)
        assert profile.source_dir is not None
        assert profile.source_dir.exists()
        assert (profile.source_dir / "device.yaml").exists()
        assert (profile.source_dir / "dialect.yaml").exists()


class TestRealLBDevAsset:
    """Tests using the real ACMER P3 2-IN-1 profile asset."""

    ASSET_PATH = ASSETS_DIR / "acmer-p3.lbdev"

    def test_asset_file_exists(self):
        assert self.ASSET_PATH.exists(), (
            f"Test asset not found: {self.ASSET_PATH}"
        )

    def test_parse_real_asset(self):
        data = parse_lbdev(self.ASSET_PATH)
        assert data["DisplayName"] == "ACMER P3 2-IN-1"
        assert data["Width"] == 400
        assert data["Height"] == 390
        assert data["Type"] == "Serial"
        assert data["HomeOnStartup"] is False

    def test_convert_real_asset(self):
        profile, summary = convert_to_profile(self.ASSET_PATH)
        assert profile.meta.name == "ACMER P3 2-IN-1"
        assert profile.machine_config.driver == "GrblSerialDriver"
        assert profile.machine_config.driver_args == {"baudrate": "115200"}
        assert profile.machine_config.axis_extents == (400.0, 390.0)
        assert profile.machine_config.home_on_start is False
        assert profile.machine_config.max_travel_speed == 400
        assert profile.machine_config.origin == Origin("bottom_left")

        assert summary.camera_calibration is True
        assert profile.machine_config.cameras is not None
        cam = profile.machine_config.cameras[0]
        assert cam["name"] == "LightBurn Camera"
        assert cam["device_id"] == "lightburn_camera"
        assert cam["camera_matrix_fx"] == pytest.approx(2480.09, rel=1e-3)
        assert cam["camera_matrix_fy"] == pytest.approx(2469.12, rel=1e-3)
        assert cam["camera_matrix_cx"] == pytest.approx(1295.10, rel=1e-3)
        assert cam["camera_matrix_cy"] == pytest.approx(956.52, rel=1e-3)
        assert cam["distortion_k1"] == pytest.approx(0.357, rel=1e-2)
        assert cam["distortion_k2"] == pytest.approx(-4.407, rel=1e-2)
        assert cam["distortion_p1"] == pytest.approx(-0.001256, abs=1e-5)
        assert cam["distortion_p2"] == pytest.approx(0.001226, abs=1e-5)
        assert cam["distortion_k3"] == pytest.approx(8.584, rel=1e-2)

        assert summary.name == "ACMER P3 2-IN-1"
        assert summary.driver == "GrblSerialDriver"
        assert summary.axis_extents == (400.0, 390.0)
        assert summary.home_on_start is False
        assert summary.max_travel_speed == 400
        assert summary.origin == "bottom_left"

    def test_summary_lines_real_asset(self):
        _, summary = convert_to_profile(self.ASSET_PATH)
        lines = summary.to_lines()
        assert any("ACMER P3 2-IN-1" in line for line in lines)
        assert any("400" in line and "390" in line for line in lines)
        assert any("GrblSerialDriver" in line for line in lines)
        assert any("115200" in line for line in lines)
        assert any("bottom_left" in line for line in lines)
        assert any("Camera calibration" in line for line in lines)

    def test_to_items_real_asset(self):
        _, summary = convert_to_profile(self.ASSET_PATH)
        items = summary.to_items()
        pairs = dict(items)
        assert pairs["Device name"] == "ACMER P3 2-IN-1"
        assert pairs["Work area"] == "400 × 390 mm"
        assert pairs["Driver"] == "GrblSerialDriver"
        assert pairs["Baud rate"] == "115200"
        assert pairs["Home on start"] == "False"
        assert pairs["Max travel speed"] == "400 mm/min"
        assert pairs["Origin"] == "bottom_left"

    def test_install_real_asset(self, tmp_path):
        install_dir = tmp_path / "installed"
        install_dir.mkdir()
        mgr = DeviceProfileManager(install_dir=install_dir)

        profile, _ = mgr.install_from_lbdev(self.ASSET_PATH)

        assert profile.meta.name == "ACMER P3 2-IN-1"
        assert profile.machine_config.driver == "GrblSerialDriver"
        assert profile.machine_config.axis_extents == (400.0, 390.0)

        manifest = install_dir / "ACMER P3 2-IN-1" / MANIFEST_FILENAME
        assert manifest.exists()
        with open(manifest) as f:
            data = yaml.safe_load(f)
        assert data["device"]["name"] == "ACMER P3 2-IN-1"
        assert data["machine"]["axis_extents"] == [400.0, 390.0]
        assert "cameras" in data["machine"]
        assert len(data["machine"]["cameras"]) == 1
        assert data["machine"]["cameras"][0]["camera_matrix_fx"] is not None

        assert mgr.get("ACMER P3 2-IN-1") is profile
