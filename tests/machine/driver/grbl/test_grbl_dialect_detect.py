from rayforge.machine.driver.grbl.grbl_dialect_detect import (
    detect_grbl_dialect,
)
from rayforge.machine.driver.grbl.grbl_probe import build_grbl_profile


class TestDetectGrblDialect:
    def test_stock_grbl_1_1(self):
        build_info = ["[VER:1.1h:]", "[OPT:VMPH,63,511]"]
        assert detect_grbl_dialect(build_info) == "grbl"

    def test_stock_grbl_with_build_name(self):
        build_info = ["[VER:1.1h.ORTUR:]", "[OPT:VMPH,63,511]"]
        assert detect_grbl_dialect(build_info) == "grbl"

    def test_grblhal_in_ver_line(self):
        build_info = ["[VER:grblHAL 1.1f:]", "[OPT:VMP,31,511]"]
        assert detect_grbl_dialect(build_info) == "grbl_dynamic"

    def test_fluidnc_in_ver_line(self):
        build_info = ["[VER:FluidNC 3.6.1:]", "[OPT:VMP,31,511]"]
        assert detect_grbl_dialect(build_info) == "grbl_dynamic"

    def test_grblhal_settings_keys(self):
        build_info = ["[VER:1.1f:]"]
        settings = ["$0=10", "$400=200.0", "$401=300.0"]
        assert detect_grbl_dialect(build_info, settings) == "grbl_dynamic"

    def test_unknown_firmware_returns_none(self):
        build_info = ["garbage"]
        assert detect_grbl_dialect(build_info) is None

    def test_empty_input_returns_none(self):
        assert detect_grbl_dialect([]) is None

    def test_comma_format_ver(self):
        build_info = ["[VER:1.0.15,20240923:]", "[OPT:VMP,31,511]"]
        assert detect_grbl_dialect(build_info) == "grbl"


class TestBuildGrblProfileDialect:
    def test_stock_grbl_gets_compat_dialect(self):
        build_info = ["[VER:1.1h:]", "[OPT:VMPH,63,511]"]
        settings = [
            "$110=3000.0",
            "$111=3000.0",
            "$120=500.0",
            "$121=500.0",
            "$130=400.0",
            "$131=300.0",
        ]
        profile, _ = build_grbl_profile(build_info, settings)
        assert profile.dialect_config
        assert "laser_on" in profile.dialect_config
        assert profile.dialect_config["laser_on"] == "M4 S{power:.0f}"

    def test_grblhal_gets_dynamic_dialect(self):
        build_info = ["[VER:grblHAL 1.1f:]", "[OPT:VMP,31,511]"]
        settings = ["$400=200.0"]
        profile, _ = build_grbl_profile(build_info, settings)
        assert profile.dialect_config
        assert profile.dialect_config["laser_on"] == "M4 S0"
        assert "s_command" in profile.dialect_config["linear_move"]

    def test_unknown_firmware_gets_empty_dialect_config(self):
        profile, _ = build_grbl_profile([], [])
        assert profile.dialect_config == {}
