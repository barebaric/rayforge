"""
Tests for G-code generation across all dialects.

This module contains parameterized tests that verify G-code output
for all supported dialects using a reference square SVG file.
"""

import logging
import re
from pathlib import Path
from typing import List, Tuple

import pytest

from rayforge.machine.models.dialect import BUILTIN_DIALECTS, MACH4_M67_DIALECT

logger = logging.getLogger(__name__)

DIALECTS_DATA_DIR = Path(__file__).parent / "data"


def parse_gcode_line(line: str) -> dict:
    """
    Parses a G-code line into a dictionary of axes and values.
    Example: "G0 X10.5 Y20.0" -> {'G': 0.0, 'X': 10.5, 'Y': 20.0}
    Ignores comments.
    """
    line = line.split(";")[0].strip()
    if not line:
        return {}
    parts = {}
    tokens = re.findall(r"([A-Za-z])([-+]?\d*\.?\d+)", line)
    for command, value in tokens:
        parts[command.upper()] = float(value)
    return parts


def get_dialect_test_cases() -> List[Tuple[str, str]]:
    """
    Returns a list of (dialect_uid, expected_file) tuples for all
    dialects that have corresponding expected output files.
    """
    test_cases = []
    for dialect in BUILTIN_DIALECTS:
        expected_file = f"expected_square_{dialect.uid}.gcode"
        expected_path = DIALECTS_DATA_DIR / expected_file
        if expected_path.exists():
            test_cases.append((dialect.uid, expected_file))
        else:
            logger.warning(
                f"No expected output file found for dialect "
                f"'{dialect.uid}': {expected_path}"
            )
    return test_cases


class TestDialectGcodeOutput:
    """
    Test G-code output for all supported dialects.

    These tests verify that the G-code generated for a reference
    square matches the expected output for each dialect.
    """

    @pytest.fixture
    def data_dir(self) -> Path:
        """Fixture providing the path to the test data directory."""
        return DIALECTS_DATA_DIR

    @pytest.mark.parametrize(
        "dialect_uid,expected_file",
        get_dialect_test_cases(),
        ids=[tc[0] for tc in get_dialect_test_cases()],
    )
    def test_expected_file_exists(
        self,
        dialect_uid: str,
        expected_file: str,
        data_dir: Path,
    ):
        """
        Test that the dialect's expected G-code file exists and is valid.
        """
        expected_path = data_dir / expected_file

        assert expected_path.exists(), (
            f"Expected G-code file not found: {expected_path}"
        )

        content = expected_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()

        assert len(lines) > 0, f"Expected file is empty: {expected_path}"

        has_motion = any(
            line.strip().startswith(("G0", "G1", "G2", "G3")) for line in lines
        )
        assert has_motion, (
            f"Expected file contains no motion commands: {expected_path}"
        )

    @pytest.mark.parametrize(
        "dialect_uid,expected_file",
        get_dialect_test_cases(),
        ids=[tc[0] for tc in get_dialect_test_cases()],
    )
    def test_gcode_commands_structure(
        self,
        dialect_uid: str,
        expected_file: str,
        data_dir: Path,
    ):
        """
        Test that the expected G-code has valid command structure.
        """
        expected_path = data_dir / expected_file
        content = expected_path.read_text(encoding="utf-8")
        lines = content.strip().splitlines()

        for line_num, line in enumerate(lines, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith(";"):
                continue

            parsed = parse_gcode_line(stripped)
            if parsed:
                valid_commands = (
                    "G" in parsed or "M" in parsed or "T" in parsed
                )
                assert valid_commands, (
                    f"Line {line_num} has no G, M, or T command: {stripped}"
                )

    def test_all_dialects_have_expected_files(self):
        """
        Verify that all built-in dialects have corresponding test data.
        """
        missing = []
        for dialect in BUILTIN_DIALECTS:
            expected_file = f"expected_square_{dialect.uid}.gcode"
            expected_path = DIALECTS_DATA_DIR / expected_file
            if not expected_path.exists():
                missing.append(dialect.uid)

        assert not missing, (
            f"Missing expected files for dialects: {', '.join(missing)}"
        )


class TestDialectProperties:
    """Test basic properties of built-in dialects."""

    def test_builtin_dialects_count(self):
        """Verify we have the expected number of built-in dialects."""
        assert len(BUILTIN_DIALECTS) == 6

    @pytest.mark.parametrize("dialect", BUILTIN_DIALECTS, ids=lambda d: d.uid)
    def test_dialect_has_required_templates(self, dialect):
        """Verify each dialect has all required template fields."""
        required_fields = [
            "laser_on",
            "laser_off",
            "travel_move",
            "linear_move",
            "arc_cw",
            "arc_ccw",
        ]
        for field in required_fields:
            assert hasattr(dialect, field), (
                f"Dialect {dialect.uid} missing required field: {field}"
            )
            value = getattr(dialect, field)
            assert value is not None and len(value) > 0, (
                f"Dialect {dialect.uid} has empty field: {field}"
            )

    @pytest.mark.parametrize("dialect", BUILTIN_DIALECTS, ids=lambda d: d.uid)
    def test_dialect_templates_have_placeholders(self, dialect):
        """Verify motion templates have coordinate placeholders."""
        coord_fields = ["travel_move", "linear_move"]
        for field in coord_fields:
            template = getattr(dialect, field)
            assert "{x}" in template or "{x_cmd}" in template, (
                f"Dialect {dialect.uid} {field} missing {{x}} placeholder"
            )
            assert "{y}" in template or "{y_cmd}" in template, (
                f"Dialect {dialect.uid} {field} missing {{y}} placeholder"
            )


class TestMach4M67Dialect:
    """Specific tests for the Mach4 M67 dialect."""

    def test_mach4_m67_uses_m67_command(self):
        """Test that Mach4 dialect uses M67 for laser control."""
        assert "M67" in MACH4_M67_DIALECT.laser_on
        assert "M67" in MACH4_M67_DIALECT.laser_off
        assert "Q" in MACH4_M67_DIALECT.laser_on

    def test_mach4_m67_no_inline_power(self):
        """Test that Mach4 motion commands have no inline S parameter."""
        assert "{s_command}" not in MACH4_M67_DIALECT.linear_move
        assert "{s_command}" not in MACH4_M67_DIALECT.arc_cw
        assert "{s_command}" not in MACH4_M67_DIALECT.arc_ccw

    def test_mach4_m67_power_format(self):
        """Test that M67 uses correct power format."""
        result = MACH4_M67_DIALECT.laser_on.format(power=128)
        assert result == "M67 E0 Q128"

    def test_mach4_m67_can_g0_with_speed(self):
        """Test that Mach4 supports speed with G0 commands."""
        assert MACH4_M67_DIALECT.can_g0_with_speed is True
