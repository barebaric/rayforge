import semver

from rayforge.shared.util.versioning import (
    check_constraint,
    check_rayforge_compatibility,
    is_newer_version,
    normalize_tilde_version,
    parse_version_constraint,
)


class TestParseVersionConstraint:
    """Tests for parse_version_constraint function."""

    def test_parse_greater_or_equal(self):
        result = parse_version_constraint(">=1.0.0")
        assert result == (">=", "1.0.0")

    def test_parse_greater_than(self):
        result = parse_version_constraint(">2.0.0")
        assert result == (">", "2.0.0")

    def test_parse_less_or_equal(self):
        result = parse_version_constraint("<=3.0.0")
        assert result == ("<=", "3.0.0")

    def test_parse_less_than(self):
        result = parse_version_constraint("<4.0.0")
        assert result == ("<", "4.0.0")

    def test_parse_equal(self):
        result = parse_version_constraint("==5.0.0")
        assert result == ("==", "5.0.0")

    def test_parse_not_equal(self):
        result = parse_version_constraint("!=6.0.0")
        assert result == ("!=", "6.0.0")

    def test_parse_caret(self):
        result = parse_version_constraint("^0.27.0")
        assert result == ("^", "0.27.0")

    def test_parse_tilde(self):
        result = parse_version_constraint("~0.27.0")
        assert result == ("~", "0.27.0")

    def test_parse_with_v_prefix(self):
        result = parse_version_constraint(">=v1.0.0")
        assert result == (">=", "1.0.0")

    def test_parse_invalid_constraint(self):
        result = parse_version_constraint("invalid")
        assert result is None

    def test_parse_empty_constraint(self):
        result = parse_version_constraint("")
        assert result is None


class TestNormalizeTildeVersion:
    """Tests for normalize_tilde_version function."""

    def test_normalize_full_version(self):
        result = normalize_tilde_version("1.0.0")
        assert result == "1.0.0"

    def test_normalize_two_part_version(self):
        result = normalize_tilde_version("0.27")
        assert result == "0.27.0"

    def test_normalize_one_part_version(self):
        result = normalize_tilde_version("1")
        assert result == "1.0.0"

    def test_normalize_with_v_prefix(self):
        result = normalize_tilde_version("v1.0.0")
        assert result == "v1.0.0"


class TestCheckConstraint:
    """Tests for check_constraint function."""

    def test_greater_or_equal_satisfied(self):
        current = semver.VersionInfo.parse("2.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, ">=") is True

    def test_greater_or_equal_not_satisfied(self):
        current = semver.VersionInfo.parse("0.9.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, ">=") is False

    def test_greater_or_equal_equal(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, ">=") is True

    def test_greater_than_satisfied(self):
        current = semver.VersionInfo.parse("2.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, ">") is True

    def test_greater_than_not_satisfied(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, ">") is False

    def test_less_or_equal_satisfied(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("2.0.0")
        assert check_constraint(current, req, "<=") is True

    def test_less_or_equal_not_satisfied(self):
        current = semver.VersionInfo.parse("3.0.0")
        req = semver.VersionInfo.parse("2.0.0")
        assert check_constraint(current, req, "<=") is False

    def test_less_than_satisfied(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("2.0.0")
        assert check_constraint(current, req, "<") is True

    def test_less_than_not_satisfied(self):
        current = semver.VersionInfo.parse("2.0.0")
        req = semver.VersionInfo.parse("2.0.0")
        assert check_constraint(current, req, "<") is False

    def test_equal_satisfied(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, "==") is True

    def test_equal_not_satisfied(self):
        current = semver.VersionInfo.parse("2.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, "==") is False

    def test_not_equal_satisfied(self):
        current = semver.VersionInfo.parse("2.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, "!=") is True

    def test_not_equal_not_satisfied(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("1.0.0")
        assert check_constraint(current, req, "!=") is False

    def test_caret_satisfied_same_major(self):
        current = semver.VersionInfo.parse("0.28.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "^") is True

    def test_caret_not_satisfied_different_major(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "^") is False

    def test_caret_not_satisfied_older(self):
        current = semver.VersionInfo.parse("0.26.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "^") is False

    def test_tilde_satisfied_same_minor(self):
        current = semver.VersionInfo.parse("0.27.5")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "~") is True

    def test_tilde_not_satisfied_different_major(self):
        current = semver.VersionInfo.parse("1.0.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "~") is False

    def test_tilde_not_satisfied_different_minor(self):
        current = semver.VersionInfo.parse("0.28.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "~") is False

    def test_tilde_not_satisfied_older(self):
        current = semver.VersionInfo.parse("0.26.0")
        req = semver.VersionInfo.parse("0.27.0")
        assert check_constraint(current, req, "~") is False


class TestCheckRayforgeCompatibility:
    """Tests for check_rayforge_compatibility function."""

    def test_no_dependencies(self):
        assert check_rayforge_compatibility([], "0.27.0") is True

    def test_no_rayforge_dependency(self):
        depends = ["other-addon>=1.0.0"]
        assert check_rayforge_compatibility(depends, "0.27.0") is True

    def test_single_greater_or_equal_satisfied(self):
        depends = ["rayforge>=0.27.0"]
        assert check_rayforge_compatibility(depends, "0.28.0") is True

    def test_single_greater_or_equal_not_satisfied(self):
        depends = ["rayforge>=0.28.0"]
        result = check_rayforge_compatibility(depends, "0.27.0")
        assert result is False

    def test_multiple_constraints_all_satisfied(self):
        depends = ["rayforge>=0.27.0,~0.27"]
        assert check_rayforge_compatibility(depends, "0.27.5") is True

    def test_multiple_constraints_one_not_satisfied(self):
        depends = ["rayforge>=0.27.0,~0.28"]
        result = check_rayforge_compatibility(depends, "0.27.5")
        assert result is False

    def test_caret_constraint_satisfied(self):
        depends = ["rayforge^0.27.0"]
        assert check_rayforge_compatibility(depends, "0.27.5") is True

    def test_caret_constraint_not_satisfied(self):
        depends = ["rayforge^0.27.0"]
        assert check_rayforge_compatibility(depends, "1.0.0") is False

    def test_tilde_partial_version(self):
        depends = ["rayforge~0.27"]
        assert check_rayforge_compatibility(depends, "0.27.5") is True

    def test_tilde_partial_version_not_satisfied(self):
        depends = ["rayforge~0.27"]
        result = check_rayforge_compatibility(depends, "0.28.0")
        assert result is False

    def test_tilde_single_part_version(self):
        depends = ["rayforge~1"]
        result = check_rayforge_compatibility(depends, "1.0.5")
        assert result is True

    def test_tilde_single_part_version_not_satisfied(self):
        depends = ["rayforge~1"]
        assert check_rayforge_compatibility(depends, "2.0.0") is False

    def test_equal_constraint_satisfied(self):
        depends = ["rayforge==0.27.0"]
        assert check_rayforge_compatibility(depends, "0.27.0") is True

    def test_equal_constraint_not_satisfied(self):
        depends = ["rayforge==0.27.0"]
        result = check_rayforge_compatibility(depends, "0.28.0")
        assert result is False

    def test_not_equal_constraint_satisfied(self):
        depends = ["rayforge!=0.27.0"]
        assert check_rayforge_compatibility(depends, "0.28.0") is True

    def test_not_equal_constraint_not_satisfied(self):
        depends = ["rayforge!=0.27.0"]
        result = check_rayforge_compatibility(depends, "0.27.0")
        assert result is False

    def test_less_than_constraint_satisfied(self):
        depends = ["rayforge<1.0.0"]
        assert check_rayforge_compatibility(depends, "0.27.0") is True

    def test_less_than_constraint_not_satisfied(self):
        depends = ["rayforge<0.27.0"]
        result = check_rayforge_compatibility(depends, "0.27.0")
        assert result is False

    def test_greater_than_constraint_satisfied(self):
        depends = ["rayforge>0.26.0"]
        assert check_rayforge_compatibility(depends, "0.27.0") is True

    def test_greater_than_constraint_not_satisfied(self):
        depends = ["rayforge>0.27.0"]
        result = check_rayforge_compatibility(depends, "0.27.0")
        assert result is False

    def test_invalid_current_version(self):
        depends = ["rayforge>=0.27.0"]
        result = check_rayforge_compatibility(depends, "invalid")
        assert result is True

    def test_invalid_constraint_fails(self):
        depends = ["rayforge>=0.27.0,invalid"]
        result = check_rayforge_compatibility(depends, "0.28.0")
        assert result is False

    def test_invalid_version_in_constraint_fails(self):
        depends = ["rayforge>=0.27.0,>=invalid"]
        result = check_rayforge_compatibility(depends, "0.28.0")
        assert result is False

    def test_version_with_v_prefix(self):
        depends = ["rayforge>=v0.27.0"]
        result = check_rayforge_compatibility(depends, "v0.28.0")
        assert result is True

    def test_complex_multiple_constraints(self):
        depends = ["rayforge>=0.27.0,<1.0.0,~0.27"]
        assert check_rayforge_compatibility(depends, "0.27.5") is True

    def test_complex_multiple_constraints_not_satisfied(self):
        depends = ["rayforge>=0.27.0,<1.0.0,~0.27"]
        assert check_rayforge_compatibility(depends, "1.0.0") is False

    def test_mixed_dependencies(self):
        depends = [
            "other-addon>=1.0.0",
            "rayforge>=0.27.0",
            "another-addon>=2.0.0",
        ]
        assert check_rayforge_compatibility(depends, "0.28.0") is True

    def test_empty_constraint_in_list(self):
        depends = ["rayforge>=0.27.0,"]
        assert check_rayforge_compatibility(depends, "0.28.0") is True


class TestIsNewerVersion:
    """Tests for is_newer_version function."""

    def test_remote_is_newer(self):
        assert is_newer_version("2.0.0", "1.0.0") is True

    def test_local_is_newer(self):
        assert is_newer_version("1.0.0", "2.0.0") is False

    def test_equal_versions(self):
        assert is_newer_version("1.0.0", "1.0.0") is False

    def test_with_v_prefix(self):
        assert is_newer_version("v2.0.0", "v1.0.0") is True

    def test_mixed_v_prefix(self):
        assert is_newer_version("2.0.0", "v1.0.0") is True

    def test_invalid_versions_fallback(self):
        assert is_newer_version("2.0.0", "1.0.0") is True
