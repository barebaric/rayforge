import tempfile
from datetime import datetime
from pathlib import Path

import pytest

from rayforge.license import LicenseStatus, LicenseResult, LicenseValidator


class TestLicenseValidator:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def validator(self, temp_dir):
        return LicenseValidator(temp_dir, "test_client_id")

    def test_validate_no_license_required(self, validator):
        result = validator.validate("test_addon", {})
        assert result.status == LicenseStatus.NOT_FOUND

    def test_validate_uses_cache(self, validator):
        validator._cache["test_addon"] = (
            LicenseResult(
                status=LicenseStatus.VALID,
                message="Cached result",
            ),
            datetime.now(),
        )

        result = validator.validate("test_addon", {"product_ids": ["P1"]})
        assert result.status == LicenseStatus.VALID
        assert result.message == "Cached result"

    def test_invalidate_cache_specific_addon(self, validator):
        validator._cache["addon1"] = (
            LicenseResult(status=LicenseStatus.VALID),
            datetime.now(),
        )
        validator._cache["addon2"] = (
            LicenseResult(status=LicenseStatus.VALID),
            datetime.now(),
        )

        validator.invalidate_cache("addon1")

        assert "addon1" not in validator._cache
        assert "addon2" in validator._cache

    def test_invalidate_cache_all(self, validator):
        validator._cache["addon1"] = (
            LicenseResult(status=LicenseStatus.VALID),
            datetime.now(),
        )
        validator._cache["addon2"] = (
            LicenseResult(status=LicenseStatus.VALID),
            datetime.now(),
        )

        validator.invalidate_cache()

        assert len(validator._cache) == 0

    def test_check_license_not_required(self, validator):
        allowed, message, url = validator.check_license("test", {})
        assert allowed is True
        assert message == ""
        assert url == ""

    def test_check_license_required_no_config(self, validator):
        allowed, message, url = validator.check_license(
            "test",
            {
                "required": True,
                "product_ids": ["P1"],
                "purchase_url": "http://example.com",
            },
        )
        assert allowed is False
        assert "Gumroad" in message
        assert url == "http://example.com"

    def test_add_gumroad_license(self, validator):
        validator.add_gumroad_license("P1", "KEY1")

        gumroad = validator.get_provider("gumroad")
        assert gumroad.has_license_for("P1")

    def test_remove_gumroad_license(self, validator):
        validator.add_gumroad_license("P1", "KEY1")
        validator.remove_gumroad_license("P1")

        gumroad = validator.get_provider("gumroad")
        assert not gumroad.has_license_for("P1")

    def test_get_gumroad_licenses(self, validator):
        validator.add_gumroad_license("P1", "KEY1")
        validator.add_gumroad_license("P2", "KEY2")

        licenses = validator.get_gumroad_licenses()
        assert licenses == {"P1": "KEY1", "P2": "KEY2"}

    def test_is_patreon_linked_false(self, validator):
        assert not validator.is_patreon_linked()

    def test_is_patreon_linked_true(self, validator):
        patreon = validator.get_provider("patreon")
        patreon._access_token = "test_token"
        assert validator.is_patreon_linked()

    def test_unlink_patreon(self, validator):
        patreon = validator.get_provider("patreon")
        patreon._access_token = "test_token"

        validator.unlink_patreon()

        assert not validator.is_patreon_linked()

    def test_validator_without_patreon(self, temp_dir):
        validator = LicenseValidator(temp_dir, None)
        assert validator.get_provider("patreon") is None
