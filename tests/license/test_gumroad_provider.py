import json
import tempfile
from datetime import datetime
from pathlib import Path
from unittest import mock

import pytest

from rayforge.license import (
    LicenseStatus,
    LicenseType,
    LicenseResult,
    GumroadProvider,
)


class TestGumroadProvider:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def provider(self, temp_dir):
        return GumroadProvider(temp_dir)

    def test_name(self, provider):
        assert provider.name == "gumroad"

    def test_is_configured_false_initially(self, provider):
        assert not provider.is_configured()

    def test_add_license(self, provider, temp_dir):
        provider.add_license("PRODUCT123", "LICENSEKEY123")
        assert provider.is_configured()
        assert provider.has_license_for("PRODUCT123")
        assert provider.get_licenses() == {"PRODUCT123": "LICENSEKEY123"}

        config_file = temp_dir / "gumroad_licenses.json"
        with open(config_file) as f:
            data = json.load(f)
        assert data["licenses"] == {"PRODUCT123": "LICENSEKEY123"}

    def test_remove_license(self, provider):
        provider.add_license("PRODUCT123", "LICENSEKEY123")
        provider.remove_license("PRODUCT123")
        assert not provider.has_license_for("PRODUCT123")
        assert not provider.is_configured()

    def test_validate_missing_product_id(self, provider):
        result = provider.validate({})
        assert result.status == LicenseStatus.ERROR
        assert "Missing product_ids" in result.message

    def test_validate_no_license_configured(self, provider):
        result = provider.validate({"product_ids": ["PRODUCT123"]})
        assert result.status == LicenseStatus.NOT_FOUND

    def test_validate_uses_cache(self, provider):
        provider.add_license("PRODUCT123", "LICENSEKEY123")

        provider._cache["PRODUCT123"] = {
            "status": "valid",
            "message": "Cached",
            "license_type": "one_time",
            "customer_email": None,
            "last_validated": datetime.now().isoformat(),
            "metadata": {},
        }

        result = provider.validate({"product_ids": ["PRODUCT123"]})
        assert result.status == LicenseStatus.VALID
        assert result.message == "Cached"

    def test_validate_tries_all_product_ids(self, provider):
        provider.add_license("PRODUCT1", "KEY1")
        provider.add_license("PRODUCT2", "KEY2")

        with mock.patch.object(provider, "_validate_single") as mock_validate:
            mock_validate.side_effect = [
                LicenseResult(
                    status=LicenseStatus.INVALID,
                    message="Invalid",
                ),
                LicenseResult(
                    status=LicenseStatus.VALID,
                    license_type=LicenseType.ONE_TIME,
                ),
            ]

            result = provider.validate(
                {"product_ids": ["PRODUCT1", "PRODUCT2"]}
            )
            assert result.status == LicenseStatus.VALID
