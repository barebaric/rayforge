import json
import tempfile
from pathlib import Path

import pytest

from rayforge.license import LicenseStatus, PatreonProvider


class TestPatreonProvider:
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def provider(self, temp_dir):
        return PatreonProvider(temp_dir, "test_client_id")

    def test_name(self, provider):
        assert provider.name == "patreon"

    def test_is_configured_false_initially(self, provider):
        assert not provider.is_configured()

    def test_is_configured_true_after_setting_token(self, provider):
        provider._access_token = "test_token"
        assert provider.is_configured()

    def test_validate_no_access_token(self, provider):
        result = provider.validate({"patreon_tier_ids": ["12345"]})
        assert result.status == LicenseStatus.NOT_FOUND
        assert "not linked" in result.message

    def test_validate_no_tier_ids(self, provider):
        provider._access_token = "test_token"
        result = provider.validate({})
        assert result.status == LicenseStatus.NOT_FOUND
        assert "does not support Patreon" in result.message

    def test_unlink(self, provider, temp_dir):
        provider._access_token = "test_token"
        config_file = temp_dir / "patreon_oauth.json"
        config_file.write_text(json.dumps({"access_token": "test_token"}))

        provider.unlink()

        assert provider._access_token is None
        assert not config_file.exists()

    def test_get_oauth_url(self, provider):
        url = provider.get_oauth_url()
        assert "patreon.com/oauth2/authorize" in url
        assert "client_id=test_client_id" in url
        assert "127.0.0.1" in url
        assert "callback" in url
        assert "scope=identity" in url
