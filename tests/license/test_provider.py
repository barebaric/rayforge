from datetime import datetime, timedelta

from rayforge.license import LicenseStatus, LicenseType, LicenseResult


class TestLicenseResult:
    def test_is_valid_for_offline_invalid_status(self):
        result = LicenseResult(status=LicenseStatus.INVALID)
        assert not result.is_valid_for_offline()

    def test_is_valid_for_offline_one_time(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.ONE_TIME,
        )
        assert result.is_valid_for_offline()

    def test_is_valid_for_offline_subscription_within_grace(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.SUBSCRIPTION,
            last_validated=datetime.now() - timedelta(days=10),
        )
        assert result.is_valid_for_offline()

    def test_is_valid_for_offline_subscription_past_grace(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.SUBSCRIPTION,
            last_validated=datetime.now() - timedelta(days=35),
        )
        assert not result.is_valid_for_offline()

    def test_is_valid_for_offline_subscription_no_timestamp(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.SUBSCRIPTION,
            last_validated=None,
        )
        assert not result.is_valid_for_offline()

    def test_is_expired_false_when_no_expiry(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.ONE_TIME,
        )
        assert not result.is_expired()

    def test_is_expired_false_when_future_expiry(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.ONE_TIME,
            expires_at=datetime.now() + timedelta(days=1),
        )
        assert not result.is_expired()

    def test_is_expired_true_when_past_expiry(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.ONE_TIME,
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        assert result.is_expired()

    def test_is_valid_for_offline_false_when_expired(self):
        result = LicenseResult(
            status=LicenseStatus.VALID,
            license_type=LicenseType.ONE_TIME,
            expires_at=datetime.now() - timedelta(seconds=1),
        )
        assert not result.is_valid_for_offline()
