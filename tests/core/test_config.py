"""Tests for the persistent Config class."""

from rayforge.core.config import Config


class TestSetupCompletedFlag:
    def test_defaults_to_false(self):
        config = Config()
        assert config.setup_completed is False

    def test_setter_emits_changed_once(self):
        config = Config()
        calls = []

        def on_changed(sender, **kwargs):
            calls.append(sender)

        config.changed.connect(on_changed, weak=False)

        config.set_setup_completed(True)
        assert len(calls) == 1

        # Setting the same value again must not re-emit.
        config.set_setup_completed(True)
        assert len(calls) == 1

    def test_round_trip(self):
        config = Config()
        config.set_setup_completed(True)
        data = config.to_dict()
        assert data["setup_completed"] is True

        restored = Config.from_dict(data, get_machine_by_id=lambda mid: None)
        assert restored.setup_completed is True

    def test_absent_key_falls_back_to_false(self):
        """Older configs without the key must not crash on load."""
        restored = Config.from_dict({}, get_machine_by_id=lambda mid: None)
        assert restored.setup_completed is False


class TestUsageConsent:
    def test_unanswered_is_not_consented(self):
        """Tracking stays off until the prompt is answered."""
        config = Config()
        assert config.has_consented_tracking is False
        assert config.has_declined_tracking is False

    def test_accept_records_consent(self):
        config = Config()
        config.set_usage_consent(True)
        assert config.has_consented_tracking is True

    def test_decline_disables_tracking(self):
        config = Config()
        config.set_usage_consent(False)
        assert config.has_consented_tracking is False
        assert config.has_declined_tracking is True

    def test_stale_consent_requires_reanswer(self):
        """Consent predating the current policy date must be re-asked."""
        config = Config()
        config.usage_consent_date = "2020-01-01T00:00:00+00:00"
        assert config.has_consented_tracking is False
        assert config.has_declined_tracking is False

    def test_round_trip(self):
        config = Config()
        config.set_usage_consent(True)
        restored = Config.from_dict(
            config.to_dict(), get_machine_by_id=lambda mid: None
        )
        assert restored.has_consented_tracking is True
