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
