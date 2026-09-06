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


class TestGestureBindings:
    def test_defaults_to_empty(self):
        config = Config()
        assert config.gesture_bindings == {}

    def test_set_binding_emits_changed_once(self):
        config = Config()
        calls = []

        def on_changed(sender, **kwargs):
            calls.append(sender)

        config.changed.connect(on_changed, weak=False)

        config.set_gesture_binding("canvas2d", "pan", "drag+primary")
        assert len(calls) == 1
        assert config.gesture_bindings == {"canvas2d": {"pan": "drag+primary"}}

        # Setting the same value again must not re-emit.
        config.set_gesture_binding("canvas2d", "pan", "drag+primary")
        assert len(calls) == 1

    def test_set_unassign_stores_none(self):
        config = Config()
        config.set_gesture_binding("canvas2d", "pan", None)
        assert config.gesture_bindings == {"canvas2d": {"pan": None}}

    def test_reset_binding_removes_entry(self):
        config = Config()
        config.set_gesture_binding("canvas2d", "pan", "drag+primary")
        config.set_gesture_binding("canvas2d", "zoom", "scroll")
        config.changed.connect(lambda sender, **kw: None, weak=False)

        config.reset_gesture_binding("canvas2d", "pan")
        assert config.gesture_bindings == {"canvas2d": {"zoom": "scroll"}}

        config.reset_gesture_binding("canvas2d", "zoom")
        assert config.gesture_bindings == {}

    def test_reset_missing_binding_is_noop(self):
        config = Config()
        config.reset_gesture_binding("canvas2d", "pan")
        assert config.gesture_bindings == {}

    def test_round_trip(self):
        config = Config()
        config.set_gesture_binding("canvas2d", "pan", "drag+primary")
        config.set_gesture_binding("canvas2d", "context_menu", None)
        data = config.to_dict()
        assert data["gesture_bindings"] == {
            "canvas2d": {"pan": "drag+primary", "context_menu": None}
        }

        restored = Config.from_dict(data, get_machine_by_id=lambda m: None)
        assert restored.gesture_bindings == {
            "canvas2d": {"pan": "drag+primary", "context_menu": None}
        }

    def test_absent_key_falls_back_to_empty(self):
        restored = Config.from_dict({}, get_machine_by_id=lambda m: None)
        assert restored.gesture_bindings == {}

    def test_malformed_entries_are_dropped(self):
        restored = Config.from_dict(
            {
                "gesture_bindings": {
                    "canvas2d": "not-a-dict",
                    "canvas3d": {"orbit": 42, "pan": "drag+middle"},
                }
            },
            get_machine_by_id=lambda m: None,
        )
        assert restored.gesture_bindings == {
            "canvas3d": {"pan": "drag+middle"}
        }
