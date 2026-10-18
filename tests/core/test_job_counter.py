from rayforge.core.job_counter import CounterSettings, JobCounter


class TestCounterSettings:
    def test_defaults(self):
        s = CounterSettings()
        assert s.current == 1
        assert s.start == 1
        assert s.step == 1
        assert s.end is None
        assert s.auto_advance is True
        assert s.advance_on_export is False

    def test_value_at_offset_uses_step(self):
        s = CounterSettings(current=10, start=1, step=5)
        assert s.value_at(0) == 10
        assert s.value_at(1) == 15
        assert s.value_at(3) == 25

    def test_value_at_wraps_after_end(self):
        s = CounterSettings(current=4, start=1, step=1, end=5)
        assert [s.value_at(i) for i in range(4)] == [4, 5, 1, 2]

    def test_value_at_wraps_with_step(self):
        s = CounterSettings(current=90, start=10, step=40, end=95)
        # Range is 10, 50, 90; 130 is past the end and wraps to 10.
        assert [s.value_at(i) for i in range(3)] == [90, 10, 50]

    def test_value_at_negative_offset_wraps_backwards(self):
        s = CounterSettings(current=1, start=1, step=1, end=3)
        assert s.value_at(-1) == 3

    def test_zero_step_keeps_value(self):
        s = CounterSettings(current=7, step=0)
        assert s.value_at(5) == 7

    def test_advanced_moves_current(self):
        s = CounterSettings(current=3, step=2)
        assert s.advanced(4).current == 11
        assert s.advanced().current == 5
        assert s.current == 3

    def test_previous_moves_back_one_step(self):
        s = CounterSettings(current=3, step=2)
        assert s.advanced(-1).current == 1

    def test_reset_returns_to_start(self):
        s = CounterSettings(current=42, start=7)
        assert s.reset().current == 7

    def test_values_for_copies(self):
        s = CounterSettings(current=8, step=1, end=9, start=1)
        assert s.values_for(3) == [8, 9, 1]

    def test_round_trip(self):
        s = CounterSettings(
            current=12,
            start=10,
            step=2,
            end=30,
            auto_advance=False,
            advance_on_export=True,
        )
        assert CounterSettings.from_dict(s.to_dict()) == s

    def test_from_dict_ignores_missing_keys(self):
        assert CounterSettings.from_dict({}) == CounterSettings()

    def test_from_dict_rejects_end_below_start(self):
        s = CounterSettings.from_dict({"start": 10, "end": 5})
        assert s.end is None


class TestJobCounter:
    def test_template_values_with_offset(self):
        vt = JobCounter()
        vt.counter = CounterSettings(current=100, step=10)
        assert vt.template_values(0) == {"counter": 100}
        assert vt.template_values(2) == {"counter": 120}

    def test_setting_counter_emits_changed(self):
        vt = JobCounter()
        received = []

        def on_changed(sender):
            received.append(sender)

        vt.changed.connect(on_changed)
        vt.counter = CounterSettings(current=5)
        assert received == [vt]

    def test_setting_equal_counter_is_silent(self):
        vt = JobCounter()
        received = []

        def on_changed(sender):
            received.append(sender)

        vt.changed.connect(on_changed)
        vt.counter = CounterSettings()
        assert received == []

    def test_names(self):
        assert "counter" in JobCounter().names

    def test_round_trip(self):
        vt = JobCounter()
        vt.counter = CounterSettings(current=3, start=2, step=4)
        restored = JobCounter.from_dict(vt.to_dict())
        assert restored.counter == vt.counter

    def test_from_dict_none(self):
        assert JobCounter.from_dict(None).counter == CounterSettings()
