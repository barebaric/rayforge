from raygeo.template import CounterSettings

from rayforge.core.job_counter import JobCounter


class TestJobCounter:
    def test_template_values_with_offset(self):
        counter = JobCounter()
        counter.counter = CounterSettings(current=100, step=10)
        assert counter.template_values(0) == {"counter": 100}
        assert counter.template_values(2) == {"counter": 120}

    def test_setting_counter_emits_changed(self):
        counter = JobCounter()
        received = []

        def on_changed(sender):
            received.append(sender)

        counter.changed.connect(on_changed)
        counter.counter = CounterSettings(current=5)
        assert received == [counter]

    def test_setting_equal_counter_is_silent(self):
        counter = JobCounter()
        received = []

        def on_changed(sender):
            received.append(sender)

        counter.changed.connect(on_changed)
        counter.counter = CounterSettings()
        assert received == []

    def test_names(self):
        assert "counter" in JobCounter().names

    def test_round_trip(self):
        counter = JobCounter()
        counter.counter = CounterSettings(current=3, start=2, step=4)
        restored = JobCounter.from_dict(counter.to_dict())
        assert restored.counter == counter.counter

    def test_from_dict_none(self):
        assert JobCounter.from_dict(None).counter == CounterSettings()
