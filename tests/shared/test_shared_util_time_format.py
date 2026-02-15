from rayforge.shared.util.time_format import format_hours_to_hm


class TestFormatHoursToHm:
    def test_hours_only(self):
        assert format_hours_to_hm(10.0) == "10h"

    def test_minutes_only(self):
        assert format_hours_to_hm(0.5) == "30m"

    def test_hours_and_minutes(self):
        assert format_hours_to_hm(10.5) == "10h 30m"

    def test_zero_hours(self):
        assert format_hours_to_hm(0.0) == "0m"

    def test_fractional_minutes_rounded(self):
        assert format_hours_to_hm(1.25) == "1h 15m"

    def test_fractional_minutes_truncated(self):
        assert format_hours_to_hm(1.1) == "1h 6m"

    def test_large_hours(self):
        assert format_hours_to_hm(100.5) == "100h 30m"

    def test_small_fraction(self):
        assert format_hours_to_hm(0.1) == "6m"

    def test_45_minutes(self):
        assert format_hours_to_hm(0.75) == "45m"

    def test_one_quarter_hour(self):
        assert format_hours_to_hm(0.25) == "15m"
