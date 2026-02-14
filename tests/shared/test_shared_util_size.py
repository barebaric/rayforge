from rayforge.shared.util.size import sizes_are_close


class TestSizesAreClose:
    def test_identical_sizes(self):
        assert sizes_are_close((100.0, 200.0), (100.0, 200.0)) is True

    def test_very_close_sizes(self):
        assert (
            sizes_are_close((100.0, 200.0), (100.0000001, 200.0000001)) is True
        )

    def test_different_sizes(self):
        assert sizes_are_close((100.0, 200.0), (100.1, 200.0)) is False

    def test_first_none(self):
        assert sizes_are_close(None, (100.0, 200.0)) is False

    def test_second_none(self):
        assert sizes_are_close((100.0, 200.0), None) is False

    def test_both_none(self):
        assert sizes_are_close(None, None) is False

    def test_width_differs(self):
        assert sizes_are_close((100.0, 200.0), (101.0, 200.0)) is False

    def test_height_differs(self):
        assert sizes_are_close((100.0, 200.0), (100.0, 201.0)) is False

    def test_zero_sizes(self):
        assert sizes_are_close((0.0, 0.0), (0.0, 0.0)) is True

    def test_negative_sizes(self):
        assert sizes_are_close((-10.0, -20.0), (-10.0, -20.0)) is True

    def test_mixed_sign_sizes(self):
        assert sizes_are_close((-10.0, 20.0), (-10.0, 20.0)) is True
