import pytest
from laser_essentials.calibration_tests import (
    FocusMode,
    FocusTestParams,
    IntervalTestParams,
    focus_commands,
    focus_offsets,
    format_offset,
    interval_values,
    lines_per_inch,
    row_positions,
)


class TestIntervalValues:
    def test_linear_including_both_ends(self):
        params = IntervalTestParams(
            min_interval=0.05, max_interval=0.25, count=5
        )
        assert interval_values(params) == pytest.approx(
            [0.05, 0.10, 0.15, 0.20, 0.25]
        )

    def test_rejects_fewer_than_two_cells(self):
        with pytest.raises(ValueError):
            interval_values(IntervalTestParams(count=1))

    @pytest.mark.parametrize("low,high", [(0.2, 0.1), (0.0, 0.1)])
    def test_rejects_bad_range(self, low, high):
        params = IntervalTestParams(min_interval=low, max_interval=high)
        with pytest.raises(ValueError):
            interval_values(params)

    def test_lines_per_inch(self):
        assert lines_per_inch(0.1) == pytest.approx(254.0)


class TestRowPositions:
    def test_cells_are_spaced_left_to_right(self):
        assert row_positions(3, 10.0, 2.0) == pytest.approx([0.0, 12.0, 24.0])


class TestFocusOffsets:
    def test_offsets(self):
        params = FocusTestParams(start_offset=-1.0, step=0.5, count=5)
        assert focus_offsets(params) == pytest.approx(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )

    def test_rejects_zero_step(self):
        with pytest.raises(ValueError):
            focus_offsets(FocusTestParams(step=0.0))

    def test_rejects_unreasonable_range(self):
        with pytest.raises(ValueError):
            focus_offsets(
                FocusTestParams(start_offset=-8.0, step=1.0, count=20)
            )

    @pytest.mark.parametrize(
        "value,text", [(-2.0, "-2.0"), (0.0, "0.0"), (0.5, "+0.5")]
    )
    def test_format_offset(self, value, text):
        assert format_offset(value) == text


class TestFocusCommands:
    def test_ramp_needs_no_commands(self):
        params = FocusTestParams(mode=FocusMode.RAMP)
        assert focus_commands(params, reverse_z=False) == []

    def test_manual_pauses_before_every_line(self):
        params = FocusTestParams(
            mode=FocusMode.MANUAL, start_offset=-0.5, step=0.5, count=3
        )
        commands = focus_commands(params, reverse_z=False)
        # (index of the line the command precedes, G-code lines)
        assert [index for index, _code, _name in commands] == [0, 1, 2]
        assert all(code == ["M0"] for _i, code, _n in commands)
        names = [name for _i, _code, name in commands]
        assert "-0.5" in names[0]
        assert "0.5" in names[1]

    def test_z_axis_moves_relatively_and_returns(self):
        params = FocusTestParams(
            mode=FocusMode.Z_AXIS, start_offset=-1.0, step=0.5, count=3
        )
        commands = focus_commands(params, reverse_z=False)
        assert [index for index, _code, _name in commands] == [0, 1, 2, 3]
        assert commands[0][1] == ["G91", "G0 Z-1", "G90"]
        assert commands[1][1] == ["G91", "G0 Z0.5", "G90"]
        assert commands[2][1] == ["G91", "G0 Z0.5", "G90"]
        # Back to the starting height after the last line.
        assert commands[3][1] == ["G91", "G0 Z0", "G90"]

    def test_z_axis_return_balances_moves(self):
        params = FocusTestParams(
            mode=FocusMode.Z_AXIS, start_offset=0.5, step=0.25, count=4
        )
        commands = focus_commands(params, reverse_z=False)
        total = sum(float(code[1].split("Z")[1]) for _i, code, _n in commands)
        assert total == pytest.approx(0.0)

    def test_z_axis_respects_reversed_axis(self):
        params = FocusTestParams(
            mode=FocusMode.Z_AXIS, start_offset=-1.0, step=0.5, count=2
        )
        commands = focus_commands(params, reverse_z=True)
        assert commands[0][1] == ["G91", "G0 Z1", "G90"]
        assert commands[1][1] == ["G91", "G0 Z-0.5", "G90"]
