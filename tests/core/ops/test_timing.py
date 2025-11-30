"""
Tests for timing estimation functionality.
"""

from rayforge.core.ops.timing import estimate_time
from rayforge.core.ops.container import Ops


class TestTiming:
    """Test cases for timing estimation."""

    def test_empty_commands(self):
        """Test that empty command list returns 0 time."""
        assert estimate_time([]) == 0.0

    def test_single_move_command(self):
        """Test timing estimation for a single move command."""
        ops = Ops()
        ops.move_to(10, 10, 0)
        # MoveToCommand is not a cutting command, so it should use travel speed
        # Distance = sqrt(10^2 + 10^2) = 14.14mm
        # At 3000mm/min = 50mm/s, time = 14.14/50 = 0.283s + acceleration
        actual_time = estimate_time(list(ops))
        # Should be around 0.33s with acceleration
        assert 0.3 < actual_time < 0.4

    def test_single_line_command(self):
        """Test timing estimation for a single line command."""
        ops = Ops()
        ops.line_to(10, 0, 0)
        # LineToCommand is a cutting command, so it should use cut speed
        # Distance = 10mm
        # At 1000mm/min = 16.67mm/s, time = 10/16.67 = 0.6s + acceleration
        actual_time = estimate_time(list(ops))
        # Should be around 0.62s with acceleration
        assert 0.6 < actual_time < 0.65

    def test_custom_speeds(self):
        """Test timing estimation with custom speeds."""
        ops = Ops()
        ops.line_to(60, 0, 0)
        # Distance = 60mm
        # At 1200mm/min = 20mm/s, time = 60/20 = 3s + acceleration
        actual_time = estimate_time(list(ops), default_cut_speed=1200.0)
        # Should be around 3.02s with acceleration
        assert 3.0 < actual_time < 3.05

    def test_speed_commands(self):
        """Test timing estimation with speed change commands."""
        ops = Ops()
        ops.set_cut_speed(600)  # 10mm/s
        ops.line_to(50, 0, 0)  # 5s at 10mm/s
        ops.set_travel_speed(1200)  # 20mm/s
        ops.move_to(50, 50, 0)  # 2.5s at 20mm/s
        actual_time = estimate_time(list(ops))
        # Should be around 7.53s with acceleration
        assert 7.5 < actual_time < 7.55

    def test_acceleration_disabled(self):
        """Test timing estimation with acceleration disabled."""
        ops = Ops()
        ops.line_to(10, 0, 0)
        # With acceleration=0, should use simple distance/speed calculation
        actual_time = estimate_time(list(ops), acceleration=0)
        expected_time = 0.6  # 10mm / (1000mm/min / 60) = 0.6s
        assert abs(actual_time - expected_time) < 0.01

    def test_acceleration_enabled(self):
        """Test timing estimation with acceleration enabled."""
        ops = Ops()
        ops.line_to(10, 0, 0)
        # With acceleration, should be slightly longer due to acceleration
        time_with_accel = estimate_time(list(ops), acceleration=1000.0)
        time_without_accel = estimate_time(list(ops), acceleration=0.0)
        assert time_with_accel > time_without_accel

    def test_scanline_power_command(self):
        """Test timing estimation for ScanLinePowerCommand."""
        ops = Ops()
        ops.scan_to(x=100, y=0, z=0, power_values=bytearray([100] * 100))
        # ScanLinePowerCommand is a cutting command
        # Distance = 100mm
        # At 1000mm/min = 16.67mm/s, time = 100/16.67 = 6s + acceleration
        actual_time = estimate_time(list(ops))
        # Should be around 6.02s with acceleration
        assert 6.0 < actual_time < 6.05

    def test_mixed_commands(self):
        """Test timing estimation for mixed command types."""
        ops = Ops()
        ops.move_to(0, 0, 0)  # Initial position
        ops.line_to(10, 0, 0)  # 10mm cut
        ops.move_to(10, 10, 0)  # 10mm travel
        ops.line_to(0, 10, 0)  # 10mm cut
        ops.move_to(0, 0, 0)  # 14.14mm travel (diagonal)

        # Cut movements: 20mm total at 1000mm/min = 16.67mm/s = 1.2s
        # Travel movements: 24.14mm total at 3000mm/min = 50mm/s = 0.48s
        # Plus acceleration effects
        actual_time = estimate_time(list(ops))
        # Should be around 1.73s with acceleration
        assert 1.7 < actual_time < 1.8

    def test_ops_integration(self):
        """Test that Ops.estimate_time() uses the timing module correctly."""
        ops = Ops()
        ops.move_to(0, 0)
        ops.line_to(10, 0)
        ops.line_to(10, 10)

        # Should match direct timing module call
        ops_time = ops.estimate_time()
        direct_time = estimate_time(list(ops))
        assert abs(ops_time - direct_time) < 0.001

    def test_negligible_movement(self):
        """Test that very small movements are skipped."""
        ops = Ops()
        ops.line_to(0.000001, 0, 0)  # Very small movement
        actual_time = estimate_time(list(ops))
        # Very small movements should have minimal time
        assert actual_time < 0.001  # Should be very small

    def test_triangular_velocity_profile(self):
        """Test timing estimation when full speed cannot be reached."""
        ops = Ops()
        ops.line_to(1, 0, 0)  # Very short distance
        # With high acceleration, full speed won't be reached
        # Should use triangular velocity profile
        time_with_high_accel = estimate_time(list(ops), acceleration=10000.0)
        time_with_low_accel = estimate_time(list(ops), acceleration=100.0)
        # Higher acceleration should result in shorter time
        assert time_with_high_accel < time_with_low_accel
