import math

import numpy as np
import pytest

from rayforge.machine.kinematic_math import KinematicMath


class TestEffectiveDiameter:
    def test_surface_z_zero(self):
        assert KinematicMath.effective_diameter(25.0, 0.0) == 25.0

    def test_negative_z(self):
        assert KinematicMath.effective_diameter(25.0, -2.0) == 21.0

    def test_positive_z(self):
        assert KinematicMath.effective_diameter(25.0, 3.0) == 31.0

    def test_matches_axis_mapper_formula(self):
        diameter = 25.0
        z = -2.0
        assert (
            KinematicMath.effective_diameter(diameter, z) == diameter + 2.0 * z
        )


class TestGearRatio:
    def test_jaws_returns_one(self):
        assert KinematicMath.gear_ratio(False, 70.0, 20.0) == 1.0

    def test_rollers_ratio(self):
        assert KinematicMath.gear_ratio(True, 70.0, 20.0) == 3.5

    def test_rollers_zero_diameter(self):
        assert KinematicMath.gear_ratio(True, 0.0, 20.0) == 1.0

    def test_rollers_zero_roller(self):
        assert KinematicMath.gear_ratio(True, 70.0, 0.0) == 1.0

    def test_matches_axis_mapper_and_opplayer(self):
        obj_d = 70.0
        roller_d = 20.0
        expected = obj_d / roller_d
        assert KinematicMath.gear_ratio(True, obj_d, roller_d) == expected


class TestMuToDegrees:
    def test_basic_conversion(self):
        diameter = 25.0
        mu = 50.0
        expected = (mu / (diameter * math.pi)) * 360.0
        assert KinematicMath.mu_to_degrees(mu, diameter) == pytest.approx(
            expected
        )

    def test_with_z_offset(self):
        diameter = 25.0
        z = -2.0
        eff_d = diameter + 2.0 * z
        mu = 50.0
        expected = (mu / (eff_d * math.pi)) * 360.0
        assert KinematicMath.mu_to_degrees(mu, eff_d) == pytest.approx(
            expected
        )

    def test_zero_diameter(self):
        assert KinematicMath.mu_to_degrees(50.0, 0.0) == 0.0

    def test_negative_diameter(self):
        assert KinematicMath.mu_to_degrees(50.0, -1.0) == 0.0

    def test_reverse(self):
        diameter = 25.0
        mu = 50.0
        forward = KinematicMath.mu_to_degrees(mu, diameter)
        backward = KinematicMath.mu_to_degrees(mu, diameter, reverse=True)
        assert backward == pytest.approx(-forward)

    def test_with_gear_ratio(self):
        diameter = 70.0
        gear_ratio = 3.5
        mu = math.pi * diameter
        expected = 360.0 * gear_ratio
        assert KinematicMath.mu_to_degrees(
            mu, diameter, gear_ratio=gear_ratio
        ) == pytest.approx(expected)

    def test_matches_assembly_formula(self):
        diameter = 50.0
        angle = 100.0
        circumference = diameter * math.pi
        expected_rad = (angle / circumference) * 2.0 * math.pi
        degrees = KinematicMath.mu_to_degrees(angle, diameter)
        actual_rad = math.radians(degrees)
        assert actual_rad == pytest.approx(expected_rad)

    def test_matches_transform_to_cylinder_formula(self):
        diameter = 10.0
        src = 5.0
        circumference = diameter * math.pi
        expected_theta = (src / circumference) * 2.0 * math.pi
        degrees = KinematicMath.mu_to_degrees(src, diameter)
        actual_theta = math.radians(degrees)
        assert actual_theta == pytest.approx(expected_theta)

    def test_numpy_array(self):
        diameter = 25.0
        src = np.array([0.0, 50.0, 100.0])
        circumference = diameter * math.pi
        expected = (src / circumference) * 360.0
        actual = KinematicMath.mu_to_degrees(src, diameter)
        np.testing.assert_array_almost_equal(actual, expected)


class TestDegreesToScaledMu:
    def test_basic(self):
        degrees = 360.0
        mu_per_rotation = 100.0
        expected = degrees * mu_per_rotation / 360.0
        assert KinematicMath.degrees_to_scaled_mu(
            degrees, mu_per_rotation
        ) == pytest.approx(expected)

    def test_zero_mu_per_rotation(self):
        assert KinematicMath.degrees_to_scaled_mu(180.0, 0.0) == 180.0

    def test_reverse(self):
        forward = KinematicMath.degrees_to_scaled_mu(180.0, 100.0)
        backward = KinematicMath.degrees_to_scaled_mu(
            180.0, 100.0, reverse=True
        )
        assert backward == pytest.approx(-forward)

    def test_with_gear_ratio(self):
        degrees = 360.0
        mu_per_rot = 100.0
        ratio = 3.5
        expected = degrees * mu_per_rot / 360.0 / ratio
        assert KinematicMath.degrees_to_scaled_mu(
            degrees, mu_per_rot, gear_ratio=ratio
        ) == pytest.approx(expected)

    def test_roundtrip_with_mu_to_degrees(self):
        mu = 50.0
        eff_d = 25.0
        mu_per_rot = 100.0
        ratio = 2.0
        degrees = KinematicMath.mu_to_degrees(mu, eff_d, gear_ratio=ratio)
        back = KinematicMath.degrees_to_scaled_mu(
            degrees, mu_per_rot, gear_ratio=ratio
        )
        assert back == pytest.approx(mu * mu_per_rot / (math.pi * eff_d))


class TestMuToScaledMu:
    def test_basic(self):
        diameter = 25.0
        mu_per_rot = 100.0
        mu = 50.0
        expected = mu * mu_per_rot / (math.pi * diameter)
        assert KinematicMath.mu_to_scaled_mu(
            mu, diameter, mu_per_rot
        ) == pytest.approx(expected)

    def test_zero_mu_per_rotation(self):
        assert KinematicMath.mu_to_scaled_mu(50.0, 25.0, 0.0) == 50.0

    def test_zero_diameter(self):
        assert KinematicMath.mu_to_scaled_mu(50.0, 0.0, 100.0) == 0.0

    def test_reverse(self):
        forward = KinematicMath.mu_to_scaled_mu(50.0, 25.0, 100.0)
        backward = KinematicMath.mu_to_scaled_mu(
            50.0, 25.0, 100.0, reverse=True
        )
        assert backward == pytest.approx(-forward)

    def test_with_gear_ratio(self):
        obj_d = 70.0
        roller_d = 20.0
        mu_per_rot = 100.0
        mu = math.pi * obj_d
        ratio = obj_d / roller_d
        expected = mu * mu_per_rot / (math.pi * obj_d) * ratio
        assert KinematicMath.mu_to_scaled_mu(
            mu, obj_d, mu_per_rot, gear_ratio=ratio
        ) == pytest.approx(expected)

    def test_matches_axis_mapper_scale_replacement(self):
        diameter = 25.0
        mu = 50.0
        z = -2.0
        mu_per_rot = 100.0
        eff_d = diameter + 2.0 * z
        expected = mu * mu_per_rot / (math.pi * eff_d)
        assert KinematicMath.mu_to_scaled_mu(
            mu, eff_d, mu_per_rot
        ) == pytest.approx(expected)


class TestScaledMuToMu:
    def test_basic(self):
        diameter = 25.0
        mu_per_rot = 100.0
        scaled = 50.0 * mu_per_rot / (math.pi * diameter)
        expected = 50.0
        assert KinematicMath.scaled_mu_to_mu(
            scaled, diameter, mu_per_rot
        ) == pytest.approx(expected)

    def test_inverse_of_mu_to_scaled_mu(self):
        mu = 50.0
        eff_d = 25.0
        mu_per_rot = 100.0
        ratio = 2.0
        scaled = KinematicMath.mu_to_scaled_mu(
            mu, eff_d, mu_per_rot, gear_ratio=ratio
        )
        back = KinematicMath.scaled_mu_to_mu(
            scaled, eff_d, mu_per_rot, gear_ratio=ratio
        )
        assert back == pytest.approx(mu)

    def test_inverse_with_reverse(self):
        mu = 50.0
        eff_d = 25.0
        mu_per_rot = 100.0
        scaled = KinematicMath.mu_to_scaled_mu(
            mu, eff_d, mu_per_rot, reverse=True
        )
        back = KinematicMath.scaled_mu_to_mu(
            scaled, eff_d, mu_per_rot, reverse=True
        )
        assert back == pytest.approx(mu)

    def test_zero_mu_per_rotation(self):
        assert KinematicMath.scaled_mu_to_mu(50.0, 25.0, 0.0) == 50.0

    def test_zero_diameter(self):
        assert KinematicMath.scaled_mu_to_mu(50.0, 0.0, 100.0) == 0.0

    def test_matches_opplayer_formula(self):
        diameter = 25.0
        mu_per_rot = 100.0
        ratio = 1.0
        fw_val = 50.0
        expected = fw_val * math.pi * diameter / mu_per_rot / ratio
        assert KinematicMath.scaled_mu_to_mu(
            fw_val, diameter, mu_per_rot, gear_ratio=ratio
        ) == pytest.approx(expected)
