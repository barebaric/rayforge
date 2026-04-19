import math


class KinematicMath:
    @staticmethod
    def effective_diameter(diameter: float, z: float) -> float:
        return diameter + 2.0 * z

    @staticmethod
    def gear_ratio(
        is_rollers: bool,
        rotary_diameter: float,
        roller_diameter: float,
    ) -> float:
        if is_rollers and roller_diameter > 0 and rotary_diameter > 0:
            return rotary_diameter / roller_diameter
        return 1.0

    @staticmethod
    def mu_to_degrees(mu, effective_diameter, gear_ratio=1.0, reverse=False):
        if effective_diameter <= 0:
            return 0.0
        circumference = effective_diameter * math.pi
        degrees = (mu / circumference) * 360.0 * gear_ratio
        if reverse:
            degrees = -degrees
        return degrees

    @staticmethod
    def degrees_to_scaled_mu(
        degrees, mu_per_rotation, gear_ratio=1.0, reverse=False
    ):
        if mu_per_rotation <= 0:
            return degrees
        scaled = degrees * mu_per_rotation / 360.0 / gear_ratio
        if reverse:
            scaled = -scaled
        return scaled

    @staticmethod
    def mu_to_scaled_mu(
        mu, effective_diameter, mu_per_rotation, gear_ratio=1.0, reverse=False
    ):
        if mu_per_rotation <= 0:
            return mu
        if effective_diameter <= 0:
            return 0.0
        scaled = (
            mu * mu_per_rotation / (math.pi * effective_diameter) * gear_ratio
        )
        if reverse:
            scaled = -scaled
        return scaled

    @staticmethod
    def scaled_mu_to_mu(
        scaled_mu,
        effective_diameter,
        mu_per_rotation,
        gear_ratio=1.0,
        reverse=False,
    ):
        if mu_per_rotation <= 0:
            return scaled_mu
        if effective_diameter <= 0:
            return 0.0
        mu = (
            scaled_mu
            * (math.pi * effective_diameter)
            / mu_per_rotation
            / gear_ratio
        )
        if reverse:
            mu = -mu
        return mu
