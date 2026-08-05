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
    def mm_to_degrees(mm, effective_diameter, gear_ratio=1.0, reverse=False):
        """Convert a surface distance (mm) on the cylinder to degrees."""
        if effective_diameter <= 0:
            return 0.0
        circumference = effective_diameter * math.pi
        degrees = (mm / circumference) * 360.0 * gear_ratio
        if reverse:
            degrees = -degrees
        return degrees

    @staticmethod
    def degrees_to_mm(degrees, mm_per_rotation, gear_ratio=1.0, reverse=False):
        """Convert degrees to linear mm via the firmware travel per
        rotation (mm)."""
        if mm_per_rotation <= 0:
            return degrees
        mm = degrees * mm_per_rotation / 360.0 / gear_ratio
        if reverse:
            mm = -mm
        return mm

    @staticmethod
    def surface_mm_to_rotation_mm(
        mm,
        effective_diameter,
        mm_per_rotation,
        gear_ratio=1.0,
        reverse=False,
    ):
        """Convert cylinder-surface mm to rotation-axis mm via the
        firmware travel per rotation."""
        if mm_per_rotation <= 0:
            return mm
        if effective_diameter <= 0:
            return 0.0
        scaled = (
            mm * mm_per_rotation / (math.pi * effective_diameter) * gear_ratio
        )
        if reverse:
            scaled = -scaled
        return scaled

    @staticmethod
    def rotation_mm_to_surface_mm(
        rotation_mm,
        effective_diameter,
        mm_per_rotation,
        gear_ratio=1.0,
        reverse=False,
    ):
        """Convert rotation-axis mm back to cylinder-surface mm."""
        if mm_per_rotation <= 0:
            return rotation_mm
        if effective_diameter <= 0:
            return 0.0
        mm = (
            rotation_mm
            * (math.pi * effective_diameter)
            / mm_per_rotation
            / gear_ratio
        )
        if reverse:
            mm = -mm
        return mm
