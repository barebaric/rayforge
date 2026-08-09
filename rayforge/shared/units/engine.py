import re
from typing import ClassVar

METERS_TO_INCH = 39.37007874
METERS_TO_FEET = 3.280839895
KW_TO_HP = 1.34102
MM_PER_INCH = 25.4


class ConversionEngine:
    _symbols: ClassVar[dict[str, str]] = {
        # SI units.
        "nanometer": "nm",
        "nanometers": "nm",
        "um": "μm",
        "micrometer": "μm",
        "micrometers": "μm",
        "millimeter": "mm",
        "millimeters": "mm",
        "centimeter": "cm",
        "centimeters": "cm",
        "meter": "m",
        "meters": "m",
        "kilometer": "km",
        "kilometers": "km",
        # Time
        "second": "s",
        "seconds": "s",
        "sec": "s",
        "minute": "min",
        "minutes": "min",
        "hour": "hr",
        "hours": "hr",
        # Imperial.
        '"': "in",
        "inch": "in",
        "inches": "in",
        "'": "ft",
        "foot": "ft",
        "feet": "ft",
        "yard": "yd",
        "yards": "yd",
        "mile": "mi",
        "miles": "mi",
    }

    _base_conversions: ClassVar[dict[tuple[str, str], float]] = {
        ("m", "in"): METERS_TO_INCH,
        ("m", "ft"): METERS_TO_FEET,
        ("kW", "HP"): KW_TO_HP,
        ("min", "s"): 60,
        ("hr", "s"): 3600,
        ("hr", "min"): 60,
    }

    _si_prefixes: ClassVar[dict[str, float]] = {
        "n": 1e-9,
        "μ": 1e-6,
        "m": 1e-3,
        "c": 1e-2,
        "d": 1e-1,
        "": 1.0,
        "k": 1e3,
    }

    _length_units: ClassVar[set[str]] = {"m", "in", "ft", "yd", "mi"}
    _time_units: ClassVar[set[str]] = {"s", "min", "hr"}
    _time_squared_units: ClassVar[set[str]] = {"s²", "min²", "hr²"}

    def __init__(self):
        self.unitmap: dict[tuple[str, str], float] = {}
        self._value_split_re = re.compile(r"^([\d\.\-eE]+)\s*(\S*)$")
        self._build_unit_map()

    def _build_unit_map(self):
        # Build SI length conversions
        for p1, f1 in self._si_prefixes.items():
            for p2, f2 in self._si_prefixes.items():
                if p1 != p2:
                    self.unitmap[(f"{p1}m", f"{p2}m")] = f1 / f2

        # Build cross-system length conversions
        for (si_unit, imp_unit), factor in self._base_conversions.items():
            if si_unit == "m":  # Length
                for p, f in self._si_prefixes.items():
                    self.unitmap[(f"{p}m", imp_unit)] = f * factor
                    self.unitmap[(imp_unit, f"{p}m")] = 1 / (f * factor)

        # Build time conversions
        for (t1, t2), factor in self._base_conversions.items():
            if t1 in self._time_units and t2 in self._time_units:
                self.unitmap[(t1, t2)] = factor
                self.unitmap[(t2, t1)] = 1 / factor

        # Build time squared conversions for acceleration
        for (t1, t2), factor in self._base_conversions.items():
            if t1 in self._time_units and t2 in self._time_units:
                # Square the factor for time squared units
                squared_factor = factor * factor
                t1_squared = f"{t1}²"
                t2_squared = f"{t2}²"
                self.unitmap[(t1_squared, t2_squared)] = squared_factor
                self.unitmap[(t2_squared, t1_squared)] = 1 / squared_factor

        # Add identity conversions
        all_units = {k[0] for k in self.unitmap} | {k[1] for k in self.unitmap}
        for unit in all_units:
            self.unitmap[(unit, unit)] = 1.0

    def _suffix_split(self, unit: str) -> tuple[str, str | None]:
        if "/" in unit:
            base, suffix = unit.split("/", 1)
            return base, suffix
        return unit, None

    def normalize_unit_symbol(self, unit: str) -> str:
        """
        Normalizes a unit symbol string by looking up synonyms.
        e.g., "inch" -> "in", "mm/second" -> "mm/s"
        """
        base_unit, suffix = self._suffix_split(unit)
        normalized_base = self._symbols.get(base_unit.lower(), base_unit)

        if suffix:
            normalized_suffix = self._symbols.get(suffix.lower(), suffix)
            return f"{normalized_base}/{normalized_suffix}"
        return normalized_base

    def parse_value(self, value_str: str) -> tuple[float, str | None]:
        if not isinstance(value_str, str):
            return value_str, None
        match = self._value_split_re.match(value_str.strip())
        if not match:
            raise ValueError(f"Could not parse value: '{value_str}'")
        value, unit_str = match.groups()
        unit = self.normalize_unit_symbol(unit_str) if unit_str else None
        return float(value), unit

    def convert(
        self, value: float, from_unit: str, to_unit: str
    ) -> tuple[float, str]:
        if from_unit == to_unit:
            return value, to_unit

        from_norm = self.normalize_unit_symbol(from_unit)
        to_norm = self.normalize_unit_symbol(to_unit)

        from_base, from_suffix = self._suffix_split(from_norm)
        to_base, to_suffix = self._suffix_split(to_norm)

        # Handle time conversion for compound units like speed and acceleration
        time_factor = 1.0
        if from_suffix != to_suffix:
            if from_suffix and to_suffix:
                time_factor = self.unitmap.get((from_suffix, to_suffix))
                if time_factor is None:
                    msg = (
                        f"Incompatible suffixes: '{from_suffix}' "
                        "and '{to_suffix}'"
                    )
                    raise ValueError(msg)
            else:
                msg = (
                    f"Incompatible suffixes: '{from_suffix}' and '{to_suffix}'"
                )
                raise ValueError(msg)

        length_factor = self.unitmap.get((from_base, to_base))
        if length_factor is None:
            raise ValueError(
                f"Unsupported conversion from '{from_base}' to '{to_base}'"
            )

        return value * length_factor / time_factor, to_norm


# Singleton instance for global use
engine = ConversionEngine()
