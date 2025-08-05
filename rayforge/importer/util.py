def to_mm(value, unit, px_factor=None):
    """Convert a value to millimeters based on its unit."""
    if unit == "cm":
        return value*10
    if unit == "mm":
        return value
    elif unit == "in":
        return value * 25.4  # 1 inch = 25.4 mm
    elif unit == "pt":
        return value * 25.4 / 72
    elif px_factor and unit in ("", "px"):
        return value * px_factor
    raise ValueError("Cannot convert to millimeters without DPI information.")
