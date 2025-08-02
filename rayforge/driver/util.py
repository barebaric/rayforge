import ipaddress


class Hostname(str):
    """A string subclass for identifying hostnames, for UI generation."""

    pass


def is_valid_hostname_or_ip(s: str) -> bool:
    """
    Checks if a string is a valid IP or a plausible hostname.

    This check is purely string-based and does not perform network
    lookups. An empty string is 'valid' here to prevent error states on
    an empty field; the driver itself will reject it upon configuration.
    """
    if not isinstance(s, str):
        return False

    # Allow empty string in UI; driver setup will handle it.
    if not s:
        return True

    # 1. Check for valid IP address (IPv4 or IPv6)
    try:
        ipaddress.ip_address(s)
        return True
    except ValueError:
        # It's not a valid IP, so proceed to hostname validation.
        pass

    # 2. Hostname validation (simplified RFC 1123)
    if len(s) > 253 or s.endswith("."):
        return False

    labels = s.split(".")

    # Rule out things that look like bad IPv4 addresses. If a name has
    # 4 parts and one part is numeric, they must all be numeric (which
    # would make it an IP, already handled and failed above).
    if len(labels) == 4 and any(label.isdigit() for label in labels):
        if not all(label.isdigit() for label in labels):
            return False

    # A string of only digits is not a valid hostname. It must be a
    # valid IP, which was already checked and failed.
    if s.replace(".", "").isdigit():
        return False

    # Check each label
    for label in labels:
        if not (1 <= len(label) <= 63):
            return False
        if label.startswith("-") or label.endswith("-"):
            return False
        if not all(c.isalnum() or c == "-" for c in label):
            return False

    return True
