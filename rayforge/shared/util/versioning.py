import logging
import re
from typing import List, Optional, Tuple

import semver

logger = logging.getLogger(__name__)


def is_newer_version(remote_str: str, local_str: str) -> bool:
    """Compares two version strings using semver."""
    try:
        remote_v = semver.VersionInfo.parse(remote_str.lstrip("v"))
        local_v = semver.VersionInfo.parse(local_str.lstrip("v"))
        return remote_v > local_v
    except ValueError:
        logger.warning(
            f"Could not parse versions '{remote_str}' or '{local_str}' "
            "with semver. Falling back to string comparison."
        )
        return remote_str != local_str


def parse_version_constraint(constraint: str) -> Optional[Tuple[str, str]]:
    """
    Parse a version constraint string into operator and version.

    Args:
        constraint: Constraint string like ">=1.0.0" or "^0.27"

    Returns:
        Tuple of (operator, version_string) or None if invalid.
    """
    op_match = re.match(r"^((?:>=|<=|==|!=|>|<|=|~|\^))(.+)$", constraint)
    if not op_match:
        return None
    op = op_match.group(1)
    version = op_match.group(2).lstrip("v")
    if not version:
        return (op, "")
    return (op, version)


def normalize_tilde_version(version_str: str) -> str:
    """
    Normalize partial versions for tilde operator.

    Args:
        version_str: Version string, possibly partial (e.g., "0.27")

    Returns:
        Normalized full semver string (e.g., "0.27.0")
    """
    version_parts = version_str.split(".")
    if len(version_parts) == 2:
        return f"{version_parts[0]}.{version_parts[1]}.0"
    elif len(version_parts) == 1:
        return f"{version_parts[0]}.0.0"
    return version_str


def check_constraint(current_v, req_v, op: str) -> bool:
    """
    Check if current version satisfies a constraint.

    Args:
        current_v: Current semver VersionInfo
        req_v: Required semver VersionInfo
        op: Operator string (>=, >, <=, <, ==, !=, ^, ~)

    Returns:
        True if constraint is satisfied, False otherwise.
    """
    if op == ">=":
        return current_v >= req_v
    elif op == ">":
        return current_v > req_v
    elif op == "<=":
        return current_v <= req_v
    elif op == "<":
        return current_v < req_v
    elif op == "==":
        return current_v == req_v
    elif op == "!=":
        return current_v != req_v
    elif op == "^":
        return current_v.major == req_v.major and current_v >= req_v
    elif op == "~":
        return (
            current_v.major == req_v.major
            and current_v.minor == req_v.minor
            and current_v >= req_v
        )
    return False


def check_rayforge_compatibility(
    depends: List[str], current_version: str
) -> bool:
    """
    Check if rayforge version satisfies all rayforge dependencies.

    Args:
        depends: List of dependency strings
        current_version: Current rayforge version string

    Returns:
        True if compatible, False otherwise.
    """
    try:
        current_v = semver.VersionInfo.parse(current_version.lstrip("v"))
    except ValueError:
        return True

    for dep in depends:
        parts = dep.split(",")
        first_part = parts[0].strip()

        pkg_name = re.split(r"[~^><=!]+", first_part)[0]
        if pkg_name != "rayforge":
            continue

        first_constraint = first_part[len(pkg_name) :].strip()
        constraints = [first_constraint] + parts[1:]

        for constraint in constraints:
            constraint = constraint.strip()
            if not constraint:
                continue

            parsed = parse_version_constraint(constraint)
            if not parsed:
                return False

            op, req_v_str = parsed

            if op == "~":
                req_v_str = normalize_tilde_version(req_v_str)

            try:
                req_v = semver.VersionInfo.parse(req_v_str)
            except ValueError:
                return False

            if not check_constraint(current_v, req_v, op):
                return False

    return True
