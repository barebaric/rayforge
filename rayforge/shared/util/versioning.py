import importlib
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import semver

logger = logging.getLogger(__name__)


class _UnknownVersion:
    """
    Sentinel class representing an unknown or undetermined version.

    Used for builtin packages where version cannot be determined from git.
    When displayed, the UI should fall back to rayforge.__version__.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "UnknownVersion"


UnknownVersion = _UnknownVersion()


def get_git_tag_version(package_dir: Path) -> str:
    """
    Gets the version from git tags in the package directory.

    Args:
        package_dir (Path): The directory containing the package.

    Returns:
        str: The version from the latest git tag.

    Raises:
        RuntimeError: If no git tags are found or git is unavailable.
    """
    try:
        importlib.import_module("git")
    except ImportError:
        raise RuntimeError("GitPython is required to get git tag version")

    from git import Repo

    try:
        repo = Repo(package_dir)
        tags = repo.tags
        if tags:
            latest_tag = sorted(
                tags, key=lambda t: t.commit.committed_datetime
            )[-1]
            return latest_tag.name
        raise RuntimeError(f"No git tags found in {package_dir}")
    except Exception as e:
        raise RuntimeError(
            f"Failed to get git tag version from {package_dir}: {e}"
        )


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


def parse_requirement(req: str) -> Tuple[str, Optional[str]]:
    """
    Parse a requirement string into package name and version constraint.

    Args:
        req: Requirement string like "rayforge>=0.27.0" or "laser-essentials"

    Returns:
        Tuple of (package_name, constraint_or_none).
        Constraint includes operator, e.g. ">=1.0.0".

    Examples:
        "laser-essentials" -> ("laser-essentials", None)
        "laser-essentials>=1.0.0" -> ("laser-essentials", ">=1.0.0")
        "rayforge >= 0.27.0" -> ("rayforge", ">=0.27.0")
    """
    req = req.strip()
    match = re.match(
        r"^([a-zA-Z0-9_-]+)\s*((?:>=|<=|==|!=|>|<|=|~|\^).*)?$", req
    )
    if match:
        name = match.group(1)
        constraint = match.group(2).strip() if match.group(2) else None
        return name, constraint
    return req, None


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

        pkg_name, constraint = parse_requirement(first_part)
        if pkg_name != "rayforge":
            continue

        constraints = [constraint] if constraint else []
        for extra in parts[1:]:
            extra = extra.strip()
            if extra:
                constraints.append(extra)

        for constraint_str in constraints:
            if not constraint_str:
                continue

            parsed = parse_version_constraint(constraint_str)
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
