"""Schema-version migrations for machines loaded from older app
versions.

When new head or machine settings are added to the model, existing
user machines have heads with the old defaults (typically 0.0). There
is no device profile to diff against for custom machines, so the
profile-review dialog cannot surface them. This module computes the
same kind of :class:`SettingDiff` entries the profile diff produces,
but from a schema-version comparison: for each new setting that was
absent at the machine's stored schema version, the diff shows the
current (sentinel) value and the suggested value (the laser-type
default or nominal default).
"""

from dataclasses import dataclass
from gettext import gettext as _

from ..models.laser import LaserHead
from .profile_diff import HEADS_SECTION, SettingDiff, apply_head_value

# Bump when new head/machine settings are added that existing machines
# should be prompted to review. Version 0 = pre-schema-version machines.
CURRENT_SCHEMA_VERSION = 1

# Sentinel indicating "follow the laser-type default". Matches the
# LaserHead default for wavelength_nm / max_power_watts.
_SENTINEL = 0.0


@dataclass(frozen=True)
class SchemaMigration:
    """Describes one setting introduced at a given schema version."""

    version: int
    head_key: str
    label: str
    suggested: float

    def applies_to(self, machine_version: int) -> bool:
        return machine_version < self.version


# Migrations added in schema version 1: the physical-power model
# (wavelength + optical power) landed on LaserHead.
_MIGRATIONS: list[SchemaMigration] = [
    SchemaMigration(
        version=1,
        head_key="wavelength_nm",
        label=_("Wavelength (nm)"),
        suggested=_SENTINEL,  # resolved per-head below
    ),
    SchemaMigration(
        version=1,
        head_key="max_power_watts",
        label=_("Max Optical Power (W)"),
        suggested=_SENTINEL,  # resolved per-head below
    ),
]


def find_schema_migrations(machine) -> list[SettingDiff]:
    """Returns one :class:`SettingDiff` per head setting the machine
    is missing relative to the current schema version.

    Only laser heads are considered. A setting is "missing" when its
    stored value is still the sentinel (0.0), meaning the user never
    configured it and no profile review filled it. Settings that
    already hold any non-zero value — whether from a profile review,
    manual entry, or a previous migration — are left alone so the
    user is not re-prompted. The suggested value for wavelength is
    the laser-type default, and for optical power is the nominal 40 W
    default — the same fallbacks
    :meth:`LaserHead.effective_wavelength_nm` and
    :meth:`LaserHead.effective_max_power_watts` produce.

    Callers should run the profile-review dialog first (for machines
    with a ``source_profile_id``) so that profile-provided values are
    written before this check — the schema migration then only
    surfaces settings the profile did not cover.
    """
    if machine.schema_version >= CURRENT_SCHEMA_VERSION:
        return []

    diffs: list[SettingDiff] = []
    for index, head in enumerate(machine.heads):
        if not isinstance(head, LaserHead):
            continue
        for mig in _MIGRATIONS:
            if not mig.applies_to(machine.schema_version):
                continue
            current = getattr(head, mig.head_key, _SENTINEL)
            if current != _SENTINEL:
                continue
            suggested = _suggested_value(head, mig.head_key)
            diffs.append(
                SettingDiff(
                    section=HEADS_SECTION,
                    key=f"head.{index}.{mig.head_key}",
                    path=(f"{HEADS_SECTION} → {head.name} → {mig.label}"),
                    current_value=current,
                    profile_value=suggested,
                )
            )
    return diffs


def apply_schema_migrations(machine, diffs: list[SettingDiff]) -> None:
    """Applies the selected migration diffs to *machine* in place and
    stamps the current schema version so the migration is not
    re-offered."""
    for diff in diffs:
        apply_head_value(machine, diff)
    machine.schema_version = CURRENT_SCHEMA_VERSION
    machine.changed.send(machine)


def _suggested_value(head: LaserHead, key: str) -> float:
    if key == "wavelength_nm":
        return head.effective_wavelength_nm()
    if key == "max_power_watts":
        return head.effective_max_power_watts()
    return _SENTINEL


__all__ = [
    "CURRENT_SCHEMA_VERSION",
    "apply_schema_migrations",
    "find_schema_migrations",
]
