"""Diff between a live :class:`Machine` and its source device profile.

Computes per-setting differences for the profile-review dialog. Each
difference is described by a :class:`SettingDiff`. Reviewable settings
are declared on :class:`~rayforge.machine.device.profile.MachineConfig`
itself, via each field's ``setting`` metadata (a
:class:`SettingBinding`); this module only walks those declarations.
Fields without a binding — connection info and ID-bearing objects like
heads — are not reviewable.
"""

import logging
from dataclasses import dataclass
from dataclasses import fields as dc_fields
from dataclasses import replace as dc_replace
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from ..models.head import head_settings
from .profile import MachineConfig

if TYPE_CHECKING:
    from ...machine.models.dialect.base import GcodeDialect
    from ..models.machine import Machine
    from .profile import DeviceProfile

logger = logging.getLogger(__name__)

MACHINE_SECTION = _("Machine Settings")
HEADS_SECTION = _("Heads")
DIALECT_SECTION = _("G-code Dialect")

DIALECT_KEY_PREFIX = "dialect."
HEAD_KEY_PREFIX = "head."


@dataclass(frozen=True)
class SettingDiff:
    """A single setting where machine and profile disagree."""

    section: str
    key: str
    path: str
    current_value: Any
    profile_value: Any


def _iter_setting_bindings():
    """Yields ``(field_name, SettingBinding)`` for reviewable fields."""
    for f in dc_fields(MachineConfig):
        binding = f.metadata.get("setting")
        if binding is not None:
            yield f.name, binding


def diff_machine_with_profile(
    machine: "Machine", profile: "DeviceProfile"
) -> list[SettingDiff]:
    """
    Returns machine-setting rows where the profile value differs from
    the machine's current value. Profile fields left unset (``None``)
    are skipped since there is nothing to apply.
    """
    cfg = profile.machine_config
    diffs: list[SettingDiff] = []
    for name, binding in _iter_setting_bindings():
        profile_value = getattr(cfg, name)
        if profile_value is None:
            continue
        current_value = binding.get(machine)
        if not _values_differ(current_value, profile_value):
            continue
        diffs.append(
            SettingDiff(
                section=MACHINE_SECTION,
                key=name,
                path=f"{binding.section} → {binding.label}",
                current_value=current_value,
                profile_value=profile_value,
            )
        )
    return diffs


def diff_heads_with_profile(
    machine: "Machine", profile: "DeviceProfile"
) -> list[SettingDiff]:
    """
    Returns one row per changed head field, matching profile heads to
    machine heads by position. Heads are only compared when the counts
    match; head objects themselves are never replaced.
    """
    cfg_heads = profile.machine_config.heads
    if not cfg_heads or len(cfg_heads) != len(machine.heads):
        return []
    diffs: list[SettingDiff] = []
    for index, (head_dict, head) in enumerate(zip(cfg_heads, machine.heads)):
        for setting in head_settings(head):
            if setting.key not in head_dict:
                continue
            profile_value = head_dict[setting.key]
            current_value = setting.yaml_value(head)
            if not _values_differ(current_value, profile_value):
                continue
            diffs.append(
                SettingDiff(
                    section=HEADS_SECTION,
                    key=f"{HEAD_KEY_PREFIX}{index}.{setting.key}",
                    path=(f"{HEADS_SECTION} → {head.name} → {setting.label}"),
                    current_value=current_value,
                    profile_value=profile_value,
                )
            )
    return diffs


def diff_dialect_with_profile(
    machine: "Machine", profile: "DeviceProfile"
) -> list[SettingDiff]:
    """
    Returns one row per changed dialect template field, comparing the
    machine's dialect against the profile's dialect template.
    """
    dialect = machine.dialect
    if dialect is None or not profile.dialect_config:
        return []
    labels = _dialect_field_labels(dialect)
    current_dict = dialect.to_template_dict()
    diffs: list[SettingDiff] = []
    for key in sorted(profile.dialect_config):
        if key not in current_dict:
            continue
        profile_value = profile.dialect_config[key]
        current_value = current_dict[key]
        if not _values_differ(current_value, profile_value):
            continue
        label = labels.get(key, _prettify_key(key))
        diffs.append(
            SettingDiff(
                section=DIALECT_SECTION,
                key=f"{DIALECT_KEY_PREFIX}{key}",
                path=f"{DIALECT_SECTION} → {label}",
                current_value=current_value,
                profile_value=profile_value,
            )
        )
    return diffs


def apply_diffs(
    machine: "Machine",
    profile: "DeviceProfile",
    diffs: list[SettingDiff],
):
    """
    Applies the selected :class:`SettingDiff` rows to *machine*.

    Machine settings are written through their field bindings, head
    fields onto the existing head objects, and dialect settings are
    merged into the machine's own dialect copy so its UID stays stable.
    The caller is responsible for persisting the machine afterwards.
    """
    bindings = dict(_iter_setting_bindings())
    dialect_values: dict[str, Any] = {}
    for diff in diffs:
        if diff.section == DIALECT_SECTION:
            dialect_key = diff.key.removeprefix(DIALECT_KEY_PREFIX)
            dialect_values[dialect_key] = diff.profile_value
            continue
        if diff.section == HEADS_SECTION:
            apply_head_value(machine, diff)
            continue
        binding = bindings.get(diff.key)
        if binding is None:
            logger.warning(f"No binding for profile setting '{diff.key}'")
            continue
        binding.apply(machine, diff.profile_value)

    if dialect_values:
        _apply_dialect_values(machine, dialect_values)


def apply_head_value(machine: "Machine", diff: SettingDiff):
    """Applies one head-field diff to the existing head object."""
    rest = diff.key.removeprefix(HEAD_KEY_PREFIX)
    index_str, _, key = rest.partition(".")
    try:
        head = machine.heads[int(index_str)]
    except (ValueError, IndexError):
        logger.warning(
            f"No head at index '{index_str}' for profile setting '{diff.key}'"
        )
        return
    setting = next((s for s in head_settings(head) if s.key == key), None)
    if setting is None:
        logger.warning(f"No reviewable head setting '{key}'")
        return
    setting.set_yaml_value(head, diff.profile_value)


def find_outdated_profiles(machines, profiles_by_id):
    """
    Returns ``(machine, profile)`` pairs whose profile content hash no
    longer matches the hash recorded at the machine's last review.
    Machines without profile provenance are ignored. A machine with a
    ``source_profile_id`` but no recorded hash (created before the
    profile-review system existed) is treated as never-reviewed and
    included so the first review can pick up new profile settings.
    """
    outdated = []
    for machine in machines:
        if not machine.source_profile_id:
            continue
        profile = profiles_by_id.get(machine.source_profile_id)
        if profile is None:
            logger.debug(
                f"Machine '{machine.name}' references unknown device "
                f"profile '{machine.source_profile_id}'; skipping "
                "profile review"
            )
            continue
        if not machine.reviewed_profile_hash:
            outdated.append((machine, profile))
            continue
        if profile.content_hash() != machine.reviewed_profile_hash:
            outdated.append((machine, profile))
    return outdated


def split_reviewable(machines_profiles):
    """
    Splits ``(machine, profile)`` pairs into those with at least one
    setting difference worth reviewing and those where the machine
    already matches the profile (nothing to apply).
    """
    reviewable = []
    nothing_to_do = []
    for machine, profile in machines_profiles:
        if (
            diff_machine_with_profile(machine, profile)
            or diff_heads_with_profile(machine, profile)
            or diff_dialect_with_profile(machine, profile)
        ):
            reviewable.append((machine, profile))
        else:
            nothing_to_do.append((machine, profile))
    return reviewable, nothing_to_do


# ----- internals -------------------------------------------------------


def _values_differ(current: Any, profile: Any) -> bool:
    return _normalize(current) != _normalize(profile)


def _normalize(value: Any) -> Any:
    if isinstance(value, tuple):
        return ("tuple", [_normalize(v) for v in value])
    if isinstance(value, (list, frozenset)):
        return sorted(_normalize(v) for v in value)
    if hasattr(value, "value"):
        return value.value
    return value


def _prettify_key(key: str) -> str:
    return key.replace("_", " ").capitalize()


def _dialect_field_labels(dialect: "GcodeDialect") -> dict[str, str]:
    labels: dict[str, str] = {}
    for varset in dialect.get_editor_varsets().values():
        for var in varset.vars:
            labels[var.key] = var.label
    return labels


def _apply_dialect_values(machine: "Machine", values: dict[str, Any]):
    dialect = machine.dialect
    if dialect is None:
        return
    if not dialect.is_custom:
        migrated_uid, _ = (
            machine.context.dialect_mgr.migrate_builtin_dialect_to_copy(
                dialect.uid, machine.name
            )
        )
        if migrated_uid:
            machine.set_dialect_uid(migrated_uid)
            dialect = machine.dialect
        if dialect is None or not dialect.is_custom:
            logger.warning(
                "Could not migrate dialect for machine "
                f"'{machine.name}'; skipping dialect update"
            )
            return
    try:
        updated = dc_replace(dialect, **values)
    except TypeError as e:
        logger.error(
            f"Invalid dialect update for machine '{machine.name}': {e}"
        )
        return
    machine.context.dialect_mgr.update_dialect(updated)
