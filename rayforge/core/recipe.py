import logging
import math
import uuid
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Optional

from .step import Step
from .step_registry import step_registry

if TYPE_CHECKING:
    from ..machine.models.machine import Machine
    from .stock import StockItem

logger = logging.getLogger(__name__)

# Specificity score contributed by the step-type axis when a recipe is
# generic (matches any step type). It must rank as the least specific
# option, so it is larger than any realistic ``len(target_step_types)``.
_GENERIC_STEP_TYPE_SCORE = 1 << 16

# Migration shim: maps the legacy ``target_capability_name`` values to
# the list of step class names that declared that capability, captured
# at the time step capabilities were removed. Used only by
# :meth:`Recipe.from_dict` to preserve the targeting of old recipe files.
_LEGACY_CAPABILITY_STEPS: dict[str, list[str]] = {
    "CUT": ["ContourStep", "FrameStep", "WavefrontStep", "ShrinkWrapStep"],
    "SCORE": ["ContourStep", "FrameStep", "ShrinkWrapStep"],
    "ENGRAVE": ["EngraveStep"],
    "MATERIAL_TEST": ["MaterialTestStep"],
}


@dataclass
class Recipe:
    """
    A preset for configuring a step based on context, such as material
    and thickness. This is a pure data object.

    A recipe applies to one or more step types (identified by their
    class name as registered in ``step_registry``). When
    :attr:`target_step_types` is empty the recipe is generic and matches
    any step type.
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Recipe"
    description: str = ""

    # --- Applicability Criteria ---
    target_step_types: list[str] = field(default_factory=list)
    target_machine_id: str | None = None
    material_uid: str | None = None
    min_thickness_mm: float | None = None
    max_thickness_mm: float | None = None

    # --- Payload ---
    # Setting entries captured by this recipe. Each dict carries
    # ``name``, ``value`` and ``recipe_apply`` (False = "Leave
    # unchanged"), structurally identical to ``transformer_dicts``.
    setting_dicts: list[dict[str, Any]] = field(default_factory=list)

    # Post-processor (transformer) settings captured by this recipe.
    # Each dict carries ``name``, ``enabled``, ``recipe_apply`` (False =
    # "Leave unchanged"), plus the transformer's own params.
    transformer_dicts: list[dict[str, Any]] = field(default_factory=list)

    # Forward compatibility: store unknown attributes
    extra: dict[str, Any] = field(default_factory=dict)

    def get_applied_settings(self) -> dict[str, Any]:
        """Flat dict of the recipe's settings with ``recipe_apply=True``.

        Entries without an explicit ``recipe_apply`` count as applied.
        """
        return {
            d["name"]: d["value"]
            for d in self.setting_dicts
            if d.get("name") is not None and d.get("recipe_apply", True)
        }

    def get_settings_for_step(self, step: "Step") -> dict[str, Any]:
        """The recipe's applied settings the given step actually owns.

        Setting names are gated through the step type's
        :meth:`~rayforge.core.step.Step.recipe_keys` allowlist;
        everything else is dropped. Recipe files are user-provided, so
        they must never be able to reach arbitrary step attributes.
        """
        allowed = type(step).recipe_keys()
        return {
            name: value
            for name, value in self.get_applied_settings().items()
            if name in allowed
        }

    def matches_step_settings(
        self,
        step: "Step",
        tolerance=1e-6,
    ) -> bool:
        """
        Compares this recipe's applied settings against a Step object's
        current settings. Only keys the recipe carries and the step type
        owns are checked; enum-backed attributes are compared through
        the step's ``recipe_value`` serialization.
        """
        for key, recipe_val in self.get_settings_for_step(step).items():
            if not hasattr(step, key):
                return False  # Step is missing an attribute the recipe defines

            step_val = type(step).recipe_value(key, getattr(step, key))

            if isinstance(step_val, float) and isinstance(recipe_val, float):
                if not math.isclose(
                    step_val, recipe_val, rel_tol=0, abs_tol=tolerance
                ):
                    return False
            elif step_val != recipe_val:
                return False
        return True

    def matches_step_transformers(
        self,
        step: "Step",
        tolerance: float = 1e-6,
    ) -> bool:
        """Compare recipe's ``recipe_apply=True`` transformers to the
        step's transformers by name + params.

        Only transformer dicts with ``recipe_apply=True`` are checked.
        Each is matched against the step's
        ``per_workpiece_transformers_dicts`` +
        ``per_step_transformers_dicts`` (deduplicated by name). For
        each matching transformer name, every key present in the recipe
        dict (except ``recipe_apply``) that the step's dict also
        declares is compared against it. Floats use ``math.isclose``
        with ``tolerance``.

        Returns ``True`` if the step has a matching transformer for
        every recipe entry with ``recipe_apply=True``.
        """
        apply_dicts = [
            d for d in self.transformer_dicts if d.get("recipe_apply", True)
        ]
        if not apply_dicts:
            return True
        step_dicts = Step._dedupe_transformer_dicts_by_name(
            list(step.per_workpiece_transformers_dicts)
            + list(step.per_step_transformers_dicts)
        )
        for recipe_dict in apply_dicts:
            name = recipe_dict.get("name")
            if not name:
                return False
            match = step_dicts.get(name)
            if match is None:
                return False
            for key, recipe_val in recipe_dict.items():
                if key == "recipe_apply":
                    continue
                if key not in match:
                    continue
                step_val = match[key]
                if isinstance(step_val, float) and isinstance(
                    recipe_val, float
                ):
                    if not math.isclose(
                        step_val, recipe_val, rel_tol=0, abs_tol=tolerance
                    ):
                        return False
                elif step_val != recipe_val:
                    return False
        return True

    def matches(
        self,
        stock_items: list["StockItem"],
        machine: Optional["Machine"] = None,
        step_type: str | None = None,
    ) -> bool:
        """
        Checks if this recipe is a valid candidate for the given context.

        Args:
            stock_items: A list of StockItems. If empty, only generic recipes
                         (without material/thickness constraints) match.
                         Returns True if recipe matches ANY item in the list.
            machine: An optional machine to filter by.
            step_type: An optional step class name (as registered in
                       ``step_registry``). Only meaningful when
                       :attr:`target_step_types` is non-empty; a recipe
                       with an empty ``target_step_types`` matches any
                       step type.

        Returns:
            True if the recipe is a valid match, False otherwise.
        """
        # 1. Check step type compatibility
        if self.target_step_types and step_type not in self.target_step_types:
            # This recipe targets specific step classes. It can only
            # match when a step type context is provided and is one of
            # the targeted classes.
            return False

        # 2. Check machine compatibility
        if self.target_machine_id and (
            not machine or machine.id != self.target_machine_id
        ):
            # This recipe requires a specific machine.
            return False

        # A recipe is considered compatible up to this point, so now check
        # secondary constraints like laser head.

        # 3. Check head compatibility (if specified in settings)
        target_head_uid = self.get_applied_settings().get("selected_head_uid")
        if target_head_uid and (
            not machine
            or not any(head.uid == target_head_uid for head in machine.heads)
        ):
            # This recipe requires a specific head. It can only match if
            # a machine context is provided and that machine has the head.
            return False

        # 4. If no stock items to check against, only match generic recipes
        # (recipes without material/thickness constraints)
        if not stock_items:
            # If recipe has material constraint, it can't match without stock
            if self.material_uid is not None:
                return False
            # If recipe has thickness constraint, it can't match without stock
            return not (
                self.min_thickness_mm is not None
                or self.max_thickness_mm is not None
            )

        # 5. Check if recipe matches ANY of the stock items
        for stock_item in stock_items:
            if self._matches_stock(stock_item):
                return True

        return False

    def _matches_stock(self, stock_item: "StockItem") -> bool:
        """
        Checks if this recipe matches a single stock item.
        """
        # Check material compatibility
        if self.material_uid and (
            not stock_item or stock_item.material_uid != self.material_uid
        ):
            # This recipe requires a specific material.
            return False

        # Check thickness compatibility
        thickness_mm = stock_item.thickness if stock_item else None
        if (
            self.min_thickness_mm is not None
            or self.max_thickness_mm is not None
        ):
            # This recipe requires a specific thickness or range.
            if thickness_mm is None:
                return False  # No thickness provided, cannot match.
            if (
                self.min_thickness_mm is not None
                and thickness_mm < self.min_thickness_mm
            ):
                return False
            if (
                self.max_thickness_mm is not None
                and thickness_mm > self.max_thickness_mm
            ):
                return False

        # If all checks passed, it's a match.
        return True

    def get_specificity_score(self) -> tuple[int, int, int, int, int]:
        """
        Calculates a score based on how specific the recipe's criteria are.
        A lower score indicates a more specific (and therefore better) match.
        The score is a tuple
        (machine, head, material, thickness, step_type).

        For the step-type axis, a recipe targeting fewer step types is
        more specific than one targeting more; a generic recipe (no
        step types) is the least specific.

        Returns:
            A tuple representing the specificity score.
        """
        # Score 0 for specific, 1 for generic (None or not present)
        machine_score = 0 if self.target_machine_id is not None else 1
        head_score = (
            0 if "selected_head_uid" in self.get_applied_settings() else 1
        )
        material_score = 0 if self.material_uid is not None else 1
        thickness_score = (
            0
            if self.min_thickness_mm is not None
            or self.max_thickness_mm is not None
            else 1
        )
        if self.target_step_types:
            # Fewer targeted step types = more specific.
            step_type_score = len(self.target_step_types)
        else:
            step_type_score = _GENERIC_STEP_TYPE_SCORE
        return (
            machine_score,
            head_score,
            material_score,
            thickness_score,
            step_type_score,
        )

    def get_icon_name(self) -> str:
        """An icon name representing this recipe's targeted step types.

        When exactly one step type is targeted, that step's icon is
        used; otherwise the generic recipe icon.
        """
        if len(self.target_step_types) == 1:
            step_class = step_registry.get(self.target_step_types[0])
            if step_class is not None:
                return step_class.ICON or "recipe-symbolic"
        return "recipe-symbolic"

    def get_step_type_label(self) -> str | None:
        """A comma-joined label of the targeted step types.

        Returns ``None`` when the recipe is generic (no step types), so
        callers can decide their own fallback (e.g. "Any"). The string
        may be long; UI labels should ellipsize it.
        """
        if not self.target_step_types:
            return None
        labels = []
        for name in self.target_step_types:
            step_class = step_registry.get(name)
            if step_class is not None:
                labels.append(step_class.TYPELABEL)
            else:
                labels.append(name)
        return ", ".join(labels)

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Recipe to a dictionary suitable for YAML."""
        result = asdict(self)
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        """Deserializes a Recipe from a dictionary.

        Migrates the legacy ``target_step_type`` (single step class) and
        ``target_capability_name`` (operation category) keys into the
        current :attr:`target_step_types` list. Legacy capabilities are
        expanded to the step class names that declared them via
        :data:`_LEGACY_CAPABILITY_STEPS`.
        """
        known_keys = {
            "uid",
            "name",
            "description",
            "target_step_types",
            "target_machine_id",
            "material_uid",
            "min_thickness_mm",
            "max_thickness_mm",
            "setting_dicts",
            "transformer_dicts",
            # Legacy keys, consumed by the migration below.
            "settings",
            "target_capability_name",
            "target_step_type",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        setting_dicts = cls._migrate_setting_dicts(data)

        target_step_types = cls._migrate_target_step_types(data)

        transformer_dicts = data.get("transformer_dicts") or []
        if transformer_dicts and not isinstance(transformer_dicts, list):
            transformer_dicts = []
        transformer_dicts = [
            dict(d) for d in transformer_dicts if isinstance(d, dict)
        ]

        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Recipe"),
            description=data.get("description", ""),
            target_step_types=target_step_types,
            target_machine_id=data.get("target_machine_id"),
            material_uid=data.get("material_uid"),
            min_thickness_mm=data.get("min_thickness_mm"),
            max_thickness_mm=data.get("max_thickness_mm"),
            setting_dicts=setting_dicts,
            transformer_dicts=transformer_dicts,
            extra=extra,
        )

    @classmethod
    def _migrate_setting_dicts(
        cls, data: dict[str, Any]
    ) -> list[dict[str, Any]]:
        """Resolve ``setting_dicts`` from new and legacy keys.

        New files store ``setting_dicts`` (a list of ``name``/``value``/
        ``recipe_apply`` dicts). Legacy files store a flat ``settings``
        dict whose entries all count as applied. The legacy
        ``selected_laser_uid`` head key is renamed to
        ``selected_head_uid`` in both forms.
        """
        raw = data.get("setting_dicts")
        if raw is None:
            legacy = data.get("settings") or {}
            entries = [
                {"name": k, "value": v, "recipe_apply": True}
                for k, v in legacy.items()
            ]
        else:
            entries = [dict(d) for d in raw if isinstance(d, dict)]

        has_head = any(d.get("name") == "selected_head_uid" for d in entries)
        migrated: list[dict[str, Any]] = []
        for d in entries:
            name = d.get("name")
            if name == "selected_laser_uid":
                if has_head:
                    continue
                name = "selected_head_uid"
            migrated.append({**d, "name": name})
        return migrated

    @staticmethod
    def _migrate_target_step_types(data: dict[str, Any]) -> list[str]:
        """Resolve ``target_step_types`` from new and legacy keys."""
        if "target_step_types" in data:
            return list(data.get("target_step_types") or [])

        step_types: list[str] = []

        legacy_step_type = data.get("target_step_type")
        if legacy_step_type:
            step_types.append(legacy_step_type)

        legacy_capability = data.get("target_capability_name")
        if legacy_capability:
            mapped = _LEGACY_CAPABILITY_STEPS.get(legacy_capability)
            if mapped:
                for name in mapped:
                    if name not in step_types:
                        step_types.append(name)
            else:
                logger.warning(
                    "Could not migrate legacy target_capability_name"
                    " '%s'; no step-type mapping is registered.",
                    legacy_capability,
                )

        return step_types
