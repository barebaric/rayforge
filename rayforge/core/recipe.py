import math
import uuid
from dataclasses import asdict, dataclass, field
from gettext import gettext as _
from typing import TYPE_CHECKING, Any, Optional

from .capability import StepCapability
from .capability_registry import step_capability_registry
from .varset import VarSet

DEFAULT_CAPABILITY_NAME = "CUT"

if TYPE_CHECKING:
    from ..machine.models.machine import Machine
    from .step import Step
    from .stock import StockItem


class _UnknownCapability(StepCapability):
    """Fallback for a recipe whose target capability is not registered."""

    @property
    def name(self) -> str:
        return "UNKNOWN"

    @property
    def label(self) -> str:
        return _("Unknown")

    @property
    def varset(self) -> VarSet:
        return VarSet(vars=[])


class _AnyCapability(StepCapability):
    """Sentinel for a recipe with no capability constraint.

    Such a recipe is either step-type-scoped or fully generic, so it
    matches any capability context.
    """

    @property
    def name(self) -> str:
        return ""

    @property
    def label(self) -> str:
        return _("Any")

    @property
    def varset(self) -> VarSet:
        return VarSet(vars=[])

    @property
    def icon_name(self) -> str:
        return "recipe-symbolic"


_UNKNOWN_CAPABILITY = _UnknownCapability()
_ANY_CAPABILITY = _AnyCapability()


@dataclass
class Recipe:
    """
    A preset for configuring a single task (capability) based on context,
    such as material and thickness. This is a pure data object.
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = "New Recipe"
    description: str = ""

    # --- Applicability Criteria ---
    target_capability_name: str = DEFAULT_CAPABILITY_NAME
    # When set, the recipe only matches steps of this class (by class
    # name, as registered in step_registry). When None, it matches by
    # capability as before.
    target_step_type: Optional[str] = None
    target_machine_id: Optional[str] = None
    material_uid: Optional[str] = None
    min_thickness_mm: Optional[float] = None
    max_thickness_mm: Optional[float] = None

    # --- Payload ---
    # A single dictionary of settings to be applied.
    settings: dict[str, Any] = field(default_factory=dict)

    # Forward compatibility: store unknown attributes
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def capability(self) -> StepCapability:
        """
        Returns the capability instance for this recipe, falling back to
        the default capability (and finally an unknown placeholder) when
        the target capability is not registered.

        A recipe without a capability constraint
        (``target_capability_name`` empty) resolves to the "Any"
        capability sentinel.
        """
        if not self.target_capability_name:
            return _ANY_CAPABILITY
        cap = step_capability_registry.get(self.target_capability_name)
        if cap is None:
            cap = step_capability_registry.get(DEFAULT_CAPABILITY_NAME)
        if cap is None:
            cap = _UNKNOWN_CAPABILITY
        return cap

    def matches_step_settings(
        self,
        step: "Step",
        tolerance=1e-6,
    ) -> bool:
        """
        Compares this recipe's settings against a Step object's current
        settings. Only keys present in the recipe are checked.
        """
        for key, recipe_val in self.settings.items():
            if not hasattr(step, key):
                return False  # Step is missing an attribute the recipe defines

            step_val = getattr(step, key)

            if isinstance(step_val, float) and isinstance(recipe_val, float):
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
        capabilities: Optional[tuple[StepCapability, ...]] = None,
        machine: Optional["Machine"] = None,
        step_type: Optional[str] = None,
    ) -> bool:
        """
        Checks if this recipe is a valid candidate for the given context.

        Args:
            stock_items: A list of StockItems. If empty, only generic recipes
                         (without material/thickness constraints) match.
                         Returns True if recipe matches ANY item in the list.
            capabilities: An optional set of capabilities to filter by.
            machine: An optional machine to filter by.
            step_type: An optional step class name (as registered in
                       ``step_registry``) to filter by. Only meaningful
                       when :attr:`target_step_type` is set.

        Returns:
            True if the recipe is a valid match, False otherwise.
        """
        # 1. Check step type compatibility
        if self.target_step_type:
            # This recipe targets a specific step class. It can only
            # match when a step type context is provided and matches.
            if not step_type or step_type != self.target_step_type:
                return False

        # 2. Check machine compatibility
        if self.target_machine_id:
            # This recipe requires a specific machine.
            if not machine or machine.id != self.target_machine_id:
                return False

        # A recipe is considered compatible up to this point, so now check
        # secondary constraints like laser head.

        # 3. Check head compatibility (if specified in settings)
        target_head_uid = self.settings.get("selected_head_uid")
        if target_head_uid:
            # This recipe requires a specific head. It can only match if
            # a machine context is provided and that machine has the head.
            if not machine or not any(
                head.uid == target_head_uid for head in machine.heads
            ):
                return False

        # 4. Check capability (only when the recipe constrains it)
        if self.target_capability_name and capabilities:
            if self.capability not in capabilities:
                return False

        # 5. If no stock items to check against, only match generic recipes
        # (recipes without material/thickness constraints)
        if not stock_items:
            # If recipe has material constraint, it can't match without stock
            if self.material_uid is not None:
                return False
            # If recipe has thickness constraint, it can't match without stock
            if (
                self.min_thickness_mm is not None
                or self.max_thickness_mm is not None
            ):
                return False
            return True

        # 6. Check if recipe matches ANY of the stock items
        for stock_item in stock_items:
            if self._matches_stock(stock_item):
                return True

        return False

    def _matches_stock(self, stock_item: "StockItem") -> bool:
        """
        Checks if this recipe matches a single stock item.
        """
        # Check material compatibility
        if self.material_uid:
            # This recipe requires a specific material.
            if not stock_item or stock_item.material_uid != self.material_uid:
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

        Returns:
            A tuple representing the specificity score.
        """
        # Score 0 for specific, 1 for generic (None or not present)
        machine_score = 0 if self.target_machine_id is not None else 1
        head_score = 0 if "selected_head_uid" in self.settings else 1
        material_score = 0 if self.material_uid is not None else 1
        thickness_score = (
            0
            if self.min_thickness_mm is not None
            or self.max_thickness_mm is not None
            else 1
        )
        step_type_score = 0 if self.target_step_type is not None else 1
        return (
            machine_score,
            head_score,
            material_score,
            thickness_score,
            step_type_score,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes the Recipe to a dictionary suitable for YAML."""
        result = asdict(self)
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Recipe":
        """Deserializes a Recipe from a dictionary."""
        known_keys = {
            "uid",
            "name",
            "description",
            "target_capability_name",
            "target_step_type",
            "target_machine_id",
            "material_uid",
            "min_thickness_mm",
            "max_thickness_mm",
            "settings",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        settings = data.get("settings", {})
        # Legacy alias: old recipe files keyed head selection as
        # "selected_laser_uid".
        if (
            "selected_laser_uid" in settings
            and "selected_head_uid" not in settings
        ):
            settings = dict(settings)
            settings["selected_head_uid"] = settings.pop("selected_laser_uid")

        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Recipe"),
            description=data.get("description", ""),
            target_capability_name=data.get(
                "target_capability_name", DEFAULT_CAPABILITY_NAME
            ),
            target_step_type=data.get("target_step_type"),
            target_machine_id=data.get("target_machine_id"),
            material_uid=data.get("material_uid"),
            min_thickness_mm=data.get("min_thickness_mm"),
            max_thickness_mm=data.get("max_thickness_mm"),
            settings=settings,
            extra=extra,
        )
