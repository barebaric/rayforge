import uuid
import math
from typing import Dict, Any, Optional, Set, Tuple, TYPE_CHECKING
from dataclasses import dataclass, field, asdict
from .capability import Capability, CAPABILITIES_BY_NAME, CUT

if TYPE_CHECKING:
    from .stock import StockItem
    from ..machine.models.machine import Machine
    from .step import Step


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
    target_capability_name: str = CUT.name
    target_machine_id: Optional[str] = None
    material_uid: Optional[str] = None
    min_thickness_mm: Optional[float] = None
    max_thickness_mm: Optional[float] = None

    # --- Payload ---
    # A single dictionary of settings to be applied.
    settings: Dict[str, Any] = field(default_factory=dict)

    @property
    def capability(self) -> Capability:
        """Returns the capability instance for this recipe."""
        return CAPABILITIES_BY_NAME.get(self.target_capability_name, CUT)

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
        stock_item: Optional["StockItem"],
        capabilities: Optional[Set[Capability]] = None,
        machine: Optional["Machine"] = None,
    ) -> bool:
        """
        Checks if this recipe is a valid candidate for the given context.

        Args:
            stock_item: The stock item to check against. Can be None.
            capabilities: An optional set of capabilities to filter by.
            machine: An optional machine to filter by.

        Returns:
            True if the recipe is a valid match, False otherwise.
        """
        # 1. Check machine compatibility
        if self.target_machine_id:
            # This recipe requires a specific machine.
            if not machine or machine.id != self.target_machine_id:
                return False

        # A recipe is considered compatible up to this point, so now check
        # secondary constraints like laser head.

        # 2. Check laser head compatibility (if specified in settings)
        target_laser_uid = self.settings.get("selected_laser_uid")
        if target_laser_uid:
            # This recipe requires a specific laser head. It can only match if
            # a machine context is provided and that machine has the head.
            if not machine or not any(
                head.uid == target_laser_uid for head in machine.heads
            ):
                return False

        # 3. Check capability
        if capabilities and self.capability not in capabilities:
            return False

        # 4. Check material compatibility
        if self.material_uid:
            # This recipe requires a specific material.
            if not stock_item or stock_item.material_uid != self.material_uid:
                return False

        # 5. Check thickness compatibility
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

    def get_specificity_score(self) -> Tuple[int, int, int, int]:
        """
        Calculates a score based on how specific the recipe's criteria are.
        A lower score indicates a more specific (and therefore better) match.
        The score is a tuple (machine, laser, material, thickness).

        Returns:
            A tuple representing the specificity score.
        """
        # Score 0 for specific, 1 for generic (None or not present)
        machine_score = 0 if self.target_machine_id is not None else 1
        laser_score = 0 if "selected_laser_uid" in self.settings else 1
        material_score = 0 if self.material_uid is not None else 1
        thickness_score = (
            0
            if self.min_thickness_mm is not None
            or self.max_thickness_mm is not None
            else 1
        )
        return (machine_score, laser_score, material_score, thickness_score)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes the Recipe to a dictionary suitable for YAML."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Recipe":
        """Deserializes a Recipe from a dictionary."""
        return cls(
            uid=data.get("uid", str(uuid.uuid4())),
            name=data.get("name", "Unnamed Recipe"),
            description=data.get("description", ""),
            target_capability_name=data.get(
                "target_capability_name", CUT.name
            ),
            target_machine_id=data.get("target_machine_id"),
            material_uid=data.get("material_uid"),
            min_thickness_mm=data.get("min_thickness_mm"),
            max_thickness_mm=data.get("max_thickness_mm"),
            settings=data.get("settings", {}),
        )
