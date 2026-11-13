from __future__ import annotations

from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from raygeo.geo.shape.text import FontConfig
from raygeo.geo.types import Point as GeoPoint
from raygeo.sketch.constraints.aspect_ratio import AspectRatioConstraint
from raygeo.sketch.constraints.distance import DistanceConstraint
from raygeo.sketch.constraints.equal_length import EqualLengthConstraint

from ..entity_ops import text_box_fourth_corner, text_box_natural_size
from ..types import EntityID
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from raygeo.sketch.registry import EntityRegistry

    from ..sketch import Sketch


def _font_properties(sketch: Sketch, text_entity_id: EntityID) -> dict:
    """The text box's font properties, overlay first, entity second."""
    entity = sketch.registry.get_entity_object(text_entity_id)
    props = sketch.flags.text_properties(text_entity_id)
    return {
        "content": props.get(
            "content",
            entity.content if entity is not None else "",
        ),
        "font_family": props.get(
            "font_family",
            entity.font_family if entity is not None else "sans-serif",
        ),
        "font_size": props.get(
            "font_size", entity.font_size if entity is not None else 10.0
        ),
        "bold": props.get("bold", entity.bold if entity is not None else False),
        "italic": props.get(
            "italic", entity.italic if entity is not None else False
        ),
    }


class ModifyTextPropertyCommand(SketchChangeCommand):
    def __init__(
        self,
        sketch: Sketch,
        text_entity_id: EntityID,
        new_content: str,
        new_font_config: FontConfig,
    ):
        super().__init__(sketch, _("Modify Text Property"))
        self.text_entity_id = text_entity_id
        self.new_content = new_content
        self.new_font_config = new_font_config
        self.old_content = ""
        self.old_font_config: FontConfig | None = None
        self.old_point_positions: dict[EntityID, GeoPoint] = {}
        self.old_aspect_ratio: float | None = None
        self.aspect_ratio_constraint_idx: int | None = None
        self._added_constraints: list[Any] = []

        self._entity_was_removed = False
        self._removed_entity_dict: dict[str, Any] | None = None
        self._removed_point_dicts: list[dict[str, Any]] = []
        self._removed_entity_dicts: list[dict[str, Any]] = []
        self._removed_constraints: list[Any] = []


    def _shed_size_constraints(self, registry: EntityRegistry) -> None:
        """
        Removes previously added hidden distance constraints and any user-added
        dimensional constraints on the text box frame to prevent over-
        constraining.
        """
        self._removed_constraints.clear()

        # Find any constraints that define the length of the box sides.
        constraints_to_remove = []
        for eid in registry.helper_ids(self.text_entity_id):
            if registry.entity_type(eid) != "line":
                continue
            endpoints = registry.entity_endpoint_ids(eid)
            if not endpoints or len(endpoints) != 2:
                continue

            for constr in self.sketch.constraints:
                if constr in constraints_to_remove:
                    continue

                is_target = False
                if isinstance(constr, EqualLengthConstraint):
                    member_ids = {
                        pid for pair in constr.pairs for pid in pair
                    }
                    if endpoints[0] in member_ids and (
                        endpoints[1] in member_ids
                    ):
                        # Equal-length constraints shedding one side
                        # can drop a pair member only by removing the
                        # whole constraint; the width re-established
                        # below keeps the frame sized.
                        constraints_to_remove.append(constr)
                        continue
                # This specifically targets hidden constraints added by this
                # command, or user-added distance constraints.
                elif isinstance(constr, DistanceConstraint) and {
                    constr.p1,
                    constr.p2,
                } == set(endpoints):
                    is_target = True

                if is_target:
                    constraints_to_remove.append(constr)

        self._removed_constraints.extend(constraints_to_remove)
        self.sketch.remove_constraints(self._removed_constraints)

    def capture_pre_edit_state(self) -> None:
        """Captures the entity state that undo should return to.

        Must be called BEFORE an interactive edit session mutates point
        positions or the aspect ratio constraint live: capturing later
        would record mid-typing mutations as the undo target."""
        registry = self.sketch.registry
        entity = registry.get_entity_object(self.text_entity_id)
        if entity is None or entity.type_name() != "text_box":
            return

        props = _font_properties(self.sketch, self.text_entity_id)
        self.old_content = props["content"]
        self.old_font_config = FontConfig.from_dict(
            {
                "font_family": props["font_family"],
                "font_size": props["font_size"],
                "bold": props["bold"],
                "italic": props["italic"],
            }
        )
        p_width = registry.get_point(entity.width_id)
        p_height = registry.get_point(entity.height_id)
        self.old_point_positions = {}
        if p_width is not None:
            self.old_point_positions[entity.width_id] = p_width
        if p_height is not None:
            self.old_point_positions[entity.height_id] = p_height
        # Find and store the old aspect ratio constraint
        for idx, constr in enumerate(self.sketch.constraints):
            if (
                isinstance(constr, AspectRatioConstraint)
                and constr.p1 == entity.origin_id
                and constr.p2 == entity.width_id
                and constr.p3 == entity.origin_id
                and constr.p4 == entity.height_id
            ):
                self.aspect_ratio_constraint_idx = idx
                self.old_aspect_ratio = constr.ratio
                break

    def _do_execute(self) -> None:
        if self._entity_was_removed:
            self._restore_text_entity()
            self._entity_was_removed = False

        registry = self.sketch.registry
        entity = registry.get_entity_object(self.text_entity_id)
        if entity is None or entity.type_name() != "text_box":
            return

        if not self.old_content and not self.old_point_positions:
            self.capture_pre_edit_state()

        # Update the entity's content and fonts.
        self.sketch.flags.set_text_property(
            self.text_entity_id, "content", self.new_content
        )
        for key, value in (
            ("font_family", self.new_font_config.font_family),
            ("font_size", self.new_font_config.font_size),
            ("bold", self.new_font_config.bold),
            ("italic", self.new_font_config.italic),
        ):
            self.sketch.flags.set_text_property(
                self.text_entity_id, key, value
            )

        # If the content is now empty, remove the entire entity.
        if not self.new_content:
            self._remove_text_entity(registry)
            return

        # Always clear any temporary constraints added by this command.
        self._added_constraints.clear()

        # Remove any existing dimensional constraints on the box frame to
        # prevent conflicts before we re-establish the width.
        self._shed_size_constraints(registry)

        # Get natural dimensions of the new text.
        natural_width, natural_height = text_box_natural_size(
            self.new_content,
            self.new_font_config.font_family,
            self.new_font_config.font_size,
            self.new_font_config.bold,
            self.new_font_config.italic,
        )

        # Always add a hidden constraint for the width. This is authoritative.
        width_constr = DistanceConstraint(
            entity.origin_id,
            entity.width_id,
            natural_width,
        )
        width_constr.set_user_visible(False)
        self._added_constraints.append(width_constr)
        self.sketch.add_constraint(width_constr)

        # Find the Aspect Ratio constraint, if it exists.
        active_ar_constraint = None
        for constr in self.sketch.constraints:
            if (
                isinstance(constr, AspectRatioConstraint)
                and constr.p1 == entity.origin_id
                and constr.p2 == entity.width_id
            ):
                active_ar_constraint = constr
                break

        # If the AR constraint exists, update its ratio. The solver will handle
        # the height automatically. If not, the height remains unconstrained.
        if active_ar_constraint and natural_height > 1e-9:
            new_ratio = natural_width / natural_height
            active_ar_constraint.set_ratio(new_ratio)

    def _remove_text_entity(
        self, registry: EntityRegistry, clear_state: bool = False
    ) -> None:
        """Removes the text entity and its associated points/constraints.

        When *clear_state* is True (undo path), the recorded
        ``_removed_*`` lists are reset first so that the state saved for
        a subsequent redo reflects only this removal.  When False
        (execute path), previously shed constraints are preserved.
        """
        entity = registry.get_entity_object(self.text_entity_id)
        if entity is None:
            return

        if clear_state:
            self._removed_point_dicts = []
            self._removed_entity_dicts = []
            self._removed_constraints = []

        frame_ids = [
            entity.origin_id,
            entity.width_id,
            entity.height_id,
        ]
        p4_id = text_box_fourth_corner(registry, entity)
        if p4_id is not None:
            frame_ids.append(p4_id)

        removed_point_ids = set()
        for pid in frame_ids:
            position = registry.get_point(pid)
            if position is None:
                continue
            self._removed_point_dicts.append(
                {
                    "id": pid,
                    "x": position[0],
                    "y": position[1],
                    "fixed": False,
                }
            )
            removed_point_ids.add(pid)

        self._removed_entity_dicts = [
            registry.entity_to_dict(eid)
            for eid in registry.helper_ids(entity.id)
            if registry.get_entity_object(eid) is not None
        ]
        self._removed_entity_dict = registry.entity_to_dict(entity.id)

        if self.aspect_ratio_constraint_idx is not None:
            constr = self.sketch.constraints[
                self.aspect_ratio_constraint_idx
            ]
            if constr not in self._removed_constraints:
                self._removed_constraints.append(constr)

        for constr in self.sketch.constraints:
            if constr in self._removed_constraints:
                continue
            if not constr.referenced_points().isdisjoint(removed_point_ids):
                self._removed_constraints.append(constr)

        self.sketch.remove_constraints(self._removed_constraints)
        registry.remove_entities(
            [entity.id]
            + [d["id"] for d in self._removed_entity_dicts]
        )
        registry.remove_points(list(removed_point_ids))

        self._entity_was_removed = True

    def _do_undo(self) -> None:
        if self._entity_was_removed:
            self._restore_text_entity()
            return

        registry = self.sketch.registry
        entity = registry.get_entity_object(self.text_entity_id)
        if entity is None or entity.type_name() != "text_box":
            return

        if not self.old_content:
            self._remove_text_entity(registry, clear_state=True)
            return

        self.sketch.flags.set_text_property(
            self.text_entity_id, "content", self.old_content
        )
        if self.old_font_config is not None:
            for key, value in (
                ("font_family", self.old_font_config.font_family),
                ("font_size", self.old_font_config.font_size),
                ("bold", self.old_font_config.bold),
                ("italic", self.old_font_config.italic),
            ):
                self.sketch.flags.set_text_property(
                    self.text_entity_id, key, value
                )

        for pid, (x, y) in self.old_point_positions.items():
            registry.set_point_position(pid, x, y)

        # Restore old aspect ratio
        if (
            self.aspect_ratio_constraint_idx is not None
            and self.old_aspect_ratio is not None
        ) and self.aspect_ratio_constraint_idx < len(self.sketch.constraints):
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            if isinstance(constr, AspectRatioConstraint):
                constr.set_ratio(self.old_aspect_ratio)

        # Remove constraints added by this command
        self.sketch.remove_constraints(self._added_constraints)
        self._added_constraints.clear()

        # Restore removed constraints
        self.sketch.add_constraints(self._removed_constraints)
        self._removed_constraints = []

        # The equal-length constraints shed by the edit were removed
        # wholesale; they come back with the removed-constraint restore.

    def undo(self) -> None:
        """
        Override undo to handle entity removal properly.
        When reverting to empty content, we remove the entity and should
        not restore the snapshot (which would restore the removed points).
        """
        self._do_undo()
        if not self._entity_was_removed:
            self.restore_undo_state()
        self.sketch.notify_update()

    def _restore_text_entity(self) -> None:
        """Restores the text entity and its associated points/constraints.

        Restores are idempotent per id: within a create-text-box
        transaction, the sibling AddItemsCommand may already have
        re-added these items during a redo. Appending unconditionally
        would create duplicate registry entries sharing one ID, which
        corrupts the sketch after a save/reload round-trip."""
        self.sketch.restore_structure(
            self._removed_point_dicts,
            self._removed_entity_dicts
            + (
                [self._removed_entity_dict]
                if self._removed_entity_dict is not None
                else []
            ),
        )

        for constraint in self._removed_constraints:
            if constraint not in self.sketch.constraints:
                self.sketch.add_constraint(constraint)

        self._entity_was_removed = False

    def should_skip_undo(self) -> bool:
        """
        Returns True if the text was empty before and after editing,
        indicating this is a no-op that should not be added to the undo stack.
        """
        return not self.old_content and not self.new_content
