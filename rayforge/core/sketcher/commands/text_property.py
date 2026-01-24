from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Tuple, Optional, List
from ....core.geo.geometry import Geometry
from ....core.geo.font_config import FontConfig
from ..constraints import (
    AspectRatioConstraint,
    EqualLengthConstraint,
    DistanceConstraint,
)
from ..entities.point import Point
from ..entities.line import Line
from ..entities.text_box import TextBoxEntity
from .base import SketchChangeCommand

if TYPE_CHECKING:
    from ..constraints import Constraint
    from ..sketch import Sketch


class ModifyTextPropertyCommand(SketchChangeCommand):
    def __init__(
        self,
        sketch: Sketch,
        text_entity_id: int,
        new_content: str,
        new_font_config: FontConfig,
    ):
        super().__init__(sketch, _("Modify Text Property"))
        self.text_entity_id = text_entity_id
        self.new_content = new_content
        self.new_font_config = new_font_config
        self.old_content = ""
        self.old_font_config: Optional[FontConfig] = None
        self.old_point_positions: Dict[int, Tuple[float, float]] = {}
        self.old_aspect_ratio: Optional[float] = None
        self.aspect_ratio_constraint_idx: Optional[int] = None
        self._added_constraints: List[Constraint] = []

        self._entity_was_removed = False
        self._removed_entity: Optional[TextBoxEntity] = None
        self._removed_points: List[Point] = []
        self._removed_entities: List[Any] = []
        self._removed_constraints: List[Constraint] = []
        self._modified_equal_length_constraints: List[
            Tuple[int, List[int]]
        ] = []

    def _calculate_natural_metrics(
        self, content: str, font_config: FontConfig
    ) -> Tuple[float, float]:
        """Calculates the natural width and height from font properties."""
        _, _, font_height = font_config.get_font_metrics()

        if not content:
            natural_width = 10.0
        else:
            natural_geo = Geometry.from_text(content, font_config)
            natural_geo.flip_y()
            min_x, _, max_x, _ = natural_geo.rect()
            natural_width = max(max_x - min_x, 1.0)

        return natural_width, font_height

    def _shed_size_constraints(self, text_entity: TextBoxEntity) -> None:
        """Removes constraints that lock the text box dimensions."""
        for eid in text_entity.construction_line_ids:
            entity = self.sketch.registry.get_entity(eid)
            if not isinstance(entity, Line):
                continue

            for idx, constr in enumerate(self.sketch.constraints):
                if constr in self._removed_constraints:
                    continue

                if isinstance(constr, EqualLengthConstraint):
                    if eid in constr.entity_ids:
                        self._modified_equal_length_constraints.append(
                            (idx, list(constr.entity_ids))
                        )
                        constr.entity_ids.remove(eid)
                elif constr.targets_segment(entity.p1_idx, entity.p2_idx, eid):
                    self._removed_constraints.append(constr)

        for c in self._removed_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)

    def _do_execute(self) -> None:
        entity = self.sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        text_entity = entity

        if not self.old_content and not self.old_point_positions:
            self.old_content = text_entity.content
            self.old_font_config = text_entity.font_config.copy()
            p_width = self.sketch.registry.get_point(text_entity.width_id)
            p_height = self.sketch.registry.get_point(text_entity.height_id)
            self.old_point_positions = {
                text_entity.width_id: (p_width.x, p_width.y),
                text_entity.height_id: (p_height.x, p_height.y),
            }

            # Find and store the old aspect ratio constraint
            for idx, constr in enumerate(self.sketch.constraints or []):
                if isinstance(constr, AspectRatioConstraint):
                    if (
                        constr.p1 == text_entity.origin_id
                        and constr.p2 == text_entity.width_id
                        and constr.p3 == text_entity.origin_id
                        and constr.p4 == text_entity.height_id
                    ):
                        self.aspect_ratio_constraint_idx = idx
                        self.old_aspect_ratio = constr.ratio
                        break

        text_entity.content = self.new_content
        text_entity.font_config = self.new_font_config.copy()

        # If the content is empty after editing, remove the entity
        if not self.new_content:
            self._remove_text_entity(text_entity)
            return

        self._added_constraints.clear()

        # --- Iteration 2: "Content as Truth" Reset ---
        natural_width, natural_height = self._calculate_natural_metrics(
            self.new_content, self.new_font_config
        )

        # --- Iteration 3: Shed size constraints ---
        self._shed_size_constraints(text_entity)

        # Create and add new implicit distance constraints to enforce size
        width_constr = DistanceConstraint(
            text_entity.origin_id,
            text_entity.width_id,
            natural_width,
            user_visible=False,
        )
        height_constr = DistanceConstraint(
            text_entity.origin_id,
            text_entity.height_id,
            natural_height,
            user_visible=False,
        )
        self._added_constraints.extend([width_constr, height_constr])
        self.sketch.constraints.extend(self._added_constraints)

        # Update aspect ratio constraint with new text dimensions
        # The constraint must exist for a text box.
        if (
            self.aspect_ratio_constraint_idx is not None
            and natural_height > 1e-9
        ):
            new_ratio = natural_width / natural_height
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            assert isinstance(constr, AspectRatioConstraint)
            constr.ratio = new_ratio

    def _remove_text_entity(self, text_entity: TextBoxEntity) -> None:
        """Removes the text entity and its associated points/constraints."""
        registry = self.sketch.registry

        self._removed_entity = text_entity

        p_origin = registry.get_point(text_entity.origin_id)
        p_width = registry.get_point(text_entity.width_id)
        p_height = registry.get_point(text_entity.height_id)

        self._removed_points = [p_origin, p_width, p_height]

        p4_id = text_entity.get_fourth_corner_id(registry)
        if p4_id:
            p4 = registry.get_point(p4_id)
            self._removed_points.append(p4)

        for eid in text_entity.construction_line_ids:
            e = registry.get_entity(eid)
            if e:
                self._removed_entities.append(e)

        if self.aspect_ratio_constraint_idx is not None:
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            self._removed_constraints.append(constr)

        point_ids = {pt.id for pt in self._removed_points}

        for constr in self.sketch.constraints:
            if constr not in self._removed_constraints:
                if constr.depends_on_points(point_ids):
                    self._removed_constraints.append(constr)

        registry.entities = [
            e for e in registry.entities if e.id != text_entity.id
        ]
        registry._entity_map = {e.id: e for e in registry.entities}

        registry.points = [p for p in registry.points if p.id not in point_ids]

        for e in self._removed_entities:
            registry.entities = [
                ent for ent in registry.entities if ent.id != e.id
            ]
        registry._entity_map = {e.id: e for e in registry.entities}

        for c in self._removed_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)

        self._entity_was_removed = True

    def _do_undo(self) -> None:
        if self._entity_was_removed:
            self._restore_text_entity()
            return

        entity = self.sketch.registry.get_entity(self.text_entity_id)
        if not isinstance(entity, TextBoxEntity):
            return

        text_entity = entity

        text_entity.content = self.old_content
        if self.old_font_config is not None:
            text_entity.font_config = self.old_font_config.copy()

        for pid, (x, y) in self.old_point_positions.items():
            p = self.sketch.registry.get_point(pid)
            p.x = x
            p.y = y

        # Restore old aspect ratio
        if (
            self.aspect_ratio_constraint_idx is not None
            and self.old_aspect_ratio is not None
        ):
            constr = self.sketch.constraints[self.aspect_ratio_constraint_idx]
            assert isinstance(constr, AspectRatioConstraint)
            constr.ratio = self.old_aspect_ratio

        # Remove constraints added by this command
        for c in self._added_constraints:
            if c in self.sketch.constraints:
                self.sketch.constraints.remove(c)
        self._added_constraints.clear()

        # Restore removed constraints
        for c in self._removed_constraints:
            self.sketch.constraints.append(c)

        # Restore modified EqualLength constraints
        for idx, old_entity_ids in reversed(
            self._modified_equal_length_constraints
        ):
            if idx < len(self.sketch.constraints):
                constr = self.sketch.constraints[idx]
                if isinstance(constr, EqualLengthConstraint):
                    constr.entity_ids = old_entity_ids

    def _restore_text_entity(self) -> None:
        """Restores the text entity and its associated points/constraints."""
        registry = self.sketch.registry

        for p in self._removed_points:
            registry.points.append(p)

        for e in self._removed_entities:
            registry.entities.append(e)

        if self._removed_entity:
            registry.entities.append(self._removed_entity)

        registry._entity_map = {e.id: e for e in registry.entities}

        for c in self._removed_constraints:
            self.sketch.constraints.append(c)

        self._entity_was_removed = False

    def should_skip_undo(self) -> bool:
        """
        Returns True if the text was empty before and after editing,
        indicating this is a no-op that should not be added to the undo stack.
        """
        return not self.old_content and not self.new_content
