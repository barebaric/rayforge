"""
Defines the Layer class: one machine configuration within a document.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Iterable
from gettext import gettext as _
from typing import (
    TYPE_CHECKING,
    Any,
    TypeVar,
)

from blinker import Signal
from raygeo.geo import Matrix

from ..context import get_context
from .group import Group
from .item import DocItem, VisibleMixin
from .step import Step
from .stock import StockItem
from .workflow import Workflow
from .workpiece import WorkPiece

if TYPE_CHECKING:
    from .material import Material

logger = logging.getLogger(__name__)

# For generic type hinting in add_child
T = TypeVar("T", bound="DocItem")


class Layer(DocItem, VisibleMixin):
    """
    Represents one machine configuration: the workpieces loaded for it,
    the stock they are cut from, and the ordered workflows that run.

    A Layer owns the job settings of its physical configuration (WCS,
    rotary attachment). A document holds one or more Layers in run
    order; the canvas and the sidebar follow the active layer. It is a
    `DocItem` and automatically manages its children and bubbles up
    signals.
    """

    def __init__(self, name: str):
        """Initializes a Layer instance.

        Args:
            name: The user-facing name of the layer.
        """
        super().__init__(name=name)
        self.rotary_enabled: bool = False
        self.rotary_diameter: float = 25.0
        self.rotary_module_uid: str | None = None
        self.stock_material_uid: str | None = None
        self.wcs: str | None = None

        # Signals for notifying other parts of the application of changes.
        # This one is special and is bubbled manually.
        self.per_step_transformer_changed = Signal()

        # Forward compatibility: store unknown attributes
        self.extra: dict[str, Any] = {}

        # A new layer gets one workflow automatically.
        workflow = Workflow(_("{name} Workflow").format(name=name))
        self.add_child(workflow)

    def to_dict(self) -> dict:
        """Serializes the layer and its children to a dictionary."""
        result = {
            "uid": self.uid,
            "type": "layer",
            "name": self.name,
            "matrix": self.matrix.to_list(),
            "visible": self.visible,
            "rotary_enabled": self.rotary_enabled,
            "rotary_diameter": self.rotary_diameter,
            "rotary_module_uid": self.rotary_module_uid,
            "stock_material_uid": self.stock_material_uid,
            "wcs": self.wcs,
            "children": [child.to_dict() for child in self.children],
        }
        result.update(self.extra)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Layer:
        """Deserializes a dictionary into a Layer instance."""
        known_keys = {
            "uid",
            "type",
            "name",
            "matrix",
            "visible",
            "rotary_enabled",
            "rotary_diameter",
            "rotary_module_uid",
            "stock_material_uid",
            "wcs",
            "children",
        }
        extra = {k: v for k, v in data.items() if k not in known_keys}

        layer = cls(name=data.get("name", "Layer"))
        layer.uid = data["uid"]
        layer.matrix = Matrix.from_list(data["matrix"])
        layer.visible = data.get("visible", True)
        layer.rotary_enabled = data.get("rotary_enabled", False)
        layer.rotary_diameter = data.get("rotary_diameter", 25.0)
        layer.rotary_module_uid = data.get("rotary_module_uid")
        layer.stock_material_uid = data.get("stock_material_uid")
        layer.wcs = data.get("wcs")
        layer.extra = extra

        children: list[DocItem] = []
        for child_data in data.get("children", []):
            child_type = child_data.get("type")
            if child_type == "workflow":
                children.append(Workflow.from_dict(child_data))
            elif child_type == "workpiece":
                children.append(WorkPiece.from_dict(child_data))
            elif child_type == "group":
                children.append(Group.from_dict(child_data))
            elif child_type == "stockitem":
                from .stock import StockItem

                children.append(StockItem.from_dict(child_data))
        if not any(isinstance(child, Workflow) for child in children):
            # The workflow created by __init__ keeps the layer runnable.
            auto_workflow = layer.workflows[0]
            children.append(auto_workflow)
        layer.set_children(children)

        return layer

    @property
    def stock_items(self) -> list[StockItem]:
        """Returns the layer's StockItem children."""
        from .stock import StockItem

        return [
            child for child in self.children if isinstance(child, StockItem)
        ]

    @property
    def workpieces(self) -> list[WorkPiece]:
        """
        Returns a list of all child items that are WorkPieces.
        Note: This only returns direct children.
        """
        return [
            child for child in self.children if isinstance(child, WorkPiece)
        ]

    @property
    def all_workpieces(self) -> list[WorkPiece]:
        """
        Recursively finds and returns a flattened list of all WorkPiece
        objects contained within this layer, including those inside
        groups.
        """
        return self.get_descendants(of_type=WorkPiece)

    @property
    def has_fills(self) -> bool:
        """Check if any workpiece in this layer has fill geometry."""
        for wp in self.all_workpieces:
            fills = wp.fills
            if fills is not None and len(fills) > 0:
                return True
        return False

    @property
    def workflows(self) -> list[Workflow]:
        """Returns the layer's workflows in run order."""
        return [
            child for child in self.children if isinstance(child, Workflow)
        ]

    @property
    def steps(self) -> list[Step]:
        """All steps of the layer's workflows, in run order."""
        return [step for workflow in self.workflows for step in workflow.steps]

    def get_content_items(self) -> list[DocItem]:
        """
        Returns a list of user-facing items in this layer (e.g.,
        WorkPieces, Groups), excluding internal objects like Workflows
        and StockItems.
        """
        from .stock import StockItem

        return [
            child
            for child in self.children
            if not isinstance(child, (Workflow, StockItem))
        ]

    @property
    def content_items(self) -> list[DocItem]:
        """Property alias for get_content_items()."""
        return self.get_content_items()

    @property
    def is_empty(self) -> bool:
        """Returns True if the layer holds no user-facing items."""
        return not self.get_content_items()

    def _on_workflow_post_transformer_changed(self, sender):
        """
        Bubbles up the post_transformer_changed signal from a workflow.
        """
        self.per_step_transformer_changed.send(self)

    def add_child(self, child: T, index: int | None = None) -> T:
        if isinstance(child, Workflow):
            child.per_step_transformer_changed.connect(
                self._on_workflow_post_transformer_changed
            )
        super().add_child(child, index)
        return child

    def remove_child(self, child: DocItem):
        if isinstance(child, Workflow):
            child.per_step_transformer_changed.disconnect(
                self._on_workflow_post_transformer_changed
            )
        super().remove_child(child)

    def set_children(self, new_children: Iterable[DocItem]):
        # Disconnect any existing workflow signal handlers
        for child in self.children:
            if isinstance(child, Workflow):
                child.per_step_transformer_changed.disconnect(
                    self._on_workflow_post_transformer_changed
                )

        # Connect to the new ones
        for child in new_children:
            if isinstance(child, Workflow):
                child.per_step_transformer_changed.connect(
                    self._on_workflow_post_transformer_changed
                )

        super().set_children(new_children)

    @property
    def active(self) -> bool:
        """
        Returns True if this layer is the currently active layer in the
        document.
        """
        return self.doc.active_layer is self if self.doc else False

    def set_name(self, name: str):
        """Sets the name of the layer.

        Not a content change, so ``updated`` stays untouched.
        """
        if self.name == name:
            return
        self.name = name

    def set_rotary_enabled(self, enabled: bool):
        """Enable or disable rotary attachment mode."""
        if self.rotary_enabled == enabled:
            return
        self.rotary_enabled = enabled
        self.updated.send(self)

    def set_rotary_diameter(self, diameter: float):
        """Set the diameter of the object on the rotary attachment."""
        if self.rotary_diameter == diameter:
            return
        self.rotary_diameter = diameter
        self.updated.send(self)

    def set_rotary_module_uid(self, uid: str | None):
        if self.rotary_module_uid == uid:
            return
        self.rotary_module_uid = uid
        self.updated.send(self)

    def set_stock_material_uid(self, uid: str | None):
        """Set the stock material rendered for this rotary layer."""
        if self.stock_material_uid == uid:
            return
        self.stock_material_uid = uid
        self.updated.send(self)

    @property
    def stock_material(self) -> Material | None:
        """The Material selected as this layer's rotary stock, if any."""
        if not self.stock_material_uid:
            return None
        return get_context().material_mgr.get_material_or_none(
            self.stock_material_uid
        )

    def get_subtitle(self, rotary_module_name: str | None = None) -> str:
        """Returns a subtitle describing the layer type.

        Args:
            rotary_module_name: The name of the selected rotary module,
                if any. The caller is responsible for resolving the
                layer's ``rotary_module_uid`` to a name.

        Returns:
            A human-readable subtitle string (e.g. ``"Flat"`` or
            ``"Rotary · Chuck Rotary"``).
        """
        if not self.rotary_enabled:
            return _("Flat")
        if rotary_module_name:
            return _("Rotary · {name}").format(name=rotary_module_name)
        return _("Rotary")

    def set_wcs(self, wcs: str | None):
        if self.wcs == wcs:
            return
        self.wcs = wcs
        self.updated.send(self)

    def get_effective_wcs(self, machine) -> str:
        """Return this layer's WCS or the machine's active WCS."""
        return self.wcs if self.wcs else machine.active_wcs

    def mm_to_degrees(self, mm: float) -> float:
        """Convert surface mm to degrees for rotary axis."""
        if self.rotary_diameter <= 0:
            return 0.0
        circumference = self.rotary_diameter * math.pi
        return (mm / circumference) * 360.0

    def add_workpiece(self, workpiece: WorkPiece):
        """Adds a single workpiece to the layer."""
        self.add_child(workpiece)

    def remove_workpiece(self, workpiece: WorkPiece):
        """Removes a single workpiece from the layer."""
        self.remove_child(workpiece)

    def set_workpieces(self, workpieces: list[WorkPiece]):
        """
        Sets the layer's workpieces to a new list, preserving the
        existing workflows, groups and stock.
        """
        others = [c for c in self.children if not isinstance(c, WorkPiece)]
        new_children: list[DocItem] = list(workpieces)
        new_children.extend(others)
        self.set_children(new_children)

    def reorder_content_items(self, new_content_order: list[DocItem]):
        """
        Reorders all content items (WorkPieces and Groups) while
        preserving the positions of workflows and stock.

        A shorter list drops the content items it does not mention.
        """
        item_iter = iter(new_content_order)
        new_children = []
        for child in self.children:
            if isinstance(child, (Workflow, StockItem)):
                new_children.append(child)
                continue
            item = next(item_iter, None)
            if item is not None:
                new_children.append(item)
        self.set_children(new_children)

    def reorder_workflows(self, new_workflow_order: list[Workflow]):
        """
        Reorders workflows while preserving the positions of the other
        children. A shorter list drops the workflows it does not
        mention.
        """
        wf_iter = iter(new_workflow_order)
        new_children = []
        for child in self.children:
            if isinstance(child, Workflow):
                workflow = next(wf_iter, None)
                if workflow is not None:
                    new_children.append(workflow)
            else:
                new_children.append(child)
        self.set_children(new_children)

    def get_renderable_items(self) -> list[tuple[Step, WorkPiece]]:
        """
        Gets a list of all visible step/workpiece pairs for rendering.

        Returns:
            A list of (Step, WorkPiece) tuples that are currently
            visible and have valid geometry for rendering.
        """
        if not self.visible:
            return []
        items = []
        for workpiece in self.all_workpieces:
            if any(s <= 0 for s in workpiece.size):
                continue
            for workflow in self.workflows:
                for step in workflow.steps:
                    if step.visible:
                        items.append((step, workpiece))
        return items
