"""
Creates calibration tests in the document and rebuilds them when their
parameters change.

Every test gets its own layer. Each cell is a workpiece with a step
that processes only that workpiece (``Step.workpiece_uids``); machine
commands between cells are Command steps. Creating a test and every
later change of its parameters are one undo entry each.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.context import get_context
from rayforge.core.asset import IAsset
from rayforge.core.color import pick_unused_color
from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.undo import Command, ListItemCommand
from rayforge.core.workpiece import WorkPiece
from rayforge.doceditor.layer_cmd import AddLayerAndSetActiveCommand

from ..calibration import GENERATORS, CalibrationGenerator
from ..calibration.builder import BuiltTest, CalibrationBuilder, bounds_of
from ..calibration.lookup import record_for_items
from ..calibration.record import CalibrationRecord, CalibrationState

if TYPE_CHECKING:
    from rayforge.doceditor.editor import DocEditor

logger = logging.getLogger(__name__)


class CalibrationCmd:
    """Creates and updates calibration tests."""

    def __init__(self, editor: "DocEditor"):
        self._editor = editor

    @property
    def _doc(self):
        return self._editor.doc

    @staticmethod
    def generator(generator_id: str) -> CalibrationGenerator:
        return GENERATORS[generator_id]

    def create(
        self, generator_id: str, params: dict[str, Any] | None = None
    ) -> CalibrationRecord:
        """
        Adds a new test with the generator's defaults, updated with
        *params*, on a new layer. Raises ``ValueError`` when the
        parameters do not make a valid test.
        """
        generator = self.generator(generator_id)
        context = self._editor.context
        if not generator.is_available(context):
            raise ValueError(_("Select a machine first."))
        values = generator.defaults(context)
        values.update(params or {})
        built = self._build(generator, values)
        dims = self._editor.machine_dimensions
        if dims:
            built.center_on(*dims)

        layer = self._new_layer(generator.title)
        for wp in built.workpieces:
            layer.add_child(wp)
        assert layer.workflow is not None
        layer.workflow.set_steps(built.steps)
        record = CalibrationRecord(
            generator_id=generator.id,
            layer_uid=layer.uid,
            state=_state(values, built),
            name=layer.name,
        )

        label = _("Add {test}").format(test=generator.title)
        with self._editor.history_manager.transaction(label) as t:
            for asset in [*built.assets, record]:
                t.execute(self._add_asset_cmd(asset))
            t.execute(AddLayerAndSetActiveCommand(self._editor, layer, label))
        logger.info("Created %s", layer.name)
        return record

    def _new_layer(self, title: str) -> Layer:
        """A layer named after the test, with an unused color."""
        names = {layer.name for layer in self._doc.layers}
        name = title
        number = 2
        while name in names:
            name = f"{title} {number}"
            number += 1
        layer = Layer(name)
        material = get_context().material_mgr.get_default_material()
        layer.set_stock_material_uid(material.uid)
        layer.set_color(
            pick_unused_color({layer.color for layer in self._doc.layers})
        )
        return layer

    @staticmethod
    def record_for_items(items: list[DocItem]) -> CalibrationRecord | None:
        """The record of the test all *items* belong to, or None."""
        return record_for_items(items)

    def update_params(
        self, record: CalibrationRecord, params: dict[str, Any]
    ) -> list[WorkPiece]:
        """
        Rebuilds the test with *params* in one undo entry, at the place
        where it is now. Steps the user added to the layer stay. Raises
        ``ValueError`` when the parameters do not make a valid test.
        """
        if params == self.params(record) and record.uid in self._doc.assets:
            return self.workpieces(record)
        generator = self.generator(record.generator_id)
        layer = self._doc.find_descendant_by_uid(record.layer_uid)
        if not isinstance(layer, Layer) or layer.workflow is None:
            raise ValueError(_("The layer of this test no longer exists."))
        built = self._build(generator, params)
        old_wps = self.workpieces(record)
        self._place(built, old_wps)
        old_steps = self._steps(record)
        changes = _reuse_steps(generator, old_steps, built)
        old = GeneratedItems(
            old_wps,
            old_steps,
            self._unused_assets(record, old_wps),
            record.state,
        )
        new = GeneratedItems(
            built.workpieces,
            built.steps,
            list(built.assets),
            _state(params, built),
        )
        label = _("Change {test}").format(test=generator.title)
        with self._editor.history_manager.transaction(label) as t:
            if record.uid not in self._doc.assets:
                t.execute(self._add_asset_cmd(record))
            t.execute(
                RebuildCalibrationCommand(
                    self._editor, layer, record, (old, new), changes, label
                )
            )
        return built.workpieces

    def params(self, record: CalibrationRecord) -> dict[str, Any]:
        """
        The parameters of the test, including changes made to its steps
        outside the Test Settings, such as an applied recipe.
        """
        generator = self.generator(record.generator_id)
        return generator.params_from_steps(record.params, self._steps(record))

    def _steps(self, record: CalibrationRecord) -> list[Step]:
        """The generated steps that are still in the document."""
        found = (
            self._doc.find_descendant_by_uid(uid)
            for uid in record.state.step_uids
        )
        return [step for step in found if isinstance(step, Step)]

    def workpieces(self, record: CalibrationRecord) -> list[WorkPiece]:
        """The workpieces of the test that are still in the document."""
        found = (
            self._doc.find_descendant_by_uid(uid)
            for uid in record.state.workpiece_uids
        )
        return [wp for wp in found if isinstance(wp, WorkPiece)]

    def _place(self, built: BuiltTest, old_wps: list[WorkPiece]):
        """
        Keeps the bottom-left corner where the old test was, but not
        beyond the work area when the test grew.
        """
        dims = self._editor.machine_dimensions
        if not old_wps:
            if dims:
                built.center_on(*dims)
            return
        min_x, min_y, _max_x, _max_y = bounds_of(old_wps)
        built.move_to(min_x, min_y)
        if dims:
            built.keep_inside(*dims)

    def _unused_assets(
        self, record: CalibrationRecord, removed: list[WorkPiece]
    ) -> list[IAsset]:
        """Assets of the old test that no remaining workpiece uses."""
        removed_uids = {wp.uid for wp in removed}
        in_use = {
            wp.source_segment.source_asset_uid
            for wp in self._doc.all_workpieces
            if wp.uid not in removed_uids and wp.source_segment is not None
        }
        found = (
            self._doc.get_asset_by_uid(uid)
            for uid in record.state.asset_uids
            if uid not in in_use
        )
        return [asset for asset in found if asset is not None]

    def _build(
        self, generator: CalibrationGenerator, params: dict[str, Any]
    ) -> BuiltTest:
        context = self._editor.context
        layout = generator.layout(params, context)
        return CalibrationBuilder(context).build(layout)

    def _add_asset_cmd(self, asset: IAsset) -> ListItemCommand:
        return ListItemCommand(self._doc, asset, "remove_asset", "add_asset")


def _state(params: dict[str, Any], built: BuiltTest) -> CalibrationState:
    return CalibrationState(
        params=dict(params),
        workpiece_uids=tuple(wp.uid for wp in built.workpieces),
        step_uids=tuple(step.uid for step in built.steps),
        asset_uids=tuple(asset.uid for asset in built.assets),
    )


@dataclass
class StepChange:
    """Settings of a kept step before and after a rebuild."""

    step: Step
    before: dict[str, Any]
    after: dict[str, Any]

    def apply(self, values: dict[str, Any]):
        for key, value in values.items():
            setattr(self.step, key, value)


def _reuse_steps(
    generator: CalibrationGenerator, old: list[Step], built: BuiltTest
) -> list[StepChange]:
    """
    For a generator that keeps its steps, puts the old steps in place of
    the new ones and returns the settings to change on them. The steps
    are kept only when they match the new ones one to one.
    """
    if not generator.keeps_steps or len(old) != len(built.steps):
        return []
    if any(type(a) is not type(b) for a, b in zip(old, built.steps)):
        return []
    changes = []
    for index, (kept, fresh) in enumerate(zip(old, built.steps)):
        after = {
            **generator.step_settings(fresh),
            "workpiece_uids": list(fresh.workpiece_uids),
        }
        before = {key: getattr(kept, key) for key in after}
        changes.append(StepChange(kept, before, after))
        built.steps[index] = kept
    return changes


@dataclass
class GeneratedItems:
    """The generated items of a test and the state they belong to."""

    workpieces: list[WorkPiece]
    steps: list[Step]
    assets: list[IAsset]
    state: CalibrationState


class RebuildCalibrationCommand(Command):
    """
    Replaces the generated items of a test in bulk. Each parent, the
    workflow and the asset list change once, so the UI and the pipeline
    update once instead of once per cell, as grouping does it. New steps
    take the place of the first old one; steps the user added stay.
    """

    def __init__(
        self,
        editor: "DocEditor",
        layer: Layer,
        record: CalibrationRecord,
        items: tuple[GeneratedItems, GeneratedItems],
        changes: list[StepChange],
        name: str,
    ):
        super().__init__(name)
        self._editor = editor
        self._layer = layer
        self._record = record
        self._old, self._new = items
        self._changes = changes
        self._parents = {wp.uid: wp.parent for wp in self._old.workpieces}

    def execute(self) -> None:
        self._swap(self._old, self._new, forward=True)

    def undo(self) -> None:
        self._swap(self._new, self._old, forward=False)

    def _swap(self, out: GeneratedItems, into: GeneratedItems, forward: bool):
        doc = self._editor.doc
        with self._editor.pipeline.paused():
            self._record.set_state(into.state)
            for change in self._changes:
                change.apply(change.after if forward else change.before)
            self._swap_workpieces(out.workpieces, into.workpieces)
            self._swap_steps(out.steps, into.steps)
            for asset in into.assets:
                doc.add_asset(asset, silent=True)
            for asset in out.assets:
                doc.remove_asset_by_uid(asset.uid, silent=True)
            for change in self._changes:
                change.step.updated.send(change.step)
        doc.updated.send(doc)

    def _swap_workpieces(self, out: list[WorkPiece], into: list[WorkPiece]):
        by_parent: dict[DocItem, list[WorkPiece]] = defaultdict(list)
        for wp in out:
            if wp.parent is not None:
                by_parent[wp.parent].append(wp)
        for parent, items in by_parent.items():
            parent.remove_children(items)
        by_parent.clear()
        for wp in into:
            by_parent[self._parents.get(wp.uid) or self._layer].append(wp)
        for parent, items in by_parent.items():
            parent.add_children(items)

    def _swap_steps(self, out: list[Step], into: list[Step]):
        workflow = self._layer.workflow
        assert workflow is not None
        current = workflow.children
        positions = [current.index(s) for s in out if s in current]
        index = min(positions) if positions else 0
        workflow.remove_children(out)
        workflow.add_children(into, index)
