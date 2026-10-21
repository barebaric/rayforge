"""
Creates calibration tests in the document and rebuilds them when their
parameters change.

Every test gets its own layer. Each cell is a workpiece with a step
that processes only that workpiece (``Step.workpiece_uids``); machine
commands between cells are Command steps. Creating a test and every
later change of its parameters are one undo entry each.
"""

import logging
from gettext import gettext as _
from typing import TYPE_CHECKING, Any

from rayforge.context import get_context
from rayforge.core.asset import IAsset
from rayforge.core.color import pick_unused_color
from rayforge.core.item import DocItem
from rayforge.core.layer import Layer
from rayforge.core.step import Step
from rayforge.core.undo import ChangePropertyCommand, ListItemCommand
from rayforge.core.undo.list_cmd import ReorderListCommand
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

    # -- Creating ---------------------------------------------------------

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

    # -- Finding ----------------------------------------------------------

    @staticmethod
    def record_for_items(items: list[DocItem]) -> CalibrationRecord | None:
        """The record of the test all *items* belong to, or None."""
        return record_for_items(items)

    # -- Updating ---------------------------------------------------------

    def update_params(
        self, record: CalibrationRecord, params: dict[str, Any]
    ) -> list[WorkPiece]:
        """
        Rebuilds the test with *params* in one undo entry, at the place
        where it is now. Steps the user added to the layer stay. Raises
        ``ValueError`` when the parameters do not make a valid test.
        """
        if params == record.params and record.uid in self._doc.assets:
            return self.workpieces(record)
        generator = self.generator(record.generator_id)
        layer = self._doc.find_descendant_by_uid(record.layer_uid)
        if not isinstance(layer, Layer) or layer.workflow is None:
            raise ValueError(_("The layer of this test no longer exists."))
        built = self._build(generator, params)
        old_wps = self.workpieces(record)
        self._place(built, old_wps)
        steps = _merge_steps(
            layer.workflow.steps, set(record.state.step_uids), built.steps
        )
        label = _("Change {test}").format(test=generator.title)
        with self._editor.history_manager.transaction(label) as t:
            if record.uid not in self._doc.assets:
                t.execute(self._add_asset_cmd(record))
            for wp in old_wps:
                t.execute(self._remove_child_cmd(wp))
            for asset in self._unused_assets(record, old_wps):
                t.execute(self._remove_asset_cmd(asset))
            for asset in built.assets:
                t.execute(self._add_asset_cmd(asset))
            for wp in built.workpieces:
                t.execute(self._add_child_cmd(layer, wp))
            t.execute(
                ReorderListCommand(layer.workflow, "steps", steps, "set_steps")
            )
            t.execute(
                ChangePropertyCommand(
                    record,
                    "state",
                    _state(params, built),
                    setter_method_name="set_state",
                    coalesce=False,
                )
            )
        return built.workpieces

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

    # -- Building blocks --------------------------------------------------

    def _build(
        self, generator: CalibrationGenerator, params: dict[str, Any]
    ) -> BuiltTest:
        context = self._editor.context
        layout = generator.layout(params, context)
        return CalibrationBuilder(context).build(layout)

    def _add_asset_cmd(self, asset: IAsset) -> ListItemCommand:
        return ListItemCommand(self._doc, asset, "remove_asset", "add_asset")

    def _remove_asset_cmd(self, asset: IAsset) -> ListItemCommand:
        return ListItemCommand(self._doc, asset, "add_asset", "remove_asset")

    @staticmethod
    def _add_child_cmd(parent: DocItem, wp: WorkPiece) -> ListItemCommand:
        return ListItemCommand(parent, wp, "remove_child", "add_child")

    @staticmethod
    def _remove_child_cmd(wp: WorkPiece) -> ListItemCommand:
        assert wp.parent is not None
        return ListItemCommand(wp.parent, wp, "add_child", "remove_child")


def _state(params: dict[str, Any], built: BuiltTest) -> CalibrationState:
    return CalibrationState(
        params=dict(params),
        workpiece_uids=tuple(wp.uid for wp in built.workpieces),
        step_uids=tuple(step.uid for step in built.steps),
        asset_uids=tuple(asset.uid for asset in built.assets),
    )


def _merge_steps(
    current: list[Step], generated_uids: set[str], new: list[Step]
) -> list[Step]:
    """
    The workflow after a rebuild: the new generated steps take the
    place of the first old one, and steps the user added keep their
    order around them.
    """
    result: list[Step] = []
    inserted = False
    for step in current:
        if step.uid not in generated_uids:
            result.append(step)
        elif not inserted:
            result.extend(new)
            inserted = True
    if not inserted:
        result = new + result
    return result
