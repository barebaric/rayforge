from __future__ import annotations

import json
import logging
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field, replace
from gettext import gettext as _
from gettext import ngettext
from pathlib import Path
from typing import TYPE_CHECKING, Any

from raygeo.geo import Geometry, Matrix

from ..core.item import DocItem
from ..core.layer import Layer
from ..core.source_asset import SourceAsset
from ..core.source_asset_segment import SourceAssetSegment
from ..core.tab import Tab
from ..core.undo import Command, ListItemCommand
from ..core.vectorization_spec import (
    LayerImportMode,
    LayerSource,
    PassthroughSpec,
    VectorizationSpec,
)
from ..core.workpiece import WorkPiece
from ..image import importer_registry
from ..image.base_importer import Importer
from ..image.structures import ImportResult, ParsingResult

if TYPE_CHECKING:
    from .editor import DocEditor

logger = logging.getLogger(__name__)

TAB_TOLERANCE_MM = 1.0

_SOURCE_DATA_FIELDS = (
    "original_data",
    "base_render_data",
    "thumbnail_data",
    "renderer",
    "metadata",
    "width_px",
    "height_px",
    "width_mm",
    "height_mm",
)


@dataclass
class ReloadReport:
    """Summary of what a reload changed in the document."""

    source_name: str
    updated: int = 0
    added: int = 0
    removed: int = 0
    skipped: int = 0
    tabs_dropped: int = 0
    error: str | None = None

    def describe(self) -> str:
        if self.error:
            return _("Could not reload {name}: {error}").format(
                name=self.source_name, error=self.error
            )
        parts = [
            _("Reloaded {name}").format(name=self.source_name),
        ]
        if self.added:
            parts.append(
                ngettext(
                    "{n} new element added",
                    "{n} new elements added",
                    self.added,
                ).format(n=self.added)
            )
        if self.removed:
            parts.append(
                ngettext(
                    "{n} element no longer in the file was removed",
                    "{n} elements no longer in the file were removed",
                    self.removed,
                ).format(n=self.removed)
            )
        if self.skipped:
            parts.append(
                ngettext(
                    "{n} edited element was left unchanged",
                    "{n} edited elements were left unchanged",
                    self.skipped,
                ).format(n=self.skipped)
            )
        if self.tabs_dropped:
            parts.append(
                ngettext(
                    "{n} tab no longer fits and was removed",
                    "{n} tabs no longer fit and were removed",
                    self.tabs_dropped,
                ).format(n=self.tabs_dropped)
            )
        return "; ".join(parts)


class ReplaceSourceDataCommand(Command):
    """Swaps the imported data of a SourceAsset, keeping its identity."""

    def __init__(
        self,
        asset: SourceAsset,
        new_values: dict[str, Any],
        name: str | None = None,
    ):
        super().__init__(name)
        self.asset = asset
        self.new_values = new_values
        self.old_values = {
            key: getattr(asset, key) for key in _SOURCE_DATA_FIELDS
        }

    def _apply(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            setattr(self.asset, key, value)
        self.asset.clear_base_image_cache()
        self.asset.updated.send(self.asset)

    def execute(self) -> None:
        self._apply(self.new_values)

    def undo(self) -> None:
        self._apply(self.old_values)


@dataclass
class _WorkPieceState:
    segment: SourceAssetSegment | None
    matrix: Matrix
    natural_size: tuple[float, float]
    tabs: list[Tab]


class ReloadWorkPieceCommand(Command):
    """Replaces the source-derived state of a WorkPiece."""

    def __init__(
        self,
        workpiece: WorkPiece,
        new_state: _WorkPieceState,
        name: str | None = None,
    ):
        super().__init__(name)
        self.workpiece = workpiece
        self.new_state = new_state
        self.old_state = _WorkPieceState(
            segment=workpiece.source_segment,
            matrix=workpiece.matrix.copy(),
            natural_size=workpiece.natural_size,
            tabs=list(workpiece.tabs),
        )

    def _apply(self, state: _WorkPieceState) -> None:
        wp = self.workpiece
        wp.natural_width_mm, wp.natural_height_mm = state.natural_size
        wp.tabs = list(state.tabs)
        wp.matrix = state.matrix
        segment_changed = wp.source_segment != state.segment
        wp.source_segment = state.segment
        if not segment_changed:
            wp.clear_render_cache()
            wp.updated.send(wp)

    def execute(self) -> None:
        self._apply(self.new_state)

    def undo(self) -> None:
        self._apply(self.old_state)


@dataclass
class _Fresh:
    """A workpiece produced by running the import pipeline once."""

    workpiece: WorkPiece
    wrapper: Layer | None


@dataclass
class _Plan:
    source_values: dict[str, Any] | None = None
    updates: list[tuple[WorkPiece, _WorkPieceState]] = field(
        default_factory=list
    )
    removals: list[WorkPiece] = field(default_factory=list)
    additions: list[tuple[DocItem, DocItem]] = field(default_factory=list)


def _spec_key(spec: VectorizationSpec) -> str:
    return json.dumps(spec.to_dict(), sort_keys=True, default=str)


def _source_of(result: ImportResult) -> SourceAsset:
    assert result.payload is not None and result.payload.source is not None
    return result.payload.source


def _fresh_items(result: ImportResult) -> dict[str | None, _Fresh]:
    items: dict[str | None, _Fresh] = {}
    if not result.payload:
        return items
    for item in result.payload.items:
        if isinstance(item, Layer):
            for child in item.get_content_items():
                if isinstance(child, WorkPiece) and child.source_segment:
                    items[child.source_segment.layer_id] = _Fresh(child, item)
        elif isinstance(item, WorkPiece) and item.source_segment:
            items[item.source_segment.layer_id] = _Fresh(item, None)
    return items


def _page_bottom_mm(parse_result: ParsingResult) -> float:
    ref = (
        parse_result.untrimmed_document_bounds or parse_result.document_bounds
    )
    return (ref[1] + ref[3]) * parse_result.native_unit_to_mm


def _anchor_shift(old: ImportResult, new: ImportResult) -> Matrix:
    """
    Fresh imports of Y-down formats measure positions from the bottom of
    the page. This shift re-anchors the new import to the top of the old
    page, so content keeps its place when the page height changes.
    """
    old_parse, new_parse = old.parse_result, new.parse_result
    if not old_parse or not new_parse:
        return Matrix.identity()
    if not (old_parse.is_y_down and new_parse.is_y_down):
        return Matrix.identity()
    shift = _page_bottom_mm(old_parse) - _page_bottom_mm(new_parse)
    return Matrix.translation(0.0, shift)


def _local_geometry(boundaries: Geometry, matrix: Matrix) -> Geometry:
    geo = boundaries.copy()
    geo.transform(matrix)
    return geo


class ReloadCmd:
    """Reloads imported sources from their files on disk."""

    def __init__(self, editor: DocEditor):
        self._editor = editor

    def workpieces_for(self, asset: SourceAsset) -> list[WorkPiece]:
        return [
            wp
            for wp in self._editor.doc.all_workpieces
            if wp.source_segment
            and wp.source_segment.source_asset_uid == asset.uid
        ]

    def reload_source(
        self, asset: SourceAsset, data: bytes | None = None
    ) -> ReloadReport:
        """
        Replaces the data of a source with the given bytes (or the current
        file on disk) and updates every workpiece made from it in place,
        as a single undoable step.
        """
        report = ReloadReport(source_name=asset.name)
        try:
            if data is None:
                data = asset.source_file.read_bytes()
            plan = self._plan(asset, data, report)
        except Exception as e:
            logger.exception(f"Reload of {asset.source_file} failed")
            report.error = str(e)
            self._editor.notification_requested.send(
                self, message=report.describe()
            )
            return report

        self._apply(asset, plan)
        self._editor.notification_requested.send(
            self, message=report.describe()
        )
        return report

    def _importer_class(self, asset: SourceAsset) -> type[Importer]:
        name = asset.metadata.get("_importer_class")
        importer_cls = importer_registry.get_by_name(name) if name else None
        if importer_cls is None:
            raise ValueError(_("the file type is not known"))
        return importer_cls

    @staticmethod
    def _run(
        importer_cls: type[Importer],
        data: bytes,
        path: Path,
        spec: VectorizationSpec | None,
    ) -> ImportResult:
        result = importer_cls(data, path).get_doc_items(spec)
        if not result or not result.payload or not result.payload.source:
            errors = result.errors if result else []
            raise ValueError(
                "; ".join(errors) or _("the file could not be read")
            )
        return result

    @staticmethod
    def _layer_ids(
        importer_cls: type[Importer],
        data: bytes,
        path: Path,
        spec: PassthroughSpec,
    ) -> list[str]:
        manifest = importer_cls(data, path).scan()
        if spec.layer_source == LayerSource.COLORS:
            layers = manifest.color_layers
        else:
            layers = manifest.layers
        return [layer.id for layer in layers]

    def _spec_for_new_data(
        self,
        importer_cls: type[Importer],
        asset: SourceAsset,
        data: bytes,
        spec: VectorizationSpec,
    ) -> VectorizationSpec:
        """
        If all layers of the old file were imported, the new file's layers
        are all imported too. A chosen subset stays as it is.
        """
        if not isinstance(spec, PassthroughSpec) or not spec.active_layer_ids:
            return spec
        path = asset.source_file
        old_ids = self._layer_ids(
            importer_cls, asset.original_data, path, spec
        )
        if not set(old_ids) <= set(spec.active_layer_ids):
            return spec
        new_ids = self._layer_ids(importer_cls, data, path, spec)
        return replace(spec, active_layer_ids=new_ids)

    def _plan(
        self, asset: SourceAsset, data: bytes, report: ReloadReport
    ) -> _Plan:
        importer_cls = self._importer_class(asset)
        plan = _Plan()
        groups: dict[str, list[WorkPiece]] = defaultdict(list)
        for wp in self.workpieces_for(asset):
            if wp.has_edited_boundaries:
                report.skipped += 1
            elif wp.source_segment:
                key = _spec_key(wp.source_segment.vectorization_spec)
                groups[key].append(wp)

        new_source = None
        for workpieces in groups.values():
            result = self._plan_group(
                importer_cls, asset, data, workpieces, plan, report
            )
            new_source = new_source or _source_of(result)

        if new_source is None:
            result = self._run(importer_cls, data, asset.source_file, None)
            new_source = _source_of(result)

        metadata = dict(new_source.metadata)
        for key in ("_importer_class", "_importer_mime"):
            if key in asset.metadata:
                metadata.setdefault(key, asset.metadata[key])
        plan.source_values = {
            key: getattr(new_source, key) for key in _SOURCE_DATA_FIELDS
        }
        plan.source_values["metadata"] = metadata
        return plan

    def _plan_group(
        self,
        importer_cls: type[Importer],
        asset: SourceAsset,
        data: bytes,
        workpieces: list[WorkPiece],
        plan: _Plan,
        report: ReloadReport,
    ) -> ImportResult:
        segment = workpieces[0].source_segment
        assert segment is not None
        spec = segment.vectorization_spec
        path = asset.source_file
        new_spec = self._spec_for_new_data(importer_cls, asset, data, spec)
        old_result = self._run(importer_cls, asset.original_data, path, spec)
        new_result = self._run(importer_cls, data, path, new_spec)
        if not _fresh_items(new_result):
            raise ValueError(_("the file contains nothing to import"))

        old_items = _fresh_items(old_result)
        new_items = _fresh_items(new_result)
        anchor = _anchor_shift(old_result, new_result)

        reference: tuple[WorkPiece, Matrix] | None = None
        for wp in workpieces:
            assert wp.source_segment is not None
            layer_id = wp.source_segment.layer_id
            old = old_items.get(layer_id)
            if old is None:
                report.skipped += 1
                continue
            delta = wp.matrix @ old.workpiece.matrix.invert()
            new = new_items.get(layer_id)
            if new is None:
                plan.removals.append(wp)
                report.removed += 1
                continue
            reference = reference or (wp, delta)
            state = self._new_state(
                wp, new.workpiece, delta @ anchor, asset, new_spec, report
            )
            plan.updates.append((wp, state))
            report.updated += 1

        if reference is not None:
            for layer_id, new in new_items.items():
                if layer_id in old_items:
                    continue
                self._plan_addition(
                    new, reference, anchor, asset, new_spec, plan
                )
                report.added += 1
        return new_result

    @staticmethod
    def _adopt_segment(
        fresh: WorkPiece,
        asset: SourceAsset,
        spec: VectorizationSpec,
        previous: SourceAssetSegment | None,
    ) -> SourceAssetSegment:
        assert fresh.source_segment is not None
        modifiers = previous.image_modifier_chain if previous else []
        return replace(
            fresh.source_segment,
            source_asset_uid=asset.uid,
            vectorization_spec=spec,
            image_modifier_chain=deepcopy(modifiers),
        )

    def _new_state(
        self,
        wp: WorkPiece,
        fresh: WorkPiece,
        delta: Matrix,
        asset: SourceAsset,
        spec: VectorizationSpec,
        report: ReloadReport,
    ) -> _WorkPieceState:
        matrix = delta @ fresh.matrix
        segment = self._adopt_segment(fresh, asset, spec, wp.source_segment)
        tabs, dropped = self._remap_tabs(wp, fresh, matrix)
        report.tabs_dropped += dropped
        return _WorkPieceState(
            segment=segment,
            matrix=matrix,
            natural_size=fresh.natural_size,
            tabs=tabs,
        )

    @staticmethod
    def _remap_tabs(
        wp: WorkPiece, fresh: WorkPiece, matrix: Matrix
    ) -> tuple[list[Tab], int]:
        """
        Moves each tab to the closest point of the new outline, measured
        in the workpiece's parent space. Tabs whose outline moved away are
        dropped.
        """
        if not wp.tabs:
            return [], 0
        old_boundaries = wp.boundaries
        new_boundaries = fresh.boundaries
        if old_boundaries is None or new_boundaries is None:
            return [], len(wp.tabs)
        old_geo = _local_geometry(old_boundaries, wp.matrix)
        new_geo = _local_geometry(new_boundaries, matrix)
        kept = []
        for tab in wp.tabs:
            point = old_geo.get_point_at(tab.segment_index, tab.pos)
            if point is None:
                continue
            closest = new_geo.find_closest_point(point[0], point[1])
            if closest is None:
                continue
            index, t, new_point = closest
            distance = (
                (new_point[0] - point[0]) ** 2 + (new_point[1] - point[1]) ** 2
            ) ** 0.5
            if distance <= TAB_TOLERANCE_MM:
                kept.append(
                    replace(
                        tab, segment_index=index, pos=min(1.0, max(0.0, t))
                    )
                )
        return kept, len(wp.tabs) - len(kept)

    def _plan_addition(
        self,
        new: _Fresh,
        reference: tuple[WorkPiece, Matrix],
        anchor: Matrix,
        asset: SourceAsset,
        spec: VectorizationSpec,
        plan: _Plan,
    ) -> None:
        ref_wp, delta = reference
        wp = new.workpiece
        wp.matrix = delta @ anchor @ wp.matrix
        wp.source_segment = self._adopt_segment(wp, asset, spec, None)
        mode = LayerImportMode.NEW_LAYERS
        if isinstance(spec, PassthroughSpec):
            mode = spec.layer_import_mode
        if new.wrapper is not None and mode == LayerImportMode.NEW_LAYERS:
            plan.additions.append((self._editor.doc, new.wrapper))
        elif ref_wp.parent is not None:
            if new.wrapper is not None:
                new.wrapper.remove_child(wp)
            plan.additions.append((ref_wp.parent, wp))

    def _apply(self, asset: SourceAsset, plan: _Plan) -> None:
        name = _("Reload {name}").format(name=asset.name)
        assert plan.source_values is not None
        with self._editor.history_manager.transaction(name) as t:
            t.execute(ReplaceSourceDataCommand(asset, plan.source_values))
            for wp, state in plan.updates:
                t.execute(ReloadWorkPieceCommand(wp, state))
            for wp in plan.removals:
                t.execute(
                    ListItemCommand(
                        owner_obj=wp.parent,
                        item=wp,
                        undo_command="add_child",
                        redo_command="remove_child",
                    )
                )
            for owner, item in plan.additions:
                t.execute(
                    ListItemCommand(
                        owner_obj=owner,
                        item=item,
                        undo_command="remove_child",
                        redo_command="add_child",
                    )
                )
        new_layers = [
            item for _owner, item in plan.additions if isinstance(item, Layer)
        ]
        if new_layers:
            self._editor.step.add_default_steps_for_layers(new_layers)
