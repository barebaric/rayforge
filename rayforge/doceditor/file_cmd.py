import asyncio
import json
import logging
import mimetypes
import warnings
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from gettext import gettext as _
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Optional,
    cast,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    import pyvips

from blinker import Signal
from raygeo.geo import Geometry, Matrix
from raygeo.geo.types import Point, Rect
from raygeo.ops.state import CoolantMode

from ..context import get_context
from ..core.doc import Doc
from ..core.doc_schema import CURRENT_DOC_SCHEMA_VERSION
from ..core.item import DocItem
from ..core.layer import Layer
from ..core.source_asset import SourceAsset
from ..core.undo import ListItemCommand
from ..core.vectorization_spec import (
    LayerImportMode,
    PassthroughSpec,
    TraceSpec,
    VectorizationSpec,
)
from ..core.workflow import Workflow
from ..core.workpiece import WorkPiece
from ..image import (
    Importer,
    ImporterFeature,
    ImportManifest,
    exporter_registry,
    importer_registry,
)
from ..image.assembler import ItemAssembler
from ..image.base_exporter import Exporter
from ..image.coreldraw.exporter import (
    CorelDrawExporter,
    MultiGeometryCorelDrawExporter,
)
from ..image.dxf.exporter import (
    GeometryDxfExporter,
    MultiGeometryDxfExporter,
)
from ..image.structures import (
    ImportPayload,
    ImportResult,
    ParsingResult,
    SourceLayerInfo,
)
from ..image.svg.exporter import (
    GeometrySvgExporter,
    MultiGeometrySvgExporter,
)
from ..pipeline.artifact import JobArtifact
from ..pipeline.artifact.handle import BaseArtifactHandle
from .layout.align import PositionAtStrategy

if TYPE_CHECKING:
    from ..core.asset import IAsset
    from ..doceditor.editor import DocEditor
    from ..machine.models.machine import Machine
    from ..shared.tasker.manager import TaskManager


logger = logging.getLogger(__name__)


_COOLANT_MODE_LABELS = {
    CoolantMode.FLOOD: _("Flood"),
    CoolantMode.MIST: _("Mist"),
}


def _unsupported_coolant_labels(
    doc: "Doc", machine: Optional["Machine"]
) -> list[str]:
    """Human-readable labels of coolant methods used by the doc's steps
    that the current machine does not support."""
    if machine is None:
        return []
    unsupported: set[CoolantMode] = set()
    for step in doc.steps:
        unsupported.update(step.get_unsupported_coolant_methods(machine))
    ordered = sorted(unsupported, key=lambda m: m.value)
    return [_COOLANT_MODE_LABELS[m] for m in ordered]


@dataclass
class PreviewResult:
    """
    Result of a preview generation operation.
    Contains the rendered image bytes, the document items to display, and the
    parsing context needed for correct rendering.
    """

    image_bytes: bytes
    payload: ImportPayload | None
    parse_result: ParsingResult | None  # Context for rendering
    aspect_ratio: float = 1.0
    warnings: list[str] = field(default_factory=list)
    content_bounds: Rect | None = None


class ImportAction(Enum):
    """Determines the workflow required to import a specific file."""

    DIRECT_LOAD = auto()
    INTERACTIVE_CONFIG = auto()
    UNSUPPORTED = auto()


class FileCmd:
    """Handles file import and export operations."""

    def __init__(
        self,
        editor: "DocEditor",
        task_manager: "TaskManager",
    ):
        self._editor = editor
        self._task_manager = task_manager
        self.gcode_exported = Signal()

    def get_importer_info(
        self, file_path: Path, mime_type: str | None
    ) -> tuple[type[Importer] | None, set[ImporterFeature]]:
        """
        Finds the importer for a file and returns its class and feature set.

        A generic MIME type carries no format information, so it never
        shadows a match by file extension.
        """
        if not mime_type:
            mime_type, _ = mimetypes.guess_type(file_path)

        importer_cls = None
        if mime_type and mime_type != "application/octet-stream":
            importer_cls = importer_registry.get_by_mime_type(mime_type)

        if not importer_cls and file_path.suffix:
            importer_cls = importer_registry.get_by_extension(
                file_path.suffix.lower()
            )

        if importer_cls:
            return importer_cls, importer_cls.features
        return None, set()

    def analyze_import_target(
        self, file_path: Path, mime_type: str | None = None
    ) -> ImportAction:
        """
        Analyzes a file path (and optional mime type) to determine how it
        should be imported.
        """
        importer_cls, features = self.get_importer_info(file_path, mime_type)

        if not importer_cls:
            return ImportAction.UNSUPPORTED

        # Any format that can be traced OR has selectable layers needs an
        # interactive dialog.
        if (
            ImporterFeature.BITMAP_TRACING in features
            or ImporterFeature.LAYER_SELECTION in features
        ):
            return ImportAction.INTERACTIVE_CONFIG

        return ImportAction.DIRECT_LOAD

    def scan_import_file(
        self, file_bytes: bytes, file_path: Path, mime_type: str
    ) -> ImportManifest:
        """
        Lightweight scan of a file to extract metadata without full processing.
        """
        importer_cls, _ = self.get_importer_info(file_path, mime_type)

        if not importer_cls:
            logger.warning(
                f"No importer found for mime type '{mime_type}' or "
                f"extension '{file_path.suffix}' during scan."
            )
            return ImportManifest(
                title=file_path.name,
                warnings=[f"Unsupported file type: {file_path.suffix}"],
            )

        try:
            importer_instance = importer_cls(
                data=file_bytes, source_file=file_path
            )
            manifest = importer_instance.scan()
            return manifest
        except Exception:
            logger.exception(
                f"Error scanning file {file_path.name} with "
                f"{importer_cls.__name__}"
            )
            return ImportManifest(
                title=file_path.name,
                warnings=[
                    "An unexpected error occurred during file analysis."
                ],
            )

    async def generate_preview(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> PreviewResult | None:
        """
        Generates a preview image and vector payload for the import dialog.
        Runs the heavy image processing in a background thread.
        """
        return await asyncio.to_thread(
            self._generate_preview_impl,
            file_bytes,
            filename,
            mime_type,
            spec,
            preview_size_px,
        )

    def _generate_preview_impl(
        self,
        file_bytes: bytes,
        filename: str,
        mime_type: str,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> PreviewResult | None:
        """Blocking implementation of preview generation."""
        importer_cls, _ = self.get_importer_info(Path(filename), mime_type)
        if not importer_cls:
            return None

        try:
            importer = importer_cls(
                data=file_bytes, source_file=Path(filename)
            )
            import_result = importer.get_doc_items(spec)

            if not import_result:
                return None

            # Even if no items were created, we might still be able to show a
            # preview of the source asset (e.g., an empty DXF).
            if not import_result.payload or not import_result.payload.items:
                logger.warning(
                    f"Import of '{filename}' produced no document items, "
                    "but attempting to generate a preview."
                )

            return self._generate_rich_preview_result(
                import_result, file_bytes, spec, preview_size_px
            )

        except Exception:
            logger.exception("Failed to generate import preview")
            return None

    def _generate_rich_preview_result(
        self,
        import_result: ImportResult,
        original_file_bytes: bytes,
        spec: VectorizationSpec,
        preview_size_px: int,
    ) -> PreviewResult | None:
        """
        Generates the final PreviewResult from a rich ImportResult.
        This is the new central logic for creating preview bitmaps.
        """
        payload = import_result.payload
        parse_result = import_result.parse_result

        if not payload or not parse_result:
            return None

        renderer = payload.source.renderer
        if not renderer:
            return None

        # 1. Generate high-res base image for the background by delegating
        # to the source's specialized renderer.
        vips_image = None
        content_bounds = None
        target_dim = 2048  # Target for the longest edge of the hi-res preview

        _, _, w_native, h_native = parse_result.document_bounds
        if w_native <= 1e-9 or h_native <= 1e-9:
            # If there's no page size, we can't render a background.
            # This is not necessarily an error; a file might have vector
            # content but no defined canvas.
            pass
        else:
            aspect = w_native / h_native
            if aspect >= 1.0:
                render_width = target_dim
                render_height = max(1, int(target_dim / aspect))
            else:
                render_height = target_dim
                render_width = max(1, int(target_dim * aspect))

            vips_image = renderer.render_preview_image(
                import_result, render_width, render_height
            )

        # 2. Calculate content bounds for vector overlays from the
        # intermediate vectorization result.
        if import_result.vectorization_result:
            all_geos = Geometry()
            for geo in (
                import_result.vectorization_result.geometries_by_layer.values()
            ):
                if geo:
                    all_geos.extend(geo)

            if not all_geos.is_empty():
                min_x, min_y, max_x, max_y = all_geos.rect()
                content_bounds = (min_x, min_y, max_x, max_y)

        # 3. Create a thumbnail for the UI.
        if not vips_image:
            # If background rendering failed or was skipped, but we have
            # vectors, create a blank image to render the vectors on.
            if payload and payload.items:
                vips_image = pyvips.Image.black(
                    preview_size_px, preview_size_px
                )
            else:
                return None  # No background and no items, nothing to show.

        aspect_ratio = (
            vips_image.width / vips_image.height if vips_image.height else 1.0
        )
        preview_vips = vips_image.thumbnail_image(
            preview_size_px, height=preview_size_px, size="both"
        )

        if isinstance(spec, TraceSpec) and spec.invert:
            bands = preview_vips.bands
            if bands == 2:
                background = [255]
            elif bands == 4:
                background = [255, 255, 255]
            else:
                background = [255, 255, 255]
            preview_vips = preview_vips.flatten(background=background).invert()

        png_bytes = preview_vips.pngsave_buffer()

        return PreviewResult(
            image_bytes=png_bytes,
            payload=payload,
            parse_result=parse_result,
            aspect_ratio=aspect_ratio,
            content_bounds=content_bounds,
        )

    def _extract_first_workpiece(
        self, items: list[DocItem]
    ) -> WorkPiece | None:
        """Recursively extract the first WorkPiece from a list of items."""
        for item in items:
            if isinstance(item, WorkPiece):
                return item
            if hasattr(item, "children"):
                res = self._extract_first_workpiece(item.children)
                if res:
                    return res
        return None

    async def _load_file_async(
        self,
        filename: Path,
        mime_type: str | None,
        vectorization_spec: VectorizationSpec | None,
    ) -> ImportResult | None:
        """
        Runs the blocking import function in a background thread and returns
        the resulting rich ImportResult.
        """
        importer_cls, _ = self.get_importer_info(filename, mime_type)
        if not importer_cls:
            return None

        file_data = filename.read_bytes()
        importer = importer_cls(file_data, source_file=filename)
        return await asyncio.to_thread(
            importer.get_doc_items, vectorization_spec
        )

    def resolve_import_mime_type(
        self, file_path: Path, mime_type: str | None
    ) -> str | None:
        """
        Returns the MIME type that identifies `mime_type` to an importer.

        System databases map some extensions to a foreign type, such as
        `.nc` (NetCDF) or `.tap` (a bitmap), which hides the importer
        the extension is meant for. When the given type is not one that
        any importer reads but the extension does resolve to an
        importer, that importer's own MIME type is used instead.

        Args:
            file_path: Path of the file being imported.
            mime_type: The MIME type guessed or reported for the file.

        Returns:
            The MIME type to hand to the import pipeline.
        """
        if mime_type and importer_registry.get_by_mime_type(mime_type):
            return mime_type

        if not file_path.suffix:
            return mime_type

        importer_cls = importer_registry.get_by_extension(
            file_path.suffix.lower()
        )
        if not importer_cls:
            return mime_type

        native_mime = importer_cls.own_mime_type()
        if not native_mime:
            return mime_type

        logger.info(
            f"Replacing foreign MIME type '{mime_type}' of "
            f"{file_path.name} with '{native_mime}'"
        )
        return native_mime

    def _position_newly_imported_items(
        self,
        items: list[DocItem],
        position_mm: Point | None,
    ):
        """
        Applies transformations to newly imported items, either positioning
        them at a specific point or fitting and centering them.
        This method modifies the items' matrices in-place.
        """
        logger.debug(
            f"_position_newly_imported_items: position_mm={position_mm}, "
            f"items={len(items)}"
        )

        # The imported items are WorkPieces; transform them directly.
        content_to_transform = list(items)
        if not content_to_transform:
            return

        scale_factor = self._scale_to_fit_if_oversized(content_to_transform)

        if position_mm:
            # Note: PositionAtStrategy needs the top-level items to calculate
            # the current group position correctly.
            strategy = PositionAtStrategy(items=items, position_mm=position_mm)
            deltas = strategy.calculate_deltas()
            if deltas:
                # All items get the same delta matrix to move the group
                delta_matrix = next(iter(deltas.values()))
                # Apply the delta to the actual content, not the containers.
                for item in content_to_transform:
                    item.matrix = delta_matrix @ item.matrix

                target_x, target_y = position_mm
                logger.info(
                    f"Positioned {len(content_to_transform)} imported "
                    f"item(s) at ({target_x:.2f}, {target_y:.2f}) mm"
                )
        else:
            self._position_at_reference_origin(content_to_transform)

        if scale_factor < 1.0:
            self._show_scale_down_notification(
                content_to_transform, scale_factor
            )

    def _resolve_destinations(
        self,
        items: list[DocItem],
        mode: LayerImportMode,
        source_layers: list[SourceLayerInfo] | None = None,
    ) -> tuple[list[tuple[DocItem, DocItem]], list[Workflow]]:
        """
        Resolve each imported workpiece to a (owner, item) pair and
        build a workflow per imported source layer, based on the import
        mode.

        All workpieces go into the active layer. NEW_WORKFLOWS creates one
        workflow per source layer; MAP_TO_EXISTING assigns each source
        layer's workpieces to the layer's existing workflow at the same
        index; FLATTEN assigns all of them to the first workflow.

        Returns (pairs, dest_workflows): the flat list of
        (destination_owner, item_to_add) tuples and the workflows that
        received imported parts.
        """
        target_layer = self._editor.doc.active_layer
        grouped = self._group_by_source_layer(items)
        if mode is LayerImportMode.MAP_TO_EXISTING:
            return self._map_to_existing(target_layer, grouped)
        if mode is LayerImportMode.NEW_WORKFLOWS:
            layers_by_id = {
                info.layer_id: info for info in (source_layers or [])
            }
            return self._new_workflows(target_layer, grouped, layers_by_id)
        return self._flatten(target_layer, items)

    @staticmethod
    def _group_by_source_layer(
        items: list[DocItem],
    ) -> dict[str | None, list[WorkPiece]]:
        """Groups imported workpieces by their source layer id."""
        grouped: dict[str | None, list[WorkPiece]] = {}
        for item in items:
            if not isinstance(item, WorkPiece):
                continue
            layer_id = (
                item.source_segment.layer_id if item.source_segment else None
            )
            grouped.setdefault(layer_id, []).append(item)
        return grouped

    @staticmethod
    def _assign(workflow: Workflow, workpieces: list[WorkPiece]) -> None:
        """Extends a workflow's part selection with imported parts."""
        uids = list(workflow.workpiece_uids)
        uids.extend(wp.uid for wp in workpieces)
        workflow.set_workpiece_uids(uids)

    def _map_to_existing(
        self,
        target_layer: Layer,
        grouped: dict[str | None, list[WorkPiece]],
    ) -> tuple[list[tuple[DocItem, DocItem]], list[Workflow]]:
        """Assigns imported parts to the layer's workflows by index."""
        pairs: list[tuple[DocItem, DocItem]] = []
        dest_workflows: list[Workflow] = []
        existing = target_layer.workflows
        for idx, workpieces in enumerate(grouped.values()):
            if idx < len(existing):
                workflow = existing[idx]
                self._assign(workflow, workpieces)
                dest_workflows.append(workflow)
            pairs.extend((target_layer, wp) for wp in workpieces)
        return pairs, dest_workflows

    def _new_workflows(
        self,
        target_layer: Layer,
        grouped: dict[str | None, list[WorkPiece]],
        layers_by_id: dict[str | None, SourceLayerInfo],
    ) -> tuple[list[tuple[DocItem, DocItem]], list[Workflow]]:
        """Creates one workflow per imported source layer.

        On a layer without parts, existing workflows that select every
        part are reused for the imported source layers. This keeps the
        generated default workflow from also claiming the imported
        parts. On a layer that already has parts, those workflows are
        scoped to the existing parts so the imported parts are not
        duplicated.
        """
        pairs: list[tuple[DocItem, DocItem]] = []
        dest_workflows: list[Workflow] = []
        existing = target_layer.workflows
        existing_uids = [wp.uid for wp in target_layer.all_workpieces]
        reusable: list[Workflow] = []
        for workflow in existing:
            if workflow.workpiece_uids:
                continue
            if existing_uids:
                workflow.set_workpiece_uids(existing_uids)
            else:
                reusable.append(workflow)
        default_workflow = existing[0] if existing else None
        first_group = True
        for layer_id, workpieces in grouped.items():
            info = layers_by_id.get(layer_id)
            if reusable:
                workflow = reusable.pop(0)
                if info is not None:
                    if info.color:
                        workflow.set_color(info.color)
                    if info.settings:
                        workflow.set_steps([])
                        ItemAssembler.apply_settings_to_workflow(
                            workflow, info.name, info.settings
                        )
                self._assign(workflow, workpieces)
                dest_workflows.append(workflow)
                pairs.extend((target_layer, wp) for wp in workpieces)
                first_group = False
                continue
            if info is not None and default_workflow is not None:
                workflow = self._editor.workflow.create_for_source_layer(
                    target_layer, info, workpieces
                )
                if workflow is not None:
                    dest_workflows.append(workflow)
                    pairs.append((target_layer, workflow))
                    pairs.extend((target_layer, wp) for wp in workpieces)
                    first_group = False
                    continue
            if first_group and default_workflow is not None:
                self._assign(default_workflow, workpieces)
                dest_workflows.append(default_workflow)
                first_group = False
            pairs.extend((target_layer, wp) for wp in workpieces)
        return pairs, dest_workflows

    def _flatten(
        self,
        target_layer: Layer,
        items: list[DocItem],
    ) -> tuple[list[tuple[DocItem, DocItem]], list[Workflow]]:
        """Assigns every imported part to the first workflow."""
        pairs: list[tuple[DocItem, DocItem]] = [
            (target_layer, item) for item in items
        ]
        default_workflow = (
            target_layer.workflows[0] if target_layer.workflows else None
        )
        if default_workflow is not None:
            self._assign(
                default_workflow,
                [item for item in items if isinstance(item, WorkPiece)],
            )
            return pairs, [default_workflow]
        return pairs, []

    @staticmethod
    def _layer_import_mode(
        vectorization_spec: VectorizationSpec | None,
    ) -> LayerImportMode:
        """Returns the layer import mode of the given spec."""
        mode = LayerImportMode.NEW_WORKFLOWS
        if isinstance(vectorization_spec, PassthroughSpec):
            mode = vectorization_spec.layer_import_mode
        return mode

    def _commit_items(
        self,
        items: list[DocItem],
        mode: LayerImportMode,
        cmd_name: str,
        source_layers: list[SourceLayerInfo] | None = None,
    ) -> list[Workflow]:
        """
        Adds the imported items to the document model using the history
        manager, resolving destinations and creating per-source-layer
        workflows.

        Returns the workflows that received items.
        """
        pairs, new_workflows = self._resolve_destinations(
            items, mode, source_layers
        )

        with self._editor.history_manager.transaction(cmd_name) as t:
            for owner, item in pairs:
                t.execute(
                    ListItemCommand(
                        owner_obj=owner,
                        item=item,
                        undo_command="remove_child",
                        redo_command="add_child",
                    )
                )

        return new_workflows

    def _commit_items_to_document(
        self,
        items: list[DocItem],
        source: SourceAsset | None,
        filename: Path,
        assets: list["IAsset"] | None = None,
        vectorization_spec: VectorizationSpec | None = None,
        source_layers: list[SourceLayerInfo] | None = None,
    ) -> list[Workflow]:
        """
        Adds the imported items and their source to the document model
        using the history manager.

        Returns the workflows that received items.
        """
        if source:
            self._editor.doc.add_asset(source)

        if assets:
            for asset in assets:
                self._editor.doc.add_asset(asset)

        cmd_name = _("Import {filename}").format(filename=filename.name)
        mode = self._layer_import_mode(vectorization_spec)
        return self._commit_items(items, mode, cmd_name, source_layers)

    def _finalize_import_on_main_thread(
        self,
        payload: ImportPayload,
        filename: Path,
        position_mm: Point | None,
        vectorization_spec: VectorizationSpec | None = None,
    ):
        """
        Performs the final steps of an import on the main thread.
        This includes positioning items (which may send UI notifications) and
        committing them to the document (which fires signals that update UI).
        """
        item_info = (
            f"{len(payload.items)} items"
            if payload and payload.items
            else "0 items"
        )
        logger.debug(f"Item_info: {item_info} position_mm: {position_mm}")
        # 1. Position the new items. This is now safe as it runs on the main
        #    thread, so any notifications it sends are valid.
        self._position_newly_imported_items(payload.items, position_mm)

        # 2. Add the positioned items to the document model. This is also
        #    safe now as all subsequent signal handling will be on the
        #    main thread.
        dest_workflows = self._commit_items_to_document(
            payload.items,
            payload.source,
            filename,
            payload.assets,
            vectorization_spec,
            payload.source_layers,
        )

        # 3. Add default steps to the destination workflows.
        if dest_workflows:
            self._editor.step.add_default_steps_for_workflows(dest_workflows)

    def load_file_from_path(
        self,
        filename: Path,
        mime_type: str | None,
        vectorization_spec: VectorizationSpec | None,
        position_mm: Point | None = None,
    ):
        """
        Public, synchronous method to launch a file import in the background.
        This is the clean entry point for the UI.

        Args:
            filename: Path to the file to import
            mime_type: MIME type of the file
            vectorization_spec: Configuration for vectorization
                (None for direct vector import)
            position_mm: Optional (x, y) tuple in world coordinates (mm)
                to center the imported item.
                        If None, items are centered on the workspace.
        """
        logger.debug(
            f"Loading file: {filename} "
            f"vectorization_spec: {vectorization_spec} "
            f"position_mm: {position_mm}"
        )

        # This wrapper adapts our clean async method to the TaskManager,
        # which expects a coroutine that accepts a 'ctx' argument.
        async def wrapper(ctx, fn, mt, vec_spec, pos_mm):
            try:
                # Update task message for UI feedback
                ctx.set_message(
                    _("Importing {filename}...").format(filename=filename.name)
                )

                # 1. Run blocking I/O and CPU work in a background thread.
                mt = self.resolve_import_mime_type(fn, mt)
                import_result = await self._load_file_async(fn, mt, vec_spec)

                # 2. Validate the result.
                if not import_result or not import_result.payload:
                    errors = import_result.errors if import_result else []
                    if errors:
                        msg = "\n".join(dict.fromkeys(errors))
                    elif mt and mt.startswith("image/"):
                        msg = _(
                            "Failed to import {filename}. The image file "
                            "may be corrupted or in an unsupported format."
                        ).format(filename=fn.name)
                    else:
                        msg = _(
                            "Import failed: No items were created "
                            "from {filename}"
                        ).format(filename=fn.name)
                    logger.warning(
                        f"Importer created no items for '{fn.name}' "
                        f"(MIME: {mt})"
                    )
                    # Schedule the error notification on the main thread.
                    self._task_manager.schedule_on_main_thread(
                        self._editor.notification_requested.send,
                        self,
                        message=msg,
                    )
                    ctx.set_message(_("Import failed."))
                    return

                # 3. Schedule finalization on main thread and wait for it to
                #    signal completion back to this (background) thread.
                loop = asyncio.get_running_loop()
                main_thread_done = loop.create_future()

                def finalizer_and_callback():
                    """Wraps finalizer to signal future on completion/error."""
                    try:
                        assert import_result.payload, "Missing import payload"
                        self._finalize_import_on_main_thread(
                            import_result.payload, fn, pos_mm, vec_spec
                        )
                        if not main_thread_done.done():
                            loop.call_soon_threadsafe(
                                main_thread_done.set_result, True
                            )
                    except Exception as e:
                        logger.exception(
                            "Failed import finalization on main thread."
                        )
                        if not main_thread_done.done():
                            loop.call_soon_threadsafe(
                                main_thread_done.set_exception, e
                            )

                self._task_manager.schedule_on_main_thread(
                    finalizer_and_callback
                )

                # Wait here until the main thread signals completion or error.
                await main_thread_done

                warnings = list(dict.fromkeys(import_result.warnings))
                if warnings:
                    self._task_manager.schedule_on_main_thread(
                        self._editor.notification_requested.send,
                        self,
                        message="\n".join(warnings),
                    )

                ctx.set_message(_("Import complete!"))
            except Exception as e:
                # This will catch failures from the importer or the finalizer.
                ctx.set_message(_("Import failed."))
                logger.error(
                    f"Import task for {fn.name} failed in wrapper.",
                    exc_info=e,
                )
                # Re-raise to ensure the task manager marks the task as failed.
                raise

        self._task_manager.add_coroutine(
            wrapper,
            filename,
            mime_type,
            vectorization_spec,
            position_mm,
            key=f"import-{filename}",
        )

    def execute_batch_import(
        self,
        files: list[Path],
        spec: VectorizationSpec,
        pos: Point | None,
    ):
        """
        Imports multiple files using the same vectorization settings.
        This spawns individual import tasks for each file.
        """
        for file_path in files:
            # We assume files are valid if passed here, or guess mime type
            # individually
            mime_type, _ = mimetypes.guess_type(file_path)
            self.load_file_from_path(file_path, mime_type, spec, pos)

    def _calculate_items_bbox(
        self,
        items: list[DocItem],
    ) -> Rect | None:
        """
        Calculates the world-space bounding box that encloses a list of
        DocItems by taking the union of their individual bboxes.
        This is more robust than item.bbox for un-parented items.
        """
        if not items:
            return None

        all_rects = []
        for item in items:
            # FIX: Use the item's matrix directly. This is robust for
            # items not yet in the document tree, as their matrix IS their
            # world transform at this point.
            item_transform = item.matrix
            item_bbox_local = item.get_local_bbox()

            if item_bbox_local:
                # Transform the four corners of the local bounding box
                corners = [
                    (item_bbox_local[0], item_bbox_local[1]),
                    (
                        item_bbox_local[0] + item_bbox_local[2],
                        item_bbox_local[1],
                    ),
                    (
                        item_bbox_local[0] + item_bbox_local[2],
                        item_bbox_local[1] + item_bbox_local[3],
                    ),
                    (
                        item_bbox_local[0],
                        item_bbox_local[1] + item_bbox_local[3],
                    ),
                ]
                world_corners = [
                    item_transform.transform_point(p) for p in corners
                ]

                min_x = min(p[0] for p in world_corners)
                min_y = min(p[1] for p in world_corners)
                max_x = max(p[0] for p in world_corners)
                max_y = max(p[1] for p in world_corners)
                all_rects.append((min_x, min_y, max_x - min_x, max_y - min_y))

        if not all_rects:
            return None

        # Calculate the union of all collected rectangles
        min_x, min_y, w, h = all_rects[0]
        max_x = min_x + w
        max_y = min_y + h

        for x, y, w, h in all_rects[1:]:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x + w)
            max_y = max(max_y, y + h)

        return min_x, min_y, max_x - min_x, max_y - min_y

    def _scale_to_fit_if_oversized(self, items: list[DocItem]) -> float:
        """
        Scales items to fit within machine work area if they are too
        large, preserving aspect ratio.

        Returns the scale factor applied (1.0 if no scaling was needed).
        """
        config = get_context().config
        if not config or not config.machine:
            logger.warning(
                "Cannot fit/position imported items: "
                "machine dimensions unknown."
            )
            return 1.0

        # The imported items are WorkPieces; measure them directly.
        content_items = list(items)
        if not content_items:
            logger.warning("No positionable content found to fit/position.")
            return 1.0

        # Calculate the bounding box of the actual content.
        bbox = self._calculate_items_bbox(content_items)
        if not bbox:
            logger.warning(
                "Cannot fit/position imported items: no bounding box."
            )
            return 1.0

        bbox_x, bbox_y, bbox_w, bbox_h = bbox
        area_x, area_y, area_w, area_h = config.machine.work_area
        logger.debug(
            f"_fit_and_position_at_reference_origin: bbox=({bbox_x:.2f}, "
            f"{bbox_y:.2f}, {bbox_w:.2f}, {bbox_h:.2f}), "
            f"work_area="
            f"({area_x:.2f}, {area_y:.2f}, {area_w:.2f}, {area_h:.2f})"
        )

        # Scale to fit if necessary, preserving aspect ratio
        scale_factor = 1.0
        if bbox_w > area_w or bbox_h > area_h:
            scale_w = area_w / bbox_w if bbox_w > 1e-9 else 1.0
            scale_h = area_h / bbox_h if bbox_h > 1e-9 else 1.0
            scale_factor = min(scale_w, scale_h)

        if scale_factor < 1.0:
            # The pivot for scaling should be the center of the bounding box
            bbox_center_x = bbox_x + bbox_w / 2
            bbox_center_y = bbox_y + bbox_h / 2

            # The transformation is: T(pivot) @ S(scale) @ T(-pivot)
            t_to_origin = Matrix.translation(-bbox_center_x, -bbox_center_y)
            s = Matrix.scale(scale_factor, scale_factor)
            t_back = Matrix.translation(bbox_center_x, bbox_center_y)
            transform_matrix = t_back @ s @ t_to_origin

            # Apply the group transform to each piece of content.
            for item in content_items:
                item.matrix = transform_matrix @ item.matrix

        return scale_factor

    def _position_at_reference_origin(self, items: list[DocItem]):
        """
        Positions items at the reference origin.

        The caller is responsible for calling _scale_to_fit_if_oversized()
        before this method.
        """
        config = get_context().config
        if not config or not config.machine:
            logger.warning(
                "Cannot fit/position imported items: "
                "machine dimensions unknown."
            )
            return

        content_items = list(items)
        if not content_items:
            return

        bbox = self._calculate_items_bbox(content_items)
        if not bbox:
            return  # Should not happen, but for safety
        bbox_x, bbox_y, bbox_w, bbox_h = bbox

        machine = config.machine
        # Position at reference origin
        # The reference origin is where the user expects (0,0) to be.
        # The panel gives us the reference origin in world coords; we use
        # world_position_from_origin to handle origin corner adjustment.
        ref_x, ref_y = machine.panel.reference_position_world
        target_x, target_y = machine.panel.world_position_from_origin(
            ref_x, ref_y, (bbox_w, bbox_h)
        )

        # Calculate translation to move bbox top-left to the target position
        delta_x = target_x - bbox_x
        delta_y = target_y - bbox_y

        # Apply the same translation to all top-level imported items
        if abs(delta_x) > 1e-9 or abs(delta_y) > 1e-9:
            translation_matrix = Matrix.translation(delta_x, delta_y)
            # Apply the group transform to each piece of content.
            for item in content_items:
                item.matrix = translation_matrix @ item.matrix

    def _show_scale_down_notification(
        self, content_items: list[DocItem], scale_factor: float
    ):
        """
        Shows a persistent notification that the imported item was scaled
        down, with an undo action that reverts the scaling.
        """
        # Notification with Undo logic
        # We define this after centering so the callback can handle the
        # final position correctly.

        def _undo_scaling_callback():
            """
            Reverts the auto-scaling applied during import.
            It scales the items back up around their CURRENT center.
            """
            # Use the content items for calculation and transformation
            current_bbox = self._calculate_items_bbox(content_items)
            if not current_bbox:
                return

            cur_x, cur_y, cur_w, cur_h = current_bbox
            cur_cx = cur_x + cur_w / 2
            cur_cy = cur_y + cur_h / 2

            inv_scale = 1.0 / scale_factor

            # Create a matrix that scales by 1/factor around the current
            # center
            undo_matrix = Matrix.scale(
                inv_scale, inv_scale, center=(cur_cx, cur_cy)
            )

            changes = []
            for item in content_items:
                current = item.matrix
                new_m = undo_matrix @ current
                changes.append((item, current, new_m))

            self._editor.transform.create_transform_transaction(changes)

        msg = _(
            "⚠️ Imported item was larger than the work area and has been "
            "scaled down to fit."
        )
        logger.info(msg)
        self._editor.notification_requested.send(
            self,
            message=msg,
            persistent=True,
            action_label=_("Reset"),
            action_callback=_undo_scaling_callback,
        )

    def assemble_job_in_background(
        self,
        when_done: Callable[
            [BaseArtifactHandle | None, Exception | None], None
        ],
    ):
        """
        Asynchronously runs the full job assembly in a background process.
        This method is non-blocking and returns immediately.

        Args:
            when_done: A callback executed upon completion. It receives
                       an ArtifactHandle on success, or (None, error) on
                       failure.
        """
        self._editor.pipeline.generate_job_artifact(when_done=when_done)

    def export_gcode_to_path(self, file_path: Path):
        """
        Asynchronously generates and exports G-code to a specific path.
        This is a non-blocking, fire-and-forget method for the UI.
        """
        artifact_store = self._editor.pipeline.artifact_store

        def _on_export_assembly_done(
            handle: BaseArtifactHandle | None,
            error: Exception | None,
        ):
            try:
                if error:
                    raise error

                with artifact_store.checkout_handle(handle) as artifact:
                    if not artifact:
                        raise ValueError(
                            "Assembly process returned no artifact."
                        )
                    if not isinstance(artifact, JobArtifact):
                        raise TypeError("Expected a JobArtifact for export.")
                    if artifact.machine_code is None:
                        raise ValueError(
                            "Final artifact is missing G-code data."
                        )

                    file_path.write_text(
                        artifact.machine_code, encoding="utf-8"
                    )

                    logger.info(f"Successfully exported G-code to {file_path}")
                    msg = _("Export successful: {name}").format(
                        name=file_path.name
                    )
                    self._editor.notification_requested.send(self, message=msg)
                    self._task_manager.schedule_on_main_thread(
                        self.gcode_exported.send, self
                    )

            except Exception as e:
                logger.error(
                    f"G-code export to {file_path} failed.", exc_info=e
                )
                self._editor.notification_requested.send(
                    self, message=_("Export failed: {error}").format(error=e)
                )

        self.assemble_job_in_background(when_done=_on_export_assembly_done)

    def export_object_to_path(self, file_path: Path, workpiece: WorkPiece):
        """
        Exports a workpiece to a file.

        Supports multiple formats based on file extension:
        - .rfs: Rayforge Sketch (parametric, sketch-based only)
        - .svg: SVG format
        - .dxf: DXF format
        - .cdr: CorelDRAW format

        This is a synchronous method for the UI.
        """
        ext = file_path.suffix.lower()
        if ext == ".rfs":
            return self._export_sketch_to_rfs(file_path, workpiece)

        geo = workpiece.get_world_geometry()
        if geo is None or geo.is_empty():
            raise ValueError(
                "Cannot export: The selected item has no geometry."
            )

        if ext == ".svg":
            exporter = GeometrySvgExporter(geo)
        elif ext == ".dxf":
            exporter = GeometryDxfExporter(geo)
        elif ext == ".cdr":
            exporter = CorelDrawExporter(geo)
        else:
            raise ValueError(f"Unsupported export format: {ext}")

        return self._do_export(file_path, exporter)

    def _export_sketch_to_rfs(
        self, file_path: Path, workpiece: WorkPiece
    ) -> bool:
        """Export a sketch-based workpiece to RFS format."""
        exporter_cls = exporter_registry.get_by_extension(file_path.suffix)
        if not exporter_cls:
            raise ValueError(
                f"No exporter registered for extension {file_path.suffix}"
            )
        exporter = cast(type[Exporter], exporter_cls)(workpiece)
        return self._do_export(file_path, exporter)

    def _do_export(self, file_path: Path, exporter) -> bool:
        """Execute the export and handle notifications."""
        try:
            data = exporter.export()
            file_path.write_bytes(data)
            logger.info(f"Successfully exported object to {file_path}")
            msg = _("Object exported successfully.")
            self._editor.notification_requested.send(self, message=msg)
            return True
        except Exception as e:
            logger.error(f"Failed to export object to {file_path}", exc_info=e)
            self._editor.notification_requested.send(
                self,
                message=_("Failed to export object: {error}").format(
                    error=str(e)
                ),
            )
            return False

    def export_document_to_path(self, file_path: Path) -> bool:
        """
        Exports all workpieces in the document to a file.

        Supports multiple formats based on file extension:
        - .svg: SVG format
        - .dxf: DXF format
        - .cdr: CorelDRAW format

        This is a synchronous method for the UI.
        """
        geometries = []
        for wp in self._editor.doc.get_descendants(WorkPiece):
            geo = wp.get_world_geometry()
            if geo is not None and not geo.is_empty():
                geometries.append(geo)

        if not geometries:
            self._editor.notification_requested.send(
                self,
                message=_("Cannot export: Document has no geometry."),
            )
            return False

        ext = file_path.suffix.lower()
        if ext == ".svg":
            exporter = MultiGeometrySvgExporter(geometries)
        elif ext == ".dxf":
            exporter = MultiGeometryDxfExporter(geometries)
        elif ext == ".cdr":
            exporter = MultiGeometryCorelDrawExporter(geometries)
        else:
            raise ValueError(f"Unsupported export format: {ext}")

        try:
            data = exporter.export()
            file_path.write_bytes(data)
            logger.info(f"Successfully exported document to {file_path}")
            msg = _("Document exported successfully.")
            self._editor.notification_requested.send(self, message=msg)
            return True
        except Exception as e:
            logger.error(
                f"Failed to export document to {file_path}", exc_info=e
            )
            self._editor.notification_requested.send(
                self,
                message=_("Failed to export document: {error}").format(
                    error=str(e)
                ),
            )
            return False

    def save_project_to_path(self, file_path: Path):
        """
        Saves the current document to a .ryp project file.
        This is a synchronous method for the UI.
        """
        try:
            doc_dict = self._editor.doc.to_dict()
            json_bytes = json.dumps(doc_dict, indent=2).encode("utf-8")
            with zipfile.ZipFile(
                file_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                zf.writestr("project.json", json_bytes)
            self._editor.set_file_path(file_path)
            self._editor.mark_as_saved()
            logger.info(f"Successfully saved project to {file_path}")
            msg = _("Project saved: {name}").format(name=file_path.name)
            self._editor.notification_requested.send(self, message=msg)
            return True
        except Exception as e:
            logger.error(f"Failed to save project to {file_path}", exc_info=e)
            self._editor.notification_requested.send(
                self, message=_("Save failed: {error}").format(error=str(e))
            )
            return False

    @staticmethod
    def _read_project_content(file_path: Path) -> str:
        if zipfile.is_zipfile(file_path):
            with zipfile.ZipFile(file_path, "r") as zf:
                return zf.read("project.json").decode("utf-8")
        return file_path.read_text(encoding="utf-8")

    def load_project_from_path(self, file_path: Path):
        """
        Loads a .ryp project file and replaces the current document.
        This is a synchronous method for the UI.
        """
        try:
            if not file_path.exists():
                msg = _("File not found: {name}").format(name=file_path.name)
                self._editor.notification_requested.send(self, message=msg)
                return False

            file_content = self._read_project_content(file_path)
            doc_dict = json.loads(file_content)

            from ..core.asset import UnknownAsset
            from ..core.doc import Doc

            new_doc = Doc.from_dict(doc_dict)

            self._editor.set_doc(new_doc)
            self._editor.set_file_path(file_path)
            self._editor.mark_as_saved()
            self._editor.doc.updated.send(self._editor.doc)

            if new_doc.newer_schema_version is not None:
                self._editor.notification_requested.send(
                    self,
                    message=_(
                        "This project uses document schema {version}, which "
                        "is newer than this version of Rayforge supports "
                        "({supported}). Some data may not load or save "
                        "correctly."
                    ).format(
                        version=new_doc.newer_schema_version,
                        supported=CURRENT_DOC_SCHEMA_VERSION,
                    ),
                    persistent=True,
                )

            labels = _unsupported_coolant_labels(
                new_doc, self._editor.context.machine
            )
            if labels:
                self._editor.notification_requested.send(
                    self,
                    message=_(
                        "This project uses cooling methods not supported by "
                        "the current machine: {methods}"
                    ).format(methods=", ".join(labels)),
                    persistent=True,
                )

            unknown_assets = [
                asset
                for asset in new_doc.get_all_assets()
                if isinstance(asset, UnknownAsset)
            ]
            if unknown_assets:
                self._editor.notification_requested.send(
                    self,
                    message=_(
                        "{count} asset(s) require disabled addon(s)"
                    ).format(count=len(unknown_assets)),
                    persistent=True,
                )

            logger.info(f"Successfully loaded project from {file_path}")
            return True
        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse project file {file_path}: {e}",
                exc_info=e,
            )
            self._editor.notification_requested.send(
                self, message=_("Invalid project file format")
            )
            return False
        except Exception as e:
            logger.error(
                f"Failed to load project from {file_path}", exc_info=e
            )
            self._editor.notification_requested.send(
                self, message=_("Load failed: {error}").format(error=str(e))
            )
            return False

    def reimport_from_source_asset(
        self,
        source_asset: SourceAsset,
        vectorization_spec: VectorizationSpec,
        position_mm: Point | None = None,
        target_layer: Layer | None = None,
    ) -> ImportResult | None:
        """
        Re-run the import pipeline for an existing SourceAsset, producing
        fresh (or additional) workpieces from the original data.

        Unlike the normal import path, no new SourceAsset is added to the
        document -- the existing one is reused.
        """
        meta = source_asset.metadata
        importer_cls_name = meta.get("_importer_class")
        if not importer_cls_name:
            logger.warning(
                "Cannot reimport: SourceAsset has no _importer_class metadata"
            )
            return None
        importer_cls = importer_registry.get_by_name(importer_cls_name)
        if not importer_cls:
            logger.warning(
                f"Cannot reimport: importer '{importer_cls_name}' not "
                f"registered"
            )
            return None

        importer = importer_cls(
            data=source_asset.original_data,
            source_file=source_asset.source_file or Path("Untitled"),
        )
        import_result = importer.get_doc_items_for_reimport(
            source_asset, vectorization_spec
        )

        if not import_result or not import_result.payload:
            return import_result

        self._finalize_reimport(
            import_result.payload,
            position_mm,
            vectorization_spec,
        )
        return import_result

    def _finalize_reimport(
        self,
        payload: ImportPayload,
        position_mm: Point | None,
        vectorization_spec: VectorizationSpec | None = None,
    ):
        """
        Commit reimported items to the document.

        Unlike _finalize_import_on_main_thread, this does NOT add a new
        SourceAsset -- the existing one is reused.
        """
        self._position_newly_imported_items(payload.items, position_mm)

        mode = self._layer_import_mode(vectorization_spec)
        dest_workflows = self._commit_items(
            payload.items,
            mode,
            _("Re-Import"),
            payload.source_layers,
        )
        if dest_workflows:
            self._editor.step.add_default_steps_for_workflows(dest_workflows)
