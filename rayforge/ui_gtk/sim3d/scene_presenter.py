"""
Scene presenter for the 3D canvas.

Owns scene compilation scheduling, the compiled artifact, the playback
OpPlayer, and the playback overlay binding.  Constructed by Canvas3D with
injected callables so it never reaches back into the widget; the canvas
keeps the GL lifecycle and per-frame rendering.
"""

import logging
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Optional

import numpy as np
from blinker import Signal
from raygeo.ops import Ops

from ...context import RayforgeContext
from ...machine.kinematic_mapping import (
    KinematicMapping,
    build_layer_assembly,
    resolve_layer_rotary,
)
from ...machine.models.laser import LaserHead
from ...pipeline.artifact.handle import BaseArtifactHandle
from ...pipeline.artifact.job import JobArtifact
from ...shared.tasker import Task, task_mgr
from ...simulator.op_player import OpPlayer, build_snapshots
from ...simulator.scene3d import (
    CompiledSceneArtifact,
    LayerRenderConfig,
    RenderConfig3D,
    compile_scene_from_job,
    compile_stock_scene,
)
from .camera import ViewDirection

if TYPE_CHECKING:
    from ...core.doc import Doc
    from ...doceditor.editor import DocEditor
    from ...machine.assembly import Assembly
    from .renderer.scene_renderer import SceneRenderer
    from .theme_resolver import ThemeResolver
    from .viewport import ViewportConfig

logger = logging.getLogger(__name__)


class ScenePresenter:
    """
    Compiles the scene, builds the playback player, and binds playback.

    The canvas owns the GL context and per-frame render state; this class
    owns everything that turns a document + job artifact into a compiled
    ``CompiledSceneArtifact`` and an ``OpPlayer``.  Dependencies are
    injected as callables so the presenter stays independent of the widget.
    """

    def __init__(
        self,
        context: RayforgeContext,
        doc_editor: "DocEditor",
        scene: "SceneRenderer",
        *,
        theme_resolver: "ThemeResolver",
        get_viewport: Callable[[], "ViewportConfig"],
        get_gl_initialized: Callable[[], bool],
        get_show_travel_moves: Callable[[], bool],
        get_camera_available: Callable[[], bool],
        make_current: Callable[[], None],
        mark_scene_dirty: Callable[[], None],
        mark_artifact_dirty: Callable[[], None],
        reset_view: Callable[[ViewDirection], None],
        request_render: Callable[[], None],
        upload_complete: Signal,
    ):
        self._context = context
        self._doc_editor = doc_editor
        self._scene = scene
        self._theme_resolver = theme_resolver
        self._get_viewport = get_viewport
        self._get_gl_initialized = get_gl_initialized
        self._get_show_travel_moves = get_show_travel_moves
        self._get_camera_available = get_camera_available
        self._make_current = make_current
        self._mark_scene_dirty = mark_scene_dirty
        self._mark_artifact_dirty = mark_artifact_dirty
        self._reset_view = reset_view
        self._request_render = request_render
        self._upload_complete = upload_complete

        self._scene_preparation_task: Task | None = None
        self._compiled_artifact: CompiledSceneArtifact | None = None
        self._current_job_handle: BaseArtifactHandle | None = None
        self._compiled_job_generation: int | None = None
        self._op_player: OpPlayer | None = None
        self._playback_assembly: Assembly | None = None
        self._playback_overlay = None

    def connect(self):
        """Subscribe to the pipeline and upload events that drive the scene.

        Called once the canvas has realized its GL context.  ``connect`` /
        ``disconnect`` pair keeps the presenter's signal wiring in one
        place instead of being threaded through the canvas.
        """
        self._upload_complete.connect(self._on_upload_complete)
        pipeline = self._doc_editor.pipeline
        if pipeline:
            pipeline.processing_state_changed.connect(
                self._on_pipeline_state_changed
            )
            pipeline.job_generation_finished.connect(
                self._on_job_generation_finished
            )

    def disconnect(self):
        """Unsubscribe from pipeline and upload events."""
        self._upload_complete.disconnect(self._on_upload_complete)
        pipeline = self._doc_editor.pipeline
        if pipeline:
            pipeline.processing_state_changed.disconnect(
                self._on_pipeline_state_changed
            )
            pipeline.job_generation_finished.disconnect(
                self._on_job_generation_finished
            )

    @property
    def doc(self) -> "Doc":
        """Returns the current document from the editor."""
        return self._doc_editor.doc

    @property
    def op_player(self) -> OpPlayer | None:
        """The current playback player, or None."""
        return self._op_player

    @property
    def playback_assembly(self) -> Optional["Assembly"]:
        """The throwaway assembly for the current playback layer, or None."""
        return self._playback_assembly

    @property
    def compiled_artifact(self) -> CompiledSceneArtifact | None:
        """The last compiled scene artifact, or None."""
        return self._compiled_artifact

    @property
    def scene_preparation_task(self) -> Task | None:
        """The in-flight scene compilation task, or None."""
        return self._scene_preparation_task

    @property
    def job_handle(self) -> BaseArtifactHandle | None:
        """The job artifact handle driving the scene, or None."""
        return self._current_job_handle

    @job_handle.setter
    def job_handle(self, handle: BaseArtifactHandle | None):
        self._current_job_handle = handle

    @property
    def playback_overlay(self):
        """The attached playback overlay widget, or None."""
        return self._playback_overlay

    def set_playback_overlay(self, overlay):
        """Store the playback overlay so players can be bound to it."""
        self._playback_overlay = overlay

    def cancel_scene_preparation(self):
        """Cancel any in-flight scene compilation task."""
        if self._scene_preparation_task:
            self._scene_preparation_task.cancel()
            self._scene_preparation_task = None

    def has_stale_job(self) -> bool:
        """True if the cached job handle is from an older generation."""
        handle = self._current_job_handle
        if handle is None:
            return True
        return (
            handle.generation_id
            != self._doc_editor.pipeline.data_generation_id
        )

    def _on_pipeline_state_changed(self, sender, *, is_processing: bool):
        """
        Handler for when the pipeline's busy state changes. When it becomes
        not busy, the document has settled and the scene should be updated.
        """
        if not is_processing and self._current_job_handle is not None:
            if self.has_stale_job():
                logger.debug(
                    "Pipeline settled with stale job. Clearing 3D scene."
                )
                self._current_job_handle = None
                self._compiled_job_generation = None
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()
            else:
                if (
                    self._current_job_handle.generation_id
                    == self._compiled_job_generation
                ):
                    logger.debug(
                        "Scene already compiled for this "
                        "generation; skipping duplicate update."
                    )
                else:
                    logger.debug("Pipeline has settled. Updating 3D scene.")
                    self.update_scene_from_doc()

    def _on_job_generation_finished(self, sender, **kwargs):
        task_status = kwargs.get("task_status")
        handle = kwargs.get("handle")
        logger.debug(
            f"_on_job_generation_finished: "
            f"status={task_status}, handle={'yes' if handle else 'none'}"
        )
        if task_status == "completed":
            if handle is not None:
                self._current_job_handle = handle
                self.update_scene_from_doc()
                self._request_render()
            else:
                logger.debug("Job completed with no output. Clearing scene.")
                self._current_job_handle = None
                self._compiled_job_generation = None
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()

    def _on_upload_complete(self, sender=None, **_kwargs):
        self._build_op_player_async()
        if self._compiled_artifact and self._op_player:
            self._scene.extract_playback_offsets(self._compiled_artifact)

    def _build_op_player_async(self):
        ops = self._get_ops_for_playback()
        time_ops = self._get_time_ops_for_playback()
        machine = self._context.machine
        if machine is None:
            return

        if ops is None or ops.is_empty():
            self._op_player = None
            self._playback_assembly = None
            for renderer in self._scene.ops_renderers:
                renderer.powered_offsets = np.array([], dtype=np.int32)
                renderer.travel_offsets = np.array([], dtype=np.int32)
            for renderer in self._scene.ring_renderers:
                renderer.ring_offsets = np.array([], dtype=np.int32)
            if self._playback_overlay:
                self._playback_overlay.set_player(None)
            self._request_render()
            return

        # Preserve the playhead and seek snapshots when the underlying
        # ops object has not changed (e.g. only the viewport moved).
        saved_index = None
        reused_snapshots = []
        if self._op_player is not None and self._op_player.ops is ops:
            saved_index = self._op_player.current_index
            reused_snapshots = self._op_player.snapshots

        player = OpPlayer(
            ops,
            machine,
            self.doc,
            build_snapshots=False,
            time_ops=time_ops,
        )
        player.set_playback_params(
            machine.max_cut_speed,
            machine.max_travel_speed,
            machine.acceleration,
        )
        player.set_snapshots(reused_snapshots)

        # Make the player available right away so that the next render
        # can dim textures that have not been reached yet.  Seeking to
        # the first layer is cheap (the first LAYER_START is near the
        # start of the ops), and reused snapshots keep restores of a
        # previous playhead fast as well.
        if saved_index is not None:
            player.seek(saved_index)
            initial_index = saved_index
        else:
            player.seek_to_first_layer()
            initial_index = 0
        self._op_player = player
        player.layer_changed.connect(self._on_playback_layer_changed)
        self._on_playback_layer_changed(player)
        if self._playback_overlay:
            self._playback_overlay.set_player(player, initial_index)
        self._request_render()

        # Build seek-acceleration snapshots in the background.  They are
        # collected into a fresh list and attached from the main thread
        # to avoid racing with concurrent seeks reading _snapshots.
        def _on_snapshots_done(task):
            if task.get_status() != "completed":
                return
            if self._op_player is player:
                player.set_snapshots(task.result())

        task_mgr.run_thread(
            build_snapshots,
            ops,
            machine,
            self.doc,
            key=(id(self), "build-snapshots"),
            when_done=_on_snapshots_done,
        )

    def _on_playback_layer_changed(self, player, layer_uid=None, **_kwargs):
        """Rebuild the throwaway playback assembly for the current layer.

        Connected to ``OpPlayer.layer_changed`` and also called once on
        player creation.  Resolves the effective layer (current or the
        first layer while in the preamble) and updates the scene's
        cylinder transform without mutating the live machine.
        """
        machine = self._context.machine
        if machine is None or player is None:
            return
        layer = player.get_effective_layer(self.doc)
        assembly = build_layer_assembly(machine, layer)
        self._playback_assembly = assembly
        if assembly.has_rotary:
            self._scene.set_cylinder_transform(
                assembly.cylinder_base_transform()
            )
        else:
            self._scene.set_cylinder_transform(np.eye(4, dtype=np.float64))
        self._request_render()

    def _on_scene_prepared(self, task: Task):
        """
        Callback for when the background scene compilation task is
        finished.  The compiled artifact is available directly as
        ``task.result_value`` since the compilation runs in-process.
        """
        if task.get_status() != "completed":
            if task.is_cancelled():
                logger.debug("Scene preparation task cancelled (superseded).")
            else:
                self._compiled_artifact = None
                self._op_player = None
                self._playback_assembly = None
                logger.error("Scene preparation task failed.")
                self._mark_artifact_dirty()
                self._request_render()
            return

        self._scene_preparation_task = None

        artifact = task.result()
        if artifact is None:
            logger.warning(
                "Scene task completed but produced no "
                "artifact (possibly empty scene)."
            )
            self._compiled_artifact = None
            self._mark_artifact_dirty()
            self._request_render()
            return

        if not isinstance(artifact, CompiledSceneArtifact):
            logger.error(
                f"Expected CompiledSceneArtifact, got "
                f"{type(artifact).__name__}"
            )
            self._compiled_artifact = None
            self._mark_artifact_dirty()
            self._request_render()
            return

        logger.debug("Scene compilation finished.")
        self._compiled_artifact = artifact
        self._mark_artifact_dirty()
        self._request_render()

    def update_renderers_from_artifact(self):
        if not self._compiled_artifact:
            for renderer in self._scene.ops_renderers:
                renderer.clear()
            for renderer in self._scene.ring_renderers:
                renderer.clear()
                renderer.ring_offsets = np.array([], dtype=np.int32)
            if self._scene.texture_renderer:
                self._scene.texture_renderer.clear()
            self._request_render()
            return

        if not self._get_gl_initialized():
            return

        self._make_current()

        self._scene.update_from_artifact(
            self._compiled_artifact, self._get_show_travel_moves()
        )

        self._theme_resolver.update_renderer_color_luts()

        logger.debug(
            "Scanline overlay uploaded. Groups: {}".format(
                ", ".join(
                    "{}:{}".format(
                        "rot" if r.is_rotary else "flat",
                        r.vertex_count,
                    )
                    for r in self._scene.ring_renderers
                )
            )
        )

        self._request_render()

    def _get_ops_for_playback(self) -> Ops | None:
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact):
                return artifact.preview_ops
        return None

    def _get_time_ops_for_playback(self) -> Ops | None:
        """Unmapped ops for the playback time model.

        The preview ops of rotary jobs keep endpoint Y at a constant
        while the real rotation lives in extra axes, which distorts
        distances and makes arcs degenerate. The raw assembled ops
        carry the true (unwrapped) path, so durations must come from
        them; command indices and order match the preview ops 1:1.
        """
        handle = self._current_job_handle
        if handle is not None:
            artifact = self._context.artifact_store.get(handle)
            if isinstance(artifact, JobArtifact):
                return artifact.ops
        return None

    def _build_stock_specs(self) -> list[dict]:
        """Collect visible stock items into plain-data compiler specs.

        The heavy triangulation runs on the background compile thread;
        here we only resolve each item's world-space geometry rings and
        its material parameters (texture path, PBR values, fallback
        color) so the spec dict stays serializable.
        """
        specs: list[dict] = []
        for item in self.doc.stock_items:
            if not item.visible:
                continue
            asset = item.stock_asset
            if asset is None or asset.geometry.is_empty():
                continue
            thickness = asset.thickness
            if thickness is not None and thickness <= 0:
                continue
            world_geo = item.get_world_geometry()
            if world_geo.is_empty():
                continue
            outers, holes = world_geo.split_inner_and_outer_polygons()
            if not outers:
                continue

            material = item.material
            if material is not None:
                appearance = material.appearance
                texture_path = material.get_texture_path()
            else:
                appearance = None
                texture_path = None

            specs.append(
                {
                    "name": item.name,
                    "thickness": (
                        float(thickness) if thickness is not None else None
                    ),
                    "outers": outers,
                    "holes": holes,
                    "texture_path": (
                        str(texture_path) if texture_path is not None else None
                    ),
                    "texture_size_mm": float(
                        appearance.texture_size_mm
                        if appearance is not None
                        else 300.0
                    ),
                    "roughness": float(
                        appearance.roughness if appearance is not None else 0.8
                    ),
                    "metallic": float(
                        appearance.metallic if appearance is not None else 0.0
                    ),
                    "color": appearance.color
                    if appearance is not None
                    else None,
                }
            )
        return specs

    def update_scene_from_doc(self):
        """
        Updates the entire scene content from the document. This is the main
        entry point for refreshing the 3D view.
        """
        if not self._get_gl_initialized():
            return
        if not self._scene.texture_renderer:
            return

        t_update_start = time.perf_counter()
        logger.debug("Canvas3D: Updating scene from document.")

        # Theme/color updates only need to happen once per theme change
        if self._theme_resolver.theme_is_dirty:
            self._theme_resolver.update_theme_and_colors()
        if not self._theme_resolver.color_set:
            logger.warning("Cannot update scene, color set not resolved.")
            return

        viewport = self._get_viewport()

        # Update cylinder renderers and camera based on layer rotary state
        any_rotary = any(layer.rotary_enabled for layer in self.doc.layers)
        self._mark_scene_dirty()
        if (
            self._scene.had_rotary_layers
            and not any_rotary
            and self._get_camera_available()
        ):
            self._reset_view(ViewDirection.ISO)
        self._scene.had_rotary_layers = any_rotary

        world_to_visual = np.identity(4, dtype=np.float32)
        world_to_cyl_local = np.identity(4, dtype=np.float32)

        machine = self._context.machine
        if machine:
            ms = viewport.margin_shift
            wcs = viewport.wcs_offset_mm
            world_to_visual[0, 3] = ms[0, 3]
            world_to_visual[1, 3] = ms[1, 3]
            world_to_visual[2, 3] = wcs[2]

            asm = machine.assembly
            if asm.has_rotary:
                self._scene.set_cylinder_transform(
                    asm.cylinder_base_transform()
                )
            else:
                self._scene.set_cylinder_transform(np.eye(4, dtype=np.float64))

        laser_dot_widths_mm: dict[str, float] = {}
        if machine:
            for head in machine.heads:
                if isinstance(head, LaserHead):
                    spot_x, _spot_y = LaserHead.get_spot_size(head)
                    laser_dot_widths_mm[head.uid] = spot_x

        layer_configs: dict[str, LayerRenderConfig] = {}
        for layer in self.doc.layers:
            axis_position = 0.0
            reverse = False
            axis_position_3d = None
            cylinder_dir = None
            if layer.rotary_enabled and machine:
                cfg = resolve_layer_rotary(layer, machine)
                module = cfg.module
                if module is not None:
                    mapping = KinematicMapping.from_rotary_module(
                        module,
                        layer.rotary_diameter,
                        apply_gear_ratio=False,
                    )
                    if mapping is not None:
                        axis_position = mapping.axis_position
                        axis_position_3d = tuple(
                            mapping.axis_position_3d.tolist()
                        )
                        cylinder_dir = tuple(mapping.cylinder_dir.tolist())
                        reverse = mapping.reverse
            layer_configs[layer.uid] = LayerRenderConfig(
                rotary_enabled=layer.rotary_enabled,
                rotary_diameter=layer.rotary_diameter,
                axis_position=axis_position,
                reverse=reverse,
                axis_position_3d=axis_position_3d,
                cylinder_dir=cylinder_dir,
            )

        render_config = RenderConfig3D(
            world_to_visual=world_to_visual,
            world_to_cyl_local=world_to_cyl_local,
            layer_configs=layer_configs,
            laser_dot_widths_mm=laser_dot_widths_mm,
            stock_specs=self._build_stock_specs(),
        )

        self._schedule_scene_preparation(render_config.to_dict())

        t_update_elapsed = (time.perf_counter() - t_update_start) * 1000
        if t_update_elapsed > 5:
            logger.debug(
                f"update_scene_from_doc took {t_update_elapsed:.1f}ms"
            )

    def _schedule_scene_preparation(
        self,
        render_config_dict: dict,
    ):
        if (
            not self._get_gl_initialized()
            or self._theme_resolver.color_set is None
        ):
            return

        job_handle = self._current_job_handle
        if job_handle is None and not render_config_dict.get("stock_specs"):
            # Nothing to compile at all; drop any previously compiled
            # content so a deleted stock does not linger.
            logger.debug("No job artifact and no stock, skipping compilation.")
            if self._compiled_artifact is not None:
                self._compiled_artifact = None
                self._mark_artifact_dirty()
                self._request_render()
            return

        if self._scene_preparation_task:
            self._scene_preparation_task.cancel()
            self._scene_preparation_task = None
            logger.debug(
                "Cancelled in-progress compilation, scheduling new one."
            )

        if job_handle is None:
            # Stock is document content and must render even without
            # an assembled job; compile a stock-only scene.
            logger.debug("Scheduling stock-only compilation.")
            self._run_scene_preparation_task(
                compile_stock_scene,
                render_config_dict,
            )
            return

        logger.debug("Scheduling scene compilation task.")
        self._compiled_job_generation = job_handle.generation_id
        self._run_scene_preparation_task(
            compile_scene_from_job,
            self._context.artifact_store,
            job_handle.to_dict(),
            render_config_dict,
        )

    def _run_scene_preparation_task(self, fn, *args):
        """Runs a scene compilation function in the worker thread."""
        task_key = (id(self), "prepare-3d-scene-vertices")
        self._scene_preparation_task = task_mgr.run_thread(
            fn,
            *args,
            key=task_key,
            when_done=self._on_scene_prepared,
        )
